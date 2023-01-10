import math
import time

import numpy as np
from numba import cuda
from tifffile import imsave

from projection.ProjectionMatrix import spin_matrices_from_xml, to_point_and_matrix, uv_from_ijk_format, \
    canonical_form_istar


@cuda.jit
def project_ellipsoids_kernel(_invAR, _S, _mu, _B, _e, _out):
    """
    Calculate analytic forward projections of ellipsoids (mu, Sigma) along geometry (invAR, S)
    :param _invAR: 3x3 projection matrix normalized to voxels. Equals (R.T @ inv(K)) shape (p, 3, 3)
    :param _S: Source points matching invAR. shape (p, 3)
    :param _mu: Mean vector of gaussian distribution. shape (n, 3)
    :param _B: Eigenvectors derived from Covariance matrix of Gaussian shape (n, 3, 3) !Row-vectors!
    :param _e: Root of eigenvalues corresponding to B. shape (n, 3)
    :param _out: Output projections in desired resolution d, shaped (p, d, d)
    :return: None
    """
    idx, v, u = cuda.grid(ndim=3)
    if idx < _out.shape[0] and v < _out.shape[1] and u < _out.shape[2]:
        # initialize output with zero
        _out[idx, v, u] = 0

        # projection matrices assume 0-975; so adapt ranges
        v_spin = v * (975 / (_out.shape[1] - 1))
        u_spin = u * (975 / (_out.shape[2] - 1))

        # create normalized ray direction r = np.norm(invAR @ uv)
        rx = _invAR[idx, 0, 0] * u_spin + _invAR[idx, 0, 1] * v_spin + _invAR[idx, 0, 2]
        ry = _invAR[idx, 1, 0] * u_spin + _invAR[idx, 1, 1] * v_spin + _invAR[idx, 1, 2]
        rz = _invAR[idx, 2, 0] * u_spin + _invAR[idx, 2, 1] * v_spin + _invAR[idx, 2, 2]
        len_r = math.sqrt(math.pow(rx, 2) + math.pow(ry, 2) + math.pow(rz, 2))
        rx, ry, rz = rx / len_r, ry / len_r, rz / len_r

        for n in range(_e.shape[0]):
            # transform into ellipsoids eigenvector basis and scale by eigenvalue
            _e0_inv, _e1_inv, _e2_inv = 1 / _e[n, 0], 1 / _e[n, 1], 1 / _e[n, 2]
            dx = (_B[n, 0, 0] * rx + _B[n, 0, 1] * ry + _B[n, 0, 2] * rz) * _e0_inv
            dy = (_B[n, 1, 0] * rx + _B[n, 1, 1] * ry + _B[n, 1, 2] * rz) * _e1_inv
            dz = (_B[n, 2, 0] * rx + _B[n, 2, 1] * ry + _B[n, 2, 2] * rz) * _e2_inv
            len_d = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2) + math.pow(dz, 2))
            dx, dy, dz = dx / len_d, dy / len_d, dz / len_d

            # transform and squish source points centered around mean
            _Sx, _Sy, _Sz = _S[idx, 0] - _mu[n, 0], _S[idx, 1] - _mu[n, 1], _S[idx, 2] - _mu[n, 2]
            ex = (_B[n, 0, 0] * _Sx + _B[n, 0, 1] * _Sy + _B[n, 0, 2] * _Sz) * _e0_inv
            ey = (_B[n, 1, 0] * _Sx + _B[n, 1, 1] * _Sy + _B[n, 1, 2] * _Sz) * _e1_inv
            ez = (_B[n, 2, 0] * _Sx + _B[n, 2, 1] * _Sy + _B[n, 2, 2] * _Sz) * _e2_inv

            # discriminant decides number of intersection points
            _b = 2 * (dx * ex + dy * ey + dz * ez)
            _c = (ex * ex + ey * ey + ez * ez) - 1
            disc = _b * _b - 4 * _c

            # if discriminant is negative, ray and sphere don't intersect
            if disc > 0:
                tmax = (-_b + math.sqrt(disc)) / 2
                tmin = (-_b - math.sqrt(disc)) / 2
                # Calculate intersection points and scale by radii
                P0x, P0y, P0z = _e[n, 0] * (ex + tmin * dx), _e[n, 1] * (ey + tmin * dy), _e[n, 2] * (ez + tmin * dz)
                P1x, P1y, P1z = _e[n, 0] * (ex + tmax * dx), _e[n, 1] * (ey + tmax * dy), _e[n, 2] * (ez + tmax * dz)
                # write length to detector p(u,v) = end-start
                _out[idx, v, u] += math.sqrt(math.pow(P1x - P0x, 2) + math.pow(P1y - P0y, 2) + math.pow(P1z - P0z, 2))


def project_ellipsoids_gpu(MU, Sigmas, res: int, matrices: np.ndarray):
    """

    :param MU: mean vectors (n, 3)
    :param Sigmas: Covariance matrices (n, 3, 3)
    :param res: desired resolution of output images
    :param matrices: in uv_from_xyz convention (mapping zero-centered mm units to detector coords in range [0, 976])
    :return: projection images. array in shape (n, res, res)
    """
    # selecting the main gpu
    cuda.select_device(0)

    # cast to float
    MU, Sigmas = MU.astype(np.float32), Sigmas.astype(np.float32)

    # Packing ellipsoids in Eigenvectors /-values representation
    B, E = np.zeros_like(Sigmas), np.zeros((Sigmas.shape[0], 3))
    for i in range(Sigmas.shape[0]):
        B1, e1, _ = np.linalg.svd(Sigmas[i])
        B[i] = B1.T  # convert to row-eigenvectors
        E[i] = np.sqrt(e1)  # convert to radii (iso-variance instead of std-dev)

    # Define detector resolution and number of projections (can be chosen arbitrary)
    out_shape = (res, res)
    n_views = matrices.shape[0]

    # Convert to uv_from_ijk and get forward matrices format
    matrices = uv_from_ijk_format(matrices)
    invAR, S = canonical_form_istar(matrices)

    # Copy arrays to device
    invAR_gpu = cuda.to_device(invAR)
    S_gpu = cuda.to_device(S)
    mu_gpu = cuda.to_device(MU)
    B_gpu = cuda.to_device(B)
    e_gpu = cuda.to_device(E)

    # Allocate output memory
    out_gpu = cuda.device_array((n_views, *out_shape))

    # Configure blocks
    threadsperblock = (8, 8, 8)
    blockspergrid_x = int(math.ceil(n_views / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(out_shape[0] / threadsperblock[1]))
    blockspergrid_z = int(math.ceil(out_shape[1] / threadsperblock[2]))
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Start kernel project_analytic_gpu(_invAR, _S, _mu, _B, _e, out)
    # print("starting gpu kernel...")
    project_ellipsoids_kernel[blockspergrid, threadsperblock](invAR_gpu, S_gpu, mu_gpu, B_gpu, e_gpu, out_gpu)
    # print("done.")

    # Obtain result
    out = out_gpu.copy_to_host()
    return out


if __name__ == '__main__':
    # selecting the main gpu
    cuda.select_device(0)

    # Define geometry used in matrices
    vox_size, vol_shape = np.array([0.313, 0.313, 0.313]), np.array([512, 512, 512])

    # Define ellipsoid 1-3; eigenvalues interpreted as radii
    mu1, Sigma1 = np.array([155.5, 155.5, 155.5]), np.array([[50., 0., 0.],
                                                             [0., 100., 0.],
                                                             [0., 0., 50.]])
    mu2, Sigma2 = np.array([255.5, 255.5, 255.5]), np.array([[50., 0., 0.],
                                                             [0., 100., 0.],
                                                             [0., 0., 50.]])
    mu3, Sigma3 = np.array([355.5, 355.5, 355.5]), np.array([[50., 0., 0.],
                                                             [0., 100., 0.],
                                                             [0., 0., 50.]])

    # Packing ellipsoids in (mu, Sigma) representation
    B1, e1, _ = np.linalg.svd(Sigma1)
    B2, e2, _ = np.linalg.svd(Sigma2)
    B3, e3, _ = np.linalg.svd(Sigma3)
    B1, B2, B3 = B1.T, B2.T, B3.T  # convert to row-eigenvectors
    MU = np.array([mu1, mu2, mu3])
    B = np.array([B1, B2, B3])
    E = np.array([e1, e2, e3])

    # Define detector resolution and number of projections (can be chosen arbitrary)
    out_shape = (976, 976)
    n_views = 400

    # Compute invAR and S
    P, _ = spin_matrices_from_xml(r"projection/SpinProjMatrix.xml")
    P = P[::(400 // n_views)]
    assert P.shape[0] == n_views
    invAR, S = to_point_and_matrix(P, vox_size, vol_shape, [0, 0, 0], 976)

    # Copy arrays to device
    print("uploading.")
    invAR_gpu = cuda.to_device(invAR)
    S_gpu = cuda.to_device(S)
    mu_gpu = cuda.to_device(MU)
    B_gpu = cuda.to_device(B)
    e_gpu = cuda.to_device(E)

    # Allocate output memory
    out_gpu = cuda.device_array((n_views, *out_shape))

    # Configure blocks
    print("configuring.")
    threadsperblock = (8, 8, 8)
    blockspergrid_x = int(math.ceil(n_views / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(out_shape[0] / threadsperblock[1]))
    blockspergrid_z = int(math.ceil(out_shape[1] / threadsperblock[2]))
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)

    # Start kernel project_analytic_gpu(_invAR, _S, _mu, _B, _e, out)
    print("invoke gpu kernels.")
    start = time.time()
    project_ellipsoids_kernel[blockspergrid, threadsperblock](invAR_gpu, S_gpu, mu_gpu, B_gpu, e_gpu, out_gpu)
    print("took {:.2f}s.".format(time.time() - start))

    # Obtain result
    print("saving.")
    out = out_gpu.copy_to_host()
    imsave("multiple_ellipsoids_gpu.tif", out.astype(np.float16))
    print("done.")
