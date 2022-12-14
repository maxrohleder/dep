import time

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch
import xmltodict
from tifffile import imwrite


def spin_matrices_from_xml(path_to_projection_matrices):
    assert str(path_to_projection_matrices).endswith('.xml')
    with open(path_to_projection_matrices) as fd:
        matrices = xmltodict.parse(fd.read())['hdr']['ElementList']['PROJECTION_MATRICES']
        projmat = [np.array(matrices[key].split(" "), order='C').reshape((3, 4)) for key in matrices.keys()]
    return np.asarray(projmat).astype(np.float32)


class AnalyticRenderer(nn.Module):
    def __init__(self, P, dec_shape=(976, 976)):
        super(AnalyticRenderer, self).__init__()
        self.nviews = P.shape[0]
        self.P = P
        self.outshape = (P.shape[0], *dec_shape)

    def forward(self, M, S):
        """
        creates a rendered image from given ellipsoid parametrisations
        :param M: centroid of ellipsoids in shape (N, 3)
        :param S: covariance of ellipsoids in shape (N, 3, 3)
        :return: rendered ellipsoid projection images
        """
        assert M.shape[0] == S.shape[0]
        # decompose into eigenvector and eigenvalues

        # create output image
        out = torch.zeros(self.outshape)

        # loop over pixel images
        for i, P in enumerate(self.P):
            print(f"rendering {i}th view")

            # calculate RTKinv and source position in world coordinates
            RTKinv = torch.linalg.inv(P[:3, :3])
            C = - RTKinv @ P[:3, 3]

            # loop over ellipsoids
            for n, (mu, sigma) in enumerate(zip(M, S)):
                print(f"starting projection of {n}th ellipsoid")
                sinv = torch.linalg.inv(sigma)

                # source position in ellipsoid eigenbasis
                Cn = sinv @ (C - mu)

                for u in range(self.outshape[1]):
                    for v in range(self.outshape[2]):

                        # calculate ray direction
                        r = RTKinv @ torch.as_tensor([u, v, 1], dtype=RTKinv.dtype)
                        r /= torch.linalg.norm(r)

                        # compute ray direction in ellipsoids eigenbasis
                        rn = sinv @ r
                        rn /= torch.linalg.norm(rn)

                        # discriminant decides number of intersection points
                        _b = 2 * torch.dot(Cn, rn)
                        _c = torch.dot(Cn, Cn) - 1
                        disc = _b * _b - 4 * _c

                        # if discriminant is negative, ray and sphere don't intersect
                        if disc > 0:
                            tmax = (-_b + np.sqrt(disc)) / 2
                            tmin = (-_b - np.sqrt(disc)) / 2
                            # Calculate intersection points and scale by radii

                            P0 = sigma @ (Cn + tmax * rn)
                            P1 = sigma @ (Cn + tmin * rn)

                            # write length to detector p(u,v) = end-start
                            out[i, u, v] += torch.linalg.norm(P0 - P1)
            if torch.all(out[i] == 0):
                print(f"projection {i} is empty")
        return out



if __name__ == '__main__':

    mu1, Sigma1 = np.array([-25., 0., 0.]), np.array([[10., 0., 0.],
                                                     [0., 50., 0.],
                                                     [0., 0., 10.]])

    mu2, Sigma2 = np.array([25., 0., 0.]), np.array([[10., 0., 0.],
                                                     [0., 50., 0.],
                                                     [0., 0., 10.]])

    # pack together in shape (n, 3) and (n, 3, 3)
    MU = torch.tensor([mu1, mu2])
    S = torch.tensor([Sigma1, Sigma2])

    # define detector shape (this adapts projection matrices
    dec_shape = (200, 200)
    P = spin_matrices_from_xml(r"SpinProjMatrix.xml")
    P = [np.diag([dec_shape[0] / 976, dec_shape[1] / 976, 1]) @ p for p in P[::40]]
    P = torch.as_tensor(P)

    # instantiate a renderer and generate projections
    A = AnalyticRenderer(P=P, dec_shape=dec_shape)
    start = time.time()
    img = A(MU, S)
    print(f'duration {time.time() - start}')

    # takes 54s for an output of shape torch.Size([10, 200, 200])
    imwrite("test_render.tif", img.numpy())
    print(img.shape)
    plt.imshow(img[0])
    plt.show()







