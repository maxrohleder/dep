import xmltodict
import numpy as np
from matplotlib import pyplot as plt


def spin_matrices_from_xml(path_to_projection_matrices):
    assert str(path_to_projection_matrices).endswith('.xml')
    with open(path_to_projection_matrices) as fd:
        matrices = xmltodict.parse(fd.read())['hdr']['ElementList']['PROJECTION_MATRICES']
        projmat = [np.array(matrices[key].split(" "), order='C').reshape((3, 4)) for key in matrices.keys()]
    return np.asarray(projmat).astype(np.float32)


def RQfactorize(A):
    A_flip = np.flip(A, axis=0).T  # shape 4x3
    Q_flip, R_flip = np.linalg.qr(A_flip)  # shapes 4x4, 4x3

    # make camera matrix look into positive direction!
    for i in range(3):
        if R_flip[i, i] < 0:
            R_flip[i, :] *= -1
            Q_flip[:, i] *= -1

    assert R_flip[0, 0] > 0
    assert R_flip[1, 1] > 0
    assert R_flip[2, 2] > 0

    R = np.flip(np.flip(R_flip.T, axis=1), axis=0)
    Q = np.flip(Q_flip.T, axis=0)
    return R, Q


def decompose(P, scaleIntrinsic=False):
    # 1. Split P in two components --> P = [M|p3] = [KR|Kt] = [KR'|-R'C] where C is the source point
    M, p3 = P[:3, :3].copy(), P[:, 3].copy()

    # 2. RQ - factorisation of m. K is upper triangle matrix, R is orthogonal
    K, R = RQfactorize(M)

    # 3. calculate translation by back substitution. p3 is Kt
    t = np.zeros(3)
    t[2] = p3[2] / K[2, 2]
    t[1] = (p3[1] - K[1, 2] * t[2]) / K[1, 1]
    t[0] = (p3[0] - K[0, 1] * t[1] - K[0, 2] * t[2]) / K[0, 0]

    # 5. Optionally scale the camera intrinsic
    if scaleIntrinsic:
        K *= (1 / K[2, 2])

    return K, R, t


def C_from_P(P):
    '''
    Computes the Camera Center C from a given Projection Matrix P
    :param P: projection matrix P in shape (3,4)
    :return: C as vector in shape (3,)
    '''
    return - np.linalg.inv(P[:3, :3]) @ P[:3, 3]

if __name__ == '__main__':
    matrices = spin_matrices_from_xml('projection/SpinProjMatrix5.xml')

    sources = np.asarray([C_from_P(P) for P in matrices])
    ts = np.asarray([-decompose(P)[1].T @ decompose(P)[2] for P in matrices])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(*sources.T, label='C')
    ax.scatter(*ts.T, label='t')
    ax.legend()
    plt.show()



