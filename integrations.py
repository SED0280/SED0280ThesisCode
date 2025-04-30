import numpy as np
from scipy.linalg import eigh_tridiagonal
import math


def w(xzeros, j):
    prod = np.polynomial.Polynomial.fromroots(
        [xzeros[i] for i in range(len(xzeros)) if i != j])
    bottom = prod(xzeros[j])
    top = prod.integ()(1)-prod.integ()(-1)
    return top/bottom


def getWeights(xzeros):
    return np.array([w(xzeros, i) for i in range(len(xzeros))])


def legendre(x, n):
    alphak = 0
    ret = [np.zeros(len(x)), np.ones(len(x))]
    for k in range(1, n+1):
        betak = 1/(4-(k)**(-2))
        pi = (x-alphak)*ret[-1]-betak*ret[-2]
        ret.append(pi)
    return np.array(ret[1:])


def legendre_normal(x, n):
    beta0 = 2
    ret = [np.zeros(x.shape), np.ones(x.shape)/math.sqrt(beta0)]
    alphak = 0

    betak = 1/(4-(1)**(-2))

    pi = ((x - alphak) * ret[-1] - math.sqrt(beta0) * ret[-2])/math.sqrt(betak)
    ret.append(pi)

    for k in range(2, n+1):
        betak_minus_one = 1/(4-(k-1)**(-2))
        betak = 1/(4-(k)**(-2))
        pi = ((x - alphak) *
              ret[-1] - math.sqrt(betak_minus_one) * ret[-2])/math.sqrt(betak)
        ret.append(pi)
    return np.array(ret[1:])


def legendre_int(x, n, a, b):
    alphak = 0
    ret = [np.zeros(len(x)), np.ones(len(x))]
    for k in range(2, n+1):
        betak = 1/(4-(k-1)**(-2))
        pi = ((((a+b)/(a-b) + (2/(b-a))*x) - alphak)*ret[-1]-betak*ret[-2])
        ret.append(pi)
    ret = np.array(ret[1:]) * math.sqrt(2/(b-a))
    return ret


def legendre_normal_int(x, n, a, b):
    beta0 = 2
    ret = [np.zeros(x.shape), np.ones(x.shape)/math.sqrt(beta0)]
    alphak = 0

    betak = 1/(4-(1)**(-2))

    pi = ((((a+b)/(a-b) + 2/(b-a)*x) - alphak) *
          ret[-1] - math.sqrt(beta0) * ret[-2])/math.sqrt(betak)
    ret.append(pi)
    for k in range(2, n+1):
        betak_minus_one = 1/(4-(k-1)**(-2))
        betak = 1/(4-(k)**(-2))

        pi = ((((a+b)/(a-b) + 2/(b-a)*x) - alphak) *
              ret[-1] - math.sqrt(betak_minus_one) * ret[-2])/math.sqrt(betak)
        ret.append(pi)
    ret = np.array(ret[1:]) * math.sqrt(2/(b-a))
    return ret


def nodes_weights(n):
    Beta0 = 2
    main_diag = np.zeros(n)
    off_diag = np.array([math.sqrt(1/(4 - k**(-2))) for k in range(1, n)])

    eigenvalues, eigenvectors = eigh_tridiagonal(main_diag, off_diag)

    lam = Beta0 * (eigenvectors[0]**2)
    return eigenvalues, lam


def nodes_weights_int(n, a, b):
    nodes, weights = nodes_weights(n)

    nodes = (a+b)/2 + (b-a)/2*nodes
    weights = weights * (b-a)/2
    return nodes, weights


def quadrature(f, n):

    nodes, weights = nodes_weights(n)
    return sum(weights*f(nodes))


def quadrature_int(f, n, a, b):
    nodes, weights = nodes_weights_int(n, a, b)

    return sum(weights*f(nodes))


def generate2DLegendre(x, y, n):
    alphak = 0
    retx = [np.zeros(np.shape(x)), np.ones(np.shape(x))]
    rety = [np.zeros(np.shape(y)), np.ones(np.shape(y))]
    for k in range(2, n+1):
        betak = 1/(4-(k-1)**(-2))
        pix = ((x-alphak)*retx[-1]-betak*retx[-2])
        piy = ((y-alphak)*retx[-1]-betak*retx[-2])
        retx.append(pix)
        rety.append(piy)
    return np.array(np.array(retx[1:])*np.array(rety[1:]))


def quadrature_2d(f, nx, ny):
    xnodes, xweights = nodes_weights(nx)
    ynodes, yweights = nodes_weights(ny)
    xnodes, ynodes = np.meshgrid(xnodes, ynodes)

    return yweights @ f(xnodes, ynodes) @ xweights


def quadrature_2d_int(f, nx, ny, a, b, c, d):
    xnodes, xweights = nodes_weights_int(nx, a, b)
    ynodes, yweights = nodes_weights_int(ny, c, d)
    xnodes, ynodes = np.meshgrid(xnodes, ynodes)

    return yweights @ f(xnodes, ynodes) @ xweights
