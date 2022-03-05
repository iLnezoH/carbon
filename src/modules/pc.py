import numpy as np
from functools import reduce
from scipy import integrate


def symbolic(v):
    if v == 0:
        return '→'
    if v < 0:
        return '↘'
    if v > 0:
        return '↗'


def get_symbolic_ind(x):
    return int(reduce(lambda x, y: x + 3 ** y[0] * (y[1] + 1),
                      enumerate(x), 0))


def retrieve_shadow_attractor(X, E, tau):
    l = X.shape[0] - (E - 1) * tau
    mx = np.empty((E, l))
    for i in range(E):
        mx[E - i - 1] = X[i * tau: i * tau + l]
        # mx[i] = X[i * tau: i * tau + l]

    return mx.T


def get_rate(mx, E):
    rate = np.empty((mx.shape[0], E - 1))
    for i in range(1, E):
        rate[:, i - 1] = mx[:, i] / mx[:, i-1] - 1

    return rate


def get_dist(mx, E, p=2):
    l = mx.shape[0]
    dist = np.empty((l, l))
    dist.fill(float('inf'))
    for i in range(l):
        for j in range(i + 1, l):
            dist[i, j] = np.linalg.norm(mx[i] - mx[j], ord=p)
            dist[j, i] = dist[i, j]

    return dist


def get_nn_weight(dist, no):
    nn_dist = np.sort(dist, axis=1)[:, :no]
    nn_index = np.argsort(dist, axis=1)[:, : no]
    nn_w = np.empty(nn_dist.shape)

    for i in range(nn_w.shape[0]):
        row = nn_dist[i]
        e_row = np.exp(np.max(row) - row)
        nn_w[i] = e_row / np.sum(e_row)

    return nn_w, nn_index


def get_signature(nn_w, rate):
    nn_w, indexes = nn_w
    l = nn_w.shape[0]
    step = rate.shape[1]
    S = np.empty((l, step))
    for t in range(l):
        S[t] = np.dot(nn_w[t], rate[indexes[t]])

    _S = np.empty(S.shape)
    _S[S > 0] = 1
    _S[S < 0] = -1
    _S[S == 0] = 0

    return S, _S


def get_accuracy(X, Y_hat, Y, E):
    n = 3**(E - 1)
    hit = np.zeros((n, n))
    all = np.zeros((n, n))
    right_ind = np.logical_and.reduce((Y_hat == Y).T)
    for i in range(X.shape[0]):

        x_ind = get_symbolic_ind(X[i])
        y_ind = get_symbolic_ind(Y[i])

        all[x_ind, y_ind] += 1

        if right_ind[i]:
            hit[x_ind, y_ind] += 1

    '''
    accuracy = np.zeros(all.shape)
    for i in range(all.shape[0]):
        for j in range(all.shape[1]):
            if all[i, j] > 0:
                accuracy[i, j] = hit[i, j] / all[i, j]
            # else:
                # accuracy[i, j] = -1
    '''

    # return accuracy
    return hit, all


def get_pc_matrix(S_X, S_Y, E, p):
    Xv, Xp = S_X
    Yv, Yp = S_Y

    n = 3 ** (E - 1)
    pc_m = np.zeros((n, n))

    def erf(x0, x1):
        return integrate.quad(lambda x: np.e ** (- x ** 2), x0, x1)[0] / np.pi ** (1/2)

    for i in range(Xv.shape[0]):
        x_ind = get_symbolic_ind(Xp[i])
        y_ind = get_symbolic_ind(Yp[i])

        r = np.linalg.norm(Yv[i], ord=p) / np.linalg.norm(Xv[i], ord=p)

        pc_m[x_ind, y_ind] += erf(-r, r)

    return pc_m


def extract_casuality(m, mL):
    positive = 0
    negative = 0
    dark = 0
    l = m.shape[0]
    for i in range(l):
        for j in range(l):
            if i == j:
                positive += m[i, j]
            elif i + j == l - 1:
                negative += m[i, j]
            else:
                dark += m[i, j]

    return positive / mL, negative / mL, dark / mL


def PC(X, Y, E, tau, no=None, p=2):

    if no is None:
        no = E + 1
    mx = retrieve_shadow_attractor(X, E=E, tau=tau)
    my = retrieve_shadow_attractor(Y, E=E, tau=tau)

    rate_x = get_rate(mx, E=E)
    rate_y = get_rate(my, E=E)

    dist_x = get_dist(mx, E=E, p=p)
    dist_y = get_dist(my, E=E, p=p)

    nn_w_x = get_nn_weight(dist_x, no=no)
    nn_w_y = get_nn_weight(dist_y, no=no)

    S_X = get_signature(nn_w_x, rate_x)
    S_Y = get_signature(nn_w_y, rate_y)

    # S_X_hat = get_signature(nn_w_y, rate_x)
    # S_Y_hat = get_signature(nn_w_x, rate_y)

    # X_Y_acc = get_accuracy(S_Y[1], S_X_hat[1], S_X[1], E=E)
    # Y_X_acc = get_accuracy(S_X[1], S_Y_hat[1], S_Y[1], E=E)

    X_Y_pc = get_pc_matrix(S_X, S_Y, E=E, p=p)
    Y_X_pc = get_pc_matrix(S_Y, S_X, E=E, p=p)

    mL = S_X[0].shape[0]

    return extract_casuality(X_Y_pc, mL), extract_casuality(Y_X_pc, mL), [X_Y_pc, Y_X_pc]
