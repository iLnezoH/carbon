import numpy as np

E = 3
tau = 1


def symbolic(v):
    if v == 0:
        return '→'
    if v < 0:
        return '↘'
    if v > 0:
        return '↗'


def retrieve_shadow_attractor(X, E=E):
    l = X.shape[0] - E + 1
    mx = np.empty((E, l))
    for i in range(E):
        mx[E - i - 1] = X[i: i+l]
        # mx[i] = X[i: i+l]

    return mx.T


def get_rate(mx, E=E):
    rate = np.empty((mx.shape[0], E - 1))
    for i in range(1, E):
        rate[:, i - 1] = mx[:, i] / mx[:, i-1] - 1

    return rate


def get_dist(mx, E=E):
    l = mx.shape[0]
    dist = np.empty((l, l))
    dist.fill(float('inf'))
    for i in range(l):
        for j in range(i + 1, l):
            dist[i, j] = np.linalg.norm(mx[i] - mx[j])
            dist[j, i] = dist[i, j]

    return dist


def get_nn_weight(dist, no=E+1):
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


def get_pc_matrix(X, Y_hat, Y, E=E):
    E2 = E**2
    hit = np.zeros((E2, E2))
    all = np.zeros((E2, E2))
    right_ind = np.logical_and.reduce((Y_hat == Y).T)
    for i in range(X.shape[0]):
        s_x = X[i]
        s_y = Y[i]
        x_ind = int((s_x[1] + 1) * 3 + (s_x[0] + 1))
        y_ind = int((s_y[1] + 1) * 3 + (s_y[0] + 1))

        all[x_ind, y_ind] += 1

        if right_ind[i]:
            hit[x_ind, y_ind] += 1

    accuracy = np.zeros(all.shape)
    for i in range(all.shape[0]):
        for j in range(all.shape[1]):
            if all[i, j] > 0:
                accuracy[i, j] = hit[i, j] / all[i, j]
            # else:
                # accuracy[i, j] = -1

    return accuracy


def extract_casuality(m):
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

    return positive, negative, dark


def PC(X, Y, E=E, tau=tau, no=E+1):
    mx = retrieve_shadow_attractor(X)
    my = retrieve_shadow_attractor(Y)

    rate_x = get_rate(mx)
    rate_y = get_rate(my)

    dist_x = get_dist(mx)
    dist_y = get_dist(my)

    nn_w_x = get_nn_weight(dist_x)
    nn_w_y = get_nn_weight(dist_y)

    _, S_X = get_signature(nn_w_x, rate_x)
    _, S_Y = get_signature(nn_w_y, rate_y)
    _, S_X_hat = get_signature(nn_w_y, rate_x)
    _, S_Y_hat = get_signature(nn_w_x, rate_y)

    X_Y = get_pc_matrix(S_Y, S_X_hat, S_X)
    Y_X = get_pc_matrix(S_X, S_Y_hat, S_Y)

    return extract_casuality(X_Y), extract_casuality(Y_X)
