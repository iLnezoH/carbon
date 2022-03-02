import numpy as np

E = 3

MODAL_INDEX = {
    'DD': 0,
    'SD': 1,
    'ID': 2,
    'DS': 3,
    'SS': 4,
    'IS': 5,
    'DI': 6,
    'SI': 7,
    'II': 8
}


def rebuild_P_S(S, dim=E):
    row = S.shape[0] - dim + 1
    n = S.shape[1]
    P = np.zeros((n, row, dim))
    for i in range(row):
        P[:, i, :] = S[i:i+dim, :][::-1].transpose(1, 0)

    return P


def get_P_Dis(P):
    n, row, _ = P.shape
    Dis = np.zeros((n, row, row))
    for i in range(row):
        for j in range(i+1, row):
            Dis[:, i, j] = np.linalg.norm(P[:, i] - P[:, j], axis=1)
            Dis[:, j, i] = Dis[:, i, j]

    Dis = np.array(
        [np.fill_diagonal(dis, val=float("inf")) or dis for dis in Dis])

    return Dis


def get_nearest_Dis(Dis, no=4):
    return np.sort(Dis, axis=2)[:, :, :4]


def get_nearest_Index(Dis, no=4):
    Index = []
    for dis in Dis:
        Index.append(np.argsort(dis, axis=1)[:, :4])

    return np.array(Index)


def get_Phase_rate(Phase):
    n, r, c = Phase.shape
    Rate = np.zeros((n, r, c - 1))
    for i in range(1, c):
        Rate[:, :, i - 1] = (Phase[:, :, i] -
                             Phase[:, :, i-1]) / Phase[:, :, i-1]

    return Rate


def get_nearest_Rate(n_Index, Rate):
    n, r, no = n_Index.shape
    c = Rate.shape[2]
    n_ind = np.tile(np.arange(n), (r * no * c, 1)).T.reshape(-1)
    r_ind = n_Index.reshape((n, -1, 1))
    r_ind = np.tile(r_ind, (1, c)).reshape(-1)
    c_ind = np.tile(np.arange(c), no * r * n)

    Ind = (n_ind, r_ind, c_ind)
    return Rate[Ind].reshape((n, r, no * c))


def get_nearest_dis_W(Dis):
    e_Dis = np.exp(-Dis)
    total = np.sum(e_Dis, axis=2)
    total = total.reshape(total.shape[0], total.shape[1], -1)
    return e_Dis / total


def get_Modal(n_R, n_W):
    M1 = np.sum(n_W * n_R[:, :, ::2], axis=2)
    M2 = np.sum(n_W * n_R[:, :, 1::2], axis=2)

    Modal = np.array([M1, M2]).transpose(1, 0, 2)

    r = np.zeros(Modal.shape, dtype=str)
    r[Modal < 0] = 'D'
    r[Modal == 0] = 'S'
    r[Modal > 0] = 'I'

    Modal_s = np.array([[''.join(t) for t in list(zip(*item))] for item in r])

    return Modal_s


def get_modal(n_r, n_w):
    m1 = np.sum(n_w * n_r[:, ::2], axis=1)
    m2 = np.sum(n_w * n_r[:, 1::2], axis=1)

    Modal = np.array([m1, m2])

    r = np.empty(Modal.shape, dtype='str_')
    r[Modal < 0] = 'D'
    r[Modal == 0] = 'S'
    r[Modal > 0] = 'I'

    modal_s = np.array([''.join(t) for t in list(zip(*r))])

    return modal_s


def get_nearest_rate(n_index, rate):
    r, no = n_index.shape
    c = rate.shape[1]
    r_ind = np.tile(n_index.reshape(-1, 1), (1, c)).reshape(-1)
    c_ind = np.tile(np.arange(c), no * r)

    Ind = (r_ind, c_ind)

    return rate[Ind].reshape(r, no * c)


def get_Predict(n_Index, Rate, n_W):

    n, r, c = n_Index.shape
    Prediction = np.empty((n, n, r), dtype='U2')

    for i in range(n):
        for j in range(n):
            n_rate = get_nearest_rate(n_Index[i], Rate[j])
            p_modal = get_modal(n_rate, n_W[i])

            Prediction[i, j] = p_modal

    return Prediction


def get_Accuracy(Prediction, E=E, save_to="outputs/"):
    n, _, r = Prediction.shape
    pos_A = -np.ones((n, n))
    neg_A = -np.ones((n, n))
    dark_A = -np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            x_modal = Prediction[i, i]
            t_modal = Prediction[j, j]
            p_modal = Prediction[i, j]
            pos_a, neg_a, dark_a = _get_count_hit(x_modal, t_modal, p_modal, E)
            pos_A[i, j] = pos_a
            neg_A[i, j] = neg_a
            dark_A[i, j] = dark_a

    np.savetxt(save_to + 'pos_A.csv', pos_A, delimiter=',')
    np.savetxt(save_to + 'neg_A.csv', neg_A, delimiter=',')
    np.savetxt(save_to + 'dark_A.csv', dark_A, delimiter=',')

    return pos_A, neg_A, dark_A


def _get_count_hit(x_modal, t_modal, p_modal, E=E):
    accuracy = -np.ones((E ** 2, E ** 2))
    count = np.zeros((E ** 2, E ** 2))
    hit = np.zeros((E ** 2, E ** 2))
    for i in range(t_modal.shape[0]):
        x_m = x_modal[i]
        t_m = t_modal[i]
        p_m = p_modal[i]
        _i = MODAL_INDEX[x_m]
        _j = MODAL_INDEX[t_m]
        count[_i][_j] += 1
        if t_m == p_m:
            hit[_i][_j] += 1

    no_Null = count != 0
    accuracy[no_Null] = hit[no_Null] / count[no_Null]

    pos_ind, neg_ind, dark_ind = _get_indcies(accuracy.shape[0])

    pos = accuracy[pos_ind]
    pos = pos[pos != -1]
    neg = accuracy[neg_ind]
    neg = neg[neg != -1]
    dark = accuracy[dark_ind]
    dark = dark[dark != -1]

    pos_acc = np.sum(pos) / len(pos) if len(pos) else -1
    neg_acc = np.sum(neg) / len(neg) if len(neg) else -1
    dark_acc = np.sum(dark) / len(dark) if len(dark) else -1

    return pos_acc, neg_acc, dark_acc


def _get_indcies(size):
    pos_ind = (np.arange(size), np.arange(size))
    neg_ind_0 = []
    neg_ind_1 = []
    dark_ind_0 = []
    dark_ind_1 = []
    for i in range(size):
        if 2 * i + 1 != size:
            neg_ind_0.append(size - i - 1)
            neg_ind_1.append(i)
        for j in range(size):
            if i == j or i + j == size:
                continue
            dark_ind_0.append(i)
            dark_ind_1.append(j)

    return pos_ind, (neg_ind_0, neg_ind_1), (dark_ind_0, dark_ind_1)


def main(data_path="data.csv"):
    data = np.loadtxt(data_path, delimiter=",")
    P = rebuild_P_S(data)
    P_Dis = get_P_Dis(P)
    n_Index = get_nearest_Index(P_Dis)
    Rate = get_Phase_rate(P)
    nearest_Dis = get_nearest_Dis(P_Dis)
    nearest_Dis_W = get_nearest_dis_W(nearest_Dis)
    Predict = get_Predict(n_Index, Rate, nearest_Dis_W)
    return get_Accuracy(Predict)
