import numpy as np
from ..common.distance import calic_dist_max
from ..common.toolbox import embedding, p_two_sided


def shreiber2000(x, y, r):
    x = np.array(x)
    y = np.array(y)
    xa = x[:-1]
    xb = x[1:]
    ya = y[:-1]
    data = np.concatenate([xb, xa, ya]).reshape(3, -1).transpose()
    select_plus = lambda x: len(x[x > 0])
    y1 = 0
    y2 = 0
    y3 = 0
    y4 = 0
    length = xa.shape[0]
    for datum in data:
        y1 += np.log2(np.sum(select_plus(r - np.max(np.abs(data - datum), axis=1))) / length)
        y2 += np.log2(np.sum(select_plus(r - np.max(np.abs(data[:, -2:] - datum[-2:]), axis=1))) / length)
        y3 += np.log2(np.sum(select_plus(r - np.max(np.abs(data[:, 0:2] - datum[0:2]), axis=1))) / length)
        y4 += np.log2(np.sum(select_plus(r - np.abs(data[:, 1] - datum[1]))) / length)
    return (y1 - y2 - y3 + y4) / length;


def kernel(x, y, eps):
    z = calic_dist_max(x, y)
    idx = (z < eps)
    z[idx] = 1
    z[~idx] = 0
    return z


def baek_brock(x, y, lag_x, lag_y, m, eps="auto"):
    """
    test if y granger cause x
    :param x: 
    :param y: 
    :param lag_x: 
    :param lag_y: 
    :param m: 
    :param eps: 
    :return: 
    """
    assert len(x) == len(y)
    if eps == "auto":
        eps_x = x.std() * 1.5
        eps_y = y.std() * 1.5
    else:
        eps_x = eps[0]
        eps_y = eps[1]
    #print(eps_x,eps_y)
    n = len(x) + 1 - m - max(lag_x, lag_y)
    emb_x_l = embedding(x, emb_dim=lag_x)
    emb_y_l = embedding(y, emb_dim=lag_y)
    emb_x_ml = embedding(x, emb_dim=lag_x + m)
    if lag_x >= lag_y:
        offset = lag_x -lag_y
        kernel_x_l = kernel(emb_x_l[:n], emb_x_l[:n], eps_x)
        kernel_y_l = kernel(emb_y_l[offset:n+offset], emb_y_l[offset:n+offset], eps_y)
        kernel_x_ml = kernel(emb_x_ml[:n], emb_x_ml[:n], eps_x)
    else:
        offset = lag_y -lag_x
        kernel_x_l = kernel(emb_x_l[offset:n + offset], emb_x_l[offset:n + offset], eps_x)
        kernel_y_l = kernel(emb_y_l[:n], emb_y_l[:n], eps_y)
        kernel_x_ml = kernel(emb_x_ml[offset:n + offset], emb_x_ml[offset:n + offset], eps_x)
    kernel_xy_l = kernel_x_l * kernel_y_l
    kernel_xy_ml = kernel_x_ml * kernel_y_l
    c1 = (np.sum(kernel_xy_ml) - np.sum(np.diag(kernel_xy_ml))) / 2 * (2 / n / (n - 1))
    c2 = (np.sum(kernel_xy_l) - np.sum(np.diag(kernel_xy_l))) / 2 * (2 / n / (n - 1))
    c3 = (np.sum(kernel_x_ml) - np.sum(np.diag(kernel_x_ml))) / 2 * (2 / n / (n - 1))
    c4 = (np.sum(kernel_x_l) - np.sum(np.diag(kernel_x_l))) / 2 * (2 / n / (n - 1))
    stat = (c1 / c2 - c3 / c4) * np.sqrt(n)
    d = np.array([1 / c2, -c1 / (c2 ** 2), -1 / c4, c3 / (c4 ** 2)])
    large_sigma = np.zeros((4, 4))
    large_k = lambda n: int(np.ceil(n ** (0.25)))
    omega = lambda k, n: 1 if k == 1 else 2 * (1 - (k - 1) / large_k(n))
    kernels = [kernel_xy_ml, kernel_xy_l, kernel_x_ml, kernel_x_l]
    cs = [c1, c2, c3, c4]
    for i in range(4):
        for j in range(4):
            temp_sum = 0
            for k in range(1, large_k(n) + 1):
                omega_kn = omega(k, n)
                for t in range(k - 1, n):  # discard idx in paper (1 ~ k-1)
                    temp_sum += omega_kn / 2 / (n - k + 1) * (
                        calic_A(i, t, n, kernels, cs) * calic_A(j, t - k + 1, n, kernels, cs) \
                        + calic_A(i, t - k + 1, n, kernels, cs) * calic_A(j, t, n, kernels, cs))
            large_sigma[i, j] = 4 * temp_sum
    sigma = d @ large_sigma @ d
    return p_two_sided(stat, std=np.sqrt(sigma)), stat, np.sqrt(sigma)


def calic_A(idx, t, n, kernels, cs):
    return 1 / (n - 1) * (np.sum(kernels[idx][t, :]) - kernels[idx][t, t]) - cs[idx]
