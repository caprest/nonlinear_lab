from .toolbox import embedding
import numpy as np
from tqdm import tqdm

def heviside(x):
    ret = np.ones_like(x)
    ret[x<0] = -1
    return ret
x = np.random.randn(100)
heviside(x)

def recurrence_matrix(x, emb_dim, delta="auto"):
    emb = embedding(x, emb_dim)
    L = emb.shape[0]
    ret = np.empty((L, L))
    for i in range(L):
        for j in range(i, L):
            ret[i, j] = np.abs(np.max(x[i] - x[j]))
            ret[j, i] = ret[i, j]
    if delta == "auto":
        rank = ret.flatten()
        rank = np.sort(rank)
        delta = rank[int(np.floor(len(rank) * 0.05))]
    ret = delta - ret
    ret = heviside(ret)
    return ret


def twin_surrogate(x, emb_dim, delta = "auto", num=1):
    r_matrix = recurrence_matrix(x, emb_dim, delta)
    twin = r_matrix @ r_matrix.T
    twin[twin < r_matrix.shape[0]] = 0
    twin[twin > 0] = 1
    emb_x = embedding(x, emb_dim)
    length = emb_x.shape[0]
    ret = []
    pbar = tqdm(total=num)
    while (num > 0):
        i = np.random.randint(length)
        sur_x = np.empty_like(emb_x)
        sur_x[0] = emb_x[i]
        cur_idx = i
        flag = True
        for j in range(1, emb_x.shape[0]):
            num_twin = int(np.sum(twin[cur_idx,:]))
            twin_idx = np.arange(length)[twin[cur_idx,:] == 1]
            temp = np.random.randint(num_twin)
            next_idx = twin_idx[temp] + 1
            if next_idx >= length:
                flag = False
                break
            else:
                cur_idx = next_idx
            sur_x[j] = emb_x[cur_idx]
        if flag == True:
            ret.append(sur_x)
            num -= 1
            pbar.update(1)
    pbar.close()
    return np.array(ret)


def make_rs_surrogate(x):
    pass


def nakamura2006(xx,n,seed=None):
    if not isinstance(xx,np.ndarray):
        xx = np.array(xx)
    assert len(xx.shape) == 1
    sorted_xx = np.sort(xx)
    np.random.seed(seed)
    arg = np.random.permutation(xx.shape[0])
    randomized_xx = xx[arg]
    yy = np.fft.fft(np.hstack((xx,xx[::-1])))
    pzz = xx
    zz = randomized_xx
    k = 0
    while np.sum((zz-pzz)**2) > 0:
        if (k>2000):
            print("Too much iteration. Broke")
            break
        k+=1
        pzz = zz
        ww = np.fft.fft(np.hstack((zz,zz[::-1])))
        phase_ww = np.exp(1j*np.angle(ww))
        m = yy.shape[0]
        mh = int(np.ceil(m/2))
        freq1 = yy[0:mh-n]
        freq2 = np.abs(yy[mh-n:mh+n]) * phase_ww[mh-n:mh+n]
        freq3 = yy[mh+n:]
        vv = np.real(np.fft.ifft(np.hstack((freq1,freq2,freq3))))
        uu = np.argsort(np.argsort(vv[:vv.shape[0]//2]))
        zz = sorted_xx[uu]
        print(k)
    freq4 = np.zeros(2*n)
    trends = np.real(np.fft.ifft(np.hstack((freq1,freq4,freq3))))
    trends = trends[0:trends.shape[0]//2]
    diff = xx- trends

    return zz,vv,trends,diff
