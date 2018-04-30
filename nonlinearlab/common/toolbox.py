import numpy as np
from scipy.stats import norm
from .distance import calic_dist_l2
import matplotlib.pyplot as plt
from collections import Counter



def p_two_sided(value,mean=0,std=1):
    p = 1 - (norm.cdf(x=abs(value), loc=mean, scale=std) - norm.cdf(x= -abs(value), loc=mean, scale=std))
    return p


def embedding(x, emb_dim, tau = 1):
    """
    input: one dimensional timeseries
    """
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    assert len(x.shape) == 1
    max_t = x.shape[0]
    emb_len = max_t  - (emb_dim - 1) * tau
    ret = np.empty((emb_len, emb_dim))
    for i in range(emb_len):
        ret[i] = x[i : i + emb_dim * tau :tau]
    return ret

def correlation(x, y):
    assert len(x) == len(y)
    varx = np.sum((x - np.mean(x)) ** 2)
    vary = np.sum((y - np.mean(y)) ** 2)
    if (varx == 0) or (vary == 0):
        print("WARNING:np.nan")
        return np.nan
    else:
        return np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sqrt(varx * vary)


def false_nearest_neighbor(x,maximum = 10,plot=True):
    num = np.zeros(maximum-1)
    maxlen = len(x) - maximum +1
    fnn_data = np.zeros(maximum-1)
    for i in np.arange(1,maximum): #1 to maximum-1
        fnn_num = 0
        emb = embedding(x,emb_dim = i)[:maxlen]
        dist = calic_dist_l2(emb,emb)
        for j,row in enumerate(dist):
            nearest_idx = np.argsort(row)[1]
            nearest_len = dist[j,nearest_idx]
            if nearest_len < 1e-10:
                fnn_num +=1
            else:
                if np.abs(x[j+i] - x[nearest_idx+i])/nearest_len > 15:
                    fnn_num+=1
        fnn_data[i-1] = fnn_num
    if plot:
        plt.plot(np.arange(1,maximum),fnn_data)
        plt.show()
    return fnn_data

def decide_dim(x,y,maximum=10):
    return max(np.argmin(false_nearest_neighbor(x, maximum=maximum, plot=False))
        , np.argmin(false_nearest_neighbor(y, maximum=maximum, plot=False)), 1) + 1

def series_to_an_ans(s,beta="xtoy",p="pxtoy"):
    return  1 if (s[beta] ==0 and s[p]  > 0.05) or (s[beta]!=0 and s[p] <0.05) else 0

def series_to_ans(s,xtoy="xtoy",ytox="ytox",pxtoy="pxtoy",pytox="pytox"):
    a1 = 1 if (s[xtoy] ==0 and s[pxtoy]  > 0.05) or (s[xtoy]!=0 and s[pxtoy] <0.05) else 0
    a2 = 1 if (s[ytox] ==0 and s["pytox"] > 0.05) or (s[ytox]!=0 and s[pytox] <0.05) else 0
    return 1 if a1 ==1 and a2 ==1 else 0


def ret_class(beta, p, pthre=0.05):
    """
    tp = 0 : prediction 1, real 1
    fp = 1 : prediction 1 ,real 0
    fn = 2 : prediction 0 , real 1
    tn = 3 : prediction 0, real 0
    """
    pred = 1 if p < pthre else 0
    real = 1 if beta != 0 else 0
    if pred == 1 and real == 1:
        return 0
    elif pred == 1 and real == 0:
        return 1
    elif pred == 0 and real == 1:
        return 2
    else:
        return 3


def series_class(s, colbeta, colp):
    return ret_class(s[colbeta], s[colp])

def ret_f(data):
    count = Counter(data)
    tp = count[0]
    fp = count[1]
    fn = count[2]
    tn = count[3]
    try:
        precision = tp /(tp+fp)
    except:
        precision = np.inf
    try:
        recall = tp / (tp + fn)
    except:
        recall =np.inf
    return 2*recall*precision/(recall+precision)


def make_grid_data(data):
    ret = []
    while(len(data)>0):
        axis = data[0]
        data = data[1:]
        if not ret:
            ret = [ [ax] for ax in axis]
            print(ret)
        else:
            new_ret = []
            for datum in ret:
                for ax in axis:
                    new_ret.append(datum+[ax])
            ret = new_ret
            print(ret)
    return ret

def make_grid(**params):
    data = [ params[key] for key in  sorted(params.keys())]
    grid = make_grid_data(data)
    ret = []
    for data in grid:
        ret.append(dict(
            [(k,data[i]) for i,k in enumerate(sorted(params.keys()))]
        ))
    return ret

