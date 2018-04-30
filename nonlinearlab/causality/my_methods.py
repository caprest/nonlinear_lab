import numpy as np
from .transfer_entropy import baek_brock
from .ccm import ccm_test
from ..common.toolbox import decide_dim
from statsmodels.tsa.stattools import grangercausalitytests


def check_validity(x,y):
    return not( np.isnan(x).any() or np.isnan(y).any() or np.isinf(x).any() or np.isinf(y).any())


def make_report(x,y,noise,maxlag,emb_dim = "auto",l_0 = "auto",l_1 = "auto"):
    data = []
    if emb_dim == "auto":
        emb_dim = decide_dim(x,y,maxlag)
    data.append([noise, "ccm", ccm_test(x, y, emb_dim=emb_dim, l_0=l_0, l_1=l_1)["p_value"],
                 ccm_test(y, x, emb_dim=emb_dim, l_0=l_0, l_1=l_1)["p_value"], emb_dim])
    ret = grangercausalitytests(np.concatenate([y, x]).reshape(2, -1).transpose(), maxlag=maxlag)
    gcxtoy = [ret[key][0]["ssr_ftest"][1] for key in ret.keys()]
    ret = grangercausalitytests(np.concatenate([x, y]).reshape(2, -1).transpose(), maxlag=maxlag)
    gcytox = [ret[key][0]["ssr_ftest"][1] for key in ret.keys()]
    cmixtoy = [baek_brock(y, x, lag_x=i, lag_y=i, m=1, eps="auto")[0] for i in range(1, maxlag + 1)]
    cmiytox = [baek_brock(x, y, lag_x=i, lag_y=i, m=1, eps="auto")[0] for i in range(1, maxlag + 1)]
    for lag in range(1, maxlag + 1):
        data.append([noise, "cmi", cmixtoy[lag - 1], cmiytox[lag - 1], lag])
        data.append([noise, "gc", gcxtoy[lag - 1], gcytox[lag - 1], lag])
    return data