import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from ..common.toolbox import embedding,correlation,decide_dim
from ..common.surrogate import twin_surrogate
from ..common.distance import calic_dist_l2
import tensorflow as tf

CCM_PARAM = {
    "save_path": "./",
    "emb_dim": 5,
    "discard": 10,
}


def set_params(**kwargs):
    if "save" in kwargs:
        CCM_PARAM["save_path"] = kwargs["save"]
    if "discard" in kwargs:
        CCM_PARAM["discard"] = kwargs["discard"]


def xmap(x, k_dist, k_idx,emb_dim,tau,eps = 1e-5):
    length = k_dist.shape[0]
    x_tilde = np.empty((length))
    for i in range(length):
        u = np.exp(-k_dist[i, :] / (k_dist[i, 0] +eps))
        w = u / np.sum(u)
        x_tilde[i] = np.sum(w * x[ k_idx[i, :] +(emb_dim -1) * tau])
    return x_tilde


def estimate(x, y, length=None, emb_dim=None,tau = 1, plot=False):
    """
    method to estimate timeseries X from information of y.
    t: the length of time you estimate, we estimate x[:t]
    discard: discard some of the time series
    note: please discard some of the time series if you use dynamic system like logisticmap.
    Some of the initial time series should be discarded. 
    :return rho and estimated x
    """
    x = np.array(x)
    y = np.array(y)
    if not emb_dim:
        emb_dim = CCM_PARAM["emb_dim"]
    emb = embedding(y, emb_dim,tau=tau)
    if not length:
        length = emb.shape[0]
    emb = emb[:length]
    rho, x_tilde = estimate_from_emb(x[:length+(emb_dim -1 ) * tau], emb,tau, plot=plot)
    return rho, x_tilde

def estimate_from_emb(x, emb, tau = 1, plot=False):
    length = emb.shape[0]
    emb_dim = emb.shape[1]
    dist_arr, dist_idx = calic_all_dist(emb)
    k_dist, k_idx = k_nearest(dist_arr, dist_idx, length, emb_dim + 1)
    x_tilde = xmap(x, k_dist, k_idx,emb_dim,tau)
    if plot:
        plt.scatter(x[(emb_dim-1)*tau:], x_tilde)
        plt.show()
    rho = correlation(x[(emb_dim-1)*tau:], x_tilde)
    return rho, x_tilde


def estimate_using_bootstrap(x,y, length="auto", emb_dim=5,tau = 1):
    """
    estimate x from y to judge x->y cause
    :param x: 
    :param y: 
    :param length: 
    :param emb_dim: 
    :param tau: 
    :return: 
    """
    emb_y = embedding(y,emb_dim,tau)
    max_length = len(emb_y)
    if length =="auto":
        length = max_length
    rho, x_tilde = estimate_from_emb_random(x,emb_y,length,max_length,emb_dim,tau)
    return rho, x_tilde

def estimate_from_emb_random(x,emb_y,length,max_length,emb_dim,tau):
    idxs = np.random.choice(np.arange(max_length), length, replace=False)
    y_selected = emb_y[idxs]
    x_selected = x[idxs + (emb_dim - 1) * tau]
    padding = np.empty((emb_dim - 1) * tau)
    x = np.concatenate([padding, x_selected])
    rho, x_tilde = estimate_from_emb(x, y_selected, tau)
    return rho, x_tilde



def convergent(x, y, start = 0, length=None, emb_dim=None, tau = 1, min_length=None, estimation_freq=1,option = "linear"):
    """
    see wheter rho increase with more use of time series. using x[start:start+length]
    :param x: 
    :param y: 
    :param start: 
    :param length: 
    :param emb_dim: 
    :param min_length: 
    :return: rho_array
    """
    x = np.array(x)
    y = np.array(y)
    x = x[start:]
    y = y[start:]
    if not emb_dim:
        emb_dim = CCM_PARAM["emb_dim"]
    if not min_length:
        min_length = CCM_PARAM["discard"]
    emb = embedding(y, emb_dim,tau)
    dist_arr, dist_idx = calic_all_dist(emb)
    length =  emb.shape[0] if not length else length
    L_array = np.arange(min_length, length, estimation_freq)
    # it is meaningless to estimate x(i) with small i
    rho_array = np.empty(L_array.shape[0])
    for i, L in tqdm(enumerate(L_array)):
        if option == "linear":
            k_dist, k_idx = k_nearest(dist_arr, dist_idx, L, emb_dim + 1,option=option)
            x_tilde = xmap(x, k_dist, k_idx, emb_dim, tau)
            rho = correlation(x_tilde, x[(emb_dim - 1) * tau:(emb_dim - 1) * tau + L])
        elif option == "random":
            k_dist, k_idx, random_cand = k_nearest(dist_arr, dist_idx, L, emb_dim + 1,option=option)
            x_tilde = xmap(x, k_dist, k_idx,emb_dim,tau) #xmap estimates x[idx_cand + offset] automatically(k_idx only includes idx in k_idx_cand
            rho = correlation(x_tilde, x[random_cand+(emb_dim-1)*tau])
        rho_array[i] = rho
    return L_array, rho_array


def convergent_random(x, y, start = 0, length=None, emb_dim=None, tau = 1, min_length=None, estimation_freq=1, num=10):
    rhos = []
    for i in range(num):
        L, rho = convergent(x, y, start=start, length=length, emb_dim=emb_dim, tau=tau, min_length=min_length, estimation_freq=estimation_freq,
                            option="random")
        rhos.append(rho)
    rhos = np.array(rhos).mean(axis=0)
    return L,rhos


def convergent_emb(x,emb_y,length = None,min_length=None, estimation_freq=1,tau = 1,option = "linear"):
    emb_dim = emb_y.shape[1]
    min_length = min_length if not min_length else (emb_dim+1) * 2
    length = emb_y.shape[0] if not length else length
    dist_arr, dist_idx = calic_all_dist(emb_y)
    L_array = np.arange(min_length, length, estimation_freq)
    rho_array = np.empty(L_array.shape[0])

    for i, L in enumerate(L_array):
        if option == "linear":
            k_dist, k_idx = k_nearest(dist_arr, dist_idx, L, emb_dim + 1)
            x_tilde = xmap(x, k_dist, k_idx,emb_dim,tau)
            rho = correlation(x_tilde, x[(emb_dim-1)*tau:(emb_dim-1)*tau+L])
        elif option == "random":
            k_dist, k_idx, random_cand = k_nearest(dist_arr, dist_idx, L, emb_dim + 1,option=option)
            x_tilde = xmap(x, k_dist, k_idx,emb_dim,tau) #xmap estimates x[idx_cand + offset] automatically(k_idx only includes idx in k_idx_cand
            rho = correlation(x_tilde, x[random_cand+(emb_dim-1)*tau])
        rho_array[i] = rho
    return L_array,rho_array

def convergent_random_emb(x, emb_y,  length=None, emb_dim=None, tau = 1, min_length=None, estimation_freq=1,num=10):
    rhos = []
    for i in range(num):
        L, rho = convergent_emb(x, emb_y,  length=length, tau=tau, min_length=min_length, estimation_freq=estimation_freq,
                            option="random")
        rhos.append(rho)
    rhos = np.array(rhos).mean(axis=0)
    return L,rhos


def convergence_plot(x, y, start=0, length=100, emb_dim = None, discard = None, save=False, sfx_xtoy="XtoY", sfx_ytox="YtoX",estimation_freq=1,tau = 1):
    L_1,rho_1 = convergent(x, y, start, length, emb_dim = emb_dim,
                           min_length= discard, estimation_freq=estimation_freq, tau = tau)
    L_2,rho_2 = convergent(y, x, start, length, emb_dim = emb_dim,
                           min_length= discard, estimation_freq=estimation_freq, tau = tau)
    _plot_rho(L_1,rho_1,save,sfx_xtoy)
    _plot_rho(L_2,rho_2,save,sfx_ytox)



def _plot_scatter(x, x_tilde):
    plt.scatter(x, x_tilde)
    plt.xlabel("real")
    plt.ylabel("predict")
    plt.show()


def _plot_rho(L_array, rho_array, save ,savesuffix):
    y = pd.Series(rho_array)
    y.fillna(method='ffill', limit=1)
    plt.plot(L_array, y)
    plt.xlabel("L")
    plt.ylabel(r"$\rho$")
    if save and savesuffix:
        plt.savefig(CCM_PARAM["save_path"] + savesuffix + ".png")
    plt.show()


def calic_all_dist(emb):
    """
    caliculate all pair of distance and sort it with idx
    :param emb: 
    :return: 
    """
    length = emb.shape[0]
    dist_arr = calic_dist_l2(emb,emb)
    dist_idx = np.zeros((length, length), dtype=int)

    for i in range(length):
        dist_idx[i, :] = np.argsort(dist_arr[i, :])
    return dist_arr, dist_idx


def k_nearest(dist_arr, dist_idx, L, k,option="linear"):
    k_idx = np.empty((L, k), dtype=int)
    k_dist = np.empty((L, k), dtype=float)
    L_max = dist_arr.shape[0]
    assert L > k
    # L = k means k-nearest includes same point thus inappropriate
    if option == "linear":
        for i in range(L):
            idx = dist_idx[i, (dist_idx[i, :] < L) & (dist_idx[i,:] != i)]
            idx = idx[0:k]
            # excluding index-0 to avoid same-point index(d(i,i) = 0)
            k_idx[i, :] = idx
            k_dist[i, :] = dist_arr[i, idx]
        return k_dist, k_idx

    elif option == "random":
        random_cand = np.sort(np.random.choice(L_max, L, replace=False))
        for i in range(L):
            idx_cand = dist_idx[random_cand[i], random_cand]
            idx = idx_cand[idx_cand != random_cand[i]][:k]
            # excluding index-0 to avoid same-point index(d(i,i) = 0)
            k_idx[i, :] = idx
            k_dist[i, :] = dist_arr[i, idx]
        return k_dist, k_idx,random_cand

    else:
        raise ValueError
        return None


def surrogate_test(x, y, emb_dim,tau=1, num=20, p=0.05,seed =None):
    if seed:
        np.random.seed(seed)
    ys = twin_surrogate(y, emb_dim, delta="auto", num=num)
    rhos = np.empty(len(ys))
    for i, emb_y in tqdm(enumerate(ys)):
        rho, x_tilde = estimate_from_emb(x, emb_y)
        rhos[i] = rho
    rho_true, x_tilde_true = estimate(x, y, emb_dim=emb_dim)
    rhos.sort()
    pvalue =  1 -len(rhos[rhos < rho_true]) / len(rhos)
    return {
        "p": pvalue,
        "judge": pvalue < p,
        "rhos": rhos,
        "true": rho_true
    }


def dual_surrogate_test(x, y, emb_dim, tau=1,num=20, p=0.05,seed =None):
    if seed:
        np.random.seed(seed)
    xtoy = surrogate_test(x, y, emb_dim, tau=tau,num = num, p = p,seed=None)
    ytox = surrogate_test(y, x, emb_dim, tau = tau,num = num, p = p,seed=None)
    return {
        "X->Y": xtoy,
        "Y->X": ytox,
    }

def regression_model(x_data,y_data,iter = 3001,plot = True,verbose = 0):
    x_train = x_data.reshape(-1,1)
    y_train = y_data.reshape(-1,1)
    #graph
    xt = tf.placeholder(tf.float32, [None,1])
    yt = tf.placeholder(tf.float32, [None,1])
    params  = tf.Variable([np.max(y_train), 0.0, 0.0],"other_variable", dtype=tf.float32)
    y = params[0] + tf.multiply(params[1],tf.exp(tf.multiply(tf.multiply(-1.0,params[2]),xt)))
    loss = tf.reduce_sum(tf.square(y-yt))
    train = tf.train.AdamOptimizer().minimize(loss)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    loss_log = []
    prev_loss = np.inf
    for step in range(iter):
        loss_log.append( sess.run(loss,feed_dict={xt:x_train,yt:y_train}))
        if step % 500 == 0:
            loss_val = sess.run(loss,feed_dict={xt:x_train,yt:y_train})
            params_val = sess.run(params)
            if verbose:
                print('Step: {},   Loss: {},   params: {}'.format(step,loss_val,params_val))
            prev_loss = loss_val
        sess.run(train,feed_dict={xt:x_train,yt:y_train})
    loss_log = np.array(loss_log)
    func = lambda a,b,c,x: a + b * np.exp(-c * x).reshape(-1,1)
    y_ret = func(*params_val,x_train)
    if plot:
        plt.scatter(x_train,y_ret)
        plt.scatter(x_train,y_train)
        plt.show()
        plt.plot(loss_log)
        plt.show()
    return loss_log,params_val


def ccm_test(x, y,emb_dim = "auto", l_0 = "auto", l_1 = "auto",  tau=1, n=10,mean_num = 10,max_dim = 10):
    """
    estimate x from y to judge x->y cause
    :param x: 
    :param y: 
    :param l_0: 
    :param l_1: 
    :param emb_dim: 
    :param tau: 
    :param n: 
    :return: 
    """


    if emb_dim == "auto":
        emb_dim = decide_dim(x,y)
    if l_0 == "auto":
        l_0 = int(np.ceil((len(x) - emb_dim + 1) * 0.1))
    if l_1 == "auto":
        l_1 = int(np.ceil((len(x) - emb_dim + 1) * 0.9))

    ys = twin_surrogate(y, emb_dim,num=n)
    raw_rhos = []
    rhos = []
    max_length = len(ys[0])
    for i in tqdm(range(n)):
        mean = 0
        for j in range(mean_num):
            rho_0, _ = estimate_using_bootstrap(x, y, length=l_0, emb_dim=emb_dim, tau=tau)
            rho_1, _ = estimate_using_bootstrap(x, y, length=l_1, emb_dim=emb_dim, tau=tau)
            rho_s_0, _ = estimate_from_emb_random(x, ys[i], length=l_0, emb_dim=emb_dim, tau=tau, max_length = max_length)
            rho_s_1, _ = estimate_from_emb_random(x, ys[i], length=l_1, emb_dim=emb_dim, tau=tau, max_length = max_length)
            raw_rhos.append([rho_0, rho_1, rho_s_0, rho_s_1])
            mean += rho_1 -rho_0 -(rho_s_1 - rho_s_0 )
        rhos.append(mean/mean_num)
    rhos = np.array(rhos)
    p = 1 - (len(rhos[rhos>0]) / n)
    return {
        "p_value" :p,
        "rhos" :rhos,
        "raw_rhos":raw_rhos
    }

def ccm_regression_test(x,y,start = 0,length = 1000 ,surrogate_num = 10,emb_dim = 2,estimation_freq =1):
    x = x[start:start+length]
    y = y[start:start+length]
    L_array, rho_array = convergent(x, y, emb_dim=emb_dim)
    loss_true, param_true  = regression_model(L_array, rho_array,plot=False)
    loss_last = loss_true[-1]
    ys = twin_surrogate(y, emb_dim=emb_dim, num=surrogate_num)
    loss_surs = np.empty(surrogate_num)
    param_data = []
    print("Caliculate surrogate's loss")
    for i,emb_y in tqdm(enumerate(ys)):
        L_array, rho_array = convergent_emb(x, emb_y, length=len(emb_y), tau=1, min_length=10, estimation_freq=estimation_freq)
        loss_log,params = regression_model(L_array, rho_array,plot = False)
        loss_surs[i] = loss_log[-1]
        param_data.append(params)
    p = 1 -(len(loss_surs[loss_surs > loss_last])/surrogate_num)
    return p,loss_last,param_true,loss_surs


def ccm_random_regression_test(x,y,start = 0,length = 1000 ,num = 10,surrogate_num = 10,emb_dim = 2,estimation_freq =1):
    x = x[start:start+length]
    y = y[start:start+length]
    L_array_true, rho_array_true = convergent_random(x, y, emb_dim=emb_dim, num = num, estimation_freq=estimation_freq)
    loss_true, param_true  = regression_model(L_array_true, rho_array_true,plot=False)
    loss_last = loss_true[-1]
    ys = twin_surrogate(y, emb_dim=emb_dim, num=surrogate_num)
    loss_surs = np.empty(surrogate_num)
    param_data = []
    print("Caliculate surrogate's loss")
    for i,emb_y in tqdm(enumerate(ys)):
        L_array, rho_array = convergent_random_emb(x, emb_y, tau=1, min_length=10, estimation_freq=estimation_freq,num = num)
        loss_log,params = regression_model(L_array, rho_array,plot = False)
        loss_surs[i] = loss_log[-1]
        param_data.append(params)
    param_data = np.array(param_data)
    gammas = param_data[:,2]
    convs = param_data[:,0]
    gamma_test = 1 -(len(gammas[gammas < param_true[2]])/surrogate_num)
    conv_test = 1 - (len(convs[convs < param_true[0]])/surrogate_num)
    p = 1 -(len(loss_surs[loss_surs > loss_last])/surrogate_num)
    return {
               "p_value" :p,
               "loss_last" :loss_last,
                "loss_surs": loss_surs,
                "param_true": param_true,
                "param_data":param_data,
                "L_array": L_array_true,
                "rho_array":rho_array_true,
                "gamma_test":gamma_test,
                "conv_test":conv_test
            }
