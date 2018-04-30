import numpy as np
from scipy.integrate import odeint



def generate_coupledlogistic_noise(init=[0.4, 0.2], r=[3.8, 3.5], xtoy = 0.1, ytox = 0.02, length=1000,noise_level = 0.01,seed =None):
    if seed:
        np.random.seed(seed)
    r = np.array(r)
    def logistic_step(prev, r, xtoy,ytox):
        x_next = prev[0] * (r[0] * (1 - prev[0]) - ytox * prev[1]) + np.random.randn() * noise_level
        y_next = prev[1] * (r[1] * (1 - prev[1]) - xtoy * prev[0]) + np.random.randn() * noise_level
        return np.array([x_next,y_next])

    X = np.zeros((length, 2))
    X[0, :] = init
    for i in range(length - 1):
        X[i + 1, :] = logistic_step(X[i, :], r, xtoy,ytox)
    print("Info: init = {}, r = {}, xtoy = {},ytox = {}".format(init,r,xtoy,ytox))
    return X[:, 0], X[:, 1]

def generate_coupledlogistic(init=[0.4, 0.2], r=[3.8, 3.5], xtoy = 0.1, ytox = 0.02, length=1000,noise_level = 0.00,seed =None,ex_period = 2,ex_level = 0,alert =True):
    if alert:
        print("Noise Level = {},xtoy = {}, ytox = {}, init = {}".format(noise_level,xtoy,ytox,init))
    if seed:
        np.random.seed(seed)
    r = np.array(r)
    Ht = ex_level *  np.cos(np.pi * 2 * np.arange(length) /ex_period )
    x = np.zeros(length)
    y = np.zeros(length)
    x[0] = init[0]
    y[0] = init[1]
    for i in range(length - 1):
        x[i + 1] = x[i] * ((r[0] +  Ht[i]) * (1 - x[i]) - ytox * y[i]) + np.random.randn() * noise_level
        y[i+1]  = y[i] * ((r[1] +   Ht[i]) * (1 - y[i]) - xtoy * x[i]) + np.random.randn() * noise_level
    print("Info: init = {}, r = {}, xtoy = {},ytox = {}".format(init,r,xtoy,ytox))
    return x,y



def generate_random(length, seed=0):
    np.random.seed(seed)
    return np.random.randn(100)


def lorentz_diff(v, t, p, r, b):
    return [-p * v[0] + p * v[1], -v[0] * v[2] + r * v[0] - v[1], v[0] * v[1] - b * v[2]]


def generate_lorentz(t = np.linspace(0, 3, 1000)):
    p = 10
    r = 28
    b = 8 / 3
    v0 = [0.1, 0.1, 0.1]
    v = odeint(lorentz_diff, v0, t, args=(p, r, b))
    return v[:, 0], v[:, 1], v[:, 2], t

def generate_mylnmodel(x_0 = 0.8, y_0 = 0.16,length=1000,xtoy= 0.05, ytox = 0.05):
    def forward(x,y):
        next_x = -0.8*x + ytox *y
        next_y = 1 - 1.5* (y**2) + xtoy *x
        return np.array([next_x,next_y])
    data = np.zeros((length,2))
    data[0,0] = x_0
    data[0,1] = y_0
    for i in range(length-1):
       data[i+1,:] = forward(data[i,0],data[i,1])
    return data[:,0], data[:,1]


def y_driven_by_x(length,xtoy = 0.3,noise_level_y = 0.1):
    x = np.random.randn(length)
    y = np.zeros(length)
    y[0] = np.random.random()
    for i in range(length - 1):
        y[i+1] = 0.7 * y[i] +  xtoy * x[i] + y[i] * x[i] + noise_level_y * np.random.randn()
    return x,y


def y_driven_by_chaos_x(length = 1000,init = [0.3, 0.4], ytox = 0,noiselevel = 0.01):
    x = np.zeros(length)
    y = np.zeros(length)
    x[0] = init[0]
    y[0] = init[1]
    for i in range(length-1):
        x[i+1] = 3.67 * x[i] * (1-x[i]) + noiselevel * np.random.randn()
        y[i+1] = 0.2 * y[i] + 0.3 * x[i] * y[i] + noiselevel * np.random.randn()
    return x,y






def lorentz961(m =10,F = 8,start=0,end=200,tic=0.05):
    num = (end - start) // tic
    t = np.linspace(start,end,num)
    def lorentz_diff(v, t, m,F):
        ret = np.zeros(m)
        idx = np.concatenate([np.arange(m)]*3)
        for j in range(m):
            iminus2,iminus1,i,iplus1 = idx[m+j-2:m+j+2]
            ret[i] = -v[iminus2]*v[iminus1] + v[iminus1] * v[iplus1] -v[i] + F
        return ret
    v0 = np.random.random(10)
    v = odeint(lorentz_diff, v0, t, args=(m,F))
    return np.array(v), t


def sugihara_external_force3(x_0=0.9, y_0=0.4, length=500, seed=None):
    if seed:
        np.random.seed(seed)
    noise =  np.random.randn(length)
    x = np.zeros(length)
    y = np.zeros(length)
    r_x = np.zeros(length)
    r_y = np.zeros(length)
    for i in range(length):
        if i == 0:
            x[i], y[i], r_x[i], r_y[i] = (x_0, y_0, 0, 0)
        elif i in [1, 2, 3]:
            x[i], y[i], r_x[i], r_y[i] = (
                0.4 * x[i - 1],
                0.35 * y[i - 1],
                x[i - 1] * (3.1 * (1 - x[i - 1])) * np.exp(-0.3 * noise[i - 1]),
                y[i - 1] * (2.9 * (1 - y[i - 1])) * np.exp(-0.36 * noise[i - 1]),
            )
        else:
            x[i], y[i], r_x[i], r_y[i] = (
                0.4 * x[i - 1] + np.max(r_x[i - 4], 0),
                0.35 * y[i - 1] + np.max(r_y[i - 4], 0),
                x[i - 1] * (3.1 * (1 - x[i - 1])) * np.exp(-0.3 * noise[i - 1]),
                y[i - 1] * (2.9 * (1 - y[i - 1])) * np.exp(-0.36 * noise[i - 1]),
            )

    return x,y,noise

def sugihara_external_force5(length=1000,init = [0.1,0.2,0.5,0.1,0.7]):
    y = np.zeros(length,5)
    y[0,:] = init
    for i in range(length-1):
        y[i+1,:] =(
            y[i,0] *(4- 4 * y[i,0] - 2* y[i,1] -0.4 * y[i,2]),
            y[i,1] * (3.1 - 0.31 * y[i,0] -3.1 * y[i,1] -0.93 * y[i,2]),
            y[i,2] * (2.12 + 0.636 * y[i,0] + 0.636 * y[i,1] + -2.12 * y[i,2]),
            y[i,3] * (3.8 -0.111 * y[i,0] - 0.011 * y[i,1] + 0.131 * y[i,2] -3.8 * y[i,3]),
            y[i,4] * (4.1 - 0.082 * y[i,0] - 0.111 * y[i,1] - 0.125 * y[i,2] -4.1 * y[i,4])
        )

    return y





def linear_coupled(length = 1000,init = [0.1,0.2],xtoy = 0.02, ytox = 0.03,noise_level = 0.05):
    x = np.zeros(length)
    y = np.zeros(length)
    x[0],y[0] = init
    for i in range(length -1):
        x[i+1] = 0.4 * x[i] + xtoy * y[i] +noise_level * np.random.randn()
        y[i+1] = 0.3 * y[i] + ytox * x[i] +noise_level * np.random.randn()

    return x,y


def my_model2(length = 2000,init = [0.1,0.2],xtoy = 0.00,ytox = 0.02,noise_level = 0.00,alert = True):
    if alert:
        print("Noise Level = {},xtoy = {}, ytox = {}, init = {}".format(noise_level,xtoy,ytox,init))
    def forward(x,y):
        x_next =  0.5 * x - 1 / (1 + np.exp(-(x) / 0.04)) + 0.24 + noise_level * np.random.randn() + ytox *y
        y_next =  0.49 * y - 1 / (1 + np.exp(-(y) / 0.042)) + 0.23 + noise_level * np.random.randn() + xtoy *x
        return x_next,y_next
    x = np.zeros(length)
    y = np.zeros(length)
    x[0],y[0] = init
    for i in range(length - 1):
        x[i + 1],y[i+1] = forward(x[i],y[i])
    return x,y

def my_model3(length = 2000,init = [0.1,0.2],xtoy = 0.00,ytox = 0.02,noise_level = 0.00,alert = True,ex_level =0,ex_period = 10):
    if alert:
        print("Noise Level = {},xtoy = {}, ytox = {}, init = {},ex_level = {},period = {}".format(noise_level,xtoy,ytox,init,ex_level,ex_period))
    Ht = ex_level * np.cos(np.pi * 2 * np.arange(length) / ex_period)
    x = np.zeros(length)
    y = np.zeros(length)
    x[0],y[0] = init
    for i in range(length - 1):
        x[i+1] =  0.5 * x[i] - 1 / (1 + np.exp(-(x[i] + Ht[i]) / 0.04)) + 0.24 + noise_level * np.random.randn() + ytox *y[i]
        y[i+1] =  0.49 * y[i] - 1 / (1 + np.exp(-(y[i] + Ht[i]) / 0.042)) + 0.23 + noise_level * np.random.randn() + xtoy *x[i]
    return x,y


def add_observation_noise(x,noise_level = 0):
    return  x + np.random.randn(len(x)) * x.std() * noise_level

