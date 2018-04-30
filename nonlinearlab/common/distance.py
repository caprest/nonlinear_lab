import numpy as np


def calic_dist_max(x,y):
    try:
        assert isinstance(x,np.ndarray)
    except AssertionError:
        x = np.array(x)
    try:
        assert isinstance(y,np.ndarray)
    except AssertionError:
        y = np.array(x)
    assert x.shape == y.shape
    z = x.reshape(x.shape[0],1,x.shape[1]) - y.reshape(1,x.shape[0],x.shape[1])
    zz = np.max(np.abs(z), axis=2)
    return zz


def calic_dist_l2(x,y):
    try:
        assert isinstance(x,np.ndarray)
    except AssertionError:
        x = np.array(x)
    try:
        assert isinstance(y,np.ndarray)
    except AssertionError:
        y = np.array(x)
    assert x.shape == y.shape
    z = x.reshape(x.shape[0],1,x.shape[1]) - y.reshape(1,x.shape[0],x.shape[1])
    zz = np.sqrt(np.sum(z**2, axis=2))

    return zz