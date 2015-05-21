import ipdb
import numpy as np
import theano.tensor as T

from cle.cle.utils.op import logsumexp


def NllBin(y, y_hat):
    """
    Binary cross-entropy

    Parameters
    ----------
    .. todo::
    """
    nll = T.nnet.binary_crossentropy(y_hat, y).sum(axis=1)
    return nll


def NllMul(y, y_hat):
    """
    Multi cross-entropy

    Parameters
    ----------
    .. todo::
    """
    nll = -T.sum(y * T.log(y_hat), axis=-1)
    return nll


def MSE(y, y_hat):
    """
    Mean squared error

    Parameters
    ----------
    .. todo::
    """
    mse = T.sum(T.sqr(y - y_hat), axis=-1)
    return mse


def Gaussian(y, mu, sig):
    """
    Gaussian negative log-likelihood

    Parameters
    ----------
    y   : TensorVariable
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    nll = 0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
                      T.log(2 * np.pi), axis=1)
    return nll


def GMM(y, mu, sig, coeff):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y = y.dimshuffle(0, 1, 'x')
    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]/coeff.shape[-1],
                     coeff.shape[-1]))
    sig = sig.reshape((sig.shape[0],
                       sig.shape[1]/coeff.shape[-1],
                       coeff.shape[-1]))
    inner = -0.5 * T.sum(T.sqr(y - mu) / sig**2 + 2 * T.log(sig) +
                         T.log(2 * np.pi), axis=1)
    nll = -logsumexp(T.log(coeff) + inner, axis=1)
    return nll

def biGauss(y, mu, sig, corr, binary):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    #y = y.dimshuffle(0, 1, 'x')
    #corr = corr.dimshuffle(0,'x',1)

    mu_1 = mu[:,0].reshape((-1,1))
    mu_2 = mu[:,1].reshape((-1,1))

    sig_1 = sig[:,0].reshape((-1,1))
    sig_2 = sig[:,1].reshape((-1,1))
    
    y0 = y[:,0].reshape((-1,1))
    y1 = y[:,1].reshape((-1,1))
    y2 = y[:,2].reshape((-1,1))

    c_b =  T.sum( T.xlogx.xlogy0(y0, binary) +
              T.xlogx.xlogy0(1 - y0, 1 - binary), axis = 1)

    inner1 =  (0.5*T.log(1-corr**2)) + \
                         T.log(sig_1) + T.log(sig_2) +\
                         T.log(2 * np.pi)

    Z = (((y1 - mu_1)/sig_1)**2) + (((y2 - mu_2) / sig_2)**2) - (2. * (corr * (y1 - mu_1)*(y2 - mu_2)) / (sig_1 * sig_2))

    inner2 = 0.5 * (1. / (1. - corr**2))
    cost = - (inner1 + (inner2 * Z))

    nll = -T.sum(cost,axis = 1) - c_b
    return nll

def biGMM(y, mu, sig, coeff, corr, binary):
    """
    Gaussian mixture model negative log-likelihood

    Parameters
    ----------
    y     : TensorVariable
    mu    : FullyConnected (Linear)
    sig   : FullyConnected (Softplus)
    coeff : FullyConnected (Softmax)
    """
    y = y.dimshuffle(0, 1, 'x')
    #corr = corr.dimshuffle(0,'x',1)

    mu = mu.reshape((mu.shape[0],
                     mu.shape[1]/coeff.shape[-1],
                     coeff.shape[-1]))
    mu_1 = mu[:,0,:]
    mu_2 = mu[:,1,:]

    sig = sig.reshape((sig.shape[0],
                       sig.shape[1]/coeff.shape[-1],
                       coeff.shape[-1]))
    sig_1 = sig[:,0,:]
    sig_2 = sig[:,1,:]

    c_b =  T.sum( T.xlogx.xlogy0(y[:,0,:], binary) +
              T.xlogx.xlogy0(1 - y[:,0,:], 1 - binary), axis = 1)

    inner1 =  (0.5*T.log(1-corr**2)) + \
                         T.log(sig_1) + T.log(sig_2) +\
                         T.log(2 * np.pi)

    Z = (((y[:,1,:] - mu_1)/sig_1)**2) + (((y[:,2,:] - mu_2) / sig_2)**2) - (2. * (corr * (y[:,1,:] - mu_1)*(y[:,2,:] - mu_2)) / (sig_1 * sig_2))

    inner2 = 0.5 * (1. / (1. - corr**2))
    cost = - (inner1 + (inner2 * Z))

    nll = -logsumexp(T.log(coeff) + cost, axis=1) - c_b
    return nll


def KLGaussianStdGaussian(mu, sig):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and standardized Gaussian dist.

    Parameters
    ----------
    mu  : FullyConnected (Linear)
    sig : FullyConnected (Softplus)
    """
    kl = T.sum(0.5 * (-2 * T.log(sig) + mu**2 + sig**2 - 1), axis=-1)
    return kl


def KLGaussianGaussian(mu1, sig1, mu2, sig2, keep_dims=0):
    """
    Re-parameterized formula for KL
    between Gaussian predicted by encoder and Gaussian dist.

    Parameters
    ----------
    mu1  : FullyConnected (Linear)
    sig1 : FullyConnected (Softplus)
    mu2  : FullyConnected (Linear)
    sig2 : FullyConnected (Softplus)
    """
    if keep_dims:
        kl = 0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) +\
             (sig1**2 + (mu1 - mu2)**2) / sig2**2 - 1)
    else:
        kl = T.sum(0.5 * (2 * T.log(sig2) - 2 * T.log(sig1) + 
                   (sig1**2 + (mu1 - mu2)**2) /
                   sig2**2 - 1), axis=-1)
    return kl
