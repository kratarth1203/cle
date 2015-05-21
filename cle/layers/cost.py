import ipdb
import copy
import numpy as np
import scipy
import theano.tensor as T

from theano.compat.python2x import OrderedDict
from cle.cle.cost import Gaussian, GMM, NllBin, NllMul, MSE
from cle.cle.layers import RandomCell, StemCell
from cle.cle.utils import sharedX, tolist, unpack, predict


class CostLayer(StemCell):
    """
    Base cost layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self, use_sum=False, **kwargs):
        super(CostLayer, self).__init__(**kwargs)
        self.use_sum = use_sum

    def fprop(self, X):
        raise NotImplementedError(
            str(type(self)) + " does not implement Layer.fprop.")

    def initialize(self):
        pass


class BinCrossEntropyLayer(CostLayer):
    """
    Binary cross-entropy layer

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, X):
        cost = NllBin(X[0], X[1])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()


class MulCrossEntropyLayer(CostLayer):
    """
    Multi cross-entropy layer

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, X):
        cost = NllMul(X[0], X[1])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()


class MSELayer(CostLayer):
    """
    Mean squared error layer

    Parameters
    ----------
    .. todo::
    """
    def fprop(self, X):
        cost = MSE(X[0], X[1])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()


class GaussianLayer(CostLayer):
    """
    Linear Gaussian layer

    Parameters
    ----------
    .. todo::
    """
    def __init__(self,
                 use_sample=False,
                 **kwargs):
        super(GaussianLayer, self).__init__(**kwargs)
        self.use_sample = use_sample
        if use_sample:
            self.fprop = self.which_fn('sample')
        else:
            self.fprop = self.which_fn('cost')

    def cost(self, X):
        if len(X) != 3:
            raise ValueError("The number of inputs does not match.")
        cost = Gaussian(X[0], X[1], X[2])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()

    def sample(self, X):
        mu = X[0]
        sig= X[1]
        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + sig * epsilon
        return z

    def __getstate__(self):
        dic = self.__dict__.copy()
        dic.pop('fprop')
        return dic
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if self.use_sample:
            self.fprop = self.which_fn('sample')
        else:
            self.fprop = self.which_fn('cost')


class GMMLayer(GaussianLayer):
    """
    Gaussian mixture model layer

    Parameters
    ----------
    .. todo::
    """
    def cost(self, X):
        if len(X) != 4:
            raise ValueError("The number of inputs does not match.")
        cost = GMM(X[0], X[1], X[2], X[3])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()

    def sample(self, X):
        mu = X[0]
        sig = X[1]
        coeff = X[2]
        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))
        sig = sig.reshape((sig.shape[0],
                           sig.shape[1]/coeff.shape[-1],
                           coeff.shape[-1]))
        idx = predict(
            self.theano_rng.multinomial(
                pvals=coeff,
                dtype=coeff.dtype
            ),
            axis=1
        )
        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = sig[T.arange(sig.shape[0]), :, idx]
        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + sig * epsilon
        return z

    def argmax_mean(self, X):
        mu = X[0]
        coeff = X[1]
        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))
        idx = predict(coeff)
        mu = mu[T.arange(mu.shape[0]), :, idx]
        return mu

    def sample_mean(self, X):
        mu = X[0]
        sig = X[1]
        coeff = X[2]
        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))
        sig = sig.reshape((sig.shape[0],
                           sig.shape[1]/coeff.shape[-1],
                           coeff.shape[-1]))
        idx = predict(
            self.theano_rng.multinomial(
                pvals=coeff,
                dtype=coeff.dtype
            ),
            axis=1
        )
        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = sig[T.arange(sig.shape[0]), :, idx]
        epsilon = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        z = mu + sig * epsilon
        return z, mu


class biGaussLayer(GaussianLayer):
    """
    bivariate Gaussian

    Parameters
    ----------
    .. todo::
    """
    def cost(self, X):
        if len(X) != 5:
            raise ValueError("The number of inputs does not match.")
        cost = GMM(X[0], X[1], X[2], X[3])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()

    def sample(self, X):
        mu = X[0]
        sig = X[1]
        corr = X[3]
        binary = X[4]
        
        mu_x = mu[:,0]
        mu_y = mu[:,1]
        sig_x = sig[:,0]
        sig_y = sig[:,1]
     
        z = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        s_x = (mu_x + sig_x * z[:,0]).dimshuffle(0,'x')
        s_y = mu_y + sig_y * ( (z[:,0] * corr) + (z[:,1] * T.sqrt(1.-corr**2) ) ).dimshuffle(0,'x')
        s = T.concatenate([binary,s_x,s_y], axis = 1)
        return s

    def sample_mean(self, X):
        mu = X[0]
        sig = X[1]
        corr = X[3]
        binary = X[4]
        
        mu_x = mu[:,0]
        mu_y = mu[:,1]
        sig_x = sig[:,0]
        sig_y = sig[:,1]
     
        z = self.theano_rng.normal(size=mu.shape,
                   avg=0., std=1.,
                   dtype=mu.dtype)
        s_x = (mu_x + sig_x * z[:,0]).dimshuffle(0,'x')
        s_y = (mu_y + sig_y * ( (z[:,0] * corr) + (z[:,1] * T.sqrt(1.-corr**2) ) )).dimshuffle(0,'x')
        s_t = T.concatenate([binary,s_x,s_y], axis = 1)
        
        return s_t, mu


class biGMMLayer(GaussianLayer):
    """
    Gaussian mixture model layer

    Parameters
    ----------
    .. todo::
    """
    def cost(self, X):
        if len(X) != 6:
            raise ValueError("The number of inputs does not match.")
        cost = GMM(X[0], X[1], X[2], X[3])
        if self.use_sum:
            return cost.sum()
        else:
            return cost.mean()

    def sample(self, X):
        mu = X[0]
        sig = X[1]
        coeff = X[2]
        corr = X[3]
        binary = X[4]
        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))
        sig = sig.reshape((sig.shape[0],
                           sig.shape[1]/coeff.shape[-1],
                           coeff.shape[-1]))
        idx = predict(
            self.theano_rng.multinomial(
                pvals=coeff,
                dtype=coeff.dtype
            ),
            axis=1
        )
        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = sig[T.arange(sig.shape[0]), :, idx]
        corr = corr[T.arange(corr.shape[0]), idx]
        
        mu_x = mu[:,0]
        mu_y = mu[:,1]
        sig_x = sig[:,0]
        sig_y = sig[:,1]
     
        z = self.theano_rng.normal(size=mu.shape,
                                         avg=0., std=1.,
                                         dtype=mu.dtype)
        s_x = (mu_x + sig_x * z[:,0]).dimshuffle(0,'x')
        s_y = mu_y + sig_y * ( (z[:,0] * corr) + (z[:,1] * T.sqrt(1.-corr**2) ) ).dimshuffle(0,'x')
        s = T.concatenate([binary,s_x,s_y], axis = 1)
        ipdb.set_trace()
        return s

    def argmax_mean(self, X):
        mu = X[0]
        coeff = X[1]
        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))
        idx = predict(coeff)
        mu = mu[T.arange(mu.shape[0]), :, idx]
        return mu

    def sample_mean(self, X):
        mu = X[0]
        sig = X[1]
        coeff = X[2]
        corr = X[3]
        binary = X[4]
        mu = mu.reshape((mu.shape[0],
                         mu.shape[1]/coeff.shape[-1],
                         coeff.shape[-1]))
        sig = sig.reshape((sig.shape[0],
                           sig.shape[1]/coeff.shape[-1],
                           coeff.shape[-1]))
        idx = predict(
            self.theano_rng.multinomial(
                pvals=coeff,
                dtype=coeff.dtype
            ),
            axis=1
        )
        mu = mu[T.arange(mu.shape[0]), :, idx]
        sig = sig[T.arange(sig.shape[0]), :, idx]
        corr = corr[T.arange(corr.shape[0]), idx]
        
        mu_x = mu[:,0]
        mu_y = mu[:,1]
        sig_x = sig[:,0]
        sig_y = sig[:,1]
     
        z = self.theano_rng.normal(size=mu.shape,
                   avg=0., std=1.,
                   dtype=mu.dtype)
        s_x = (mu_x + sig_x * z[:,0]).dimshuffle(0,'x')
        s_y = (mu_y + sig_y * ( (z[:,0] * corr) + (z[:,1] * T.sqrt(1.-corr**2) ) )).dimshuffle(0,'x')
        s_t = T.concatenate([binary,s_x,s_y], axis = 1)
        
        return s_t, mu
