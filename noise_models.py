import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from jax.scipy.special import logsumexp
from jax.scipy.special import gamma
from scipy.stats.qmc import Sobol
import scipy
from functools import partial
from collections import namedtuple

# TODO - Latent space must norm to 1, unit check noise models

class IsotropicGaussian:
    def __init__(self, log_std):
        self.log_std = log_std

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        s = jax.nn.softplus(self.log_std)
        s2 = s ** 2
        z = (x - loc) / s
        z2 = z ** 2
        return jnp.sum(-0.5 * (jnp.log(s2 * 2 * jnp.pi) + z2), axis=-1)
    
    @partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        return loc + jax.nn.softplus(self.log_std) * jax.random.normal(key, shape=loc.shape)

class Gaussian:
    def __init__(self, log_std):
        self.log_std = log_std

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        return jnp.sum(jax.scipy.stats.norm.logpdf(x, loc=loc, scale = self.log_std), axis = -1)
    
    
    @partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        return loc + self.log_std * jax.random.normal(key, shape=loc.shape)

class Poisson:
       
    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, k: int):
        
        return jnp.sum(k * jnp.log(loc) - loc - jax.scipy.special.gammaln(k + 1), axis=-1)

    @partial(jit, static_argnums=(0,))
    def sample(self, key, loc):

        return jax.random.poisson(key, loc)


class Beta:

    def __init__(self, var, scale):
        self.var = var
        self.scale = scale

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        loc = jnp.clip(loc/self.scale, 0.2, 0.8)
        x = jnp.clip(x/self.scale, 0.2, 0.8)
        sum_ = (loc*(1. - loc)/self.var)-1.
        a = jax.nn.relu(sum_ * loc)
        b = jax.nn.relu(sum_ * (1 - loc))
        return jnp.sum(jax.scipy.stats.beta.logpdf(x, a, b), axis=-1)
    
    @partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        loc = jnp.clip(loc/self.scale, 0.2, 0.8)
        sum_ = (loc*(1. - loc)/self.var)-1.
        a = jax.nn.relu(sum_ * loc)
        b = jax.nn.relu(sum_ * (1 - loc))
        return jax.random.beta(key, a, b, shape=loc.shape)*self.scale

class Uniform:

    def __init__(self, minval, maxval):
        assert maxval > minval
        self.minval = minval
        self.maxval = maxval
    
    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        num_dims = x.shape[1]
        return jnp.sum(jnp.full(
            x.shape,
            -num_dims * jnp.log(self.maxval - self.minval)
        ), axis=-1)

    @partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        """
        Parameters
        ----------
        minval,maxval : float64
            Minimum calue of uniform dist

        Returns
        -------
        X: Array
            (num_mc_samples x num_dimensions)
        """
        return jax.random.uniform(
            key,
            minval=self.minval,
            maxval=self.maxval,
            shape=loc.shape
        )

class UniformSobol:

    def __init__(self, minval, maxval):
        assert maxval > minval
        self.minval = minval
        self.maxval = maxval

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        num_dims = x.shape[1]
        return jnp.sum(jnp.full(
                x.shape,
                -num_dims * jnp.log(self.maxval - self.minval)
            ), axis=-1)

    
    #@partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        """
        Parameters
        ----------
        minval,maxval : float64
            Range of samples

        Returns
        -------
        X: Array
            (num_mc_samples x num_dimensions)
        """
        n_samples, n_dims = loc.shape
        qrng = Sobol(n_dims, seed=0)
        xs = jnp.array(qrng.random(n=n_samples) * \
                        (self.maxval-self.minval))  + self.minval

        return xs

class ProjectedNormal:
    """
    TODO - Note: assumes no correlation between dimensions
    TODO - Add an average function -> circular average
    Draw random samples from a projected normal distribution for a 1d
    circular variable, theta. The generative model is

        x ~ N(m1, 1)
        y ~ N(m2, 1)
        theta = arctan2(y, x)
    """
    def __init__(self, conc):
        self.conc = conc

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        """
        See Equation 1 of Wang & Gelfand, "Directional data analysis under
        the general projected normal distribution"
        """
        cos_loc = jnp.cos(loc)
        sin_loc = jnp.sin(loc)
        D = self.conc * (cos_loc * jnp.cos(x) + sin_loc * jnp.sin(x))
        F = self.conc * (cos_loc * jnp.sin(x) - sin_loc * jnp.cos(x))
        return jnp.sum(jnp.log(
            jax.scipy.stats.norm.pdf(cos_loc) * jax.scipy.stats.norm.pdf(sin_loc)
            + D * jax.scipy.stats.norm.cdf(D) * jax.scipy.stats.norm.pdf(F)
        ), axis=-1)

    @partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        k1, k2 = jax.random.split(key)
        x = self.conc * jnp.cos(loc) + jax.random.normal(k1, shape=loc.shape)
        y = self.conc * jnp.sin(loc) + jax.random.normal(k2, shape=loc.shape)
        return jnp.arctan2(y, x)

class ProjectedNormalNormed:
    """
    TODO - Note: assumes no correlation between dimensions
    TODO - Add an average function -> circular average
    Draw random samples from a projected normal distribution for a 1d
    circular variable, theta. The generative model is

        x ~ N(m1, 1)
        y ~ N(m2, 1)
        theta = arctan2(y, x)
    """
    def __init__(self, conc):
        self.conc = conc

    #@partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        """
        See Equation 1 of Wang & Gelfand, "Directional data analysis under
        the general projected normal distribution"
        """
        # TODO - add assertion - distance between mc samples cant be larger than concentration
        # or things will get weird
        x = (x *2*jnp.pi)-jnp.pi
        loc = (loc *2*jnp.pi)-jnp.pi
        # TODO - Pretty sure there is an issue here, look into this, go with von mis for now
        
        cos_loc = jnp.cos(loc)
        sin_loc = jnp.sin(loc)
        D = self.conc * (cos_loc * jnp.cos(x) + sin_loc * jnp.sin(x))
        F = self.conc * (cos_loc * jnp.sin(x) - sin_loc * jnp.cos(x))
        return jnp.sum(jnp.log(
            jax.scipy.stats.norm.pdf(cos_loc) * jax.scipy.stats.norm.pdf(sin_loc)
            + D * jax.scipy.stats.norm.cdf(D) * jax.scipy.stats.norm.pdf(F)
        ), axis=-1)

    #@partial(jit, static_argnums=(0,))
    def sample(self, key, loc):

        loc = (loc *2*jnp.pi)-jnp.pi
        k1, k2 = jax.random.split(key)
        x = self.conc * jnp.cos(loc) + jax.random.normal(k1, shape=loc.shape)
        y = self.conc * jnp.sin(loc) + jax.random.normal(k2, shape=loc.shape)
        return (jnp.arctan2(y, x)+jnp.pi)/(2*jnp.pi)

class VonMises:
    def __init__(self, kappa):
        self.kappa = kappa

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        S_centered = x - loc
        S_centered = (S_centered - jnp.pi) % (2 * jnp.pi)+ jnp.pi
        f = jax.scipy.stats.vonmises.logpdf(S_centered, self.kappa)
        
        return jnp.sum(f, axis=-1)

    #@partial(jit, static_argnums=(0,))
    def sample(self, key,loc):
        # TODO - Cant find jax.random implementation of von mises :(
        num_seed = jax.random.randint(key, (), 0, 2**32).item()
        np.random.seed(num_seed)

        n_samps = loc.shape
        noise = jnp.array(scipy.stats.vonmises(loc=np.zeros((n_samps)), 
                                                kappa=self.kappa).rvs((n_samps)))
        return (loc+noise)%(2*jnp.pi)

class VonMisesNormed:
    def __init__(self, kappa):
        self.kappa = kappa

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        x = (x *2*jnp.pi)
        loc = (loc *2*jnp.pi)
        
        S_centered = x - loc
        S_centered = (S_centered - jnp.pi) % (2 * jnp.pi)+ jnp.pi
        f = jax.scipy.stats.vonmises.logpdf(S_centered, self.kappa)
        
        return jnp.sum(f, axis=-1)

    
    def sample(self, key,loc):
        num_seed = jax.random.randint(key, (), 0, 2**32).item()
        np.random.seed(num_seed)

        loc = loc * 2*jnp.pi
        n_samps = loc.shape
        noise = jnp.array(scipy.stats.vonmises(loc=np.zeros((n_samps)), 
                                                kappa=self.kappa).rvs((n_samps)))
        return ((loc+noise)/(2*jnp.pi))%1


class EIVNoiseModel:
   
    def __init__(self, noise_models):
        self.noise_models = noise_models 

    @partial(jit, static_argnums=(0,))
    def log_density(self, locs, xs):
        evals = jnp.array([noise.log_density(loc, x) for noise, loc, x in zip(self.noise_models, locs, xs)])
        
        return jnp.sum(evals, axis=0)
        
    
    def sample(self, key, locs):
        
        keys = jax.random.split(key, num=2)
        samples = [no.sample(k, l) for no, k, l in zip(self.noise_models, keys, locs)]     
        
        return samples
 


class CompoundNoiseModel:
   
    def __init__(self, noise_models):
        self.noise_models = noise_models if isinstance(noise_models, list) else [noise_models]
    
    @partial(jit, static_argnums=(0,))
    def log_density(self, locs, xs):
        evals = jnp.array([noise.log_density(loc, x) for noise, loc, x in zip(self.noise_models, locs, xs)])
        return jnp.sum(evals, axis=0)
        
    def sample(self, key, locs):
        # TODO Sample shouldnt be jit compiled - seeding scikit issue
        keys = jax.random.split(key, num=2)
        samples = [no.sample(k, l) for no, k, l in zip(self.noise_models, keys, locs)]     
        
        return samples
 