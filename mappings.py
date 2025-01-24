"""
Bijective mapping functions for probabilistic layers.

Each function takes in `(params, x)` and returns `x_next`.

`params` is anything jax can differentiate (i.e. a PyTree).

`x` and `x_next` are jax arrays, matrices with shape (num_particles, num_dims).
"""

import jax
from jax import jit
import jax.numpy as jnp
from functools import partial
from collections import namedtuple


def identity(params, x):
    return x

def linear(A, x):
    return x @ A.T

def affine(params, x):
    A, b = params
    return x @ A.T + b[None, :]


class IdentityMapping:
    def __call__(self, params, x):
        return x

    def sample(self, key, params):
        return params

    def log_density(self, params):
        return 0



class EivMapping:
    def __init__(self, neural_mapping = None, behavioral_mapping = None):
        self.neural_mapping = neural_mapping
        self.behavioral_mapping = behavioral_mapping
    
    def __call__(self, params, xs):

        evals = [mapping(p, xs) for mapping, p in zip(self.mappings, params)]
        
        return evals

    def sample(self, key, params):

        keys = jax.random.split(key, num=len(params))

        samples = [mapping.sample(k, p) for mapping, k, p in zip(self.mappings, keys, params)]
        
        return [samples]

    def log_density(self, params):
        
        dense = [mapping.log_density(p) for mapping, p in zip(self.mappings, params)]

        return jnp.sum(jnp.array(dense))

class CompoundMapping:
    def __init__(self, mappings,  dim_names):
        self.mappings = mappings# if isinstance(mappings, list) else [mappings]
        self.dim_names = dim_names
        
    def __call__(self, params, xs):
        # TODO - this is HACKEY fix this
        params = [params, None]
        
        evals = [mapping(p, xs) for mapping, p in zip(self.mappings, params)]
        '''evals = jax.tree.map(lambda p, mapping: mapping(None, xs) if p is None else mapping(p, xs), tuple(params), tuple(self.mappings),
                            is_leaf=lambda p: p is None)'''
        #Maps = namedtuple('Maps', self.dim_names)
        return evals#Maps(*evals)

    def sample(self, key, params):
        # TODO - this is HACKEY fix this
        params = [params, None]
        keys = jax.random.split(key, num=len(params))
        
        '''samples = jax.tree.map(lambda p, mapping, keys: mapping.sample(k, None) if p is None else mapping.sample(k, p), 
                                params, self.mappings, keys,
                                is_leaf=lambda p: p is None)'''
        samples = [mapping.sample(k, p) for mapping, k, p in zip(self.mappings, keys, params)]
        return samples

    def log_density(self, params):
        # TODO - this is HACKEY fix this
        params = [params, None]
        # TODO - change this back i hate this. 
        '''dense = jax.tree.map(lambda p, mapping: mapping.log_density(None) if p is None else mapping.log_density(p), 
                                tuple(params), tuple(self.mappings),
                                is_leaf=lambda p: p is None)'''
        dense = [mapping.log_density(p) for mapping, p in zip(self.mappings, params)]
        #print("dense, mappings")
        #print(dense)
        return jnp.sum(jnp.array(dense))

class WeightedFourierBasisMapping:

    def __init__(self, params):
        self.max_freq = params['max_freq']
        self.num_dims = params['num_dims']
        self.len_scale = params['len_scale']
        self.out_scale = params['out_scale']
        self.bias_mean = params['bias_mean']
        self.bias_std = params['bias_std']
        self.num_neurons = params['num_neurons']
        self.tol = params['tol']
        self.nonlinearity = params['nonlinearity']
        
        grid = jnp.meshgrid(
            *[jnp.arange(self.max_freq + 1) for _ in range(self.num_dims)]
        )

        # shape is (max_freq,) x  num_dims
        shape = tuple(self.max_freq for _ in range(self.num_dims))
        # enumerate freqs
        idx = jnp.arange(self.max_freq ** self.num_dims)
        # shape is # params x num_dims 
        F = 2*jnp.pi * (
            jnp.column_stack(jnp.unravel_index(idx, shape=shape)))[1:]

        lam = jnp.sum(F ** 2, axis=1)

        # specify kernel
        kern = self.out_scale * jnp.exp(-0.5 * (self.len_scale ** 2) * lam)
        tau = jnp.sqrt(kern)

        # truncate frequencies below tolerance
        thres = jnp.max(tau) * self.tol
        idx = tau > thres
        self.tF, self.ttau = F[idx], tau[idx]
        self.params_per_neuron = 1 + 2 * len(self.ttau)

    @partial(jit, static_argnums=(0,))
    def __call__(self, params, x):
        """
        lattice.shape = [num_basis_funcs, dim_in]
        x.shape = [observations, dim_in]
        sin_coeffs.shape = [num_basis_funcs, dim_out]
        cos_coeffs.shape = [num_basis_funcs, dim_out]
        bias.shape = [dim_out]
        """
        # coeffs.shape = n_neurons x n_bases
        # x.shape = n_samples x n_dims
        # lattice.shape = n_bases x n_dims
        # z.shape = n_samples x n_bases
        bias = params[0]
        sin_coeffs, cos_coeffs = jnp.array_split(params[1:], 2)
        z = x @ self.tF.T # [observations, num_basis_funcs]
        
        wsin = jnp.sin(z) @ sin_coeffs # [observations, dim_out]
        wcos = jnp.cos(z) @ cos_coeffs # [observations, dim_out]
        
        return self.nonlinearity(bias + wsin + wcos) # [observations, dim_out]

    def sample(self, key, FIXTHIS):
        # TODO - dont like this
        return (jax.random.normal(
                        key, shape=(self.num_neurons, self.params_per_neuron)
                    )).T
        
    @partial(jit, static_argnums=(0,))
    def log_density(self, params):
        """l0 = jax.scipy.stats.norm.logpdf(
            params['bias'],
            loc=self.bias_mean,
            scale=self.bias_std
        )
        l1 = jax.scipy.stats.norm.logpdf(
            params['cos_coeffs'], loc=0.0, scale=self.tau
        )
        l2 = jax.scipy.stats.norm.logpdf(
            params['sin_coeffs'], loc=0.0, scale=self.tau
        )
        return jnp.sum(l0) + jnp.sum(l1) + jnp.sum(l2)"""
        return jnp.sum(jax.scipy.stats.norm.logpdf(params))



class WeightedLinearMapping:

    def __init__(self, params):
        self.dim_in = params['dim_in']
        self.w_variance = params['w_variance']
        self.nonlinearity = params['nonlinearity']

    def __call__(self, params, x):
        """
        x.shape = [observations, dim_in]
        """
        wx = x @ params["w"].T  # [observations, dim_in]
        return self.nonlinearity(wx) # [observations, dim_in]

    def sample(self, key, params):
        
        k0, k1, k2 = jax.random.split(key, num=3)

        params = {
            "w": jax.random.normal(
                            k1, shape=(params['num_neurons'], params['dim_in'])
                        )
        }

        return params

    def log_density(self, params):
        
        l0 = jax.scipy.stats.norm.logpdf(
            params['w'], loc=0.0, scale=self.w_variance
        )

        return jnp.sum(l0)

