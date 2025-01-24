import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
from functools import partial
import smc
import mappings
import noise_models
import inference
import matplotlib.pyplot as plt
from mc_samplers import Roberts

from functools import partial
from jax import jit

class AbstractGPLVM:
    def __init__(
            self, observation=None,  num_samples=2**10,
            method = "LPFGS", opt_params = None
        ):

        # Construct truncated bases functions
        self.observation = observation
        #self.params_per_neuron = observation.mapping.params_per_neuron
        # TODO - make this a model param, problem pulling this from compound mapping


        # TODO - maybe move this to inference class
        self.num_samples = num_samples
        self.method = method
        self.opt_params = opt_params

    def marginal_log_likelihood_params(self, params, y, key):
        # TODO - could maybe rename, calculating boith samples and likelihood in this step
        raise NotImplementedError(
            "marginal_log_likelihood_params function must be implemented by subclass."
        )

    def simulate(self, params, y, key):
        raise NotImplementedError(
            "Simulate function must be implemented by subclass."
        )

    @partial(jit, static_argnums=(0,))
    def logp_y_given_x(self, params, y, xs):
        
        log_pdf = self.observation.log_density(
            params, xs, y
        )
        return log_pdf

    @partial(jit, static_argnums=(0,))
    def log_posterior_params(self, params, y, key):
        return (
            jnp.sum(self.marginal_log_likelihood_params(params, y, key)) +
            self.observation.mapping.log_density(params)
        )
    
    def fit(self, Y, method, opt_params):
        
        _fitting_methods = \
            dict(adam=inference.Adam,
                 lbfgs=inference.LBFGS,
                 ula = inference.ULA
                 )

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        method = _fitting_methods[method](self, opt_params)

        return method.fit(Y)


class GPLVM(AbstractGPLVM):

    def __init__(self, *args, sampler = Roberts(),
                 **kwargs):
        super().__init__(*args, **kwargs)
        # TODO - messy
        #self.params_per_neuron = self.observation.mapping.mappings[0].params_per_neuron

        # Not dynamic, have to specify sampling procedure
        self.sampler = sampler

    def marginal_log_likelihood_params(self,params, y, key):
        @partial(jax.vmap, in_axes=(None, 0, None), out_axes=0)
        def _marginal_log_likelihood_params(params, y, key):
            # might need to separate this out
            xs = self.sampler.sample(key, self.num_samples)
            return self.logp_y_given_x(params, y, xs)
        return  logsumexp(_marginal_log_likelihood_params(params, y, key), axis=1)
    
    def simulate(self, key, params, num_observations):
        k1, k2 = jax.random.split(key, 2)
        
        x = self.sampler.sample(
            k1, num_observations
        )

        y = self.observation.sample(
            k2, params, x
        )

        return x, y
        
class DynamicGPLVM(AbstractGPLVM):

    def __init__(self, *args, transition=None, proposal=None,**kwargs):
        super().__init__(*args, **kwargs)
        self.transition = transition
        #self.params_per_neuron = self.observation.mapping.mappings[0].params_per_neuron
        self.proposal = proposal

    def log_transition_prob(self, params, x, x_last):
        log_pdf = self.transition.log_density(
            params, x_last, x
        )
        return log_pdf

    def marginal_log_likelihood_params(self, params, y, key):
        

        log_avg_weights = smc.log_marginal_likelihood(
                    params, y, key, self.proposal.initialize,  
                    self.proposal.sample, self.log_transition_prob,
                    self.logp_y_given_x, self.num_samples,
            )

        return log_avg_weights


    def simulate(self, key, params, x_init, num_timesteps):
        
        def scanned_func(x_last, k):
            k1, k2 = jax.random.split(k, 2)
            x = self.transition.sample(
                k1, params, x_last
            )
            
            y = self.observation.sample(
                k2, params, x
            )
            return x, (x, y)
        return jax.lax.scan(
            scanned_func, x_init, jax.random.split(key, num_timesteps)
        )[1]

class Layer:
    """
    Probabilistic mapping layer, f(x) + noise.
    """

    def __init__(self, mapping, noise):
        self.mapping = mappings.CompoundMapping(mapping,None) if isinstance(mapping, list) else mapping
        self.noise = noise_models.CompoundNoiseModel(noise,None)  if isinstance(noise, list) else noise

    def log_density(self,params, x, y):
        loc = self.mapping(params, x) 
        #TODO - double check that specifying noise is correct

        return self.noise.log_density(loc, y)
    
    def sample(self, key, params, x):

        loc = self.mapping(params, x)
        return self.noise.sample(key, loc)
    

class Proposal:
    """
    Proposal distribution (used for sequential monte carlo).
    """
    #TODO - pretty much got rid of init params, do something about this
    def __init__(self, layer, params, init_params, init_loc):
        self.params = params
        self.layer = layer
        self.init_params = init_params
        self.init_loc = init_loc

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key, x_last):
        x = self.layer.sample(key, self.params, x_last)
        log_pdf = self.layer.log_density(self.init_params, x_last, x)

        return x, log_pdf

    @partial(jax.jit, static_argnums=(0,2))
    def initialize(self, key, num_particles):
        loc = jnp.tile(self.init_loc, (num_particles, 1))
        # loc = (n_particles x 1)
        
        x = self.layer.sample(key, self.init_params, loc)
        # x = (n_particles x n_dims)

        log_pdf = self.layer.log_density(self.init_params, loc, x)
        
        return x, log_pdf

    @partial(jax.jit, static_argnums=(0,))
    def initialize_samp(self, key, num_particles):
        
        loc = jnp.tile(self.init_loc, (num_particles, 1))
        # loc = (n_particles x 1)
        
        x = self.layer.sample(key, self.init_params, loc)
        # x = (n_particles x n_dims)

        return x
