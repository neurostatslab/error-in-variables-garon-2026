from mc_samplers import Roberts, Sobol
import jax
import jaxopt
import jax.numpy as jnp
from tqdm import trange, tqdm

from functools import partial
from jax import jit
'''class AbstractInference:
    def __init__(self, mc_sampler=Roberts, batching=None, save=):

    def _fit():
        raise NotImplementedError(
            "Must be implemented by subclass"
        )'''

class LBFGS:
    def __init__(self, model, opt_params):
        
        '''if opt_params["init_params"][0]:
            
        else:
            raise Exception("Implement this")'''

        self.model = model
        self.init_params = opt_params["init_params"]
        self.save_prior = opt_params["save_prior"]
        self.opt_key = opt_params["opt_key"]
        self.init_key = opt_params["init_key"]
        self.n_iters = opt_params["n_iters"]


    def fit(self, Y):
        def objective(*args):
            return -1 * self.model.log_posterior_params(*args)

        # Initialize optimization method
        solver = jaxopt.LBFGS(fun=objective)
        est_params = self.init_params
        state = solver.init_state(
            est_params, Y, self.init_key
        )

        objhist = []
        priorhist = []

        for key in tqdm(jax.random.split(self.opt_key, self.n_iters)):
            est_params, state = solver.update(
                est_params, state, Y, key
            )
            if self.save_prior:
                priorhist.append(self.model.observation.mapping.log_density(est_params))
            objhist.append(state.value)

        self.model.params_ = est_params
        self.model.objhist_ = objhist
        
        if self.save_prior:
            self.model.priorhist_ = priorhist
 
# TODO have some sort of struct for saving probs


class ULA:
    def __init__(self, model, opt_params):
        

        self.model = model
        self.opt_key = opt_params["opt_key"]
        self.init_key = opt_params["init_key"]
        self.n_iters = opt_params["n_iters"]
        self.n_chains = opt_params["n_chains"]
        self.friction = opt_params["friction"]
        self.min_lr = opt_params["min_lr"]
        self.max_lr = opt_params["max_lr"]
        self.noise_multiplier = opt_params["noise_multiplier"]
        self.burn_in = opt_params["burn_in"]
        self.save_every = opt_params["save_every"]
        self.params_per_neuron = opt_params["params_per_neuron"]
        self.num_neurons = opt_params["num_neurons"]


        if jnp.any(opt_params["init_params"]):
            self.init_params = opt_params["init_params"]
        else:
            self.init_params = jax.random.normal(
                jax.random.PRNGKey(99),
                shape=(self.n_chains, self.params_per_neuron, self.num_neurons)
            )

    @partial(jit, static_argnums=(0,))
    def mcmc_update(self, params, velocity, key, Y):

        # Update random keys
        k1, k2, k3, k4, key = jax.random.split(key, num=5)
        
        objective = jax.value_and_grad(self.model.log_posterior_params)
        # Compute log joint probability and gradient
        val, grads = objective(params, Y, k1)

        # Sample learning rate from a log uniform distribution
        lr = self.min_lr * ((self.max_lr / self.min_lr) ** jax.random.uniform(k2))

        # Set noise scale according to underdamped Langevin equation.
        noise_scale = self.noise_multiplier * jnp.sqrt(2 * self.friction * lr)

        # Sample noise.
        noise = noise_scale * jax.random.normal(k3, shape=grads.shape)

        # Update parameters and velocity
        new_params = params + velocity
        new_velocity = velocity + (
            -(self.friction * velocity) + (lr * grads) + noise
        )

        return val, new_params, new_velocity, grads

    def fit(self, Y):
        
        # Initialize optimization method
        est_params = self.init_params
        velocity = jnp.zeros_like(est_params)

        
        batched_update = jax.jit(
            jax.vmap(
                self.mcmc_update, in_axes=(0, 0, 0, None)
            )
        )

        objhist = []
        saved_params = []
        saved_grads = []

        for i in trange(self.n_iters):
            keys = jax.random.split(jax.random.PRNGKey(i), num=self.n_chains)
    
            vals, est_params, velocity, grads = batched_update(
                est_params, velocity, keys, Y
            )
            objhist.append(vals)

            if (i > self.burn_in) and (i % self.save_every) == 0:
                saved_params.append(est_params)
                saved_grads.append(grads)

        self.model.saved_params_ = saved_params
        self.model.saved_grads_ = saved_grads
        self.model.objhist_ = jnp.array(objhist)

 