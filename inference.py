from mc_samplers import Roberts, Sobol
import jax
import optax
import jaxopt
import jax.numpy as jnp
from tqdm import trange, tqdm
from jax.scipy.special import logsumexp

from functools import partial
from jax import jit
from distrax import MultivariateNormalDiag

class Adam:
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
        self.learning_rate = opt_params["lr"]

    # TODO - Pynapple for fit transform?
    
    def fit(self, Y):
        
        objective = jax.value_and_grad(self.model.log_posterior_params)

        # Initialize optimization method
        optimizer = optax.chain(
            optax.adam(learning_rate=self.learning_rate),
            optax.scale(-1.0)
        )
        est_params = self.init_params

        opt_state = optimizer.init(est_params)

        objhist = []
        priorhist = []

        for key in tqdm(jax.random.split(self.opt_key, self.n_iters)):
            val, grads = objective(est_params, Y, key)
            updates, opt_state = optimizer.update(grads, opt_state)
            est_params = optax.apply_updates(est_params, updates)
                
            if self.save_prior:
                priorhist.append(self.model.observation.mapping.log_density(est_params))
            objhist.append(val)

        self.model.params_ = est_params
        self.model.objhist_ = objhist
        # TODO - rename this point estimate?
        
        if self.save_prior:
            self.model.priorhist_ = priorhist

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
        stim_hist = []
        cond_hist = []

        for key in tqdm(jax.random.split(self.opt_key, self.n_iters)):
            est_params, state = solver.update(
                est_params, state, Y, key
            )
            if self.save_prior:
                priorhist.append(self.model.observation.mapping.log_density(est_params))
                objhist.append(state.value)

        self.model.params_ = est_params
        self.model.objhist_ = objhist
        # TODO - rename this point estimate?
        
        if self.save_prior:
            self.model.priorhist_ = priorhist
 
# TODO have some sort of struct for saving probs


class ULA:
    def __init__(self, model, opt_params):
        

        self.model = model
        self.opt_key = opt_params["opt_key"]
        self.init_key = opt_params["init_key"]

        # TODO - just split this from opt key?
        self.is_key = opt_params["is_key"]
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


        if jnp.any(opt_params["init_params"][0]):
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

        vals, grads = objective(params, Y, k1)

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
        return vals, new_params, new_velocity, grads

    def multiple_importance_sampler(self, func, prms, noise_dist, key, Y):
        n, d1, d2 = prms.shape
        
        k1, k2 = jax.random.split(key, num=2)
        noise = noise_dist.sample(seed=k1, sample_shape=n)
        log_p = func(prms + noise.reshape(-1, d1, d2), Y, k2)
        log_q = noise_dist.log_prob(noise)
        log_w = log_p - log_q
        est_log_Z = logsumexp(log_w) - jnp.log(n)

        indep_is_w = log_w-logsumexp(log_w)
        # marginal
        est_elbo = jnp.mean(log_w)

        return est_log_Z, est_elbo, indep_is_w

    def fit(self, Y, n_chains_keep = 2, kde_bandwidth = (-2, -1), kde_reso = 150, num_importance_iters = 500):
        # TODO - move these to optimization dict
        # TODO - set default optimization dict?
        # Initialize optimization method
        est_params = self.init_params
        velocity = jnp.zeros((self.n_chains, self.params_per_neuron, self.num_neurons))
        
        #### TODO - 
        batched_update = jax.jit(
            jax.vmap(
                self.mcmc_update, in_axes=(0, 0, 0, None)
            )
        )

        objhist = []
        saved_params = []
        saved_grads = []
        saved_vals = []

        for i in trange(self.n_iters):

            keys = jax.random.split(jax.random.PRNGKey(i), num=self.n_chains)
    
            vals, est_params, velocity, grads = batched_update(
                est_params, velocity, keys, Y
            )
            objhist.append(vals)

            if (i > self.burn_in) and (i % self.save_every) == 0:
                saved_params.append(est_params)
                saved_grads.append(grads)
                saved_vals.append(vals)


        self.model.saved_params_ = jnp.array(saved_params)
        self.model.saved_grads_ = jnp.array(saved_grads)
        self.model.objhist_ = jnp.array(objhist)
        rank_order_chains = jnp.argsort(jnp.mean(self.model.objhist_[self.burn_in:], axis=0))[::-1]
        self.model.rank_order_chains_ = rank_order_chains

        # Create batched posterior for MIS.
        batched_posterior = jax.jit(jax.vmap(
            self.model.log_posterior_params, in_axes=(0, None, None)
        ))

        hs = jnp.logspace(kde_bandwidth[0], kde_bandwidth[1], kde_reso)
        log_Z_ests = [[] for _ in range(n_chains_keep)]
        elbos = [[] for _ in range(n_chains_keep)]

        for c in range(n_chains_keep):
            for i, h in enumerate(tqdm(hs)):
                subkey, key = jax.random.split(self.is_key, 2)
                noise_dist = MultivariateNormalDiag(
                    jnp.zeros(self.params_per_neuron * self.num_neurons),
                    h * jnp.ones(self.params_per_neuron * self.num_neurons)
                )
                lgZ, elb, _ = self.multiple_importance_sampler(
                    batched_posterior,
                    self.model.saved_params_[:, rank_order_chains[c]],
                    noise_dist,
                    subkey, Y
                )
                log_Z_ests[c].append(lgZ)
                elbos[c].append(elb)
        log_Z_ests = jnp.array(log_Z_ests)
        elbos = jnp.array(elbos)

        self.model.best_h_ = hs[jnp.argmax(elbos[0])]
        best_noise_dist = MultivariateNormalDiag(
            jnp.zeros(self.params_per_neuron * self.num_neurons),
            self.model.best_h_ * jnp.ones(self.params_per_neuron * self.num_neurons)
        )

        
        best_lgZs = [[] for _ in range(n_chains_keep)]

        key = self.is_key
        for c in range(n_chains_keep):
            for i in trange(num_importance_iters):
                subkey, key = jax.random.split(key, 2)
                lgZ, _, indep_is_w = self.multiple_importance_sampler(
                    batched_posterior,
                    self.model.saved_params_[:, rank_order_chains[c]],
                    noise_dist,
                    subkey, Y
                )
                best_lgZs[c].append(lgZ)

        self.model.indep_is_w_ = indep_is_w
        self.model.best_lgZs_ = jnp.array(best_lgZs)

                