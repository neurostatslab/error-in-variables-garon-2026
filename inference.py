from mc_samplers import Roberts, Sobol
import jax
import optax
import jaxopt
import smc
import time
import jax.numpy as jnp
import numpy as np
from tqdm import trange, tqdm
from jax.scipy.special import logsumexp
from functools import partial
from jax import jit
from distrax import MultivariateNormalDiag

"""
inference.py
=============
Gradient-based and MCMC optimization routines for EIV model.
 
All optimizers share a common interface:
 
    opt = OptimizerClass(model, opt_params)
    opt.fit(Y)
 
After calling ``fit``, the trained parameters are stored in ``model.params_``
(point-estimate methods) or ``model.saved_params_`` (sampling methods), along
with diagnostic likelihood histories stored in ``model.objhist_`` and ``model.priorhist_``.
 
Point-estimate optimizers
-------------------------
- ``Adam``                  – single-trial Adam, supports batching and stopping conditions
- ``SGD``                   – SGD with Nesterov momentum and cosine-warmup schedule
- ``LBFGS``                 – L-BFGS via jaxopt
 
Sampling / variational methods
-------------------------------
- ``ULA``                   – Underdamped Langevin Algorithm with Multiple
                              Importance Sampling (MIS) for marginal-likelihood
                              estimation
 
Dependencies
------------
- JAX / JAXopt
- Optax
- tqdm

Notes
-----
- All optimizers expect ``model`` to expose a ``log_posterior_params(params, Y, key)``
  method returning a scalar log-posterior value.
- Random-number keys should be JAX ``PRNGKey`` objects.
- ``Y`` may be a single array or a tuple of arrays depending on whether the model is a GPLVM or an EIV model.

To - Do
-----
- Fix save syntax for sampling method params to be consistent
- Consolidate batching to be separate from inference methods, include random sampling method
- Simplify initialization procedure:
    if opt_params["init_params"][0]: use params else: initialize randomly
- Clarify where point estimate or full posterior
- Make pytree handling more consistent
- Make opt_params dictionary syntax more consistent
- ADD DIMENSIONS
"""


class Adam:
    """Adam optimizer for MAP estimation, with either full-batch or
     minibatch a fixed set of trial windows.

    ---> Primarily used for particle filter, smoothing over fixed windows of time
         smaller than the full recording duration

    Parameters
    ----------
    model : object
        Model instance that exposes:
        - ``log_posterior_params(params, Y_single, key) -> scalar``
        - ``observation.mapping.log_density(params) -> scalar``
    opt_params : dict
        Configuration dictionary with the following keys:

        save_prior : bool
            If ``True``, record ``log_density(params)`` at every step.
        opt_key : jax.random.PRNGKey
            Key used to generate per-step random keys and epoch shuffles.
        init_key : jax.random.PRNGKey
            Reserved initialisation key (unused when ``init_params`` is given).
        n_iters : int
            Number of gradient steps in full-batch mode, or number of
            *epochs* (full passes over all windows) in minibatch mode.
        batch_size : int or False
            Window length (in timesteps) used to reshape ``Y`` into
            ``(num_batch, batch_size, n)``. Required (not ``False``) when
            ``minibatch=True``.
        minibatch : bool
            If ``True``, train with classic minibatch SGD: each step uses
            exactly one randomly-selected window (no replacement), cycling
            through all windows once per epoch. If ``False`` (default),
            uses the original full-batch behavior: every step computes the
            gradient over *all* windows at once via ``jax.vmap``.
        scale_likelihood : bool
            Only used when ``minibatch=True``. If ``True`` (default), the
            single-window log-likelihood is scaled by ``num_batch`` so its
            expectation over a shuffled epoch matches the true full-data
            log-likelihood sum -- this keeps gradient step sizes and the
            relative weight of the prior comparable to full-batch training.
            Set ``False`` to use the raw, unscaled per-window log-likelihood.
        lr : float or optax schedule
            Learning rate for ``optax.adam``.
        tol_loss : float or False
            Relative loss-change stopping tolerance. In full-batch mode this
            is checked every step; in minibatch mode it's checked once per
            epoch (against the mean loss over that epoch), since per-step
            loss in SGD is inherently noisy. If ``False``, disables early
            stopping and runs for exactly ``n_iters`` steps/epochs.

    Attributes
    ----------
    model.params_ : pytree
        Estimated parameters after optimisation.
    model.objhist_ : list of float
        Per-step loss (full-batch mode) or per-window loss (minibatch mode).
    model.priorhist_ : list of float
        Log-prior at every step (only if ``save_prior=True``).
    model.g_norms_ : list of float
        Relative gradient norm per step, diagnostics only (full-batch,
        ``tol_loss`` set) -- not populated in minibatch mode.
    model.rel_loss_hist_ : list of float
        Relative loss change -- per step (full-batch) or per epoch (minibatch).
    model.iters_run_ : int
        Number of steps (full-batch) or epochs (minibatch) actually run.
    """

    DEFAULTS = {
            "init_params":None,
            "save_prior": True,
            "n_iters": 1000,
            "tol_loss": False,
            "minibatch": False,
            "scale_likelihood": True,
            "batch_size": False,
            "lr": 1e-1,
        }
    REQUIRED = ("opt_key", "init_key")

    def __init__(self, model, opt_params):

        allowed = set(self.DEFAULTS) | set(self.REQUIRED)
        unknown = set(opt_params) - allowed
        if unknown:
            raise ValueError(
                f"Unrecognized opt_params key(s): {sorted(unknown)}. "
                f"Allowed keys: {sorted(allowed)}"
            )

        missing = [k for k in self.REQUIRED if k not in opt_params]
        if missing:
            raise ValueError(f"opt_params is missing required key(s): {missing}")

        self.model = model
        self.opt_key = opt_params["opt_key"]
        self.init_key = opt_params["init_key"]

        for key, default in self.DEFAULTS.items():
            setattr(self, key, opt_params.get(key, default))
        
        if self.minibatch and self.batch_size == False:
            raise ValueError("minibatch=True requires a non-False `batch_size`.")

    def batch_time_series(self, x, window_size):
        """
        Reshape (t, n) -> (b, window_size, n), dropping any trailing
        timesteps that don't fill a full window.
        """
        t = x.shape[0]
        num_batch = t // window_size
        remainder = t - num_batch * window_size

        if remainder > 0:
            print(
                f"batch_time_series: dropping last {remainder} timestep(s) "
                f"(t={t} not divisible by window_size={window_size}); "
                f"kept {num_batch * window_size}/{t} samples across {num_batch} batches."
            )

        x = x[: num_batch * window_size]
        return x.reshape((num_batch, window_size) + x.shape[1:])

    def _make_optimizer_and_state(self, est_params):
        optimizer = optax.chain(
            optax.adam(learning_rate=self.lr),
            optax.scale(-1.0),
        )
        opt_state = optimizer.init(est_params)
        return optimizer, opt_state

    def fit(self, Y):
        """Run Adam optimisation (full-batch or minibatch) and store
        results on ``self.model``.

        Parameters
        ----------
        Y : tuple or array
            Batch of observed data. The first axis of each element is
            expected to index individual trials/windows.
        """

        if self.minibatch and self.batch_size == False:
            raise ValueError("minibatch=True requires a non-False `batch_size`.")


        # Full-batch mode): every step sees all windows at once via vmap.
        
        if not self.minibatch:

            if self.batch_size == False:
                objective = jax.value_and_grad(self.model.log_posterior_params)
            else:
                ys = self.batch_time_series(jnp.asarray(Y[0]), self.batch_size)
                s = self.batch_time_series(jnp.asarray(Y[1]), self.batch_size)
                Y = (ys, s)

                mapped_post = jax.vmap(
                    self.model.log_posterior_params, in_axes=(None, 0, None), out_axes=0
                )

                def total_mll(params, y, key):
                    temp = mapped_post(params, y, key)
                    return jnp.sum(temp) + self.model.observation.mapping.log_density(params)

                objective = jax.value_and_grad(total_mll)
            if not self.init_params:
                est_params = self.model.random_init(self.init_key)
            else: 
                est_params = self.init_params
            optimizer, opt_state = self._make_optimizer_and_state(est_params)

            log_density_fn = self.model.observation.mapping.log_density
            save_prior = self.save_prior

            @jax.jit
            def step(est_params, opt_state, Y, key):
                val, grads = objective(est_params, Y, key)
                updates, opt_state = optimizer.update(grads, opt_state)
                est_params = optax.apply_updates(est_params, updates)

                prior_val = log_density_fn(est_params) if save_prior else None
                g_norm = optax.global_norm(grads)

                return est_params, opt_state, val, prior_val, g_norm

            objhist = []
            priorhist = []

            if self.tol_loss == False:
                for key in tqdm(jax.random.split(self.opt_key, self.n_iters)):
                    est_params, opt_state, val, prior_val, _ = step(
                        est_params, opt_state, Y, key
                    )
                    if save_prior:
                        priorhist.append(prior_val)
                    objhist.append(val)
            else:
                g_norms = []
                rel_loss_hist = []

                i = 0
                key = self.opt_key

                est_params, opt_state, val, prior_val, g_norm = step(
                    est_params, opt_state, Y, key
                )
                g_normer = g_norm

                objhist.append(val)
                if save_prior:
                    priorhist.append(prior_val)
                g_norms.append(g_norm / g_normer)
                rel_loss_hist.append(jnp.inf)

                prev_val = val
                
                converged = False

                while jnp.logical_and(i < self.n_iters, jnp.logical_not(converged)):
                    key, subkey = jax.random.split(key)

                    est_params, opt_state, val, prior_val, g_norm = step(
                        est_params, opt_state, Y, subkey
                    )

                    if save_prior:
                        priorhist.append(prior_val)
                    objhist.append(val)

                    rel_loss_change = jnp.abs(val - prev_val) / (jnp.abs(prev_val) + 1e-8)
                    rel_loss_hist.append(rel_loss_change)
                    converged = rel_loss_change < self.tol_loss

                    g_norms.append(g_norm / g_normer)

                    prev_val = val
                    i = i + 1

                self.model.g_norms_ = g_norms
                self.model.rel_loss_hist_ = rel_loss_hist
                self.model.iters_run_ = i

            self.model.params_ = est_params
            self.model.objhist_ = objhist
            if self.save_prior:
                self.model.priorhist_ = priorhist

            return

        # One window per step, shuffled once per epoch, no replacement, full pass = one epoch.
        
        ys = self.batch_time_series(jnp.asarray(Y[0]), self.batch_size)
        s = self.batch_time_series(jnp.asarray(Y[1]), self.batch_size)
        num_batch = ys.shape[0]

        log_posterior = self.model.log_posterior_params
        log_density_fn = self.model.observation.mapping.log_density
        save_prior = self.save_prior
        scale = float(num_batch) if self.scale_likelihood else 1.0

        def total_mll_minibatch(params, y, key):
            """Scaled single-window log-likelihood + log-prior.

            Scaling by `num_batch` makes E[this] over a shuffled epoch equal
            the true full-data log-likelihood sum, so gradient magnitude and
            prior/likelihood balance stay comparable to full-batch training.
            """
            return scale * log_posterior(params, y, key) + log_density_fn(params)

        objective = jax.value_and_grad(total_mll_minibatch)

        if not opt_params["init_params"]:
                est_params = self.model.random_init(self.init_key)
        else: 
            est_params = opt_params["init_params"]
        optimizer, opt_state = self._make_optimizer_and_state(est_params)

        @jax.jit
        def step(est_params, opt_state, y_window, key):
            val, grads = objective(est_params, y_window, key)
            updates, opt_state = optimizer.update(grads, opt_state)
            est_params = optax.apply_updates(est_params, updates)

            prior_val = log_density_fn(est_params) if save_prior else None

            return est_params, opt_state, val, prior_val

        objhist = []
        priorhist = []
        rel_loss_hist = []

        key = self.opt_key
        epoch = 0
        prev_epoch_loss = None
        converged = False

        while epoch < self.n_iters and not converged:
            key, perm_key = jax.random.split(key)
            # Pull the shuffle to host once, then index with plain ints so
            # `step` never retraces (only index *values* change, not shapes).
            perm = np.asarray(jax.random.permutation(perm_key, num_batch))

            epoch_losses = []
            for b in tqdm(perm, desc=f"epoch {epoch}", leave=False):
                key, subkey = jax.random.split(key)
                y_window = (ys[b], s[b])

                est_params, opt_state, val, prior_val = step(
                    est_params, opt_state, y_window, subkey
                )

                objhist.append(val)
                epoch_losses.append(val)
                if save_prior:
                    priorhist.append(prior_val)

            mean_epoch_loss = jnp.mean(jnp.stack(epoch_losses))

            if self.tol_loss != False and prev_epoch_loss is not None:
                rel_loss_change = jnp.abs(mean_epoch_loss - prev_epoch_loss) / (
                    jnp.abs(prev_epoch_loss) + 1e-8
                )
                rel_loss_hist.append(rel_loss_change)
                converged = rel_loss_change < self.tol_loss
            else:
                rel_loss_hist.append(jnp.inf)

            prev_epoch_loss = mean_epoch_loss
            epoch += 1

        self.model.params_ = est_params
        self.model.objhist_ = objhist
        self.model.rel_loss_hist_ = rel_loss_hist
        self.model.iters_run_ = epoch
        if self.save_prior:
            self.model.priorhist_ = priorhist


class SGD:
    """SGD with Nesterov momentum and a cosine warm-up learning-rate schedule.
 
    Uses ``optax.sgd`` with ``nesterov=True`` and
    ``optax.warmup_cosine_decay_schedule`` to anneal the learning rate from
    ``init_value`` up to ``peak_value`` over ``warmup_steps`` steps, then
    decay back toward zero by ``total_steps``.
 
    Parameters
    ----------
    model : object
        Model instance exposing ``log_posterior_params(params, Y, key) -> scalar``.
    opt_params : dict
        Configuration dictionary with the following keys:
 
        save_prior : bool
            If ``True``, record ``log_density(params)`` at every iteration.
        opt_key : jax.random.PRNGKey
            Key for per-iteration random keys.
        init_key : jax.random.PRNGKey
            Reserved initialisation key.
        init_value : float
            Starting learning rate (before warm-up).
        peak_value : float
            Peak learning rate reached at the end of warm-up.
        warmup_steps : int
            Number of steps over which to linearly increase the LR.
        total_steps : int
            Total number of gradient steps (also the decay horizon).
 
    Attributes
    ----------
    model.params_ : pytree
        Estimated parameters after optimisation.
    model.objhist_ : list of float
        Log-posterior value at every iteration.
    model.priorhist_ : list of float
        Log-prior at every iteration (only if ``save_prior=True``).
    """

    def __init__(self, model, opt_params):
        
        self.model = model
        self.init_params = opt_params["init_params"]
        self.save_prior = opt_params["save_prior"]
        self.opt_key = opt_params["opt_key"]
        self.init_key = opt_params["init_key"]
        self.init_value = opt_params["init_value"]
        self.peak_value = opt_params["peak_value"]
        self.warmup_steps = opt_params["warmup_steps"]
        self.total_steps = opt_params["total_steps"]

    def fit(self, Y):
        """Run SGD optimisation and store results on ``self.model``.
 
        Parameters
        ----------
        Y : array-like
            Observed data passed to ``model.log_posterior_params``.
        """
        
        objective = jax.value_and_grad(self.model.log_posterior_params)
        
        warmup_exponential_decay_scheduler = \
        optax.warmup_cosine_decay_schedule(init_value=self.init_value, peak_value=self.peak_value,
                                                warmup_steps=self.warmup_steps,
                                                decay_steps=self.total_steps,
                                                end_value=0) 

        optimizer = optax.chain(
            optax.sgd(learning_rate=warmup_exponential_decay_scheduler, momentum=.99, nesterov=True),
            optax.scale(-1.0)
        )

        if not self.init_params:
                est_params = self.model.random_init(self.init_key)
        else: 
            est_params = self.init_params

        opt_state = optimizer.init(est_params)

        objhist = []
        priorhist = []

        for key in tqdm(jax.random.split(self.opt_key, self.total_steps)):
            val, grads = objective(est_params, Y, key)
            updates, opt_state = optimizer.update(grads, opt_state)
            est_params = optax.apply_updates(est_params, updates)
                
            if self.save_prior:
                priorhist.append(self.model.observation.mapping.log_density(est_params))
            objhist.append(val)

        self.model.params_ = est_params
        self.model.objhist_ = objhist
        
        
        if self.save_prior:
            self.model.priorhist_ = priorhist



class LBFGS:
    """L-BFGS quasi-Newton optimizer via ``jaxopt``.
 
    Wraps ``jaxopt.LBFGS`` for second-order MAP estimation.  Compared to
    first-order methods, L-BFGS typically converges in far fewer iterations
    but requires more memory per step and may be less robust to noisy
    stochastic objectives.
 
    Parameters
    ----------
    model : object
        Model instance exposing ``log_posterior_params(params, Y, key) -> scalar``.
    opt_params : dict
        Configuration dictionary with the following keys:
 
        init_params : pytree
            Initial parameter values.
        save_prior : bool
            If ``True``, record ``log_density(params)`` at every iteration.
        opt_key : jax.random.PRNGKey
            Key for per-iteration random keys.
        init_key : jax.random.PRNGKey
            Key used to initialise the solver state.
        n_iters : int
            Number of L-BFGS update steps.
 
    Attributes
    ----------
    model.params_ : pytree
        Estimated parameters after optimisation.
    model.objhist_ : list of float
        Negative log-posterior (solver value) at every iteration
        (only populated when ``save_prior=True``).
    model.priorhist_ : list of float
        Log-prior at every iteration (only if ``save_prior=True``).
    """

    def __init__(self, model, opt_params):

        self.model = model
        self.init_params = opt_params["init_params"]
        self.save_prior = opt_params["save_prior"]
        self.opt_key = opt_params["opt_key"]
        self.init_key = opt_params["init_key"]
        self.n_iters = opt_params["n_iters"]


    def fit(self, Y):
        """Run L-BFGS optimisation and store results on ``self.model``.
 
        Parameters
        ----------
        Y : array-like
            Observed data passed to ``model.log_posterior_params``.
        """

        def objective(*args):
            """Negate the log-posterior for minimisation."""
            return -1 * self.model.log_posterior_params(*args)

        # Initialize optimization method
        solver = jaxopt.LBFGS(fun=objective)
        if not self.init_params:
                est_params = self.model.random_init(self.init_key)
        else: 
            est_params = self.init_params
        
        state = solver.init_state(
            est_params, Y, self.init_key
        )

        objhist = []
        priorhist = []

        for i, key in enumerate(tqdm(jax.random.split(self.opt_key, self.n_iters))):
            
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
 

class ULA:
    """Underdamped Langevin Algorithm (ULA) sampler with Multiple Importance Sampling.
 
    Runs ``n_chains`` parallel underdamped Langevin MCMC chains to draw
    approximate posterior samples, then uses Multiple Importance Sampling (MIS)
    to estimate the model's log marginal likelihood (log Z).
 
    The underdamped Langevin update incorporates a velocity/momentum variable,
    a friction term, and additive Gaussian noise scaled according to the
    fluctuation-dissipation theorem.  The learning rate is randomised at every
    step by drawing uniformly on a log scale between ``min_lr`` and ``max_lr``.
 
    After the MCMC phase, a bandwidth search over a log-spaced grid selects the
    Gaussian IS kernel width that maximises the ELBO, and then a final round of
    importance-weighted estimates is computed with that bandwidth.
 
    Parameters
    ----------
    model : object
        Model instance exposing ``log_posterior_params(params, Y, key) -> scalar``.
    opt_params : dict
        Configuration dictionary with the following keys:
 
        opt_key : jax.random.PRNGKey
            Key for the main optimisation loop.
        init_key : jax.random.PRNGKey
            Key for parameter initialisation (used when ``init_params`` is falsy).
        is_key : jax.random.PRNGKey
            Key for the importance-sampling phase.
        n_iters : int
            Total number of Langevin steps per chain.
        n_chains : int
            Number of parallel chains.
        friction : float
            Friction coefficient ``γ`` in the underdamped Langevin equation.
        min_lr : float
            Minimum step size (log-uniform lower bound).
        max_lr : float
            Maximum step size (log-uniform upper bound).
        noise_multiplier : float
            Scalar multiplier on the diffusion noise:
            ``noise_scale = noise_multiplier * sqrt(2 * friction * lr)``.
        burn_in : int
            Number of initial steps discarded before saving samples.
        save_every : int
            Thinning interval: save one sample every ``save_every`` steps
            (after burn-in).
        params_per_neuron : int
            Number of parameters per neuron (first non-chain dimension).
        num_neurons : int
            Number of neurons (second non-chain dimension).
        init_params : pytree or falsy
            If truthy, use these as the starting parameters; otherwise draw
            from ``N(0, 1)`` with shape
            ``(n_chains, params_per_neuron, num_neurons)``.
 
    Attributes
    ----------
    model.saved_params_ : jnp.ndarray, shape (n_saved, n_chains, params_per_neuron, num_neurons)
        Thinned posterior samples collected after burn-in.
    model.saved_grads_ : jnp.ndarray
        Gradients at each saved sample (same shape as ``saved_params_``).
    model.objhist_ : jnp.ndarray, shape (n_iters, n_chains)
        Log-posterior values at every iteration for every chain.
    model.rank_order_chains_ : jnp.ndarray
        Chain indices sorted by mean post-burn-in log-posterior (best first).
    model.best_h_ : float
        IS kernel bandwidth that maximised the ELBO during bandwidth selection.
    model.indep_is_w_ : jnp.ndarray
        Normalised importance weights from the final MIS round.
    model.best_lgZs_ : jnp.ndarray, shape (n_chains_keep, num_importance_iters)
        Log marginal-likelihood estimates across the final importance-sampling
        iterations for each kept chain.
    """

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
        """Single underdamped Langevin update step for one chain.
 
        Performs one step of the underdamped (kinetic) Langevin integrator:
 
        .. math::
 
            \\theta_{t+1} &= \\theta_t + v_t \\\\
            v_{t+1}      &= v_t - \\gamma v_t + h \\nabla \\log p(\\theta_t)
                            + \\sqrt{2 \\gamma h}\\, \\epsilon, \\quad
                            \\epsilon \\sim \\mathcal{N}(0, I)
 
        where ``h`` is drawn log-uniformly from ``[min_lr, max_lr]`` at each call.
 
        Parameters
        ----------
        params : jnp.ndarray
            Current parameter vector for this chain.
        velocity : jnp.ndarray
            Current velocity (momentum) for this chain, same shape as ``params``.
        key : jax.random.PRNGKey
            PRNG key consumed to generate the step size and noise.
        Y : array-like
            Observed data.
 
        Returns
        -------
        vals : float
            Log-posterior evaluated at the *current* (pre-update) ``params``.
        new_params : jnp.ndarray
            Updated parameter vector.
        new_velocity : jnp.ndarray
            Updated velocity.
        grads : jnp.ndarray
            Gradient of the log-posterior w.r.t. ``params``.
        """

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
        """Estimate log Z via Multiple Importance Sampling (MIS).
 
        Perturbs each of the ``n`` saved parameter samples with noise drawn
        from ``noise_dist``, evaluates the (unnormalised) log-posterior at the
        perturbed locations, and uses importance weighting to estimate the
        log normalising constant.
 
        Parameters
        ----------
        func : callable
            Batched log-posterior function with signature
            ``func(params_batch, Y, key) -> jnp.ndarray`` of shape ``(n,)``.
        prms : jnp.ndarray, shape (n, params_per_neuron, num_neurons)
            Saved posterior samples used as IS proposal centres.
        noise_dist : distribution
            A ``tensorflow_probability``-compatible distribution with
            ``log_prob`` and ``sample`` methods.  Should have event shape
            ``(params_per_neuron * num_neurons,)``.
        key : jax.random.PRNGKey
            PRNG key for noise sampling.
        Y : array-like
            Observed data.
 
        Returns
        -------
        est_log_Z : float
            Log-normalising-constant estimate (log-mean-exp of importance weights).
        est_elbo : float
            Evidence Lower Bound (ELBO) – the mean of the log importance weights.
        indep_is_w : jnp.ndarray, shape (n,)
            Normalised importance weights (log scale, sums to 0 in log space).
        """

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
        """Run ULA sampling followed by MIS-based log-Z estimation.
 
        Phase 1 – MCMC:
            Runs ``n_chains`` underdamped Langevin chains for ``n_iters`` steps,
            discarding the first ``burn_in`` steps and saving a thinned set of
            samples thereafter.
 
        Phase 2 – Bandwidth selection:
            For each of the ``n_chains_keep`` best chains (ranked by mean
            post-burn-in log-posterior), evaluates the ELBO over a log-spaced
            grid of ``kde_reso`` IS kernel bandwidths.  The bandwidth maximising
            the ELBO for the best chain is selected as ``model.best_h_``.
 
        Phase 3 – Final MIS:
            Runs ``num_importance_iters`` MIS rounds with the selected bandwidth
            to obtain stable log-Z estimates.
 
        Parameters
        ----------
        Y : array-like
            Observed data.
        n_chains_keep : int, optional
            Number of top-ranked chains used in the MIS phase (default 2).
        kde_bandwidth : tuple of (float, float), optional
            ``(log10_min, log10_max)`` range for the bandwidth grid search
            (default ``(-2, -1)``).
        kde_reso : int, optional
            Number of bandwidth candidates in the grid (default 150).
        num_importance_iters : int, optional
            Number of MIS iterations in the final estimation phase (default 500).
        """

        # Initialize optimization method
        if not opt_params["init_params"]:
            est_params = self.model.random_init(self.init_key)
        else: 
            est_params = opt_params["init_params"]
        velocity = jnp.zeros((self.n_chains, self.params_per_neuron, self.num_neurons))
        
        batched_update = jax.jit(
            jax.vmap(
                self.mcmc_update, in_axes=(0, 0, 0, None)
            )
        )

        objhist = []
        saved_params = []
        saved_grads = []
        saved_vals = []

        # ------------------------------------------------------------------ #
        # Phase 1: MCMC                                                       #
        # ------------------------------------------------------------------ #
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
        
        # Rank chains by mean log-posterior after burn-in (best chain first).
        rank_order_chains = jnp.argsort(jnp.mean(self.model.objhist_[self.burn_in:], axis=0))[::-1]
        self.model.rank_order_chains_ = rank_order_chains

        # Create batched posterior for MIS.
        batched_posterior = jax.jit(jax.vmap(
            self.model.log_posterior_params, in_axes=(0, None, None)
        ))

        # ------------------------------------------------------------------ #
        # Phase 2: Bandwidth selection via ELBO maximisation                  #
        # ------------------------------------------------------------------ #
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

        # Select the bandwidth that maximises the ELBO on the best chain.
        self.model.best_h_ = hs[jnp.argmax(elbos[0])]
        best_noise_dist = MultivariateNormalDiag(
            jnp.zeros(self.params_per_neuron * self.num_neurons),
            self.model.best_h_ * jnp.ones(self.params_per_neuron * self.num_neurons)
        )

        
        # ------------------------------------------------------------------ #
        # Phase 3: Final MIS estimation                                       #
        # ------------------------------------------------------------------ #
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

