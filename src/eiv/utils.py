import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np
from typing import Literal
from functools import partialmethod
import tqdm
from . import mappings

def get_obs_tuning(n_bins, n_neurons, spikes, headdir):
    bins = np.linspace(0, 1, n_bins+1)
    tuning_curves = np.zeros((n_neurons, n_bins))
    digi = np.digitize(headdir, bins, right = True)
    for i in range(n_neurons):
        tuning_curves[i, :] = [np.nanmean(spikes[np.where(digi==j)[0],i]) for j in range(n_bins)]
    return tuning_curves, bins
    
def make_xgrid(num_latent_dims, num_grid_pts, grid_max = 1):
    X = jnp.array(jnp.meshgrid(
        *[jnp.linspace(0,grid_max, num_grid_pts) for _ in range(num_latent_dims)]
    ))
    X = jnp.moveaxis(X, 0, -1)
    return X

def plot_simulated_data_1D(xs_true, true_weights, ys, model, grid_reso = 100, n_neurs = 5, n_timesteps=500, grid_max = 1):
    
    eiv_flag = True if isinstance(model.observation.mapping, mappings.EIVMapping) else False
    fig, axes = plt.subplots(1, 4, figsize=(15,5))

    axes[0].set_title("Latent")
    axes[0].plot(xs_true[:n_timesteps])
    if eiv_flag  & (len(ys)>1):
        axes[0].plot(ys[1][:n_timesteps])
    axes[0].set_xlabel("Time")

    axes[1].set_title("Observations")
    if eiv_flag:
        axes[1].plot(ys[0][:n_timesteps,:n_neurs])
    else:
        axes[1].plot(ys[:n_timesteps,:n_neurs])
    axes[1].set_xlabel("Time")

    x_grid = jnp.linspace(0, grid_max, grid_reso)[:, None]
    if eiv_flag:
        true_tunings = model.observation.mapping(true_weights, x_grid)[0]
    else:
        true_tunings = model.observation.mapping(true_weights, x_grid)

    axes[2].set_title("Tuning Curves")
    axes[2].plot(x_grid, true_tunings[:,:n_neurs])
    axes[2].set_xlabel("Stimulus")

    axes[3].set_title("Noisy Samples")
    if eiv_flag:
        axes[3].scatter(xs_true, ys[0][:,0], lw=0, alpha=.5)
    else:
        axes[3].scatter(xs_true, ys[:,0], lw=0, alpha=.5)

    axes[3].plot(x_grid, true_tunings[:,0], "k", lw=2)
    axes[3].set_xlabel("Stimulus")
    plt.show()
    return axes

def plot_objhist(model, show_prior = True):
    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    if show_prior:
        axes[1].plot(model.priorhist_)
        axes[1].set(xlabel="Iteration", ylabel="Prior Log Likelihood")
    
    axes[0].set(xlabel="Iteration", ylabel="Marg Log Likelihood")
    axes[0].plot(model.objhist_)
    plt.show()
    return axes

def plot_objhist_sim(model, ys, key, show_prior = True):
    fig, axes = plt.subplots(1, 2, figsize=(15,5))
    if show_prior:
        axes[1].plot(model.priorhist_)
        axes[1].set(xlabel="Iteration", ylabel="Prior Log Likelihood")
    
    axes[0].set(xlabel="Iteration", ylabel="Marg Log Likelihood")
    axes[0].plot(model.objhist_)

    axes[0].axhline(jnp.sum(model.marginal_log_likelihood_params(model.true_params, ys, key)), color = "k", linestyle="--")
    
    plt.show()
    return axes

def plot_latent_recon_sim(model, ys, xs_true, grid_reso = 100, window = 500, grid_max = 1, ula_flag=False):
    eiv_flag = True if isinstance(model.observation.mapping, mappings.EIVMapping) else False
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    
    x_grid = make_xgrid(1, grid_reso, grid_max = grid_max)

    
    if eiv_flag:
        if ula_flag:
            chain_ind=model.rank_order_chains_[0]
            curr_param = model.saved_params_[-1][chain_ind,:,:]
            est_logpost = model.logp_x(curr_param, ys[0], x_grid)
            
        else:
            est_logpost = model.logp_x(model.params_, ys[0], x_grid)
    else:
        if ula_flag:
            chain_ind=model.rank_order_chains_[0]
            curr_param = model.saved_params_[-1][chain_ind,:,:]
            est_logpost = model.logp_x(curr_param, ys, x_grid)
            
        else:
            est_logpost = model.logp_x(model.params_, ys, x_grid)

    axes[2].imshow(jnp.exp(est_logpost)[:window].T, aspect='auto')
    axes[2].plot(xs_true[:window]*grid_reso, marker='o', color = 'r', linestyle="", markersize = 1, label="MAP Est.")
    axes[2].legend()
    axes[2].set(xlabel="Time", ylabel="Latent", title="Latent posterior")
    # MAP estimates of x

    est_x_map = x_grid[jnp.argmax(est_logpost, axis=1)].ravel()
    axes[1].set(xlabel="Time", ylabel="Latent", title="Reconstruction Over Time")
    if eiv_flag:
        axes[1].plot((ys[1][:window]), color="orange", lw=2, label = "Observed")
    axes[1].plot(est_x_map[:window], label = "Estimate",  color='#0081ff')
    axes[1].plot(xs_true[:window], label = "True",  color='k')
    axes[1].legend()
    
    axes[0].set(xlabel="True Latent", ylabel="Reconstructed Latent & Observed Behavior", title="Reconstruction Performance")
    
    if eiv_flag:
        axes[0].scatter(xs_true, ys[1], label="Observed Behavior")
    axes[0].scatter(xs_true, est_x_map, label="Reconstructed Latent")
    axes[0].legend()


def plot_latent_recon_real(model, ys, grid_reso = 100, window = 500, grid_max = 1, ula_flag = False):
    eiv_flag = True if isinstance(model.observation.mapping, mappings.EIVMapping) else False
    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    x_grid = make_xgrid(1, grid_reso, grid_max = grid_max)

    if eiv_flag:
        if ula_flag:
            chain_ind=model.rank_order_chains_[0]
            curr_param = model.saved_params_[-1][chain_ind,:,:]
            est_logpost = model.logp_x(curr_param, ys[0], x_grid)
            
        else:
            est_logpost = model.logp_x(model.params_, ys[0], x_grid)
    else:
        if ula_flag:
            chain_ind=model.rank_order_chains_[0]
            curr_param = model.saved_params_[-1][chain_ind,:,:]
            est_logpost = model.logp_x(curr_param, ys, x_grid)
            
        else:
            est_logpost = model.logp_x(model.params_, ys, x_grid)

    axes[2].imshow(jnp.exp(est_logpost)[:window].T, aspect='auto')
    axes[2].set(xlabel="Time", ylabel="Angle", title="Latent posterior")
    # MAP estimates of x

    est_x_map = x_grid[jnp.argmax(est_logpost, axis=1)].ravel()
    axes[1].set(xlabel="Time", ylabel="Angle", title="Reconstruction")
    axes[1].plot((ys[1][:window]), color="orange", lw=2, label = "Observed")
    axes[1].plot(est_x_map[:window], label = "Estimate",  color='#0081ff')
    axes[1].legend()
    
    axes[0].set(xlabel="Measuered Latent", ylabel="Recon", title="Reconstruction vs Measured")
    
    axes[0].scatter(ys[1],est_x_map, label="Observed")
    axes[0].legend()

def plot_real_tuning(model, true_tuning, ys, grid_max = 1, grid_reso=100, ula_flag = False):
    eiv_flag = True if isinstance(model.observation.mapping, mappings.EIVMapping) else False
    
    fig, axes = plt.subplots(5, 3, sharex=True, figsize=(10,10))
    obs_tuning, bins = get_obs_tuning(20, ys[0].shape[1], ys[0], ys[1])
    x_grid = make_xgrid(1, grid_reso, grid_max)
    if ula_flag:

        chain_ind=model.rank_order_chains_[0]
        for j in range(20):
            curr_param = model.saved_params_[j][chain_ind,:,:]
            est_tunings = model.observation.mapping(curr_param, x_grid)
            for i, ax in enumerate(axes.ravel()):
                ax.plot(jnp.linspace(0, 1, true_tuning.shape[0]), true_tuning[:,i], color="k", alpha=.8, label="generative")
                if eiv_flag:
                    ax.plot(x_grid, jnp.roll(est_tunings[0][:,i], 0), color="limegreen", alpha=.8, dashes=[2, 2], label="eiv")
                else:    
                    ax.plot(x_grid, jnp.roll(est_tunings[:,i], 0), color="limegreen", alpha=.8, dashes=[2, 2], label="eiv")


    else: 
        est_tunings = model.observation.mapping(model.params_, x_grid)
        for i, ax in enumerate(axes.ravel()):
            ax.plot(jnp.linspace(0, 1, true_tuning.shape[1]), true_tuning[i,:], color="k", alpha=1., lw = 2, label="generative")
            ax.plot(bins[:-1], obs_tuning[i, :], color="silver", alpha=.8,  label="observed", lw=2)
            if eiv_flag:
                ax.plot(x_grid, jnp.roll(est_tunings[0][:,i], 0), color="limegreen", lw=2, alpha=.8, dashes=[2, 2], label="eiv")
            else:    
                ax.plot(x_grid, jnp.roll(est_tunings[:,i], 0), color="limegreen", alpha=.8, dashes=[2, 2], label="eiv")
    [ax.set_xlabel("Latent or Observed") for ax in axes[-1, :]]
    [ax.set_ylabel("Firing Rate") for ax in axes[:, 0]]
    axes[-1, -1].legend()
    fig.suptitle("observed vs. estimated tuning")
    fig.tight_layout()
    return axes

def plot_real_tuning_2d(model, tuning_curves,nan_mask = None, num_plot = 100, grid_reso = 20,grid_max=0.8):
    if nan_mask is None:
        mask = np.ones(tuning_curves[:,0,:].shape)
    x_grid = make_xgrid(2, grid_reso, grid_max = grid_max)
    est_tunings = model.observation.mapping(model.params_, x_grid)[0]
    
    
    n_cols = 4
    n_rows = num_plot//n_cols + num_plot%n_cols
    fig, axes = plt.subplots(n_rows, n_cols*2, figsize=(n_cols*4, n_rows*2))
    
    n_p_c = 0
    col = 0
    for i in range(num_plot):
      tuning_curve = tuning_curves[:,i,:]
      mask = np.ones(tuning_curve.shape)
      mask[nan_mask] = np.nan
      mask = mask.reshape(tuning_curve.shape)
    
      axes[n_p_c,0+(col*2)].imshow(tuning_curve*mask)
      axes[n_p_c,1+(col*2)].imshow(est_tunings[:,i,:].T*mask, vmin = jnp.nanmin(tuning_curve), vmax = jnp.nanmax(tuning_curve))
      axes[n_p_c,0+(col*2)].set_title("Neuron #: "+str(i))
      axes[n_p_c,1+(col*2)].set_title("EIV Est")
    
      n_p_c = n_p_c + 1
    
      if n_p_c == n_rows:
          col = col + 1
          n_p_c = 0
      plt.tight_layout()
    return axes


def jax_to_numpy_dict(d):
    """Converts all JAX arrays in a dictionary to NumPy arrays.
    In place! .copy() if you want to leave the orig dict the same"""
    new_dict = {}
    for key, value in d.items():
        if isinstance(value, jnp.ndarray):
            new_dict[key] = np.array(value)
        else:
            new_dict[key] = value
    return new_dict

def jax_array_to_list(d):
    """Recursively convert JAX arrays to lists in a dictionary.
    In place! .copy() if you want to leave the orig dict the same"""

    for key, value in d.items():
        if isinstance(value, jnp.ndarray):
            d[key] = value.tolist()
        elif isinstance(value, dict):
            jax_array_to_list(value)

    return d

#####################################################################
#################### Figure 2 Plotting Functions ####################
#####################################################################
    
def plot_hist_fig2(model, y_and_s, xs_gen = None, grid_reso=100, grid_max = 1):
    fig, axes = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(2,2))
    x_grid = utils.make_xgrid(1, grid_reso, grid_max)
    est_logpost = model.logp_x(model.params_, y_and_s[0], x_grid)
    est_x_map = x_grid[jnp.argmax(est_logpost, axis=1)].ravel()

    if xs_gen is not None:
        diff_true_meas = ((y_and_s[1] - xs_gen) + .5) % 1
        axes.hist(diff_true_meas, weights=jnp.ones((diff_true_meas.shape[0]))/diff_true_meas.shape[0], alpha = .5, color="k", label = "Observed",range=(0,1),bins=30)

    diff_est_meas =  ((jnp.squeeze(y_and_s[1]) - est_x_map) + .5) % 1

    axes.hist(diff_est_meas, weights=jnp.ones((diff_est_meas.shape[0]))/diff_est_meas.shape[0], alpha = .5, color="g", label = "Latent",range=(0,1),bins=30)

    axes.legend(loc="lower left")

    axes.set_xticks([0, 0.5, 1], [r'$0$',r'$\pi$',  r'2$\pi$'])
    axes.set(xlabel="Error (X-S)", ylabel="Density")
    axes.legend()

    return axes

def plot_behav_fig2(model, y_and_s, xs_gen = None, window = [0, -1]):
    fig, axes = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(2,2))
    start, end = window
    axes.plot(y_and_s[1][start:end], color="k")
    if xs_gen is not None:
      axes.plot(xs_true[start:end],label = "Inferred Latent", color = "g")
    axes.legend(loc="lower left")
    axes.set_yticks([0, 0.5, 1], [r'$0$',r'$\pi$',  r'2$\pi$'])
    axes.set(xlabel="Time", ylabel="Inferred Latent (X)")

    return axes

def plot_kappa_sweep_fig2(scores, gen_kappa = None, resolution = 20, kappa_bounds = [0.5, 1000]):

    kappas = jnp.geomspace(kappa_bounds[0], kappa_bounds[1], resolution)
    fig, axes = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(2,2))
    axes.plot(kappas, np.mean(scores, axis = 0), color = "g")
    if gen_kappa is not None:
      axes.axvline(gen_kappa, color = "r", lw = 2, label = "Generative Kappa")
    axes.axvline(kappas[np.argmax(np.mean(scores, axis = 0))], color = "k", lw = 2, linestyle = "--", label = "Estimated Kappa")
    axes.legend(loc="lower left")
    axes.set(xlabel="Kappa", ylabel="Heldout MLL", xscale="log")

    return axes


def plot_recon_fig2(model, y_and_s, xs_gen = None, grid_reso=100, grid_max = 1):

    fig, axes = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(2,2))

    x_grid = utils.make_xgrid(1, grid_reso, grid_max)
    est_logpost = model.logp_x(model.params_, y_and_s[0], x_grid)
    est_x_map = x_grid[jnp.argmax(est_logpost, axis=1)].ravel()

    axes.scatter(est_x_map, y_and_s[1], alpha = .5, color="k", label = "Observed", s = 4)

    if xs_gen is not None:
      axes.scatter(est_x_map, xs_gen, alpha = .5, color="g", label = "Latent", s = 4)

    axes.legend(loc="lower left")
    axes.set_yticks([0, 0.5, 1], [r'$0$',r'$\pi$',  r'2$\pi$'])

    axes.set_xticks([0, 0.5, 1], [r'$0$',r'$\pi$',  r'2$\pi$'])
    axes.set(xlabel="Reconstruction", ylabel="Observed or True")
    axes.legend()

    return axes

def plot_tuning_fig2(model, y_and_s, grid_reso=20, max_plot = 5, grid_max = 1):

    num_timepoints, num_neurons = y_and_s[0].shape

    x_grid = utils.make_xgrid(1, grid_reso, grid_max)


    if hasattr(model, "true_params"):
        true_tunings = model.observation.mapping(model.true_params, x_grid)[0]

    fig, axes = plt.subplots(max_plot, 4, sharey=True, sharex=True, figsize=(8,2*max_plot))

    obs_tuning, obs_bins = utils.get_obs_tuning(grid_reso, num_neurons, y_and_s[0], y_and_s[1])


    if hasattr(model, "params_"):
        est_tunings = model.observation.mapping(model.params_, x_grid)[0]
    else:
        raise ValueError("Model has not been fit yet.")

    lw = 3
    for i, ax in enumerate(axes.ravel()):
        if hasattr(model, "true_params"):
            ax.plot(x_grid, true_tunings[:,i],label = "Generative", lw=lw, color="k", alpha = .8)

        ax.plot(obs_bins[:-1], obs_tuning[i,:], color = "k", alpha = .2, lw = lw, label="Observed")
        ax.plot(x_grid, est_tunings[:,i],label = "Estimated", lw=lw, color="g", linestyle = "--")

        ax.set_xticks([0, 0.5, 1], [r'$0$',r'$\pi$',  r'2$\pi$'])
        ax.set(xlabel="Latent (X)", ylabel="Firing Rate")
    axes[0,0].legend(loc="lower left")
    return axes


def cross_val_kappa(y_and_s, kappa_bounds =  [0.5, 1000.], out_scale =100., len_scale = .2,
                    resolution = 20, scale: Literal["linear", "geometric"] = "geometric",
                    n_folds = 1, split = .5):

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    if scale == "geometric":
        kappa_range = jnp.geomspace(kappa_bounds[0], kappa_bounds[1], resolution)
    elif scale == "linear":
        kappa_range = jnp.linspace(kappa_bounds[0], kappa_bounds[1], resolution)
    else:
        raise ValueError(f"Invalid Scale Key")


    OPT_KEY = jax.random.PRNGKey(1)
    INIT_KEY = jax.random.PRNGKey(5)# 4

    TEST_KEY = jax.random.PRNGKey(0)

    num_steps, num_neurons = y_and_s[0].shape
    num_train = int(jnp.floor(num_steps*split))

    scores = np.zeros((n_folds, resolution,))
    total_runs = n_folds*resolution
    run_count = 0
    for n in range(n_folds):
        SHUFFLE_KEY = jax.random.PRNGKey(int(n))
        shuffled_inds = jax.random.permutation(SHUFFLE_KEY, num_steps)
        ys_test = tuple([y_and_s[0][shuffled_inds[num_train:], :], y_and_s[1][shuffled_inds[num_train:], :]])
        ys = tuple([y_and_s[0][shuffled_inds[:num_train], :], y_and_s[1][shuffled_inds[:num_train], :]])

        for k, kappa in enumerate(kappa_range):
            model = EIV(len_scale = len_scale,
                out_scale = out_scale,
                kappa = kappa,
                num_dims=num_dims,
                num_neurons=num_neurons)

            opt_params = {
                "opt_key": OPT_KEY,
                "init_key": INIT_KEY,
                #"freq_march":True,
                "lr":1e-2,
                "n_iters":8000
                #"n_iters_per_freq":1000,
                #"freqs": model.observation.mapping.mappings[0].F_unscaled
            }

            model.fit(ys, "adam", opt_params)
            held_out_mll = model.marginal_log_likelihood_params(model.params_, ys_test, TEST_KEY)
            scores[n,k] = jnp.mean(held_out_mll)
            print("Run: "+str(run_count)+"/"+str(total_runs))
            run_count +=1

    tqdm.__init__ = partialmethod(tqdm.__init__, disable=False)
    return scores
