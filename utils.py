import matplotlib.pyplot as plt
import jax.numpy as jnp
import mappings

def make_xgrid(num_latent_dims, num_grid_pts, grid_max = 1):
    X = jnp.array(jnp.meshgrid(
        *[jnp.linspace(0,grid_max, num_grid_pts) for _ in range(num_latent_dims)]
    ))
    X = jnp.moveaxis(X, 0, -1)
    return X

def plot_simulated_data_1D(xs_true, true_weights, ys, model, grid_reso = 100, n_neurs = 5, n_timesteps=300, grid_max = 1):
    
    eiv_flag = True if isinstance(model.observation.mapping, mappings.EIVMapping) else False
    fig, axes = plt.subplots(1, 4, figsize=(15,5))

    axes[0].set_title("Latent")
    axes[0].plot(xs_true)
    if eiv_flag  & (len(ys)>1):
        axes[0].plot(ys[1])
    axes[0].set_xlabel("Time")

    axes[1].set_title("Observations")
    if eiv_flag:
        axes[1].plot(ys[0][:,:n_neurs])
    else:
        axes[1].plot(ys[:,:n_neurs])
    axes[1].set_xlabel("Time")

    x_grid = jnp.linspace(0, grid_max, grid_reso)[:, None]
    if eiv_flag:
        true_tunings = model.observation.mapping(true_weights, x_grid)[0]
    else:
        true_tunings = model.observation.mapping(true_weights, x_grid)

    axes[2].set_title("Tuning Curves")
    axes[2].plot(x_grid, true_tunings)
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

def plot_simulated_data_2D(xs_true, true_weights, ys, model, grid_reso = 100, n_neurs = 5, n_timesteps=300, grid_max = 1):
    
    eiv_flag = True if isinstance(model.observation.mapping, mappings.EIVMapping) else False
    fig, axes = plt.subplots(1, 4, figsize=(15,5))

    axes[0].set_title("Latent")
    axes[0].plot(xs_true, color='r', label="True")
    if eiv_flag  & (len(ys)>1):
        axes[0].plot(ys[1], color='k', label="Measured")
    axes[0].set_xlabel("Time")

    axes[1].set_title("Observations")
    if eiv_flag:
        axes[1].plot(ys[0][:,:n_neurs])
    else:
        axes[1].plot(ys[:,:n_neurs])
    axes[1].set_xlabel("Time")

    x_grid = make_xgrid(2, 100, grid_max = grid_max)
    if eiv_flag:
        true_tunings = model.observation.mapping(true_weights, x_grid)[0]
    else:
        true_tunings = model.observation.mapping(true_weights, x_grid)

    axes[2].set_title("Tuning Curves")
    axes[2].imshow(true_tunings[:,0,:])
    
    axes[3].set_title("Tuning Curves")
    axes[3].imshow(true_tunings[:,1,:])
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

def plot_3d_neurons(model, true_weights, grid_reso = 100, neurs = [0,1,2]):
    #TODO mame general, this will break with compound
    eiv_flag = True if isinstance(model.observation.mapping, mappings.EIVMapping) else False
    x_grid = jnp.linspace(0, 1, grid_reso)[:, None]

    if eiv_flag:
        true_tunings = model.observation.mapping(true_weights, x_grid)[0]
        est_tunings = model.observation.mapping(model.params_, x_grid)[0]

    else:
        true_tunings = model.observation.mapping(true_weights, x_grid)
        est_tunings = model.observation.mapping(model.params_, x_grid)

    fig, ax = plt.subplots(1, 1, figsize=(15,5), subplot_kw={'projection': '3d'})
    ax.plot(est_tunings[:,neurs[0]], est_tunings[:,neurs[1]], est_tunings[:,neurs[2]], c='green', alpha=.5)
    ax.plot(true_tunings[:,neurs[0]], true_tunings[:,neurs[1]], true_tunings[:,neurs[2]], c='k', alpha=.5)
    plt.tight_layout()
    plt.show()
    return ax

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

def plot_real_data_1D(xs_obs, ys, tuning, n_neurs = 5, n_timesteps=300, grid_max = 1):
    fig, axes = plt.subplots(1, 4, figsize=(15,5))

    axes[0].set_title("Latent")
    axes[0].plot(xs_obs)
    axes[0].set_xlabel("Time")

    axes[1].set_title("Observations")
    axes[1].plot(ys[:,:n_neurs])
    axes[1].set_xlabel("Time")

    axes[2].set_title("Tuning Curves")
    axes[2].plot(tuning[:n_neurs, :].T)
    axes[2].set_xlabel("Time")

    axes[3].set_title("Noisy Samples")
    axes[3].plot(jnp.linspace(0, grid_max, tuning.shape[1]), tuning[0,:], color='k')
    
    axes[3].scatter(xs_obs, ys[:,0], lw=0, alpha=.5)

    axes[3].set_xlabel("Stimulus")
    plt.show()
    return axes

def plot_real_tuning(model, true_tuning, grid_max = 1, grid_reso=100, ula_flag = False):
    eiv_flag = True if isinstance(model.observation.mapping, mappings.EIVMapping) else False
    
    fig, axes = plt.subplots(5, 3, sharex=True, figsize=(10,10))

    x_grid = make_xgrid(1, grid_reso, grid_max)
    if ula_flag:

        chain_ind=model.rank_order_chains_[0]
        for j in range(20):
            curr_param = model.saved_params_[j][chain_ind,:,:]
            est_tunings = model.observation.mapping(curr_param, x_grid)
            for i, ax in enumerate(axes.ravel()):
                ax.plot(jnp.linspace(0, 1, true_tuning.shape[0]), true_tuning[:,i], color="k", alpha=.8, label="true")
                if eiv_flag:
                    ax.plot(x_grid, jnp.roll(est_tunings[0][:,i], 0), color="g", alpha=.8, dashes=[2, 2], label="est")
                else:    
                    ax.plot(x_grid, jnp.roll(est_tunings[:,i], 0), color="g", alpha=.8, dashes=[2, 2], label="est")


    else: 
        est_tunings = model.observation.mapping(model.params_, x_grid)
        for i, ax in enumerate(axes.ravel()):
            ax.plot(jnp.linspace(0, 1, true_tuning.shape[1]), true_tuning[i,:], color="k", alpha=.8,  label="true")
            if eiv_flag:
                ax.plot(x_grid, jnp.roll(est_tunings[0][:,i], 0), color="g", alpha=.8, dashes=[2, 2], label="est")
            else:    
                ax.plot(x_grid, jnp.roll(est_tunings[:,i], 0), color="g", alpha=.8, dashes=[2, 2], label="est")
    [ax.set_xlabel("Latent or Observed") for ax in axes[-1, :]]
    [ax.set_ylabel("Firing Rate") for ax in axes[:, 0]]
    axes[-1, -1].legend()
    fig.suptitle("true vs. estimated tuning")
    fig.tight_layout()
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
    
