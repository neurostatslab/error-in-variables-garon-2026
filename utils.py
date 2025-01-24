import matplotlib.pyplot as plt
import jax.numpy as jnp
import mappings

def plot_simulated_data_1D(xs_true, true_weights, ys, model, grid_reso = 100, n_neurs = 5, n_timesteps=300):
    eiv_flag = True if isinstance(ys, list) else False
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

    x_grid = jnp.linspace(0, 1, grid_reso)[:, None]
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


def plot_objhist(model, show_prior = True):
    if show_prior:
        fig, axes = plt.subplots(1, 2, figsize=(15,5))
        axes[1].plot(model.priorhist_)
        axes[1].set(xlabel="Iteration", ylabel="Prior Log Likelihood")
    
    axes[0].set(xlabel="Iteration", ylabel="Marg Log Likelihood")
    axes[0].plot(model.objhist_)
    plt.show()
    return axes


def plot_3d_neurons(model, true_weights, grid_reso = 100, neurs = [0,1,2]):
    eiv_flag = True if isinstance(model.observation.mapping, mappings.CompoundMapping) else False
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


