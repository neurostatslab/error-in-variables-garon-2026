import jax
import jax.numpy as jnp
import pynapple as nap
import pandas as pd
import numpy as np

class FentonHeaddir:
    def __init__(self, filepath):
        data = pd.read_pickle(filepath)
        spikes_key = 0
        yaw_key = 1
        locomotion_key = 2

        light_fr = 0
        dark_fr = 1
        light_hf = 2
        dark_hf = 3

        self.data = {"spikes" : {
                    "fr_light": data[spikes_key][light_fr][:],
                    "fr_dark": data[spikes_key][dark_fr][:],
                    "hf_light": data[spikes_key][light_hf][:],
                    "hf_dark": data[spikes_key][dark_hf][:]
                },
                "headdir": {
                    "fr_light":  data[yaw_key][light_fr][:],
                    "fr_dark":  data[yaw_key][dark_fr][:],
                    "hf_light":  data[yaw_key][light_hf][:],
                    "hf_dark":  data[yaw_key][dark_hf][:]
                },
                "locomotion": {
                    "fr_light":  data[locomotion_key][light_fr][:],
                    "fr_dark":  data[locomotion_key][dark_fr][:],
                    "hf_light":  data[locomotion_key][light_hf][:],
                    "hf_dark":  data[locomotion_key][dark_hf][:]
                }
            }
        

    def load_data(self, animal, epoch, bin_size, filter_locomotion=False, filter_nan = True, normed = True):
        
        max_ind = jnp.min(jnp.array([jnp.array(self.data["headdir"][epoch][animal]).shape[0], self.data["spikes"][epoch][animal].shape[1]]))
        if filter_locomotion:
            X_obs = np.radians(jnp.array(self.data["headdir"][epoch][animal])[:max_ind][np.where(self.data["locomotion"][epoch][animal])[0]])
            Y = self.data["spikes"][epoch][animal][:,np.where(self.data["locomotion"][epoch][animal][:max_ind])[0]].T
            # Y and X are sometimes mismatched by a few timepoints, this ensures
            # were taking the minimum timepoint to align, and then keeping a y that divides 
            # evenly by the bin size
            min_ind = np.min([X_obs.shape[0], Y.shape[0]])
            leftover = min_ind%bin_size
            Y = Y[:min_ind-leftover,:]
            X_obs = X_obs[:min_ind-leftover]
        else:
            X_obs = np.radians(self.data["headdir"][epoch][animal][:max_ind])
            Y = self.data["spikes"][epoch][animal][:max_ind].T
            # TODO - check on this with jose
            min_ind = np.min([X_obs.shape[0], Y.shape[0]])
            leftover = min_ind%bin_size
            Y = Y[:min_ind-leftover,:]
            X_obs = X_obs[:min_ind-leftover]
        
        X_obs = jnp.mean(X_obs.reshape(int(X_obs.shape[0]/bin_size), bin_size), axis=1)
        
        Y = jnp.mean(Y.reshape(int(Y.shape[0]/bin_size), bin_size, Y.shape[1]), axis=1)

        if filter_nan:
            X_obs = jnp.array(X_obs)
            Y = Y[~np.isnan(X_obs)]
            X_obs = X_obs[~np.isnan(X_obs)]
        
        print("Timesteps x Neurons: " +str(Y.shape))
        #TODO - confirm this
        if normed:
            X_obs = X_obs/(2*jnp.pi)

        return [Y, X_obs]

    def get_tuning(self, n_bins, n_neurons, spikes, headdir, return_bins = False):
        # TODO reload data? avoid passing? 
        # TODO make so matched pynapple returns to simplify plotting
        
        #bins = np.linspace(0, 2*np.pi, n_bins+1)
        bins = np.linspace(np.min(headdir), np.max(headdir), n_bins+1)
        tuning_curves = np.zeros((n_neurons, n_bins))
        for i in range(n_neurons):
            digi = np.digitize(headdir, bins)
            tuning_curves[i, :] = [np.nanmean(spikes[np.where(digi==j)[0],i]) for j in range(1, n_bins+1)]
        
        if return_bins:
            return tuning_curves, bins[1:]
        else:
            return tuning_curves