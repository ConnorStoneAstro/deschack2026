# Setting this up as a separate file to really drive home what parts are
# simulated data generation (generate_sims.py script), and what parts are the
# fiducial simulation and fitting (fit_sim_data.py).

import jax
import jax.numpy as jnp
import cosmographi as cg
import caskade as ck
import matplotlib.pyplot as plt
import emcee
import corner
import numpy as np
from generate_sims import NaiveLikelihood, M_mean, M_std, m_threshold

# Initialize objects
# --------------------------------------------------------------------
z_max = 2.0
cosmology = cg.Cosmology()
cosmology.Omega_m = 0.3
cosmology.w0 = -1.0
rateIa = cg.rates.RateConst(cosmology, r=1e-4, z_min=0.0, z_max=z_max, name="rateIa")

# Fiducial simulation, this is the reference simulation for the P(D|Cosmology) calculation
#####################################################################
Nsamp = 5000
M_mean = -19.3
mu_scat_std = 0.3  # note Fiducial simulation scatter larger than the forward model PDF
m_threshold = -4
m_std = 0.1
z_std = 0.05


def main(savename="posterior_samples", key=jax.random.PRNGKey(43), savefigs=True):
    z_max = 2.0
    cosmology = cg.Cosmology()
    # In principle can use any Cosmology for fiducial simulation
    cosmology.Omega_m = 0.35
    cosmology.w0 = -1.2
    rateIa = cg.rates.RateConst(cosmology, r=1e-4, z_min=0.0, z_max=z_max, name="rateIa")

    # Sample fiducial data
    # --------------------------------------------------------------------
    key, subkey = jax.random.split(key)
    z_true = rateIa.sample_z(subkey, Nsamp)

    DL = jax.jit(jax.vmap(cosmology.luminosity_distance))(z_true)
    mu_true = 5 * jnp.log10(DL) - 5
    key, subkey = jax.random.split(key)
    mu_scat = jax.random.normal(subkey, shape=(Nsamp,)) * mu_scat_std + mu_true

    m_true = mu_scat + M_mean

    z_noise = z_true + jax.random.normal(key, shape=(Nsamp,)) * z_std
    m_noise = m_true + jax.random.normal(key, shape=(Nsamp,)) * m_std

    select = m_noise < m_threshold

    P_D_C = jnp.mean(select)  # P(D|Cosmology) = fraction of SNe that pass the selection cut

    print(f"Fraction of SNe that pass the selection cut: {P_D_C:.4f}")
    mu_scat_import = mu_scat[select]
    mu_true_import = mu_true[select]
    z_true_import = z_true[select]
    importance_weights_logdenom = -0.5 * (mu_scat_import - mu_true_import) ** 2 / mu_scat_std**2

    # Make normalized Likelihood
    #####################################################################
    class NormalizedLikelihood(NaiveLikelihood):
        @ck.forward
        def log_P_D_C(self):
            update_mu_true = (
                5 * jnp.log10(jax.vmap(self.cosmology.luminosity_distance)(z_true_import)) - 5
            )
            importance_weights_numer = -0.5 * (mu_scat_import - update_mu_true) ** 2 / M_std**2
            logP_D_C = jax.scipy.special.logsumexp(
                importance_weights_numer - importance_weights_logdenom
            ) - jnp.log(Nsamp)
            return logP_D_C

        @ck.forward
        def log_likelihood(self):
            ll = super().log_likelihood()
            return ll - len(self.z_obs) * self.log_P_D_C()

    opt_cosmology = cg.Cosmology()
    opt_cosmology.Omega_m.to_dynamic()
    opt_cosmology.w0.to_dynamic()
    load_dat = np.load("simulated_dataset.npz")
    NLL = NormalizedLikelihood(
        load_dat["z_sel"], load_dat["m_sel"], load_dat["z_std"], load_dat["m_std"], opt_cosmology
    )

    # Make map of the log-likelihood over a grid of cosmological parameters
    # --------------------------------------------------------------------
    if savefigs:
        OM_grid, w0_grid = np.meshgrid(np.linspace(0.1, 0.5, 50), np.linspace(-1.5, -0.5, 50))
        params = np.column_stack([OM_grid.ravel(), w0_grid.ravel()])
        nll = jax.vmap(NLL.log_likelihood)(jnp.array(params))
        plt.imshow(
            nll.reshape(OM_grid.shape),
            extent=(0.1, 0.5, -1.5, -0.5),
            origin="lower",
            aspect="auto",
            vmin=np.max(nll) - 50,
        )
        plt.colorbar(label="Log-Likelihood")
        plt.xlabel("Omega_m")
        plt.ylabel("w0")
        plt.title(f"Normalized Log-Likelihood Heatmap")
        plt.savefig(f"normalized_ll_heatmap.png")
        plt.close()

        pdc = jax.vmap(NLL.log_P_D_C)(jnp.array(params))
        plt.imshow(
            pdc.reshape(OM_grid.shape),
            extent=(0.1, 0.5, -1.5, -0.5),
            origin="lower",
            aspect="auto",
            vmin=np.max(pdc) - 1,
        )
        plt.colorbar(label="Log P(D|Cosmology)")
        plt.xlabel("Omega_m")
        plt.ylabel("w0")
        plt.title(f"Log P(D|Cosmology) Heatmap")
        plt.savefig(f"normalized_pdc_heatmap.png")
        plt.close()

    # Sample the normalized posterior using emcee
    # --------------------------------------------------------------------
    true_params_plot = np.array(NLL.get_values())
    omega_bounds = (0.01, 1.0)
    w0_bounds = (-2.0, 0.0)

    batched_log_likelihood = jax.jit(jax.vmap(NLL.log_likelihood))

    def log_probability_batch(theta_batch):
        theta_batch = np.asarray(theta_batch, dtype=float)
        theta_batch_jax = jnp.asarray(theta_batch)

        in_bounds = (
            (theta_batch_jax[:, 0] > omega_bounds[0])
            & (theta_batch_jax[:, 0] < omega_bounds[1])
            & (theta_batch_jax[:, 1] > w0_bounds[0])
            & (theta_batch_jax[:, 1] < w0_bounds[1])
        )
        ll = batched_log_likelihood(theta_batch_jax)
        return np.where(np.asarray(in_bounds), np.asarray(ll), -np.inf)

    ndim = 2
    nwalkers = 8
    nsteps = 5000
    nburn = 500
    thin = 250

    p0 = np.tile(true_params_plot, (nwalkers, 1)) + 1e-12 * np.random.standard_normal(
        (nwalkers, ndim)
    )
    p0[:, 0] = np.clip(p0[:, 0], omega_bounds[0] + 1e-5, omega_bounds[1] - 1e-5)
    p0[:, 1] = np.clip(p0[:, 1], w0_bounds[0] + 1e-5, w0_bounds[1] - 1e-5)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_batch, vectorize=True)
    sampler.run_mcmc(p0, nsteps, progress=True)

    samples = sampler.get_chain(discard=nburn, thin=thin, flat=True)

    if savefigs:
        fig = corner.corner(
            samples,
            labels=["Omega_m", "w0"],
            truths=true_params_plot,
            show_titles=True,
            title_fmt=".3f",
        )
        fig.suptitle(f"Cosmology Posterior Corner Plot (Proper Fit Selected Data)", y=1.02)
        fig.savefig(f"cosmology_corner_plot_proper-fit-selected-data.png", bbox_inches="tight")
        plt.close(fig)

    np.savez(f"{savename}.npz", samples=samples, true_params=true_params_plot)


if __name__ == "__main__":
    main()
