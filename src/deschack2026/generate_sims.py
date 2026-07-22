import jax
import jax.numpy as jnp
import cosmographi as cg
import caskade as ck
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import emcee
import corner
from scipy.optimize import minimize
import numpy as np

# Define sampling setup
# --------------------------------------------------------------------
Nsamp = 500
M_mean = -19.3
M_std = 0.1
m_threshold = -4
m_std = 0.1
z_std = 0.05

# Initialize objects
# --------------------------------------------------------------------
z_max = 2.0
cosmology = cg.Cosmology()
cosmology.Omega_m = 0.3
rateIa = cg.rates.RateConst(cosmology, r=1e-4, z_min=0.0, z_max=z_max, name="rateIa")

# Choose which dataset is used for cosmology fitting: "observed" or "detected".
# --------------------------------------------------------------------
FIT_DATA_MODE = "observed"
assert FIT_DATA_MODE in [
    "observed",
    "detected",
], "FIT_DATA_MODE must be either 'observed' or 'detected'."

# Sample redshifts
# --------------------------------------------------------------------
key = jax.random.PRNGKey(42)
key, subkey = jax.random.split(key)
z_true = rateIa.sample_z(subkey, Nsamp)

hist, bins = jnp.histogram(z_true, bins=30, range=(0.0, z_max))
plt.bar(bins[:-1], hist, width=bins[1] - bins[0], align="edge")
plt.xlabel("True Redshift")
plt.ylabel("Count")
plt.title("True Redshift Histogram")
plt.savefig(f"true_redshift_histogram_fit-{FIT_DATA_MODE}.png")
plt.close()

# Compute distance modulus
# --------------------------------------------------------------------
DL = jax.jit(jax.vmap(cosmology.luminosity_distance))(z_true)
mu_true = 5 * jnp.log10(DL) - 5

plt.scatter(z_true, mu_true, s=5)
plt.xlabel("True Redshift")
plt.ylabel("True Distance Modulus")
plt.title("True Distance Modulus vs Redshift")
plt.savefig(f"true_distance_modulus_scatter_fit-{FIT_DATA_MODE}.png")
plt.close()

# Sample absolute magnitudes
# --------------------------------------------------------------------
key, subkey = jax.random.split(key)
M_true = jax.random.normal(subkey, shape=(Nsamp,)) * M_std + M_mean

# Compute apparent magnitudes
# --------------------------------------------------------------------
m_true = mu_true + M_true

plt.scatter(z_true, m_true, s=5)
plt.xlabel("True Redshift")
plt.ylabel("True Apparent Magnitude")
# plt.gca().invert_yaxis()  # Invert y-axis for magnitudes
plt.grid()
plt.title("True Apparent Magnitude vs Redshift")
plt.savefig(f"true_apparent_magnitude_scatter_fit-{FIT_DATA_MODE}.png")
plt.close()

# Generate observed values with noise
# --------------------------------------------------------------------
z_obs = z_true + jax.random.normal(key, shape=(Nsamp,)) * z_std
m_obs = m_true + jax.random.normal(key, shape=(Nsamp,)) * m_std

# Plot observed values
# --------------------------------------------------------------------
plt.scatter(z_obs, m_obs, s=5, color="orange")
plt.xlabel("Redshift")
plt.ylabel("Apparent Magnitude")
plt.grid()
plt.title("Observed Apparent Magnitude vs Redshift")
plt.savefig(f"observed_apparent_magnitude_scatter_fit-{FIT_DATA_MODE}.png")
plt.close()

# Apply detection threshold
# --------------------------------------------------------------------
detect = m_obs < m_threshold
z_det = z_obs[detect]
m_det = m_obs[detect]

plt.scatter(z_det, m_det, s=5, color="green")
plt.xlabel("Detected Redshift")
plt.ylabel("Detected Apparent Magnitude")
plt.grid()
plt.title("Detected Apparent Magnitude vs Redshift")
plt.savefig(f"detected_apparent_magnitude_scatter_fit-{FIT_DATA_MODE}.png")
plt.close()

if FIT_DATA_MODE == "observed":
    z_fit = z_obs
    m_fit = m_obs
    fit_data_label = "Observed"
elif FIT_DATA_MODE == "detected":
    z_fit = z_det
    m_fit = m_det
    fit_data_label = "Detected"
else:
    raise ValueError("FIT_DATA_MODE must be either 'observed' or 'detected'.")


# Run a fit, without accounting for selection effects
# --------------------------------------------------------------------
class NaiveLikelihood(ck.Module):
    def __init__(self, z_obs, m_obs, z_std, m_std, cosmology: cg.Cosmology):
        super().__init__()
        self.z_obs = z_obs
        self.m_obs = m_obs
        self.z_std = z_std
        self.m_std = m_std
        self.cosmology = cosmology

    @ck.forward
    def single_log_likelihood(self, z, mu, z_obs, z_std, m_obs, m_std):
        z_term = -0.5 * ((z_obs - z) / z_std) ** 2
        m_term = -0.5 * ((m_obs - mu - M_mean) ** 2 / (m_std**2 + M_std**2))
        return z_term + m_term

    @ck.forward
    def log_likelihood(self):
        z_model = jnp.linspace(0.001, z_max, 100)
        mu_model = 5 * jnp.log10(jax.vmap(self.cosmology.luminosity_distance)(z_model)) - 5
        ll = jax.vmap(self.single_log_likelihood, in_axes=(None, None, 0, None, 0, None))(
            z_model, mu_model, self.z_obs, self.z_std, self.m_obs, self.m_std
        )
        return jnp.sum(jax.scipy.special.logsumexp(ll, axis=1))

    @ck.forward
    def neg_log_likelihood(self):
        return -self.log_likelihood()


# Setup naive likelihood and map out the log-likelihood surface
#####################################################################

opt_cosmology = cg.Cosmology()
opt_cosmology.Omega_m.to_dynamic()
opt_cosmology.w0.to_dynamic()
print(opt_cosmology)
NLL = NaiveLikelihood(z_fit, m_fit, z_std, m_std, opt_cosmology)
print(NLL.dynamic_params)
true_params = NLL.get_values()

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
plt.title(f"Naive Log-Likelihood Heatmap ({fit_data_label} Fit Data)")
plt.savefig(f"naive_ll_heatmap_fit-{FIT_DATA_MODE}.png")
plt.close()


# MCMC sampling with emcee and posterior corner plot
# --------------------------------------------------------------------
true_params_plot = np.array(NLL.get_values())
omega_bounds = (0.01, 1.0)
w0_bounds = (-2.0, 0.0)


def make_sigma_ellipse(mean, cov, chi2_value=2.30, **kwargs):
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    width = 2.0 * np.sqrt(chi2_value * eigvals[0])
    height = 2.0 * np.sqrt(chi2_value * eigvals[1])
    angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

    return Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)


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
thin = 1

# Start walkers at the true cosmology; tiny jitter avoids a singular initial ensemble.
# --------------------------------------------------------------------
p0 = np.tile(true_params_plot, (nwalkers, 1)) + 1e-12 * np.random.standard_normal((nwalkers, ndim))
p0[:, 0] = np.clip(p0[:, 0], omega_bounds[0] + 1e-5, omega_bounds[1] - 1e-5)
p0[:, 1] = np.clip(p0[:, 1], w0_bounds[0] + 1e-5, w0_bounds[1] - 1e-5)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_batch, vectorize=True)
sampler.run_mcmc(p0, nsteps, progress=True)

samples = sampler.get_chain(discard=nburn, thin=thin, flat=True)

fig = corner.corner(
    samples,
    labels=["Omega_m", "w0"],
    truths=true_params_plot,
    show_titles=True,
    title_fmt=".3f",
)
fig.suptitle(f"Cosmology Posterior Corner Plot ({fit_data_label} Fit Data)", y=1.02)
fig.savefig(f"cosmology_corner_plot_fit-{FIT_DATA_MODE}.png", bbox_inches="tight")
plt.close(fig)

# Setup proper normalized likelihood
#####################################################################
