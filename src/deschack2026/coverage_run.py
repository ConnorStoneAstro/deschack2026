# This script is intended to alternately run the generate and fit scripts to
# produce data for a coverage test.

from generate_sims import main as generate_main
from fit_sim_data import main as fit_main
import numpy as np
import jax
from pted import pted_coverage_test

Nsims = 10
run_sims = False
key = jax.random.PRNGKey(124)

g = []
s = []
for n in range(Nsims):
    if run_sims:
        print(f"Running simulation {n + 1}/{Nsims}...")
        key, subkey = jax.random.split(key)
        generate_main("dummy", key=subkey, just_generate=True)
        key, subkey = jax.random.split(key)
        fit_main(savename=f"posterior_samples_{n + 1}", key=subkey, savefigs=False)
    dat = np.load(f"posterior_samples_{n + 1}.npz")
    g.append(dat["true_params"])
    s.append(dat["samples"])

g = np.stack(g)
s = np.stack(s, axis=1)
pvalue = pted_coverage_test(g, s, pit_plot="HBI_coverage_test_pit.png")
print(f"Coverage test p-value: {pvalue:.4f}")
