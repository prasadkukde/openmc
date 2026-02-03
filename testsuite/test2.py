import openmc
import matplotlib.pyplot as plt
import numpy as np

# 1. Load Data
sp = openmc.StatePoint('statepoint.50.h5')
t_truth = sp.get_tally(name='tally_truth')
truth_scores = t_truth.mean.flatten()

t_zernike = sp.get_tally(name='zernike_tally')
zn_coeffs = t_zernike.mean.flatten()

t_col=sp.get_tally(name='tally_col')
col_scores=t_col.mean.flatten()

# 2. Physics Constants
r_max = 0.392
# Total Truth (Sum of all neutrons in the pin)
total_neutrons_truth = np.sum(truth_scores)
# Target Value (The fraction of neutrons in the last ring)
truth_fraction_last_ring = truth_scores[-1] / total_neutrons_truth
col_fraction=col_scores[-1]/np.sum(col_scores)
baseline_error=abs(col_fraction-truth_fraction_last_ring)/truth_fraction_last_ring
# 3. Calculation Loop
errors = []
terms = range(1, len(zn_coeffs) + 1)

# Create a dense grid for integration (0 to r_max)
r_grid = np.linspace(0, r_max, 1000)
dr = r_grid[1] - r_grid[0]

# Define the start of the last ring
r_inner_last = r_max * np.sqrt(9.0/10.0)

for i in terms:
    # A. Build the function
    func = openmc.ZernikeRadial(zn_coeffs[:i], radius=r_max)
    vals = func(r_grid)
    
    # B. Calculate "Total Integral" (Weighted by radius r!)
    # Integral = Sum( f(r) * r * dr )
    integrand = vals * r_grid * dr
    total_integral_zernike = np.sum(integrand)
    
    # C. Calculate "Last Ring Integral"
    # We only sum the parts where r > r_inner_last
    mask = r_grid > r_inner_last
    last_ring_integral = np.sum(integrand[mask])
    
    # D. Calculate the Fraction (Integral Last Ring / Total Integral)
    # This automatically cancels out all units/normalizations!
    zernike_fraction_last_ring = last_ring_integral / total_integral_zernike
    
    # E. Calculate Error
    err = abs(zernike_fraction_last_ring - truth_fraction_last_ring) / truth_fraction_last_ring
    errors.append(err)



# 4. Plot
plt.figure(figsize=(10,6))
plt.semilogy(terms, errors, 'o', color='orange', label='ZN vs TKL', markersize=10)
plt.axhline(y=baseline_error,color='green',label='COL vs TKL',markersize=10)
plt.xlabel('Number of Terms', fontsize=12)
plt.ylabel('Relative Error (Log Scale)', fontsize=12)
plt.title('Convergence of Zernike Method (Area Matched)', fontsize=14)
plt.grid(True, which="both", alpha=0.3)
plt.legend()
plt.show()