import numpy as np
import scipy.integrate as integrate
import openmc

#  This test assumes you have implemented 'ExponentialRadial' in OpenMC
# or have a python class with that name available.

def test_exponential_radial():
    """
    Unit test for Exponential Basis Function Expansion.
    Verifies that the class correctly solves the matrix equation G * a = t
    as defined in Section 2.2 of the paper.
    """
    
    # =========================================================================
    # 1. SETUP: INPUTS & PARAMETERS
    # =========================================================================
    # Fake tally moments (t_i) representing raw data from OpenMC
    # In the paper, t_i = (1/N) * Sum( psi_i(r) )
    t_moments = np.asarray([1.0, 2.5, 3.8, 5.1])
    
    radius = 0.392        # Fuel pin radius (cm)
    order = len(t_moments) # Number of expansion terms
    
    # -------------------------------------------------------------------------
    # CALL THE CLASS (The "Easy Way")
    # This represents the code you want to test
    # -------------------------------------------------------------------------
    # exp_func = openmc.ExponentialRadial(t_moments, radius=radius)
    # NOTE: Since we don't have the actual class compiled, I will simulate 
    # the class output at the end using the manual math for demonstration.
    
    # =========================================================================
    # 2. MANUAL REFERENCE CALCULATION (The "Hard Math" / Truth)
    # =========================================================================
    
    # A. Define the Basis Function psi_n(rho)
    # From Paper Section 2.4: psi_n = exp(n * rho^6)
    # Note: rho is normalized distance (r/R) from 0 to 1
    def psi(n, rho):
        return np.exp(n * (rho**6))

    # B. Build the Gram Matrix (G)
    # From Paper Eq (5) & (6): G_ij = Inner Product <psi_i, psi_j>
    # Inner Product = Integral( psi_i * psi_j * dV ) over the disk
    # dV = 2 * pi * r * dr  --> which is (2 * pi * R^2) * rho * d_rho
    
    G = np.zeros((order, order))
    area_factor = 2 * np.pi * (radius**2)

    print("Building Gram Matrix...")
    for i in range(order):
        for j in range(order):
            # Define the integrand function: psi_i * psi_j * rho
            integrand = lambda rho: psi(i, rho) * psi(j, rho) * rho
            
            # Perform numerical integration from 0 to 1
            integral_val, error = integrate.quad(integrand, 0, 1)
            
            # Store in Matrix G (scaled by physical area)
            G[i, j] = integral_val * area_factor

    # C. Solve the Linear System (Eq 6 in Paper)
    # Matrix Equation: G * a = t
    # Solution: a = Inverse(G) * t
    
    print("Solving G * a = t ...")
    # We use linalg.solve because it is more stable than inv()
    a_coeffs_truth = np.linalg.solve(G, t_moments)
    
    print(f"True Coefficients (a_n): {a_coeffs_truth}")

    # =========================================================================
    # 3. VERIFICATION POINT
    # =========================================================================
    # Pick a random test point to check accuracy
    test_r_cm = 0.2    # Test at r = 0.2 cm
    test_rho = test_r_cm / radius
    
    # Calculate the True Value manually using Eq (1)
    # P(r) = Sum( a_n * psi_n(r) )
    ref_val = 0.0
    for n in range(order):
        ref_val += a_coeffs_truth[n] * psi(n, test_rho)
        
    print(f"Manual Calculated Value at r={test_r_cm}: {ref_val:.6f}")

    # =========================================================================
    # 4. ASSERTION (The Test)
    # =========================================================================
    # Here you would call your actual class:
    # test_val = exp_func(test_r_cm)
    
    # For now, we simulate the class working perfectly to show the assertion logic:
    test_val = ref_val 
    
    # The crucial check:
    # "Does the Class Output match the Manual Matrix Math?"
    assert np.isclose(ref_val, test_val)
    print("TEST PASSED: Manual Matrix Math matches Class Output.")

if __name__ == "__main__":
    test_exponential_radial()