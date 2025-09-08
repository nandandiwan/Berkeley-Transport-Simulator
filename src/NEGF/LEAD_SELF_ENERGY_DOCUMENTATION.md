# Lead Self-Energy Implementation - Cleaned and Robust

## Overview

This document describes the cleaned and robust implementation of lead self-energy calculations using surface Green's functions. The implementation is based on the proven algorithms from OpenMX's `TRAN_Calc_SurfGreen.c` and provides three different methods for calculating surface Green's functions.

## Key Features

### 1. Single Clean Interface
- **One main method**: `self_energy(side, E, ky, method)`
- **Three robust algorithms**: Sancho-Rubio, Iterative, Transfer Matrix
- **Automatic error handling**: Fallback mechanisms for numerical issues
- **Consistent results**: All methods should give equivalent results for the same physical system

### 2. Surface Green's Function Methods

#### A. Sancho-Rubio Method (`method="sancho_rubio"`)
- **Based on**: `TRAN_Calc_SurfGreen_Normal` from OpenMX
- **Algorithm**: Standard iterative decimation technique
- **Advantages**: Well-established, numerically stable
- **Best for**: General purpose, most systems

```python
# Usage
sigma_L = lead_se.self_energy("left", E=0.1, ky=0.0, method="sancho_rubio")
```

#### B. Iterative Method (`method="iterative"`)
- **Based on**: `TRAN_Calc_SurfGreen_Multiple_Inverse` from OpenMX
- **Algorithm**: Direct inversion with recursive coupling
- **Advantages**: Sometimes faster convergence
- **Best for**: Systems with weak inter-layer coupling

```python
# Usage
sigma_R = lead_se.self_energy("right", E=0.1, ky=0.0, method="iterative")
```

#### C. Transfer Matrix Method (`method="transfer"`)
- **Based on**: `TRAN_Calc_SurfGreen_transfer` from OpenMX
- **Algorithm**: Transfer matrix approach with accumulated terms
- **Advantages**: Good for strongly coupled systems
- **Best for**: Systems with significant inter-layer interactions

```python
# Usage
sigma_L = lead_se.self_energy("left", E=0.1, ky=0.0, method="transfer")
```

## Mathematical Foundation

### Surface Green's Function
The surface Green's function describes the response of a semi-infinite lead:

$$G^{\text{surf}}(E) = \left(E \cdot S_{00} - H_{00} - \Sigma_{\text{coupling}}\right)^{-1}$$

Where:
- $H_{00}$: On-site Hamiltonian of the lead layer
- $H_{01}$: Coupling to next layer
- $S_{00}$, $S_{01}$: Overlap matrices (usually identity)

### Self-Energy Calculation
The self-energy describes how the leads affect the device:

**Left Lead:**
$$\Sigma_L = H_{10} \, G^{\text{surf}}_L \, H_{01}$$

**Right Lead:**
$$\Sigma_R = H_{01} \, G^{\text{surf}}_R \, H_{10}$$

### Algorithm Details

#### Sancho-Rubio Method
The iterative decimation algorithm:

1. Initialize: $\varepsilon_s^{(0)} = \varepsilon^{(0)} = E \cdot I - H_{00}$, $\alpha^{(0)} = H_{01}$, $\beta^{(0)} = H_{10}$

2. Iterate until convergence:
   $$\begin{align}
   \text{Inverse: } &\quad g^{(n)} = \left(E \cdot I - \varepsilon^{(n)}\right)^{-1} \\
   \text{Update surface: } &\quad \varepsilon_s^{(n+1)} = \varepsilon_s^{(n)} - \alpha^{(n)} g^{(n)} \beta^{(n)} \\
   \text{Update bulk: } &\quad \varepsilon^{(n+1)} = \varepsilon^{(n)} - \beta^{(n)} g^{(n)} \alpha^{(n)} - \alpha^{(n)} g^{(n)} \beta^{(n)} \\
   \text{Update couplings: } &\quad \alpha^{(n+1)} = \alpha^{(n)} g^{(n)} \alpha^{(n)}, \quad \beta^{(n+1)} = \beta^{(n)} g^{(n)} \beta^{(n)}
   \end{align}$$

3. Final result: $G^{\text{surf}} = \left(E \cdot I - \varepsilon_s^{(\infty)}\right)^{-1}$

#### Iterative Method
Direct closed-form iteration:

1. Initialize: $h_0 = E \cdot I - H_{00}$, $h_l = H_{01}$, $h_r = H_{01}^\dagger$

2. Iterate: $G^{(n+1)} = \left(h_0 - h_l G^{(n)} h_r\right)^{-1}$

#### Transfer Matrix Method
Accumulated transfer matrix approach:

1. Initialize transfer matrices: $t_0 = g_{00} H_{10}$, $\bar{t}_0 = g_{00} H_{01}$

2. Iterate transfer matrices:
   $$\begin{align}
   \text{Denominator: } &\quad D^{(n)} = \left(I - t^{(n-1)} \bar{t}^{(n-1)} - \bar{t}^{(n-1)} t^{(n-1)}\right)^{-1} \\
   \text{Update: } &\quad t^{(n)} = D^{(n)} \left(t^{(n-1)}\right)^2, \quad \bar{t}^{(n)} = D^{(n)} \left(\bar{t}^{(n-1)}\right)^2
   \end{align}$$

3. Accumulate: $T^{(n)} = T^{(n-1)} + \bar{T}^{(n-1)} t^{(n)}$

4. Final: $G^{\text{surf}} = \left(E \cdot I - H_{00} - H_{01} T^{(\infty)}\right)^{-1}$

## Implementation Details

### Error Handling
1. **Numerical Stability**: Automatic addition of small imaginary part ($\eta = 10^{-6}$)
2. **Matrix Inversion**: Uses `linalg.solve()` with `linalg.pinv()` fallback
3. **Large Energy Handling**: Returns zero matrix for $|E| > 5 \times 10^5$
4. **Convergence Monitoring**: Warns if iterations don't converge

### Voltage Application
- **Left lead energy**: $E_{\text{lead}} = E - V_{\text{source}}$
- **Right lead energy**: $E_{\text{lead}} = E - V_{\text{drain}}$

### Matrix Size Handling
- Automatically extracts relevant device-coupling blocks
- Size: $2 \times N_z \times 10$ (assuming 10 orbitals per layer)

## Usage Examples

### Basic Usage
```python
from lead_self_energy import LeadSelfEnergy

# Initialize
lead_se = LeadSelfEnergy(device, hamiltonian)

# Calculate self-energies
sigma_L = lead_se.self_energy("left", E=0.1, ky=0.0)
sigma_R = lead_se.self_energy("right", E=0.1, ky=0.0)
```

### Method Comparison
```python
methods = ["sancho_rubio", "iterative", "transfer"]
results = {}

for method in methods:
    results[method] = lead_se.self_energy("left", E=0.1, ky=0.0, method=method)
    
# Compare results
for i, method1 in enumerate(methods):
    for method2 in methods[i+1:]:
        diff = np.max(np.abs(results[method1] - results[method2]))
        print(f"{method1} vs {method2}: max difference = {diff:.2e}")
```

### Energy Sweep
```python
energies = np.linspace(-2.0, 2.0, 100)
self_energies = []

for E in energies:
    sigma = lead_se.self_energy("left", E, ky=0.0)
    self_energies.append(sigma[0,0])  # Take diagonal element

# Plot
import matplotlib.pyplot as plt
plt.plot(energies, np.real(self_energies), label='Real')
plt.plot(energies, np.imag(self_energies), label='Imaginary')
plt.xlabel('Energy')
plt.ylabel(r'Self-Energy $\Sigma_{L}(0,0)$')
plt.legend()
plt.show()
```

## Testing and Validation

### Unit Tests
Run the test script to verify functionality:
```bash
python test_lead_self_energy_cleaned.py
```

The test script verifies:
1. All three methods run without errors
2. Results are numerically reasonable
3. Methods give consistent results
4. Energy sweep produces smooth curves

### Expected Behavior
- **Real part**: Should vary smoothly with energy
- **Imaginary part**: Should be negative (provides broadening)
- **Method consistency**: Different methods should agree within numerical precision
- **Physical bounds**: Self-energy should not have unreasonably large magnitudes

## Troubleshooting

### Common Issues
1. **Convergence warnings**: Increase `iteration_max` or adjust `tolerance`
2. **Matrix singularities**: Check that your Hamiltonian is physically reasonable
3. **Large self-energies**: May indicate problems with energy scale or Hamiltonian setup
4. **Method disagreements**: Check for numerical precision issues or edge cases

### Performance Tips
- **Sancho-Rubio**: Generally most reliable
- **Iterative**: Try for weakly coupled systems
- **Transfer**: Use for strongly coupled or problematic systems
- **Energy range**: Avoid extremely large $|E|$ values

## References

1. OpenMX TRAN_Calc_SurfGreen.c implementation
2. Sancho, M. P. L. et al. "Highly convergent schemes for the calculation of bulk and surface Green functions" J. Phys. F: Met. Phys. 15, 851 (1985)
3. Datta, S. "Electronic Transport in Mesoscopic Systems" Cambridge University Press (1995)

## API Reference

### Class: `LeadSelfEnergy`

#### Constructor
```python
LeadSelfEnergy(device: Device, hamiltonian: Hamiltonian)
```

#### Main Method
```python
self_energy(side, E, ky, method="sancho_rubio")
```
- **side**: "left" or "right"
- **E**: Energy (float or complex)
- **ky**: Transverse momentum (float)
- **method**: "sancho_rubio", "iterative", or "transfer"
- **Returns**: Self-energy matrix $\Sigma$ (complex numpy array)

#### Surface Green's Function
```python
surface_greens_function(E, H00, H01, method="sancho_rubio", iteration_max=1000, tolerance=1e-6)
```
- **E**: Energy (complex)
- **H00**: On-site Hamiltonian $H_{00}$
- **H01**: Inter-layer coupling $H_{01}$
- **method**: Algorithm choice
- **iteration_max**: Maximum iterations
- **tolerance**: Convergence criterion
- **Returns**: Surface Green's function matrix $G^{\text{surf}}$
