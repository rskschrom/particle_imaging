## Particle Imaging

This repository contains codes to simulate maximum dimension of aggregates from the [Kuo et al. (2016)](https://doi.org/10.1175/JAMC-D-15-0130.1) database and derive mass-dimensional coefficients for a variety of probe configurations and particle orientation distributions. The data files needed for these codes are stored [here](https://zenodo.org/doi/10.5281/zenodo.11642006). To run the codes, [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), and [SciPy](https://scipy.org) must be installed.

### Code description
- ```dmax_err.py```: calculate and plot the errors in maximum dimension for each probe configuration.
- ```dmax_err_orient.py```: calculate and plot the errors in maximum dimension for probes following the Lebedev order 7 quadrature nodes for varying orientation distributions.
- ```dmax_err_pdf.py```: separate the database particles into tertiles and plot histograms of the percentage error for each number of probe views.
- ```md_orient_sim.py```: calculate and plot the estimated ```a``` and ```b``` coefficients.
- ```md_orient_truth.py```: calculate and plot the true ```a``` and ```b``` coefficients.
- ```mass_err.pt```: calculate and plot the errors in ice water content for different orientation distributions and number of probe views.
- ```bf_dest```: plot the differences between ```D``` for [Brown and Francis (1995)](https://doi.org/10.1175/1520-0426(1995)012%3C0410:IMOTIW%3E2.0.CO;2) and our definition of maximum dimension.
