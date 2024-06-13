import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
from netCDF4 import Dataset
from scipy.special import beta as beta_func
from scipy.stats import linregress

# get spherical beta distribution
def beta_spherical(a, b, beta):
    fdist = 1./(4.*np.pi*beta_func(a,b))*((1.-np.cos(beta))/2.)**(a-1.)*\
                                         ((1.+np.cos(beta))/2.)**(b-1.)
    return fdist

# read netcdf file
ncfile = Dataset('aggregate_standard.nc', 'r')
dmax_view = ncfile.variables['dmax_view'][:]
view_az = ncfile.variables['view_az'][:]
view_ze = ncfile.variables['view_ze'][:]
npar = dmax_view.shape[0]

# probe view indices
iv_2dvd = [0,3]
iv_masc = [0,1,2]
iv_sing = [0]
iv_3ort = [0,3,4]

# get apparent dmax values
dmax_2dvd = np.max(dmax_view[:,0,iv_2dvd], axis=1)
dmax_masc = np.max(dmax_view[:,0,iv_masc], axis=1)
dmax_sing = np.max(dmax_view[:,0,iv_sing], axis=1)
dmax_3ort = np.max(dmax_view[:,0,iv_3ort], axis=1)

# read lebedev 7 view file
ncfile = Dataset('aggregate_leb.nc', 'r')

dmax_true = ncfile.variables['dmax'][:]
mass_true = ncfile.variables['mass_true'][:]
dmax_view = ncfile.variables['dmax_view'][:]
view_az = ncfile.variables['view_az'][:]
view_ze = ncfile.variables['view_ze'][:]

nleb = len(view_az)

# get error in dmax for each additional view (lebedev)
rmse_step_leb7 = np.empty(nleb-1)
rmse_step2_leb7 = np.empty(nleb-1)

for i in range(nleb-1):
    dmax_step_leb7 = np.max(dmax_view[:,0,:i+1], axis=1)
    rmse_step_leb7[i] = np.sqrt(np.mean((dmax_true[:npar]-dmax_step_leb7[:npar])**2.))
    
    view2_inds = np.clip(2*np.arange(i+1), 0, nleb-1)
    
    dmax_step2_leb7 = np.max(dmax_view[:,0,view2_inds], axis=1)
    rmse_step2_leb7[i] = np.sqrt(np.mean((dmax_true[:npar]-dmax_step2_leb7[:npar])**2.))
    
dmax_leb7 = np.max(dmax_view[:,0,:], axis=1)
dmax_leb0 = dmax_view[:,0,0]
                   
# errors
rmse_2dvd = np.sqrt(np.mean((dmax_true[:npar]-dmax_2dvd[:npar])**2.))
rmse_masc = np.sqrt(np.mean((dmax_true[:npar]-dmax_masc[:npar])**2.))
rmse_sing = np.sqrt(np.mean((dmax_true[:npar]-dmax_sing[:npar])**2.))
rmse_3ort = np.sqrt(np.mean((dmax_true[:npar]-dmax_3ort)**2.))
rmse_leb7 = np.sqrt(np.mean((dmax_true[:npar]-dmax_leb7[:npar])**2.))

# plot stuff
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]

# test plot
plt.scatter(dmax_true[:npar], dmax_sing, c='g', s=5., label=f'Single - RMSE = {rmse_sing:.2f} mm')
plt.scatter(dmax_true[:npar], dmax_2dvd[:npar], c='m', s=5., label=f'2DVD - RMSE = {rmse_2dvd:.2f} mm')
plt.scatter(dmax_true[:npar], dmax_masc[:npar], c='y', s=5., label=f'MASC - RMSE = {rmse_masc:.2f} mm')
plt.scatter(dmax_true[:npar], dmax_3ort, c='b', s=5., label=f'3ORT - RMSE = {rmse_3ort:.2f} mm')
plt.scatter(dmax_true[:npar], dmax_leb7[:npar], c='k', s=5., label=f'LEB7 - RMSE = {rmse_leb7:.2f} mm')

ax = plt.gca()
ax.set_xlabel('D (mm)', fontsize=16)
ax.set_ylabel('D$_{\sf{est}}$ (mm)', fontsize=16)
plt.legend()
plt.savefig('md_single_orient.pdf')
plt.close()