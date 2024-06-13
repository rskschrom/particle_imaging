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

# read lebedev 7 view, multi-orientation file
ncfile = Dataset('aggregate_leb.nc', 'r')

dmax_true = ncfile.variables['dmax'][:]
mass_true = ncfile.variables['mass_true'][:]
dmax_view = ncfile.variables['dmax_view'][:]
view_az = ncfile.variables['view_az'][:]
view_ze = ncfile.variables['view_ze'][:]
alp = ncfile.variables['alpha'][:]
bet = ncfile.variables['beta'][:]
wgt = ncfile.variables['weight'][:]

npar = dmax_view.shape[0]
norient = dmax_view.shape[1]

# use only even nodes
dmax_view = dmax_view[:,:,::2]

nview = dmax_view.shape[2]

# get dmean estimate for all random orientation for each additional view
odist = beta_spherical(1., 1., bet*np.pi/180.)
dmean_est = np.empty([npar,norient,nview-1])
dmax_est = np.empty([npar,norient,nview-1])

for i in range(nview-1):
    dmean_est[:,:,i] = np.mean(dmax_view[:,:,:i+1], axis=2)
    dmax_est[:,:,i] = np.max(dmax_view[:,:,:i+1], axis=2)

dmean_est_oavg = 4.*np.pi*np.einsum('j,ijk->ik', wgt*odist, dmean_est)
dmax_est_oavg = 4.*np.pi*np.einsum('j,ijk->ik', wgt*odist, dmax_est)

# get relation between dmean and dtrue
bmn, imn, rmn, p, se, = linregress(np.log10(dmax_true), np.log10(dmean_est_oavg[:,1]))

# plot
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]

plt.scatter(dmax_true, dmean_est_oavg[:,1], c='k', s=1., label='d - two views')
plt.scatter(dmax_true, dmax_est_oavg[:,1], c='c', s=1., label='D - two views')

ax = plt.gca()
ax.set_xlabel('D truth (mm)', fontsize=14)
ax.set_ylabel('D estimate (mm)', fontsize=14)

plt.legend()
plt.savefig('dmean.pdf')