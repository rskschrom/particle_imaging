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

# read lebedev 7 view file
ncfile = Dataset('aggregate_leb.nc', 'r')

dmax_true = ncfile.variables['dmax'][:]
mass_true = ncfile.variables['mass_true'][:]
dmax_view = ncfile.variables['dmax_view'][:]
view_az = ncfile.variables['view_az'][:]
view_ze = ncfile.variables['view_ze'][:]
alp = ncfile.variables['alpha'][:]
bet = ncfile.variables['beta'][:]
wgt = ncfile.variables['weight'][:]

norient = dmax_view.shape[1]
npar = dmax_view.shape[0]

# get dmax errors for each additional view
nleb = dmax_view.shape[2]
uni_inds = np.clip(2*np.arange(nleb), 0, nleb-1)
uni_inds = uni_inds[uni_inds<nleb-2]

dmax_view_uni = dmax_view[:,:,uni_inds]
nuni = len(uni_inds)

# set orientation distributions
bvals = 10.**np.linspace(0., 2., 101)
nbv = len(bvals)

# calculate orientation averaged dmax estimates for each particle
dmax_step_leb7_oavg = np.empty([npar,nbv,nuni-1])
rmse_step_oavg = np.empty([nbv,nuni-1])

for i in range(nbv):
    odist = beta_spherical(1., bvals[i], bet*np.pi/180.)

    for j in range(nuni-1):
        dmax_step_leb7 = np.max(dmax_view_uni[:,:,:j+1], axis=2)
        dmax_step_leb7_oavg[:,i,j] = 4.*np.pi*np.einsum('j,ij->i', wgt*odist, dmax_step_leb7)
        rmse_step_oavg[i,j] = np.sqrt(np.mean((dmax_true-dmax_step_leb7_oavg[:,i,j])**2.))    

# plot
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]

plt.figure(figsize=(5,4))

bv2, nv2 = np.meshgrid(bvals, np.arange(nuni-1)+1, indexing='ij')
plt.contourf(nv2, bv2, rmse_step_oavg, levels=11, cmap='coolwarm')
cb = plt.colorbar()
cb.set_label('RMSE (mm)', fontsize=14)

ax = plt.gca()
ax.set_xlabel('Number of probe views', fontsize=14)
ax.set_ylabel('h', fontsize=14)

plt.tight_layout()
plt.savefig('dmax_err_orient.pdf')