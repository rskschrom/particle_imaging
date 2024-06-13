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

# split data into thirds of dmax_true
dthirds = np.percentile(dmax_true, [33.,66.])

dinds = [dmax_true<dthirds[0],
        (dmax_true>=dthirds[0])&(dmax_true<dthirds[1]),
         dmax_true>=dthirds[1]]

# get dmax errors for each additional view
nleb = dmax_view.shape[2]
uni_inds = np.clip(2*np.arange(nleb), 0, nleb-1)
uni_inds = uni_inds[uni_inds<nleb-2]

dmax_view_uni = dmax_view[:,:,uni_inds]
nuni = len(uni_inds)

# calculate maximum dimension estimates and percentage errors
dmax_est = np.empty([npar,norient,nuni-1])
dmax_true.shape = (npar,1,1)

for si in range(nuni-1):
    dmax_est[:,:,si] = np.max(dmax_view_uni[:,:,:si+1], axis=2)

dmax_perr = (dmax_est-dmax_true)/dmax_true*100.

# histogram percentage errors
nbin = 40
perr_edges = np.linspace(-40.,0.,nbin+1)
perr_hist = np.empty([3,nuni-1,nbin])

for i in range(nuni-1):
    for j in range(3):
        perr_hist[j,i,:],_ = np.histogram(dmax_perr[dinds[j],:,i].flatten(), bins=perr_edges)
    
# plot
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]

# mask data and create grid coordinates
perr_hist = perr_hist/np.sum(perr_hist)
perr_hist = np.ma.masked_where(perr_hist<0.0001, perr_hist)
v2, p2 = np.meshgrid(np.arange(nuni)+0.5, perr_edges, indexing='ij')

# tertile labels
dtert_lbl = [f'D < {dthirds[0]:.1f}mm',
             f'{dthirds[0]:.1f}mm ≤ D < {dthirds[1]:.1f}mm',
             f'D > {dthirds[1]:.1f}mm']

fig = plt.figure(figsize=(12,5))

# loop over tertiles
panels = ['a)','b)','c)']

for i in range(3):
    ax = fig.add_subplot(1,3,1+i)
    plt.pcolormesh(v2, p2, perr_hist[i,:,:], cmap='Spectral_r')

    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=12)

    ax.set_xlim([1,nuni-1])
    ax.set_xlabel('Number of probe views', fontsize=16)
    if i==0:
        ax.set_ylabel('Percentage error', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(f'{panels[i]} {dtert_lbl[i]}', fontsize=18, ha='left', x=0.)
    
    if i>0:
        ax.axes.yaxis.set_ticklabels([])

plt.suptitle('Relative frequency of $D$ % error', fontsize=18)
plt.tight_layout()
plt.savefig('perr_hist.pdf', bbox_inches='tight')