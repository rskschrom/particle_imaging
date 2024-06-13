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
nview = dmax_view.shape[2]

# get dmax estimate for all orientations for each additional view
dmax_est = np.empty([npar,norient,nview-1])
for i in range(nview-1):
    dmax_est[:,:,i] = np.max(dmax_view[:,:,:i+1], axis=2)

# average dmax estimates over orientation distributions
bvals = 10.**np.linspace(0., 2., 101)
nbv = len(bvals)
dmax_est_oavg = np.empty([npar,nbv,nview-1])

for i in range(nbv):
    odist = beta_spherical(1., bvals[i], bet*np.pi/180.)
    dmax_est_oavg[:,i,:] = 4.*np.pi*np.einsum('j,ijk->ik', wgt*odist, dmax_est)
    
# calculate orientation averaged rmse
dmax_true.shape = (npar,1,1)

vind = np.arange(nview-1)+1
b2, v2 = np.meshgrid(bvals, vind[::2], indexing='ij')

# get m-d errors
# find particles with a given m-d relationship
dmax_true.shape = (npar)
dmax_true = dmax_true*1.e-3
md_ind = 1

# m-d relations used in Heymsfield et al. 2023
a_h10_cgs = 6.1e-3
b_h10 = 2.05
a_h10 = a_h10_cgs*1.e-3/(1.e-2**b_h10)

a_bf_cgs = 2.94e-3
b_bf = 1.9 # 2.05
a_bf = a_bf_cgs*1.e-3/(1.e-2**b_bf)

# Mitchell 1990 m-d relation (agg. of rad. side planes)
a_m90_mgmm = 0.023
b_m90 = 1.8
a_m90 = a_m90_mgmm*1.e-6*(1.e3)**b_m90

a_vals = [a_m90,a_h10]
b_vals = [b_m90,b_h10]
md_names = ['m90','h10']
nmd = len(a_vals)

b_true = np.empty([nmd,nbv,nview-1])
i_true = np.empty([nmd,nbv,nview-1])
r_true = np.empty([nmd,nbv,nview-1])

for mi in range(nmd):
    a = a_vals[mi]
    b = b_vals[mi]
    
    # get apparent particle dmax close to m-d relation
    m_est_oavg = a*(1.e-3*dmax_est_oavg)**b
    mass_true.shape = (npar,1,1)
    err = np.abs(m_est_oavg-mass_true)/mass_true*100.
    mass_true.shape = (npar)

    for i in range(nbv):
        for j in range(nview-1):
            md_inds = np.arange(npar)[err[:,i,j]<5.]

            if (len(md_inds)>5):
                mass_md = mass_true[md_inds]
                dmax_md = dmax_true[md_inds]

                # get simulated coefficients of m-d relation
                b_true[mi,i,j], i_true[mi,i,j], r_true[mi,i,j], p, se = linregress(np.log10(dmax_md), np.log10(mass_md))
                
# plot
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
da = 0.001
b_levs = [1.5+np.arange(100)*0.02, 1.5+np.arange(100)*0.01]

cnames = ['M90', 'H10']

fig = plt.figure(figsize=(6,5))

fig_panels = ['a)','b)']

# plot each m-d coefficient set
for mi in range(2):
    # get a levels
    ax = fig.add_subplot(2,1,mi+1)
    a_true = 10.**i_true[mi,:,::2]
    a_min = np.min(a_true)
    a_max = np.max(a_true)
    a_min_round = int(a_min/da)*da

    numa = int((a_max-a_min)/da+3)

    a_levs = np.arange(numa)*da+a_min_round
    
    c1 = plt.contour((v2-1)/2+1, b2, b_true[mi,:,::2], levels=b_levs[mi], colors='c')
    plt.contourf((v2-1)/2+1, b2, a_true, levels=a_levs, cmap='magma')
    cb = plt.colorbar()
    cb.set_label('Estimated a (kg m$^{-b}$)', fontsize=12)

    ax.set_yscale('log')
    ax.clabel(c1, inline=True, fontsize=10, fmt='%1.2f')

    ax.set_title(f'{fig_panels[mi]} {cnames[mi]} - a = {a_vals[mi]:.3f}, b = {b_vals[mi]:.2f}', fontsize=12, ha='left', x=0.)
    ax.set_ylabel('h', fontsize=12)
        
    if mi==1:
        ax.set_xlabel('Number of probe views', fontsize=12)

plt.tight_layout()
plt.savefig(f'md_params_true.pdf', bbox_inches='tight')

# save luts
a_true = 10.**i_true[:,:,::2]
b_true = b_true[:,:,::2]
np.save('a_true.npy', a_true)
np.save('b_true.npy', b_true)
