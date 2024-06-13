import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# integrate psd
def psd_integrate(psd_data, sizes):
    ds = sizes[1:]-sizes[:-1]
    ds.shape = (len(sizes)-1,1)
    dint = np.sum(0.5*(psd_data[1:,:]+psd_data[:-1,:])*ds, axis=0)
    return dint
    
# create psds
# n(d) = d**mu*exp(-d*lam), mu = 2
nlam = 51
lam_vals = 10.**np.linspace(np.log10(3.e2), np.log10(5.e3), nlam)

nsize = 51
min_size = 1.e-6
max_size = 2.4e-2
sizes = 10.**np.linspace(np.log10(min_size), np.log10(max_size), nsize)

s2, l2 = np.meshgrid(sizes, lam_vals, indexing='ij')
psds = s2**2.*np.exp(-s2*l2)

# get normalization factors to mass for bf coefficients
a_bf = 0.019
b_bf = 1.9
mass_bf = a_bf*s2**b_bf
nfac_bf = 1./psd_integrate(psds*mass_bf, sizes)

# Mitchell 1990 m-d relation (agg. of rad. side planes)
a_m90_mgmm = 0.023
b_m90 = 1.8
a_m90 = a_m90_mgmm*1.e-6*(1.e3)**b_m90
mass_m90 = a_m90*s2**b_m90
nfac_m90 = 1./psd_integrate(psds*mass_m90, sizes)

# get normalization factors to mass for hw coefficients
a_hw = 0.077
b_hw = 2.05
mass_hw = a_hw*s2**b_hw
nfac_hw = 1./psd_integrate(psds*mass_hw, sizes)

# read in mass-dimension relation coefficients
a_true = np.load('a_true.npy')
b_true = np.load('b_true.npy')

# recalculate integrate mass for psd using estimated coefficients
a_m90_est = a_true[0,0,:]
b_m90_est = b_true[0,0,:]
a_hw_est = a_true[1,0,:]
b_hw_est = b_true[1,0,:]
nprobe = len(a_m90_est)

mtot_m90_est = np.empty([nprobe,nlam])
mtot_bf_est = np.empty([nprobe,nlam])
mtot_hw_est = np.empty([nprobe,nlam])

for pi in range(nprobe):
    mtot_m90_est[pi,:] = psd_integrate(nfac_m90*psds*a_m90_est[pi]*s2**b_m90_est[pi], sizes)
    mtot_hw_est[pi,:] = psd_integrate(nfac_hw*psds*a_hw_est[pi]*s2**b_hw_est[pi], sizes)

# plot
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Nimbus Sans"]
fig_panels = ['b)','c)']
cnames = ['M90','H10']

fig = plt.figure(figsize=(12,5))
a_vals = [a_m90, a_hw]
b_vals = [b_m90, b_hw]

# plot psd first
ax = fig.add_subplot(1,2,1)
plt.contourf(s2, l2, np.log10(1.e-3*nfac_bf*psds), levels=np.arange(-33.,12.,3), cmap='Spectral_r', vmin=-30, vmax=9.)
cb = plt.colorbar()
cb.set_label('(log$_{10}$[$\sf{m^{-4}}$])', fontsize=14)

ax.set_title(f'a) Normalized PSDs', fontsize=16, ha='left', x=0.)
ax.set_xlabel('D (m)', fontsize=14)
ax.set_ylabel('$\lambda$ (m$^{-1}$)', fontsize=14)
ax.set_xticks(np.arange(0., 0.024, 0.004), minor=False)
ax.set_xlim([0.,0.02])
ax.xaxis.set_major_formatter(FormatStrFormatter('%0.3f'))

# plot each m-d coefficient set
p2p, l2p = np.meshgrid(np.arange(nprobe)+1, lam_vals, indexing='ij')
mtot = [mtot_m90_est, mtot_hw_est]

for mi in range(2):
    ax = fig.add_subplot(2,2,2*mi+2)

    plt.contourf(p2p, l2p, mtot[mi], levels=11, cmap='magma')
    cb = plt.colorbar()
    cb.set_label('(g m$^{-3}$)', fontsize=14)

    ax.set_title(f'{fig_panels[mi]} {cnames[mi]} - a = {a_vals[mi]:.3f}, b = {b_vals[mi]:.2f}', fontsize=16, ha='left', x=0.)
    if mi==1:
        ax.set_xlabel('Number of probe views', fontsize=14)
    ax.set_ylabel('$\lambda$ (m$^{-1}$)', fontsize=14)

plt.tight_layout()
plt.savefig('mass_err.pdf', bbox_inches='tight')
