from itertools import count
import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt

# from Roger
def plot_maps(map, mask=[], min=0, max=1, output_fig='', title='', nest=False, rot=(0,0,0), coord=[]):
    # coord: 'C': equatorial, 'G': galactic, 'E': ecliptic
    if False: map_plot = hp.ma(map)
    else: map_plot = map
    if mask!=[]: map_plot.mask = np.logical_not(mask)
    plt.figure(figsize=(9,6), dpi=300)
    if coord==[]: hp.mollview(map_plot, cmap='viridis', title=title, min=min, max=max, nest=nest, rot=rot)
    else: hp.mollview(map_plot, cmap='viridis', title=title, min=min, max=max, nest=nest, rot=rot, coord=coord)
    hp.graticule(dmer=360,dpar=360,alpha=0)
    plt.savefig(output_fig, bbox_inches='tight')
    plt.close()

# from : https://stackoverflow.com/questions/44443498/how-to-convert-and-save-healpy-map-to-different-coordinate-system
def change_coord(m, coord):
    """ Change coordinates of a HEALPIX map
    m : map or array of maps (map(s) to be rotated)
    coord : sequence of two character
      First character is the coordinate system of m, second character is the coordinate system of the output map.
      As in HEALPIX, allowed coordinate systems are 'G' (galactic), 'E' (ecliptic) or 'C' (equatorial)
    """
    # Basic HEALPix parameters
    npix = m.shape[-1]
    nside = hp.npix2nside(npix)
    ang = hp.pix2ang(nside, np.arange(npix))
    # Select the coordinate transformation
    rot = hp.Rotator(coord=reversed(coord))
    # Convert the coordinates
    new_ang = rot(*ang)
    new_pix = hp.ang2pix(nside, *new_ang)

    return m[..., new_pix]

def filter_lowell_from_map(map, nside, lowell_min=20):
    def filter_lowell(ell):
        return [ell > lowell_min]
    map_alm = hp.map2alm(map)
    new_alm = hp.almxfl(map_alm, filter_lowell(np.arange(3*nside))[0])
    return hp.alm2map(new_alm, nside=nside)

def filter_highell_from_map(map, nside, lmax=1500):
    def filter(ell):
        return [ell < lmax]
    map_alm = hp.map2alm(map)
    new_alm = hp.almxfl(map_alm, filter(np.arange(3*nside))[0])
    return hp.alm2map(new_alm, nside=nside)

# from https://namaster.readthedocs.io/en/latest/sample_bins.html
def get_correlation_from_map(map1, mask1, map2=[], mask2=[], nside=256, nb_multipoles_per_bins=50, lowell_min=20, pixel_window=False, shotnoise=0.0):
    field1 = nmt.NmtField(mask1, [map1])
    if len(map2)!=0:
        field2 = nmt.NmtField(mask2, [map2])
        auto = False
    else:
        field2 = field1
        auto = True

    ells = np.arange(3 * nside, dtype='int32')  # Array of multipoles
    weights = 1./nb_multipoles_per_bins * np.ones_like(ells)  # Array of weights
    if pixel_window:
        pixwin = hp.pixwin(nside)
        if auto: pixwin *= pixwin
    else: pixwin = 1.0
    #weights /= pixwin # ????? is that correct ?????
    weights[ells < lowell_min] = 0
    bpws = -1 + np.zeros_like(ells)  # Array of bandpower indices
    i = 0
    while nb_multipoles_per_bins * (i + 1) + 2 < 3 * nside:
        bpws[nb_multipoles_per_bins * i + 2:nb_multipoles_per_bins * (i + 1) + 2] = i
        i += 1
    bins = nmt.NmtBin(nside=nside, bpws=bpws, ells=ells, weights=weights)
    # for linear spacing with constant weights, equivalent to: bins = nmt.NmtBin.from_nside_linear(nside, nb_multipoles_per_bins)

    lbin = bins.get_effective_ells()
    if pixel_window: pixwin = pixwin[lbin.astype('int')]

    Cl = nmt.compute_full_master(field1, field2, bins)

    Cl = Cl[0]/pixwin - shotnoise # ???? is that correct ??????

    return lbin, Cl, bins

# modified from Roger
#https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.anafast.htmlhttps://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.anafast.html
def get_correlation_from_map_hp(map1, mask1, map2=[], mask2=[], lmin=30, lmax=1200, nb_bin=15, pixel_window=False, shotnoise=0.0, nside=256):
    map1_masked = hp.ma(map1)*mask1
    if len(map2)!=0:
        map2_masked = hp.ma(map2)*mask2
        fsky, _, _, _  = get_fsky(mask1*mask2)
        auto = False
    else:
        map2_masked = map1_masked
        fsky, _, _, _  = get_fsky(mask1)
        auto = True
    C = (1/fsky)*hp.anafast(map1_masked, map2_masked, nspec=lmax) #nspec or lmax???

    width = int((lmax - lmin)/nb_bin)
    bin_Cl = np.zeros((nb_bin))
    bin_l = np.zeros((nb_bin))
    for i in range(nb_bin):
        ell_min = lmin + width*i
        ell_max = lmin + (i+1)*width
        ell_seq = np.arange(ell_min, ell_max)
        weights = (2*ell_seq + 1)/np.sum(2*ell_seq + 1)
        bin_l[i] = np.sum(weights*ell_seq)
        bin_Cl[i] = np.sum(weights*C[ell_min:ell_max])

    if pixel_window:
        pixwin = hp.pixwin(nside, lmax=lmax)
        if auto: pixwin *= pixwin
        pixwin = pixwin[bin_l.astype('int')]
    else: pixwin = 1.0

    bin_Cl = bin_Cl/pixwin - shotnoise # ???? is that correct ??????

    return bin_l, bin_Cl, C


def get_fsky(mask, weights=[], threshold=0.0):
    # if weights!=[]:
    keep = (mask > threshold).astype(bool)
    weights = mask.copy() if weights==[] else weights
    # wi defined in eqt 9 of Hivon et al. 2001
    w2 = np.sum(weights[keep]**2) / np.sum(weights[keep])
    w4 = np.sum(weights[keep]**4) / np.sum(weights[keep])
    # else: keep, w2, w4 = True, 1.0, 1.0
    fsky = np.sum(mask[keep]) / len(mask)
    fsky_cov = fsky * w2**2 / w4 # "factor w2^2/w4 accounts for the loss of modes induced by the pixel weighting"
    print('fsky = ' + str(fsky), 'fsky_cov = ' + str(fsky_cov))
    return fsky, fsky_cov, w2, w4

# From Krowelski: section 3.5 of Hivon et al. 2001 / Knox 1995
def get_gaussian_errors(Cl1, Cl2, Clcross, mask, bins):
    fsky, _, _, _  = get_fsky(mask)
    lmax_bin = np.array([bins.get_ell_list(i)[-1] for i in range(bins.get_n_bands())])
    lmin_bin = np.array([bins.get_ell_list(i)[0] for i in range(bins.get_n_bands())])
    Nmodes_bin = lmax_bin**2 - lmin_bin**2 # from Alex code, = Delta l * 2l (where Delta l = lmax-lmin; l = 0.5*(lmax+lmin))
    Nmodes_bin = (lmax_bin - lmin_bin)*(lmax_bin + lmin_bin + 1) # from Alex paper
    return np.sqrt(np.abs( (Clcross**2 + Cl1 * Cl2) / (fsky * Nmodes_bin) ))

def get_gaussian_errors_hp(Cl1, Cl2, Clcross, mask, delta_bins, l):
    fsky, _, _, _  = get_fsky(mask)
    Nmodes_bin = delta_bins*(2*l + 1) # from Alex paper
    return np.sqrt(np.abs( (Clcross**2 + Cl1 * Cl2) / (fsky * Nmodes_bin) ))


def get_mask_map_from_radec(ra, dec, nside_mask, nside_out):
    sc = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
    gc = sc.galactic  # galactic coordinate system
    l = gc.l.degree
    b = gc.b.degree

    indices = hp.pixelfunc.ang2pix(nside_mask, l, b, lonlat=True)
    mask = np.zeros(hp.nside2npix(nside_mask), dtype=np.float64)
    mask[np.unique(indices)] = 1

    mask = hp.pixelfunc.ud_grade(mask,nside_out=nside_out) # upgrade the mask resolution to nside_out

    return mask

def get_overdensity_map_from_radec(ra, dec, nside, mask=[], count_correction=1):
    # count_correction multiplies the counts
    sc = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
    gc = sc.galactic  # galactic coordinate system
    l = gc.l.degree
    b = gc.b.degree

    indices = hp.pixelfunc.ang2pix(nside, l, b, lonlat=True)
    idx, counts = np.unique(indices, return_counts=True) # each non empty healpix indices and corresponding number of object
    pixarea  = hp.nside2pixarea(nside, degrees=True)
    if mask==[]: nb_pixels = len(idx) # in this case n_mean = np.mean(counts)/pixarea
    else: nb_pixels = np.sum(mask) # if nside_mask == nside_out (during build mask from get_mask_map_from_radec): np.sum(mask) == len(idx)
    n_mean = len(ra)/(pixarea*nb_pixels)
    shotnoise = 1./(n_mean * (180./np.pi)**2.)
    print(f'Average number of counts in pixel of size {nside}: {np.mean(counts)} .')
    print(f'Average number density: {n_mean} .')

    map_counts = np.zeros(hp.nside2npix(nside), dtype=np.float64)
    map_counts[idx] = counts

    map = map_counts/(pixarea*n_mean)
    if not(isinstance(count_correction, int)): count_correction[count_correction==0.0] = 1.0
    map = map*count_correction

    if mask==[]: pass
    else: map[mask!=1.0] = 0.0

    return map - 1.0, n_mean, shotnoise, map_counts, pixarea

def get_shotnoise_from_map(map, nside, mask_binary=[]):
    pixarea  = hp.nside2pixarea(nside, degrees=True)
    if mask_binary==[]: nmean = np.mean(map)
    else: nmean = np.sum(map*mask_binary)/np.sum(mask_binary)/pixarea
    shotnoise = 1./(nmean * (180./np.pi)**2.)
    return shotnoise

# from Roger # calculate the shared masked region for the two maps
# smoothing kernel in arcminutes
def get_shared_mask_map(mask1, mask2, smoothing_kernel):
    smooth_mask1 = hp.sphtfunc.smoothing(mask1,fwhm=np.deg2rad(smoothing_kernel/60.))
    smooth_mask2 = hp.sphtfunc.smoothing(mask2,fwhm=np.deg2rad(smoothing_kernel/60.))
    mask = smooth_mask1 * smooth_mask2
    return mask

def get_cross_mask_map(mask1, mask2):
    return mask1*mask2

def get_field(density, mask, counts=0., compute_field=True):
    if compute_field: f = nmt.NmtField(mask, [density])
    else: f=0
    fsky = np.sum(mask)/len(mask)

    tot_map = np.sum(counts * mask)
    dens = tot_map/(fsky * 4*np.pi)
    if dens == 0.: dens = 1.
    SN = 1./dens * (180./np.pi)**2.

    return f, fsky, SN

def get_gaussian_errs(cl_cross, cl_auto1, cl_auto2, fsky, Nmodes_bin):
    return np.sqrt(np.abs( (cl_cross**2 + cl_auto1 * cl_auto2) / (fsky * Nmodes_bin) ))

def get_bins(nside, nb_ells=1500, nb_multipoles_per_bins=20, nb_ells_removed=0):
    ells = np.arange(nb_ells, dtype='int32')
    bpws = -1 + np.zeros_like(ells)  # Array of bandpower indices
    weights = np.ones_like(ells)/nb_multipoles_per_bins

    i = 0
    while nb_multipoles_per_bins * (i + 1) + nb_ells_removed < nb_ells:
        bpws[nb_multipoles_per_bins * i + nb_ells_removed:nb_multipoles_per_bins * (i + 1) + nb_ells_removed] = i
        i += 1
    #for i in range(round((nb_ells-nb_ells_removed)/nb_multipoles_per_bins)):
    #	bpws[i*nb_multipoles_per_bins+nb_ells_removed:(i+1)*nb_multipoles_per_bins+nb_ells_removed] = i

    bins = nmt.NmtBin(bpws=bpws, ells=ells, weights=weights)  # Default run, lmax = 1000.  3000 for highL

    lbin = bins.get_effective_ells()
    lmin, lmax = [], []
    for i in range(np.max(bpws)-1):
        lmin.append(bins.get_ell_list(i)[0])
        lmax.append(bins.get_ell_list(i+1)[0])
    lmin, lmax = np.array(lmin).astype('float'), np.array(lmax).astype('float')

    lmax_bin = np.array([bins.get_ell_list(i)[-1] for i in range(bins.get_n_bands())])
    lmin_bin = np.array([bins.get_ell_list(i)[0] for i in range(bins.get_n_bands())])
    Nmodes_bin = (lmax_bin - lmin_bin + 1)*(lmax_bin + lmin_bin + 1)

    return bins, lbin, lmin, lmax, lmax_bin, lmin_bin, Nmodes_bin

def get_coupled_cells(f1, f2, nside, nb_ells=1500, nb_multipoles_per_bins=20, nb_ells_removed=0, auto1=True, auto2=True, cross=True):

    bins, lbin, _, _, _, _, _ = get_bins(nside, nb_ells=nb_ells, nb_multipoles_per_bins=nb_multipoles_per_bins, nb_ells_removed=nb_ells_removed)

    Cl_cross_coupled, Cl_auto_gg_coupled, Cl_auto_kk_coupled = 0,0,0
    if cross:
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f1, f2, bins)
        windows = w.get_bandpower_windows()
        Cl_cross_coupled = nmt.compute_coupled_cell(f1, f2)

    if auto1:
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f1, f1, bins)
        windows = w.get_bandpower_windows()
        Cl_auto_gg_coupled = nmt.compute_coupled_cell(f1, f1)

    if auto2:
        w = nmt.NmtWorkspace()
        w.compute_coupling_matrix(f2, f2, bins)
        windows = w.get_bandpower_windows()
        Cl_auto_kk_coupled = nmt.compute_coupled_cell(f2, f2)

    return Cl_cross_coupled, Cl_auto_gg_coupled, Cl_auto_kk_coupled, lbin


def get_cells(f1, f2, nside, nb_ells=1500, nb_multipoles_per_bins=20, nb_ells_removed=0, auto1=True, auto2=True, cross=True):

    bins, lbin, _, _, _, _, _ = get_bins(nside, nb_ells=nb_ells, nb_multipoles_per_bins=nb_multipoles_per_bins, nb_ells_removed=nb_ells_removed)

    Cl_cross, Cl_auto_gg, Cl_auto_kk = 0,0,0

    if cross: Cl_cross   = nmt.compute_full_master(f1, f2, bins)
    if auto1: Cl_auto_gg = nmt.compute_full_master(f1, f1, bins)
    if auto2: Cl_auto_kk = nmt.compute_full_master(f2, f2, bins)

    return Cl_cross, Cl_auto_gg, Cl_auto_kk, lbin


def SNR_computation(ells_theory, cl_theory, ells_data, cl_data, errs_data, lmin=20, lmax=1000):
    # Cut the data
    mask_th = (ells_theory > lmin) & (ells_theory < lmax)
    ells_theory = ells_theory[mask_th]
    cl_theory = cl_theory[mask_th]
    mask_data = (ells_data > lmin) & (ells_data < lmax)
    ells_data = ells_data[mask_data]
    cl_data = cl_data[mask_data]
    errs_data = errs_data[mask_data]

    th  = np.interp(ells_data, ells_theory, cl_theory)
    chi2 = np.sum((th - cl_data)**2/errs_data**2)
    chi2_null = np.sum((np.zeros(np.shape(th)) - cl_data)**2/errs_data**2)
    return np.sqrt(chi2_null - chi2)


#import camb
# https://camb.readthedocs.io/en/latest/CAMBdemo.html
def get_pk_camb(z, h, omega_b, omega_cdm, omega_k, tau_reio, A_s, n_s, lmax, Pkmax, nonlinear=True, pk_interp=True, minkh=1e-4, maxkh=1, npoints=200, z_min=2.0, z_max=3.5):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100, ombh2=omega_b*h**2, omch2=omega_cdm*h**2, mnu=0.06, omk=omega_k, tau=tau_reio)
    pars.InitPower.set_params(As=A_s, ns=n_s, r=0)
    #pars.set_for_lmax(lmax, lens_potential_accuracy=0);
    pars.set_matter_power(redshifts=z, kmax=Pkmax)
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    if nonlinear:
        pars.NonLinear = camb.model.NonLinear_both
        results.calc_power_spectra(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=npoints)
    s8 = np.array(results.get_sigma8())
    if pk_interp: pk_nl_interp =  camb.get_matter_power_interpolator(pars,nonlinear=nonlinear, hubble_units=True, k_hunit=True,kmax=20, zmin=0, zmax=z_max)
    else: pk_nl_interp = 0
    return kh, z, pk, pk_nl_interp


from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
def get_dndz(data_z, zmin, zmax, nbins, window_length=51, polyorder=3):
    dNdz, z_edges = np.histogram(data_z, range=(zmin, zmax), bins=nbins, density=True)
    dNdz_smooth = savgol_filter(dNdz, window_length, polyorder)
    z, dz = (z_edges[:-1]+z_edges[1:])/2, z_edges[1:]-z_edges[:-1]
    f_dNdz = interp1d(z, dNdz_smooth)
    return z_edges, z, dz, dNdz, dNdz_smooth, f_dNdz


def get_model(cosmo, f_dndz, Pk, lmax, z_min, z_max, nb_bins_in_z=200, s_mu=2/5):
    c = 299792458./1000.           #in km/s
    z_cmb = 1100
    def H(z):
        return cosmo.H(z).value

    def chi(z):
        return cosmo.comoving_distance(z).value * cosmo.h

    def W_k(z):
        f = 3/2*cosmo.Om0*(cosmo.H0.value)**2/c**2
        W_k_chi = f*(1+z)*chi(z)*(chi(z_cmb)-chi(z))/chi(z_cmb)
        W_k_z = c/H(z)*W_k_chi
        return W_k_z

    def bq_dMdB(z):
    # Eq. 25 from https://arxiv.org/pdf/1708.02225.pdf   (dMdB 2017)
        return 0.53 + 0.289*(1. + z)**2

    def W_g(z):
        return f_dndz(z)*bq_dMdB(z)

    def W_mu(z):
        z_range = np.linspace(z_min, z_max, 500) # can I use z_max instead of z_cmb??? effectively we can go larger than z_max with dN ????
        dz = (z_max-z_min)/(500-1)
        g_int = np.ones(len(z))
        for i,z_i in enumerate(z):
            g_i = chi(z_i)*(chi(z_range)-chi(z_i))/chi(z_range)*f_dndz(z_range)
            g_int[i] = np.sum(dz*g_i[z_range>z_i])
        f = (5*s_mu-2)*3/2*cosmo.Om0*(cosmo.H0.value)**2/c**2
        W_mu_chi = f*(1+z)*g_int
        W_mu_z = c/H(z)*W_mu_chi
        return W_mu_z

    def integrand(z, k):
        f = H(z)/c/chi(z)**2
        g = f*W_k(z)*(W_g(z)+W_mu(z))*Pk(z, k, grid=False)
        return g

    zvec = np.linspace(z_min, z_max, nb_bins_in_z+1)
    dz = zvec[1:] - zvec[:-1]
    z = zvec[:-1] + dz/2
    ells = np.ones((len(z), lmax))*np.arange(lmax)
    k = (ells.T+0.5)/chi(z)
    Cl = np.sum(integrand(z,k)*dz, axis=1)
    return ells[0], Cl



def compute_cl_and_cov(map1, mask1, map2=[], mask2=[], nside=256, nb_multipoles_per_bins=50, lowell_min=20, pixel_window1=False, pixel_window2=False, shotnoise1=0.0, shotnoise2=0.0):
    field1 = nmt.NmtField(mask1, [map1])
    if len(map2)!=0:
        field2 = nmt.NmtField(mask2, [map2])
        auto = False
    else:
        field2 = field1
        auto = True

    ells = np.arange(3 * nside, dtype='int32')  # Array of multipoles
    weights = 1./nb_multipoles_per_bins * np.ones_like(ells)  # Array of weights
    weights[ells < lowell_min] = 0
    bpws = -1 + np.zeros_like(ells)  # Array of bandpower indices
    i = 0
    while nb_multipoles_per_bins * (i + 1) + 2 < 3 * nside:
        bpws[nb_multipoles_per_bins * i + 2:nb_multipoles_per_bins * (i + 1) + 2] = i
        i += 1
    bins = nmt.NmtBin(nside=nside, bpws=bpws, ells=ells, weights=weights)
    # for linear spacing with constant weights, equivalent to: bins = nmt.NmtBin.from_nside_linear(nside, nb_multipoles_per_bins)

    lbin = bins.get_effective_ells()

    pixwin = hp.pixwin(nside)
    pixwin_auto = pixwin*pixwin
    pixwin = pixwin[lbin.astype('int')]
    pixwin_auto = pixwin_auto[lbin.astype('int')]
    pixwin11, pixwin12, pixwin22 = 1.0, 1.0, 1.0
    if pixel_window1: pixwin11 = pixwin_auto
    if pixel_window1 or pixel_window2: pixwin12 = pixwin
    if pixel_window2: pixwin22 = pixwin_auto

    w11 = nmt.NmtWorkspace()
    w11.compute_coupling_matrix(field1, field1, bins)
    cl_coupled11 = nmt.compute_coupled_cell(field1, field1)
    Cl11 = w11.decouple_cell(cl_coupled11)
    Cl11 = Cl11[0]/pixwin11 - shotnoise1
    cl_coupled, Cl = [cl_coupled11], [Cl11]
    if auto==False:
        w12 = nmt.NmtWorkspace()
        w12.compute_coupling_matrix(field1, field2, bins)
        w22 = nmt.NmtWorkspace()
        w22.compute_coupling_matrix(field2, field2, bins)
        cl_coupled12 = nmt.compute_coupled_cell(field1, field2)
        Cl12 = w12.decouple_cell(cl_coupled12)
        Cl12 = Cl12[0]/pixwin12
        cl_coupled22 = nmt.compute_coupled_cell(field2, field2)
        Cl22 = w22.decouple_cell(cl_coupled22)
        Cl22 = Cl22[0]/pixwin22 - shotnoise2
        cl_coupled.extend([cl_coupled12, cl_coupled22])
        Cl.extend([Cl12, Cl22])

    # Pk11 x Pk11
    cw = nmt.NmtCovarianceWorkspace()
    # (fl_a1, fl_a2, fl_b1=None, fl_b2=None, lmax=None, n_iter=3, l_toeplitz=-1, l_exact=-1, dl_band=-1)
    cw.compute_coupling_coefficients(field1, field1, field1, field1)
    # (cw, spin_a1, spin_a2, spin_b1, spin_b2, cl_a1b1, cl_a1b2, cl_a2b1, cl_a2b2, wa, wb=None, coupled=False)
    cov1111 = nmt.gaussian_covariance(cw, 0, 0, 0, 0,  cl_coupled11,  cl_coupled11, cl_coupled11, cl_coupled11, w11, wb=w11)
    cov = [cov1111]
    if auto==False:
            # Pk11 x Pk12
            cw.compute_coupling_coefficients(field1, field1, field1, field2)
            cov1112 = nmt.gaussian_covariance(cw, 0, 0, 0, 0,  cl_coupled11,  cl_coupled12, cl_coupled11, cl_coupled12, w11, wb=w12)
            # Pk11 x Pk22
            cw.compute_coupling_coefficients(field1, field1, field2, field2)
            cov1122 = nmt.gaussian_covariance(cw, 0, 0, 0, 0,  cl_coupled12,  cl_coupled12, cl_coupled12, cl_coupled12, w11, wb=w22)
            # Pk12 x Pk12
            cw.compute_coupling_coefficients(field1, field2, field1, field2)
            cov1212 = nmt.gaussian_covariance(cw, 0, 0, 0, 0,  cl_coupled11,  cl_coupled12, cl_coupled12, cl_coupled22, w12, wb=w12)
            # Pk12 x Pk22
            cw.compute_coupling_coefficients(field1, field2, field2, field2)
            cov1222 = nmt.gaussian_covariance(cw, 0, 0, 0, 0,  cl_coupled12,  cl_coupled12, cl_coupled22, cl_coupled22, w12, wb=w22)
            # Pk22 x Pk22
            cw.compute_coupling_coefficients(field2, field2, field2, field2)
            cov2222 = nmt.gaussian_covariance(cw, 0, 0, 0, 0,  cl_coupled22,  cl_coupled22, cl_coupled22, cl_coupled22, w22, wb=w22)

            cov.extend([cov1112, cov1122, cov1212, cov1222, cov2222])


    d = np.sqrt(np.diag(cov1111))
    f_d = np.ones(cov1111.shape)
    f_d = (d*f_d).T*d
    corr1111 = cov1111/(f_d)

    return lbin, Cl, bins, cl_coupled, cov
