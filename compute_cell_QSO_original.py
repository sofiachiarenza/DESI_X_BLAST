import matplotlib
matplotlib.use('Agg')

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import pymaster as nmt
from astropy.io import fits
import time
from astropy.coordinates import SkyCoord
from astropy import units as u
import sys

from functions_AT_new import *


print('=== COMPUTE CELL ===')

#HEALPix map resolution
nside = 2048

PATH_d = '/global/cfs/cdirs/desi/users/akrolew/QSO_maps/' # path for desi data
PATH_m = '/global/cfs/cdirs/desi/users/akrolew/QSO_maps/' # path for mocks
PATH_p = '/pscratch/sd/r/rmvd2/CMBxLya/data/COM_Lensing_4096_R3.00/' # path for planck data
abacus_dir ='/global/cfs/cdirs/desi/users/akrolew/AbacusSummit_huge_c000_ph201_QSO/data/'

PATH_of = '/global/homes/s/schiaren/CMBxQSO/results/plots'
PATH_oc = '/global/homes/s/schiaren/CMBxQSO/results'

data_type = str(sys.argv[1])

if data_type == 'data':
    # read DESI DR1 data 
    zmin = float(sys.argv[2]) # zmin
    zmax = float(sys.argv[3]) # zmax
    version = 'v1.5' #Krolewski generated DESI DR1 data 29072024
    # lensing_str='PR3'
    lensing_str='PR4'
    print(f'use Planck {lensing_str}')
    data_str=f'desi_dr1_{zmin:.2f}_z_{zmax:.2f}_{lensing_str}_PR4mask'

elif data_type == 'mock':
    ######read Gaussian mocks
    i_min = int(sys.argv[2]) # first mock
    i_max = int(sys.argv[3]) # last mock
    # version = 'v1.1' #--> old version pre June 24 2024
    version = 'v1.2' #Krolewski generated new Gaussian mocks 
    mock_version = '_v2'# Krolewski generated new Gaussian mocks with correct dNdz splining and magnification bias--> use these
    add_nlqq=True
    data_str='gaussian_mocks'
    if add_nlqq: 
        data_str += f'_nlqq{mock_version}'
        print("INLCUDE NOISE FOR GAUSSIAN MOCKS")

elif 'abacus' in data_type:
    ######read Abacus mocks
    i_min = int(sys.argv[2]) # first mock
    i_max = int(sys.argv[3]) # last mock
    version='v1.5'
    data_str=data_type

ap_scale = float(sys.argv[4]) # apodization scale in degrees
comp_s = float(sys.argv[5]) # completeness
cut_off = float(sys.argv[6]) # cut off for completeness mask * binary mask 
filter_lowell_alm = sys.argv[7].lower() in ['true', '1', 't', 'y', 'yes']
filter_highell_alm = sys.argv[8].lower() in ['true', '1', 't', 'y', 'yes']

mask_name = sys.argv[9] # N: NGC, S: SGC, A: NGC+SGC

sepnorm = False
sys_wts = True # if True: data with systematics
do_cov = False # update theory paths as well
compute_cl = True
compute_coupled_cls = False
coupled_str=''
if compute_coupled_cls: coupled_str='coupled_'
compute_field = True # (usually) should be True if compute_cl is True

filter_str=''
if filter_lowell_alm:
    filter_str = '_filter_lowell'
if filter_highell_alm:
    filter_str += '_filter_highell'
    
if sepnorm:
    sepnorm_str = 'True'
else:
    sepnorm_str = 'False'

compute_lmin = 5
binsize = int(sys.argv[10])

opt_ = f''
opt1 = f'_{version}_lmax6144_mbinary' 
opt2 = f'_{version}_lmax6144_mcompleteness'
opt3 = f'_{version}_lmax6144_mapC2s{ap_scale}{filter_str}_comp{comp_s:.1f}_cutoff{cut_off:.1f}_{data_str}_lmin{compute_lmin:.0f}_binsize{binsize:.0f}_sepnorm{sepnorm_str}'

print('REGION:          ', mask_name)
print('number of mocks: ', i_max) if data_type == 'mock' or 'abacus' in data_type else None
print('apo scale (deg): ', ap_scale)
print('cut off mask:    ', cut_off)
print('data set :       ', data_str)
print('completeness:    ', comp_s)
print('nside:           ', nside)
print('systematics:     ', sys_wts)
print('opt:             ', opt3)
print('lmin for Cell:   ', compute_lmin)
print('binsize:         ', binsize)
print('')

SNtot1, SNtot2, SNtot3 = [], [], []

t0 = time.time()
# --------------------------------------------
# masks
# --------------------------------------------
mask_pl = hp.read_map(f'/pscratch/sd/r/rmvd2/CMBxLya/data/COM_Lensing_4096_R3.00/mask.fits.gz')
# mask_pl = hp.read_map(f'/global/homes/s/sferraro/maps/unWISE/MASKS/mask_Planck_full_v10.fits')



comp ='{:.2f}'.format(comp_s)
if data_type == 'data':
    masked_count_dn_N = hp.read_map(f'{PATH_d}/QSO_z{zmin:.2f}_{zmax:.2f}_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_DELTA_MAP.fits') #Directly read delta_map
    bin_mask_N = hp.read_map(f'{PATH_d}/QSO_z{zmin:.2f}_{zmax:.2f}_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_BINARY_MASK.fits') #
    ran_map_N = hp.read_map(f'{PATH_d}/QSO_z{zmin:.2f}_{zmax:.2f}_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_RAN_MAP.fits') # why not exactly same dep completeness?
    ran_mean_N = np.loadtxt(f'{PATH_d}QSO_z{zmin:.2f}_{zmax:.2f}_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_ran_mean.txt') # doesn't dep on completeness
    completeness_N = ran_map_N/ran_mean_N
    masked_count_dn_S = hp.read_map(f'{PATH_d}/QSO_z{zmin:.2f}_{zmax:.2f}_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_DELTA_MAP.fits') #Directly read delta_map
    bin_mask_S = hp.read_map(f'{PATH_d}/QSO_z{zmin:.2f}_{zmax:.2f}_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_BINARY_MASK.fits') #
    ran_map_S = hp.read_map(f'{PATH_d}/QSO_z{zmin:.2f}_{zmax:.2f}_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_RAN_MAP.fits') # why not exactly same dep completeness?
    ran_mean_S = np.loadtxt(f'{PATH_d}QSO_z{zmin:.2f}_{zmax:.2f}_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_ran_mean.txt') # doesn't dep on completeness
    completeness_S = ran_map_S/ran_mean_S
    if sys_wts:
        numcounts_map_N = hp.read_map(f'{PATH_d}/QSO_z{zmin:.2f}_{zmax:.2f}_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_DATA_MAP.fits', field=[0])
        numcounts_map_S = hp.read_map(f'{PATH_d}/QSO_z{zmin:.2f}_{zmax:.2f}_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_DATA_MAP.fits', field=[0])
    elif (sys_wts == False):
        numcounts_map_N = hp.read_map(f'{PATH_d}/QSO_z{zmin:.2f}_{zmax:.2f}_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_DATA_MAP_UNW.fits', field=[0])
        numcounts_map_S = hp.read_map(f'{PATH_d}/QSO_z{zmin:.2f}_{zmax:.2f}_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_DATA_MAP_UNW.fits', field=[0])

    if mask_name == 'N':
        masked_count_dn, bin_mask, ran_map, completeness, numcounts_map = masked_count_dn_N, bin_mask_N, ran_map_N, completeness_N, numcounts_map_N
        keep_bin_mask = (bin_mask>cut_off) 
        bin_mask[keep_bin_mask] = True
    if mask_name == 'S':
        masked_count_dn, bin_mask, ran_map, completeness, numcounts_map = masked_count_dn_S, bin_mask_S, ran_map_S, completeness_S, numcounts_map_S
        keep_bin_mask = (bin_mask>cut_off) 
        bin_mask[keep_bin_mask] = True
    if mask_name == 'A':
        numcounts_map = numcounts_map_N + numcounts_map_S
        overlap = (ran_map_S/ran_mean_S>cut_off) & (ran_map_N/ran_mean_N>cut_off)
        bin_mask = np.full(bin_mask_S.shape, False)
        keep_bin_mask = (bin_mask_N>cut_off) | (bin_mask_S>cut_off)
        bin_mask[keep_bin_mask] = True
        masked_count_dn, ran_map, completeness = 1.*masked_count_dn_N, 1.*ran_map_N, 1.*completeness_N
        masked_count_dn[masked_count_dn_S>-1] = masked_count_dn_S[masked_count_dn_S>-1]
        ran_map[ran_map_S>cut_off] = ran_map_S[ran_map_S>cut_off]
        completeness[ran_map_S>cut_off] = completeness_S[ran_map_S>cut_off]

    # only in mock so far - check RB 29072024
    completeness = completeness*bin_mask.astype(np.float64)
    completeness[completeness<cut_off] = 0

    #use the numbercount maps to compute deltas
    if sepnorm == False:
        data_map = numcounts_map / completeness
        data_map[completeness < cut_off] = 0
        data_map[completeness == 0] = 0
        masked_count = data_map * bin_mask
        mean_count = np.nansum(masked_count)/np.nansum(bin_mask)
        masked_count_dn = data_map / mean_count - 1.
        masked_count_dn[np.isnan(masked_count_dn)] = 0 # same as masked_count_dn from DELTA_MAP for S and N

elif 'abacus' in data_type:
    sys_wts=False #no sys_weights applied to abacus mocks
    bin_mask_N = hp.read_map(f'{PATH_d}/QSO_z0.80_2.10_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_BINARY_MASK.fits') 
    ran_map_N = hp.read_map(f'{PATH_d}/QSO_z0.80_2.10_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_RAN_MAP.fits') 
    ran_mean_N = np.loadtxt(f'{PATH_d}QSO_z0.80_2.10_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_ran_mean.txt') 
    completeness_N = ran_map_N/ran_mean_N
    bin_mask_S = hp.read_map(f'{PATH_d}/QSO_z0.80_2.10_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_BINARY_MASK.fits') #
    ran_map_S = hp.read_map(f'{PATH_d}/QSO_z0.80_2.10_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_RAN_MAP.fits')
    ran_mean_S = np.loadtxt(f'{PATH_d}QSO_z0.80_2.10_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_ran_mean.txt') 
    completeness_S = ran_map_S/ran_mean_S
    
    if mask_name == 'A':
        overlap = (ran_map_S/ran_mean_S>cut_off) & (ran_map_N/ran_mean_N>cut_off)
        bin_mask = np.full(bin_mask_S.shape, False)
        keep_bin_mask = (bin_mask_N>cut_off) | (bin_mask_S>cut_off)
        bin_mask[keep_bin_mask] = True
        ran_map, completeness = 1.*ran_map_N, 1.*completeness_N
        ran_map[ran_map_S>cut_off] = ran_map_S[ran_map_S>cut_off]
        completeness[ran_map_S>cut_off] = completeness_S[ran_map_S>cut_off]
    elif mask_name == 'N' or mask_name == 'S':
        assert False, "abacus only mask=ALL possible"

    # only in mock so far - check RB 29072024
    completeness = completeness*bin_mask.astype(np.float64)
    completeness[completeness<cut_off] = 0

elif data_type == 'mock':
    bin_mask_N = hp.read_map(f'{PATH_d}/QSO_z0.80_2.10_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_BINARY_MASK.fits')
    #bin_mask = hp.pixelfunc.ud_grade(bin_mask,nside_out=2048) # upgrade the mask resolution to nside_out		
    #bin_mask_n1024 = hp.pixelfunc.ud_grade(bin_mask,nside_out=1024) # upgrade the mask resolution to nside_out		
    ran_map_N = hp.read_map(f'{PATH_d}/QSO_z0.80_2.10_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_RAN_MAP.fits') # why not exactly same dep completeness?
    ran_mean_N = np.loadtxt(f'{PATH_d}QSO_z0.80_2.10_N__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_ran_mean.txt') # doesn't dep on completeness
    completeness_N = ran_map_N/ran_mean_N
    bin_mask_S = hp.read_map(f'{PATH_d}/QSO_z0.80_2.10_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_BINARY_MASK.fits')
    ran_map_S = hp.read_map(f'{PATH_d}/QSO_z0.80_2.10_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_RAN_MAP.fits') # why not exactly same dep completeness?
    ran_mean_S = np.loadtxt(f'{PATH_d}QSO_z0.80_2.10_S__HPmapcut_default_addLIN_nside2048_{version}_comp{comp}_galactic_ran_mean.txt') # doesn't dep on completeness
    completeness_S = ran_map_S/ran_mean_S

    if mask_name == 'N':
        bin_mask, ran_map, completeness = bin_mask_N, ran_map_N, completeness_N
        keep_bin_mask = (bin_mask>cut_off) 
        bin_mask[keep_bin_mask] = True
    if mask_name == 'S':
        bin_mask, ran_map, completeness = bin_mask_S, ran_map_S, completeness_S
        keep_bin_mask = (bin_mask>cut_off)
        bin_mask[keep_bin_mask] = True
    if mask_name == 'A':
        overlap = (ran_map_S/ran_mean_S>cut_off) & (ran_map_N/ran_mean_N>cut_off)
        bin_mask = np.full(bin_mask_S.shape, False)
        keep_bin_mask = (bin_mask_N>cut_off) | (bin_mask_S>cut_off)
        bin_mask[keep_bin_mask] = True
        ran_map, completeness = 1.*ran_map_N, 1.*completeness_N
        ran_map[ran_map_S>cut_off] = ran_map_S[ran_map_S>cut_off]
        completeness[ran_map_S>cut_off] = completeness_S[ran_map_S>cut_off]

    completeness = completeness*bin_mask.astype(np.float64)
    completeness[completeness<cut_off] = 0
    # np.savetxt(f'{PATH_oc}/completeness_{mask_name}{opt3}.txt',completeness)
    # np.savetxt(f'{PATH_oc}/binary_mask_{mask_name}{opt3}.txt',bin_mask)
    # assert(False)
    # mock way
    density = 114
    mean_galaxy_counts = density * 41253./(12*nside**2)

    # plt.figure();hp.mollview(bin_mask, title='bin_mask');plt.savefig(f'plots/bin_mask{opt3}.pdf')
    # plt.figure();hp.mollview(completeness, title='comp_mask');plt.savefig(f'plots/comp_mask{opt3}.pdf')
    # plt.figure();plt.hist(completeness[completeness>0], bins=100);plt.savefig(f'plots/comp_hist{opt3}.pdf')
    # assert(False)

if ap_scale > 0.0:
    ap_mask_c2 = nmt.mask_apodization(completeness, ap_scale, apotype="C2")
else:
    ap_mask_c2 = completeness.copy()

if data_type == 'data':
    if lensing_str=='PR3':
        lensing_alm = hp.read_alm(f'{PATH_p}/MV/dat_klm.fits')
    elif lensing_str=='PR4':
        lensing_alm = hp.read_alm('/global/cfs/cdirs/cmb/data/planck2020/PR4_lensing/PR4_klm_dat_p.fits')
        lensing_alm[0]=0+0j
    else:
        assert False, "Choose between lensing map PR3 or PR4!"
    
    lensing_map = hp.alm2map(lensing_alm, nside)        

    if filter_lowell_alm: 
        print('filter low-ell from maps')
        lensing_map = filter_lowell_from_map(lensing_map, nside, lowell_min=compute_lmin)# filter PLANCK map
        masked_count_dn = filter_lowell_from_map(masked_count_dn, nside, lowell_min=compute_lmin) # filter QSO delta map

    if filter_highell_alm: 
        print('filter high-ell from maps')
        lensing_map = filter_highell_from_map(lensing_map, nside, nside)# filter PLANCK map
        masked_count_dn = filter_highell_from_map(masked_count_dn, nside, nside) # filter QSO delta map

    opt = f'{opt_}'
    if not(sys_wts):
        opt = f'{opt}_unw'

    print('read maps and masks',time.time()-t0)

    # First field
    f1_ap, fsky1_ap, SN1_ap = get_field(masked_count_dn, ap_mask_c2, numcounts_map, compute_field=compute_field)             
    # Second field
    f2, fsky2, _ = get_field(lensing_map, mask_pl, compute_field=compute_field)
    print(f'fsky1_ap : {fsky1_ap:.2f}, fsky2 : {fsky2:.2f}')

    print('nmt fields',time.time()-t0)

    # --------------------------------------------
    # correlations
    # --------------------------------------------
    if compute_cl:
        if compute_coupled_cls:
            Cl_cross_ap, Cl_auto_gg_ap, Cl_auto_kk, lbin = get_coupled_cells(f1_ap, f2, nside, nb_ells=3*nside, nb_multipoles_per_bins=binsize, nb_ells_removed=compute_lmin, auto1=True, auto2=True, cross=True)
        else:
            Cl_cross_ap, Cl_auto_gg_ap, Cl_auto_kk, lbin = get_cells(f1_ap, f2, nside, nb_ells=3*nside, nb_multipoles_per_bins=binsize, nb_ells_removed=compute_lmin, auto1=True, auto2=True, cross=True)
        print('Correlations',time.time()-t0)

        bins, lbin, _, _, _, _, _ = get_bins(nside, nb_ells=3*nside, nb_multipoles_per_bins=binsize, nb_ells_removed=compute_lmin)
        sigma_kk = get_gaussian_errors(Cl_auto_kk, Cl_auto_kk, Cl_auto_kk, mask_pl, bins)
        sigma_kg = get_gaussian_errors(Cl_auto_kk, Cl_auto_gg_ap, Cl_cross_ap, mask_pl*ap_mask_c2, bins)
        sigma_gg = get_gaussian_errors(Cl_auto_gg_ap, Cl_auto_gg_ap, Cl_auto_gg_ap, ap_mask_c2, bins)
        print('Gaussian error bars',time.time()-t0)

        np.savetxt(f'{PATH_oc}/{coupled_str}Clkg_{mask_name}{opt3}{opt}.txt', np.array([lbin,np.squeeze(Cl_cross_ap), np.squeeze(sigma_kg)]).T)
        np.savetxt(f'{PATH_oc}/{coupled_str}Clgg_{mask_name}{opt3}{opt}.txt', np.array([lbin,np.squeeze(Cl_auto_gg_ap), np.squeeze(sigma_gg)]).T)
        np.savetxt(f'{PATH_oc}/{coupled_str}Clkk_{mask_name}{opt1}{opt}.txt', np.array([lbin,np.squeeze(Cl_auto_kk), np.squeeze(sigma_kk)]).T)

# --------------------------------------------
# theory
# --------------------------------------------
if do_cov:
    clgg_theory_no_SN = np.loadtxt('/global/cfs/cdirs/desi/users/akrolew/QSO_maps/clgg_desi_quasars_QSO_z0.80_2.10_NGC__HPmapcut_default.txt' )[:,1]
    density = 114
    clgg_SN = 1./(density * (180./np.pi)**2.)
    clgg_noise = np.loadtxt('/global/cfs/cdirs/desi/users/akrolew/QSO_maps/QSO_z0.80_2.10_N__HPmapcut_default_addLIN_nside2048_galactic_DELTA_MAP_BINARY_MASK_nlqq.txt')
    clgg_tot = clgg_theory_no_SN + clgg_SN + clgg_noise

    # clgg_tot = np.loadtxt(f'/global/cfs/cdirs/desi/users/akrolew/desi_qso_clkg/clgg_desi_quasars_QSO_z0.80_2.10_NGC__HPmapcut_default.txt')[:,1]
    clkg = np.loadtxt(f'/global/cfs/cdirs/desi/users/akrolew/QSO_maps/clkg_desi_quasars_QSO_z0.80_2.10_NGC__HPmapcut_default.txt')[:,1]
    ells = np.linspace(0,len(clgg_tot)-1, len(clgg_tot))+0.5

    nlkk = np.loadtxt('/global/cfs/cdirs/desi/users/akrolew/unWISE/PLANCK_LENSING/COM_Lensing_4096_R3.00/MV/nlkk.dat')[:,1]
    ells_clkk = np.loadtxt('/global/cfs/cdirs/desi/users/akrolew/unWISE/PLANCK_LENSING/COM_Lensing_4096_R3.00/MV/nlkk.dat')[:,0]
    clkk = np.loadtxt('/global/cfs/cdirs/desi/users/akrolew/unWISE/PLANCK_LENSING/COM_Lensing_4096_R3.00/MV/nlkk.dat')[:,2] - nlkk
    # clkk = np.loadtxt('/global/homes/s/sferraro/lensing_pipeline/transfer/PS_for_transfer/clkk_synthetic.txt')
    clkk_s = np.loadtxt('/global/cfs/cdirs/desi/users/akrolew/lensing_pipeline/transfer/PS_for_transfer/clkk_synthetic.txt')

    clgg_tot[0] = 0

    noise = np.zeros(len(clgg_tot))
    noise[:4097] = nlkk
    noise[4097:] = nlkk[-1]
    clkktot = np.zeros_like(noise)
    clkktot[:4097]= np.loadtxt('/global/cfs/cdirs/desi/users/akrolew/unWISE/PLANCK_LENSING/COM_Lensing_4096_R3.00/MV/nlkk.dat')[:4097,2]
    clkktot[4097:] = np.loadtxt('/global/cfs/cdirs/desi/users/akrolew/unWISE/PLANCK_LENSING/COM_Lensing_4096_R3.00/MV/nlkk.dat')[:,2][-1]

    # bin theory 
    bins, lbin, _, _, _, _, _ = get_bins(nside, nb_ells=3*nside, nb_multipoles_per_bins=binsize, nb_ells_removed=compute_lmin)
    ell_eff = bins.get_effective_ells()
    cl_kk_binned = bins.bin_cell(clkktot)
    cl_kg_binned = bins.bin_cell(clkg)
    cl_gg_binned = bins.bin_cell(clgg_tot)
    np.savetxt(f'{PATH_oc}/binned_theory_Cl_kk_kg_gg_lmin{compute_lmin:.0f}.txt',np.array([ell_eff,cl_kk_binned,cl_kg_binned,cl_gg_binned]).T)
    print('save theory Cl')

if data_type == 'mock' or 'abacus' in data_type:
    i_m = i_min
    while i_m <= i_max:
        # --------------------------------------------
        # data maps and fields
        # --------------------------------------------
        print(i_m)
        if 'abacus' in data_type:
            if mask_name == 'A':
                if data_type == 'abacus_nodownsample_fullsky':
                    # full-sky, no downsample. very high density, use the Abacus dndz rather than the data dndz: /global/cfs/cdirs/desi/users/akrolew/AbacusSummit_huge_c000_ph201_QSO_no_downsample/cat.fits
                    numcounts_map = hp.read_map(f'/global/cfs/cdirs/desi/users/akrolew/AbacusSummit_huge_c000_ph201_QSO/data/numcounts_QSO_nodownsample_galactic_coord.fits')
                    ap_mask_c2=np.ones_like(ap_mask_c2)
                    mask_pl=np.ones_like(mask_pl)
                    completeness = np.ones_like(completeness)
                    bin_mask = np.ones_like(bin_mask)
                elif data_type == 'abacus_nodownsample_fullsky_lowz':
                    # full-sky, no downsample. very high density, use the Abacus dndz rather than the data dndz: /global/cfs/cdirs/desi/users/akrolew/AbacusSummit_huge_c000_ph201_QSO_no_downsample/cat.fits
                    numcounts_map = hp.read_map(f'/global/cfs/cdirs/desi/users/akrolew/AbacusSummit_huge_c000_ph201_QSO/data/numcounts_QSO_nodownsample_galactic_coord_0.80_z_1.30.fits')
                    ap_mask_c2=np.ones_like(ap_mask_c2)
                    mask_pl=np.ones_like(mask_pl)
                    completeness = np.ones_like(completeness)
                    bin_mask = np.ones_like(bin_mask)
                elif data_type == 'abacus_downsample_fullsky':
                    # full-sky, downsample. density matches data, use the data dndz: /global/cfs/cdirs/desi/users/akrolew/AbacusSummit_huge_c000_ph201_QSO/cat_downsample_%i.fits
                    # numcounts_map = hp.read_map(f'/global/cfs/cdirs/desi/users/akrolew/AbacusSummit_huge_c000_ph201_QSO/numcounts_QSO_fullsky_downsample_galactic_coord_{i_m}.fits')
                    numcounts_map = hp.read_map(f'/global/cfs/cdirs/desi/users/akrolew/AbacusSummit_huge_c000_ph201_QSO/numcounts_QSO_fullsky_downsample_galactic_coord_{i_m}_new.fits')
                    ap_mask_c2=np.ones_like(ap_mask_c2)
                    mask_pl=np.ones_like(mask_pl)
                    completeness = np.ones_like(completeness)
                    bin_mask = np.ones_like(bin_mask)

                elif data_type == 'abacus_downsample_cutsky':
                    # downsample & cut to DESI geometry, with completeness applied. use the data dndz. should be analyzed in the same way as the data: /global/cfs/cdirs/desi/users/akrolew/AbacusSummit_huge_c000_ph201_QSO/data/QSO_Y1_%i.fits
                    numcounts_map = hp.read_map(f'/global/cfs/cdirs/desi/users/akrolew/AbacusSummit_huge_c000_ph201_QSO/numcounts_QSO_cutsky_downsample_galactic_coord_{i_m}.fits')

            else:
                assert False, "abacus only NGC+SGC possible"

            #use the numbercount maps to compute deltas -- like DESI DR1 data 
            data_map = numcounts_map / completeness
            data_map[completeness < cut_off] = 0
            masked_count = data_map * bin_mask
            mean_count = np.nansum(masked_count)/np.nansum(bin_mask)
            masked_count_dn = data_map / mean_count - 1.
            masked_count_dn[np.isnan(masked_count_dn)] = 0 

            # load lensing maps corresponding to abacus mocks
            # lensing_map = hp.read_map(f'{abacus_dir}/AbacusSummit_huge_c000_ph201_CMBkappa_map_nside2048.fits')
            lensing_map = hp.read_map(f'{abacus_dir}/AbacusSummit_huge_c000_ph201_CMBkappa_map_nside2048_galactic_coord.fits')

            # if True:
            #     # ROTATING TO GALACTIC COORDINATES
            #     rot = hp.Rotator(coord=['C', 'G'])
            #     lensing_map = rot.rotate_map_alms(lensing_map)

        elif data_type == 'mock':
            if add_nlqq:
                sim_name = 'galaxy+nlqq2048_poisson_'
            else:
                sim_name = 'galaxy2048_poisson_'
            if mask_name in {'N', 'S'}: 
                numcounts_map = hp.read_map(f'{PATH_m}/{sim_name}{mask_name}{mock_version}/{i_m}.fits')
            if mask_name == 'A':
                numcounts_map_N = hp.read_map(f'{PATH_m}/{sim_name}N{mock_version}/{i_m}.fits')
                numcounts_map_S = hp.read_map(f'{PATH_m}/{sim_name}S{mock_version}/{i_m}.fits')
                numcounts_map = 1.*numcounts_map_N
                numcounts_map[numcounts_map_S>0.] = numcounts_map_S[numcounts_map_S>0.]
            # mock way
            masked_count_dn = (numcounts_map / completeness) / mean_galaxy_counts - 1.
            # RB 23072024 - update with AK
            # masked_count_dn[ran_map == 0] = -1
            #  this fixes the NaN problem!
            masked_count_dn[completeness == 0] = 0.

            lensing_map = hp.read_map(f'{PATH_m}kappa2048/{i_m}.fits', field=[0])

        if filter_lowell_alm: 
            print('filter low-ell from maps')
            lensing_map = filter_lowell_from_map(lensing_map, nside)# filter PLANCK map
            masked_count_dn = filter_lowell_from_map(masked_count_dn, nside) # filter QSO delta map

        if filter_highell_alm: 
            print('filter high-ell from maps')
            lensing_map = filter_highell_from_map(lensing_map, nside, nside)# filter PLANCK map
            masked_count_dn = filter_highell_from_map(masked_count_dn, nside, nside) # filter QSO delta map

        #clgg_anafast = hp.anafast(bin_mask * masked_count_dn)/fsky
        #clgg_anafast_l1500_1 = (1/fsky)*hp.anafast(bin_mask * masked_count_dn, nspec=1500)
        opt = f'{opt_}'
        if data_type == 'mock' or 'abacus' in data_type:
            opt = f'{opt}_m{i_m}'

        print('read maps and masks',time.time()-t0)

        # First field
        # f1, fsky1, SN1 = get_field(masked_count_dn, bin_mask, numcounts_map, compute_field=compute_field)   
        #f1_c5, fsky1_c5, SN1_c5 = get_field(masked_count_dn, completeness*(completeness>=0.5), numcounts_map)   
        #f1_5, fsky1_5, SN1_5 = get_field(masked_count_dn, np.ones(np.shape(completeness))*(completeness>=0.5), numcounts_map)   
        # f1_c, fsky1_c, SN1_c = get_field(masked_count_dn, completeness, numcounts_map, compute_field=compute_field)   

        f1_ap, fsky1_ap, SN1_ap = get_field(masked_count_dn, ap_mask_c2, numcounts_map, compute_field=compute_field)             
        # Second field
        f2, fsky2, _ = get_field(lensing_map, mask_pl, compute_field=compute_field)
        print(f'fsky1_ap : {fsky1_ap:.2f}, fsky2 : {fsky2:.2f}')

        # SNtot1.append(SN1)
        # SNtot2.append(SN1_c)
        SNtot3.append(SN1_ap)

        print('nmt fields',time.time()-t0)


        # --------------------------------------------
        # correlations
        # --------------------------------------------
        if compute_cl:
            if compute_coupled_cls:
                Cl_cross_ap, Cl_auto_gg_ap, Cl_auto_kk, lbin = get_coupled_cells(f1_ap, f2, nside, nb_ells=3*nside, nb_multipoles_per_bins=binsize, nb_ells_removed=compute_lmin, auto1=True, auto2=True, cross=True)
            else:
                Cl_cross_ap, Cl_auto_gg_ap, Cl_auto_kk, lbin = get_cells(f1_ap, f2, nside, nb_ells=3*nside, nb_multipoles_per_bins=binsize, nb_ells_removed=compute_lmin, auto1=True, auto2=True, cross=True)

            print('Correlations',time.time()-t0)
            bins, lbin, _, _, _, _, _ = get_bins(nside, nb_ells=3*nside, nb_multipoles_per_bins=binsize, nb_ells_removed=compute_lmin)
            sigma_kk = get_gaussian_errors(Cl_auto_kk, Cl_auto_kk, Cl_auto_kk, mask_pl, bins)
            sigma_kg = get_gaussian_errors(Cl_auto_kk, Cl_auto_gg_ap, Cl_cross_ap, mask_pl*ap_mask_c2, bins)
            sigma_gg = get_gaussian_errors(Cl_auto_gg_ap, Cl_auto_gg_ap, Cl_auto_gg_ap, ap_mask_c2, bins)

            print('Gaussian error bars',time.time()-t0)

            np.savetxt(f'{PATH_oc}/{coupled_str}Clkg_{mask_name}{opt3}{opt}.txt', np.array([lbin,np.squeeze(Cl_cross_ap), np.squeeze(sigma_kg)]).T)
            np.savetxt(f'{PATH_oc}/{coupled_str}Clgg_{mask_name}{opt3}{opt}.txt', np.array([lbin,np.squeeze(Cl_auto_gg_ap), np.squeeze(sigma_gg)]).T)
            np.savetxt(f'{PATH_oc}/{coupled_str}Clkk_{mask_name}{opt1}{opt}.txt', np.array([lbin,np.squeeze(Cl_auto_kk), np.squeeze(sigma_kk)]).T)


        # if i_m==i_min:
        #     #ell_eff,cl_kk_binned,cl_kg_binned,cl_gg_binned
        #     # Plotting with subplots
        #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16), sharex=True, gridspec_kw={'hspace': 0., 'wspace': 0})

        #     # Plot theory curves initially
        #     ax1.plot(ell_eff, cl_kg_binned, 'k-',lw=2,label='theory')
        #     ax1.set_ylabel(r'$C_{\ell}^{kg}$')
        #     ax2.plot(ell_eff, cl_gg_binned, 'k-',lw=2, label='theory')
        #     ax2.set_xlabel(r'multipole $\ell$')
        #     ax2.set_ylabel(r'$C_{\ell}^{gg}$')

        #     SNRkg = SNR_computation(ell_eff, cl_kg_binned, lbin, np.squeeze(Cl_cross_ap), np.squeeze(sigma_kg))
        #     SNRgg = SNR_computation(ell_eff, cl_gg_binned, lbin, np.squeeze(Cl_auto_gg_ap), np.squeeze(sigma_gg))
        #     # Plot mean values with error bars
        #     ax1.errorbar(lbin, np.squeeze(Cl_cross_ap), yerr=np.squeeze(sigma_kg), label=f'apo {ap_scale} comp {comp_s} SNR={SNRkg:.1f}')
        #     ax2.errorbar(lbin, np.squeeze(Cl_auto_gg_ap), yerr=np.squeeze(sigma_gg), label=f'apo {ap_scale} comp {comp_s} SNR={SNRgg:.1f}')
        #     # Final adjustments to the plots
        #     ax1.set_xlim(0, 1000)
        #     ax1.set_ylim(-1e-7, 7e-7)
        #     ax1.legend()
        #     ax1.axvline(20, color='k', linestyle='--')

        #     ax2.set_xlim(0, 1000)
        #     ax2.set_ylim(1e-6, 7e-6)
        #     ax2.legend()
        #     ax2.axvline(20, color='k', linestyle='--')
        #     lmin, lmax = 20, 1000
        #     mask = (ell_eff > lmin) & (ell_eff < lmax)

        #     ax2.set_xlabel(r'multipole $\ell$')

        #     plt.savefig(f'{PATH_oc}/plots/Clkg_Clgg_{mask_name}{opt3}{opt}_SNR_isim{i_m}.pdf', bbox_inches='tight')
        #     plt.show()


        # --------------------------------------------
        # covariances
        # --------------------------------------------



        if data_type == 'mock' or 'abacus' in data_type:
            i_m += 1
        else: i_m = i_max + 1

if True:
    if do_cov:
        print('compute Gaussian cov')
        bins, lbin, _, _, _, _, _ = get_bins(nside, nb_ells=3*nside, nb_multipoles_per_bins=binsize, nb_ells_removed=2)
        # k x g
        w12_ap = nmt.NmtWorkspace()
        w12_ap.compute_coupling_matrix(f1_ap, f2, bins)

        # g x g
        w11_ap = nmt.NmtWorkspace()
        w11_ap.compute_coupling_matrix(f1_ap, f1_ap, bins)

        # kg x kg
        cw1212_ap = nmt.NmtCovarianceWorkspace()
        cw1212_ap.compute_coupling_coefficients(f1_ap, f2, f1_ap, f2) 

        # gg x gg
        cw1111_ap = nmt.NmtCovarianceWorkspace()
        cw1111_ap.compute_coupling_coefficients(f1_ap, f1_ap, f1_ap, f1_ap)       

        # kg x gg
        cw1211_ap = nmt.NmtCovarianceWorkspace()
        cw1211_ap.compute_coupling_coefficients(f1_ap, f2, f1_ap, f1_ap)

        # kg kg 
        covar_tot1212_ap = nmt.gaussian_covariance(cw1212_ap, 0, 0, 0, 0, [clgg_tot], [clkg], [clkg],  [clkktot],  wa=w12_ap, wb=w12_ap) 
        # gg gg 
        covar_tot1111_ap = nmt.gaussian_covariance(cw1111_ap, 0, 0, 0, 0, [clgg_tot], [clgg_tot], [clgg_tot],  [clgg_tot],  wa=w11_ap, wb=w11_ap)
        # kg gg
        covar_tot1211_ap = nmt.gaussian_covariance(cw1211_ap, 0, 0, 0, 0, [clgg_tot], [clgg_tot], [clkg],  [clkg],  wa=w12_ap, wb=w11_ap) 

        np.savetxt(f'{PATH_oc}/Cov_kgkg_{mask_name}{opt3}{opt}.txt', covar_tot1212_ap)
        np.savetxt(f'{PATH_oc}/Cov_gggg_{mask_name}{opt3}{opt}.txt', covar_tot1111_ap)
        np.savetxt(f'{PATH_oc}/Cov_kggg_{mask_name}{opt3}{opt}.txt', covar_tot1211_ap)
        # np.savetxt(f'{PATH_oc}/Cov_kgkk_{mask_name}{opt3}{opt}.txt', covar_tot1222_ap)



"""
nside_new = 1024
bin_mask_n1024 = hp.pixelfunc.ud_grade(bin_mask,nside_out=nside_new)
ratio = len(bin_mask)/len(bin_mask_n1024)
numcounts_map_n1024 = hp.pixelfunc.ud_grade(numcounts_map*1.0,nside_out=nside_new)*ratio
ran_map_n1024 = hp.pixelfunc.ud_grade(ran_map*1.0,nside_out=nside_new)*ratio
ran_mean_n1024 = ran_mean*ratio
completeness_n1024 = ran_map_n1024/ran_mean_n1024
# mock way
density = 114
mean_galaxy_counts_n1024 = density * 41253./(12*nside_new**2)
masked_count_dn_n1024 = (numcounts_map_n1024/ (ran_map_n1024/ran_mean_n1024)) / mean_galaxy_counts_n1024 - 1.
masked_count_dn_n1024[ran_map_n1024 == 0] = -1
bin_mask_n1024_2 = 1*bin_mask_n1024
bin_mask_n1024_2[completeness_n1024<=0.8] = 0
bin_mask_n1024_2[completeness_n1024>0.8] = 1
ap_mask_c2_n1024 = nmt.mask_apodization(completeness_n1024, ap_scale, apotype="C2")

f1_n1024, fsky1_n1024, SN1_n1024 = get_field(masked_count_dn_n1024, bin_mask_n1024_2, numcounts_map_n1024)   
f1_c_n1024, fsky1_c_n1024, SN1_c_n1024 = get_field(masked_count_dn_n1024, completeness_n1024, numcounts_map_n1024)   
f1_ap_n1024, fsky1_ap_n1024, SN1_ap_n1024 = get_field(masked_count_dn_n1024, ap_mask_c2_n1024, numcounts_map_n1024)   
"""

"""
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)
redshift = 2.1
d_A = cosmo.angular_diameter_distance(z=redshift) # angular diameter distance in Mpc
theta = 1.0 # arcdeg
theta_radian = theta * np.pi / 180
distance_Mpc = d_A * theta_radian
print(distance_Mpc)
rs_drag = 150
rs_drag_at_z = rs_drag/(1+redshift) # proper distance
theta_rs = rs_drag_at_z/d_A
print('in rad:', theta_rs, 'in deg:', theta_rs*180/np.pi)
"""


