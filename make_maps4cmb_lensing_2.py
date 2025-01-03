import healpy as hp
from astropy.io import fits
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
import time
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()

def select_regressis_DES(ra, dec):    
    '''
    input_array with RA, DEC given by ra_col,dec_col
    return selection for DES as defined by regressis
    '''

    from regressis import footprint
    import healpy as hp
    foot = footprint.DR9Footprint(256, mask_lmc=False, clear_south=True, mask_around_des=False, cut_desi=False)
    north, south, des = foot.get_imaging_surveys()
    th,phi = (-dec+90.)*np.pi/180.,ra*np.pi/180.
    pix = hp.ang2pix(256,th,phi,nest=True)
    sel_des = des[pix]
    return sel_des



t0 = time.time()

outdir = '/global/homes/s/schiaren/CMBxQSO/QSO_maps/'


def write_data_and_ran_map(version = 'v1.5', zmin = 0.80, zmax = 2.10, region = 'N', 
						   weight_opt = 'default_addLIN', nside = 2048,
						   comp = 0.00, coord = 'galactic'):
	def cut_to_region_and_zrange(fname_NGC, fname_SGC):
		data_file1 = fits.open(fname_NGC)[1].data
		data_file2 = fits.open(fname_SGC)[1].data

		if (region == 'NGC') or (region == 'S_NGC'):
			ra = data_file1['RA']
			dec = data_file1['DEC']
			z = data_file1['Z']
			weight = data_file1['WEIGHT']
			weight_sys = data_file1['WEIGHT_SYS']
			photsys = data_file1['PHOTSYS']
			targetid = data_file1['TARGETID']
		elif (region == 'SGC') or (region == 'S_SGC-noDES') or (region == 'DES'):
			ra = data_file2['RA']
			dec = data_file2['DEC']
			z = data_file2['Z']
			weight = data_file2['WEIGHT']
			weight_sys = data_file2['WEIGHT_SYS']
			photsys = data_file2['PHOTSYS']
			targetid = data_file2['TARGETID']
		else:
			ra = np.concatenate((data_file1['RA'],data_file2['RA']))
			dec = np.concatenate((data_file1['DEC'],data_file2['DEC']))
			z = np.concatenate((data_file1['Z'],data_file2['Z']))
			weight = np.concatenate((data_file1['WEIGHT'],data_file2['WEIGHT']))
			weight_sys = np.concatenate((data_file1['WEIGHT_SYS'],data_file2['WEIGHT_SYS']))
			photsys = np.concatenate((data_file1['PHOTSYS'],data_file2['PHOTSYS']))
			targetid = np.concatenate((data_file1['TARGETID'],data_file2['TARGETID']))

		# Redshift cut
		print('length before zcut',np.shape(ra)[-1])
		len_before_cut = np.shape(ra)[-1]
		ra = ra[(z >= zmin) & (z <= zmax)]
		dec = dec[(z >= zmin) & (z <= zmax)]
		weight = weight[(z >= zmin) & (z <= zmax)]
		weight_sys = weight_sys[(z >= zmin) & (z <= zmax)]
		photsys = photsys[(z >= zmin) & (z <= zmax)]
		targetid = targetid[(z >= zmin) & (z <= zmax)]
		z = z[(z >= zmin) & (z <= zmax)]
		print('length after zcut',np.shape(ra)[-1])
		len_after_cut = np.shape(ra)[-1]
		norm_fac = len_after_cut / len_before_cut

		# Hemisphere cut
		if (region == 'N') or (region == 'S') or (region == 'S_NGC'):
			# Handle Northern and Southern hemisphere cuts
			if (region == 'N') or (region == 'S'):
				regsel = region
			elif region == 'S_NGC':
				regsel = 'S'

			# Filter data based on the photometric system
			ra = ra[photsys == regsel]
			dec = dec[photsys == regsel]
			weight = weight[photsys == regsel]
			weight_sys = weight_sys[photsys == regsel]
			z = z[photsys == regsel]
			targetid = targetid[photsys == regsel]
			photsys = photsys[photsys == regsel]

		elif region == 'DES':
			# Select data within the DES region
			is_DES_dat = select_regressis_DES(ra, dec)

			ra = ra[is_DES_dat]
			dec = dec[is_DES_dat]
			weight = weight[is_DES_dat]
			weight_sys = weight_sys[is_DES_dat]
			z = z[is_DES_dat]
			photsys = photsys[is_DES_dat]
			targetid = targetid[is_DES_dat]

		elif region == 'S_SGC-noDES':
			# Filter out DES data specifically from the Southern region
			is_DES_dat = select_regressis_DES(ra, dec)

			# Apply both 'S' (Southern) and not in DES filters
			mask = (photsys == 'S') & (~is_DES_dat)
			ra = ra[mask]
			dec = dec[mask]
			weight = weight[mask]
			weight_sys = weight_sys[mask]
			z = z[mask]
			photsys = photsys[mask]
			targetid = targetid[mask]

		return ra, dec, weight, weight_sys, z, photsys, targetid, norm_fac


	ra, dec, weight, weight_sys, z, photsys, targetid, norm_fac = cut_to_region_and_zrange('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/%s/QSO_NGC_clustering.dat.fits' % version,
		'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/%s/QSO_SGC_clustering.dat.fits' % version)

	full = fits.open('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/%s/QSO_full_HPmapcut.dat.fits' % version)[1].data


	if weight_opt == 'default':
		weight = weight / weight_sys
	
	elif weight_opt == 'default_addLIN':
		targetid_as = np.argsort(full['TARGETID'])
		full_inds = np.searchsorted(full['TARGETID'][targetid_as],targetid)
		weight_lin = full[targetid_as][full_inds]['WEIGHT_IMLIN']
		weight = weight * weight_lin / weight_sys
	elif weight_opt == 'default_addRF':
		pass

	coords = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
	pix = hp.ang2pix(nside, coords.galactic.l.value, coords.galactic.b.value, lonlat=True)
	data_map = np.bincount(pix,minlength=12*nside**2,weights=weight)
	hp.write_map(outdir + 'QSO_z%.2f_%.2f_%s__HPmapcut_%s_nside%i_%s_comp0.00_galactic_DATA_MAP.fits' % (zmin, zmax, region, weight_opt, nside, version),data_map,overwrite=True)
	map_unw = np.bincount(pix,minlength=12*nside**2)
	hp.write_map(outdir + 'QSO_z%.2f_%.2f_%s__HPmapcut_%s_nside%i_%s_comp0.00_galactic_DATA_MAP_UNW.fits' % (zmin, zmax, region, weight_opt, nside, version),map_unw,overwrite=True)

	counts, zbin = np.histogram(z, range=(0,5), bins=500, weights=weight)
	counts_unw, _ = np.histogram(z, range=(0,5), bins=500)

	np.savetxt(outdir + 'QSO_z%.2f_%.2f_%s__HPmapcut_%s_dndz.txt' % (zmin, zmax, region, weight_opt), np.array([ 0.5 * (zbin[1:] + zbin[:-1]),
		counts, counts_unw]).T, header = '# zcenter weighted_counts unweighted_counts')

	ran_map = np.zeros(12*2048**2)

	for i in range(18):
		ra, dec, weight, weight_sys, z, photsys, targetid, _ = cut_to_region_and_zrange('/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/%s/QSO_NGC_%i_clustering.ran.fits' % (version, i),
			'/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/%s/QSO_SGC_%i_clustering.ran.fits' % (version,i))

		coords = SkyCoord(ra=ra*u.deg,dec=dec*u.deg)
		pix = hp.ang2pix(nside, coords.galactic.l.value, coords.galactic.b.value, lonlat=True)
		ran_map += np.bincount(pix,minlength=12*nside**2,weights=weight)

	hp.write_map(outdir + 'QSO_z%.2f_%.2f_%s__HPmapcut_%s_nside%i_%s_comp0.00_galactic_RAN_MAP.fits' % (zmin, zmax, region, weight_opt, nside, version),ran_map,overwrite=True)


	random_density = 2500 * 18 * norm_fac
	ran_mean = random_density * 41253. / (12 * nside**2.)
	f = open(outdir + 'QSO_z%.2f_%.2f_%s__HPmapcut_%s_nside%i_%s_comp0.00_galactic_ran_mean.txt' % (zmin, zmax, region, weight_opt, nside, version), 'w')
	f.write('%.5f' % ran_mean)
	f.close()

	mask = np.ones_like(ran_map)
	mask[ran_map <= ran_mean * 0.00] = 0
	hp.write_map(outdir + 'QSO_z%.2f_%.2f_%s__HPmapcut_%s_nside%i_%s_comp0.00_galactic_BINARY_MASK.fits' % (zmin, zmax, region, weight_opt, nside, version),mask,overwrite=True)

	mask_lost = ran_map / ran_mean
	data_map = data_map / mask_lost
	data_map[mask_lost == 0] = 0
	masked_count = data_map * mask
	mean_count = np.nansum(masked_count)/np.nansum(mask)

	delta_map = data_map / mean_count - 1.
	delta_map[np.isnan(delta_map)] = 0

	hp.write_map(outdir + 'QSO_z%.2f_%.2f_%s__HPmapcut_%s_nside%i_%s_comp0.00_galactic_DELTA_MAP.fits' % (zmin, zmax, region, weight_opt, nside, version),delta_map,overwrite=True)


	print(time.time()-t0)
	
	
regions = ['N','S','DES','S_NGC','S_SGC-noDES']
weight_opts = ['default','default_addLIN','default_addRF']
zmins = [0.8, 0.8, 1.5, 2.1, 2.1, 2.5]
zmaxs = [2.1, 1.5, 2.1, 3.5, 2.5, 3.5]

cnt = 0
for reg in regions:
	for weight_opt in weight_opts:
		for kk in range(len(zmins)):
			if rank == cnt:
				write_data_and_ran_map(zmin=zmins[kk],zmax=zmaxs[kk],weight_opt=weight_opt,region=reg)
			cnt += 1