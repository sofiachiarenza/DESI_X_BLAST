import numpy    as np
import pymaster as nmt

readme = " "

def get_bins(ledges,nside):
    """
    Takes an input set of ledges and sets up a NaMaster bin object.
    
    ledges : list of ell-bin edges
    nside  : healpix nside
    """
    # set up ell-bins
    Nbin = len(ledges)-1
    ells = np.arange(ledges[-1],dtype='int32')
    bpws = np.zeros_like(ells) - 1
    for i in range(Nbin): bpws[ledges[i]:ledges[i+1]] = i
    bins = nmt.NmtBin(nside,bpws=bpws,ells=ells,weights=np.ones_like(ells))
    return bins

def master_cl(ledges,map1,msk1,map2,msk2):
    """
    A bare-bones pseudo-cell calculator. ledges (list)
    defines the edges of the ell-bins. The maps and masks
    are assumed to be in healpix format with the same nside.
    
    Returns the effective-ells, the window function, and the
    pseudo-cells.
    """
    nside = int((len(map1)/12)**0.5)
    bins  = get_bins(ledges,nside)
    field1 = nmt.NmtField(msk1,[map1])
    field2 = nmt.NmtField(msk2,[map2])
    wsp = nmt.NmtWorkspace()
    wsp.compute_coupling_matrix(field1,field2,bins)
    ell = bins.get_effective_ells()
    w12 = wsp.get_bandpower_windows()[0,:,0,:]
    c12 = nmt.compute_full_master(field1,field2,bins,workspace=wsp)[0,:]
    return ell,w12,c12


def master_cov(ledges,map1,msk1,map2,msk2,map3,msk3,map4,msk4,c13,c14,c23,c24):
    """
    A bare-bones covariance calculator. ledges (list)
    defines the edges of the ell-bins. The maps and masks
    are assumed to be in healpix format with the same nside.
    """
    nside = int((len(map1)/12)**0.5)
    bins  = get_bins(ledges,nside)
    field1 = nmt.NmtField(msk1,[map1])
    field2 = nmt.NmtField(msk2,[map2])
    field3 = nmt.NmtField(msk3,[map3])
    field4 = nmt.NmtField(msk4,[map4])
    wsp12 = nmt.NmtWorkspace()
    wsp12.compute_coupling_matrix(field1,field2,bins)
    wsp34 = nmt.NmtWorkspace()
    wsp34.compute_coupling_matrix(field3,field4,bins)
    cw = nmt.NmtCovarianceWorkspace()
    cw.compute_coupling_coefficients(field1,field2,field3,field4)
    cov = nmt.gaussian_covariance(cw,0,0,0,0,[c13],[c14],[c23],[c24],wa=wsp12,wb=wsp34)  
    return cov

