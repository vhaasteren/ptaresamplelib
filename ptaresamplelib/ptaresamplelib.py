from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl
import scipy.stats as stats
import scipy.special as ss
import scipy.ndimage.filters as filters
import scipy.interpolate as interp
import bounded_kde
import piccard as pic
import glob, os, sys

pic_spd = 86400.0       # Seconds per day
pic_spy =  31557600.0   # Seconds per year (yr = 365.25 days, so Julian years)
pic_T0 = 53000.0        # MJD to which all HDF5 toas are referenced


def getPlSpectrum(Apl, gpl, Tmax, ufreqs):
    """
    Convert power-law parameters to residual-power
    
    @param Apl:     Power-law amplitude
    @param gpl:     Power-law spectral index
    @param Tmax:    Duration experiment
    @param ufreqs:  Log10 of frequencies
    """
    
    pfreqs = 10**ufreqs
    Apl = 10**Apl
    ypl = (Apl**2 * pic_spy**3 / (12*np.pi*np.pi * (Tmax))) * ((pfreqs * pic_spy) ** (-gpl))
    
    return np.log10(ypl)

def getWnSpectrum(Wn, ufreqs):
    """
    Get the PSD for white-noise
    
    @param Wn:      RMS white-noise leven in sec
    @param ufreqs:  Log10 of frequencies
    """
    psd = Wn**2 * np.ones(len(ufreqs))
    return np.log10(psd)

def spectrumToHc(ufreqs, rho, Tmax):
    """
    Convert residual power per frequency to h_c
    
    @param ufreqs:  Log10 of frequencies
    @param rho:     Residual power
    @param Tmax:    Duration of experiment
    """
    
    pfreqs = 10**ufreqs
    norm = 12*Tmax*np.pi**2 / (pic_spy**3)
    return np.log10(np.sqrt(norm * (10**rho) * (pfreqs * pic_spy)**3))


def gw_turnover_spectrum(freqs, Tmax, pars):
    # get Amplitude and spectral index
    Amp = 10 ** pars[0]
    gamma = pars[1]
    f0 = 10 ** pars[2]
    kappa = pars[3]

    freqpy = freqs
    f1yr = 1.0 / pic_spy
    hcf = Amp * (freqpy / f1yr) ** ((3 - gamma) / 2) / \
        (1 + (f0 / freqpy) ** kappa) ** (1 / 2)
    return hcf ** 2 / 12 / np.pi ** 2 / freqpy ** 3 / Tmax


def gw_pl_spectrum(freqs, lh_c=0.0, si=4.33, Tmax=None):
    """
    Given an array of frequencies, return the power-law spectrum for h_c and si
    """
    spy = 31557600.0
    fr = freqs * spy
    if Tmax is None:
        Tmax = 1.0 / freqs[0]
    
    amp = 10**lh_c
    
    return ((amp**2) * spy**3 / (12*np.pi*np.pi * Tmax)) * fr ** (-si)

def get_kde(samples):
    """
    Given 1D samples 'samples', return a function
    that returns a kde estimate of the distribution
    
    hmm, seems too trivial actually :)
    """
    return bounded_kde(samples)

def transform_xy(x, y):
    """
    Make the transformation from (x,y) to z
    """
    return 0.5 * np.log10(10**(2*x) + 10**(2*y))

def smooth_hist(hist, xedges, sigma=0.75, kind='cubic'):
    """
    Create a smoothed version of the histogram
    """
    ghist = filters.gaussian_filter(hist, sigma=sigma)
    return interp.interp1d(xedges, ghist, kind=kind)

def kde_gwprior(low=-18.0, high=-10.0, bins=250, kind='cubic'):
    """
    The GW parameter is log(amp). We need a prior flat in amp.
    Return the KDE of that prior
    """
    gwamp = np.linspace(low, high, bins)
    prior = 10**gwamp
    return interp.interp1d(gwamp, prior, kind=kind)

def kde_multiply(kde_list, low=-18.0, high=-10.0, bins=250, kind='cubic'):
    """
    Multiply a bunch of KDEs, and return the resulting KDE
    """
    pdf = np.ones(bins)
    x = np.linspace(low, high, bins)
    for kde in kde_list:
        pdf *= kde(x)
    
    pdf *= bins / (np.sum(pdf) * (high-low))
    return smooth_hist(pdf, x, kind=kind)

def kde_logsum(kde_list, low=-18.0, high=-10.0, bins=250, kind='cubic'):
    """
    Multiply a bunch of KDEs, and return the resulting KDE
    """
    lpdf = np.zeros(bins)
    x = np.linspace(low, high, bins)
    for kde in kde_list:
        pdf = kde(x)
        if np.any(pdf < 0.0):
            #print("kde(x) = ", kde(x))
            raise ValueError("Probabilities cannot get negative! kde(x) = {0}".format(kde(x)))
            #pdf[pdf < 0.0] = 1.0e-99
        lpdf += np.log(pdf)
    
    #pdf *= bins / (np.sum(pdf) * (high-low))
    return smooth_hist(lpdf, x, kind=kind)

def make_ul_kde(kde, minz, maxz, bins=250, kind='cubic'):
    """
    Given a kernel density estimator on the logarithmic interval [minz, maxz],
    return the kde of the upper-limit estimate, produced by changing the model
    in two components:
    10**(2*z) = 10**(2*x) + 10**(2*y)
    
    Marginalize over y, and get the kde for x
    """
    # Obtain the 2D distribution
    X, Y = np.mgrid[minz:maxz:bins*1j, minz:maxz:bins*1j]
    Z = transform_xy(X.ravel(), Y.ravel())
    dist_xy = np.reshape(kde(Z), X.shape)
    
    # Marginalize
    X1D = np.mgrid[minz:maxz:bins*1j]
    dx = X1D[1]-X1D[0]
    xedges = np.append(X1D+0.5*dx, [X1D[0]-0.5*dx])
    dist_x = np.sum(dist_xy, axis=1)
    dist_x *= bins / (np.sum(dist_x) * (maxz-minz))
    
    return smooth_hist(dist_x, X1D, kind=kind)

def create_psr_1D_kdes(psrdir, burnin=5000, low=-18.0, high=-10.0, bins=250, numfreqs=None, kind='cubic'):
    """
    For a single pulsar (chain directory), create the kernel density estimate distributions
    of the per-frequency posteriors. Also create the upper-limit equivalent
    """
    lp, ll, chain, labels, pulsars, pulsarnames, stypes, mlpso, mlpsopars = pic.ReadMCMCFile(psrdir, incextra=True)
    inds = np.where(np.array(stypes) == 'spectrum')[0]
    
    if numfreqs is not None:
        inds = inds[:numfreqs]

    freqs = np.float64(np.array(labels)[inds])
    
    kdes = []
    kdes_ul = []
    for ii in inds:
        samples = chain[burnin:,ii]
        kde_b = bounded_kde.Bounded_kde(samples, low=low)

        # The bounded KDE is super slow. Speed it up by using an interpolator
        xx = np.linspace(low-0.1, high + np.log10(np.sqrt(2))+0.1, 2*bins)
        yy = kde_b(xx)
        kde = interp.interp1d(xx, yy, kind=kind)

        kdes.append(kde)
        kde_ul = make_ul_kde(kde, low, high, bins=bins, kind=kind)
        kdes_ul.append(kde_ul)
    
    return freqs, kdes, kdes_ul

def process_psrdirs(psrdirs, low=-18.0, high=-10.0, bins=100, numfreqs=None,
        burnin=5000, verbose=False, kind='cubic'):
    """
    Create the result list of dicts of the psr dictionaries
    """
    resdict = []
    for psrdir in psrdirs:
        psrname = os.path.basename(psrdir.rstrip('/'))
        fr, kd, kdu = create_psr_1D_kdes(psrdir, burnin=burnin,
                                       low=low, high=high, bins=bins,
                                       numfreqs=numfreqs, kind=kind)
        resdict.append({'freqs':fr, 'kdes':kd, 'kdes_ul':kdu, 'psrname':psrname})
        if verbose:
            print("Done with {0}".format(psrdir))

    return resdict

def write_resdict(dirname, resdict, low=-18.0, high=-10.0, niter=1000):
    """
    Given a list of result dictionaries 'resdict', write all the kde's to text
    files

    @param dirname:     Name of output directory
    @param resdict:     List of result dictionaries
    @param low:         Lowest PSD value allowed
    @param high:        Highest PSD value allowed
    @param niter:       Number of iterations/bins when interpolating
    """
    # Sample the kde's and save them to disk
    for ii, rd in enumerate(resdict):
        amp = np.linspace(low, high, niter)
        fr = rd['freqs']
        psrname = rd['psrname']
        pdf_sin = np.zeros((niter, len(fr)))
        pdf_ul = np.zeros((niter, len(fr)))

        for jj in range(len(fr)):
            pdf_sin[:,jj] = rd['kdes'][jj](amp)
            pdf_ul[:,jj] = rd['kdes_ul'][jj](amp)
             
            filename_sin = os.path.join(dirname, psrname + '-sin-kde-df.txt')
            filename_ul = os.path.join(dirname, psrname + '-ul-kde-df.txt')
            filename_fr = os.path.join(dirname, psrname + '-gwfreqs.txt')
            np.savetxt(filename_sin, pdf_sin)
            np.savetxt(filename_ul, pdf_ul)
            np.savetxt(filename_fr, fr)

def read_resdict(dirname, psrlist=None, low=-18.0, high=-10.0, niter=1000):
    """
    Read all the result dictionaries present in 'dirname'. Only use selected
    pulsars if psrlist is provided

    @param dirname:     Name of input directory
    @param psrlist:     List of pulsars to read (None=all)
    @param low:         Lowest PSD value allowed
    @param high:        Highest PSD value allowed
    @param niter:       Number of iterations/bins when interpolating

    @return resdict:    List of result dictionaries
    """
    resdict = []

    # Create list of pulsars
    pfiles = glob.glob(os.path.join(dirname, '*-sin-kde-df.txt'))
    new_psrlist = []
    for pfile in pfiles:
        psrname = os.path.basename(pfile.rstrip('/'))[:-15]
        if psrlist is None or psrname in psrlist:
            new_psrlist.append(psrname)
        elif psrlist is not None:
            pass

    if len(new_psrlist) < len(psrlist):
        raise IOError("Not all requested pulsars were found")

    for pp, psrname in enumerate(new_psrlist):
        amp = np.linspace(low, high, niter)
        filename_sin = os.path.join(dirname, psrname + '-sin-kde-df.txt')
        filename_ul = os.path.join(dirname, psrname + '-ul-kde-df.txt')
        filename_fr = os.path.join(dirname, psrname + '-gwfreqs.txt')

        kde_sin_pdf = np.loadtxt(filename_sin)
        kde_ul_pdf = np.loadtxt(filename_ul)
        gwfreqs = np.loadtxt(filename_fr)
        kdes = []
        kdes_ul = []
        for ii in range(len(gwfreqs)):
            # Cubic takes so long....
            kde_sin = smooth_hist(kde_sin_pdf[:,ii], amp, kind='linear')
            kde_ul = smooth_hist(kde_ul_pdf[:,ii], amp, kind='linear')
            kdes.append(kde_sin)
            kdes_ul.append(kde_ul)
        resdict.append({'freqs':gwfreqs, 'kdes':kdes, 'kdes_ul':kdes_ul, 'psrname':psrname})
    
    return resdict
 


def gw_ul_spectrum(resdict, confidence=0.95, low=-18.0, high=-10.0, bins=100):
    """
    Using the list of result dictionaries, create the spectrum upper-limits
    """
    gwfreqs = resdict[0]['freqs']      # Assume all frequencies of all pulsars are the same
    #prior_kde = kde_gwprior(low=low, high=high, bins=bins)
    ul = np.zeros_like(gwfreqs)
    gwamps = np.linspace(low, high, bins)

    for ii, freq in enumerate(gwfreqs):
        #kdes = [prior_kde]
        kdes = []
        for pp, psrres in enumerate(resdict):
            kdes.append(psrres['kdes_ul'][ii])

        #fullkde = kde_multiply(kdes, low, high, bins=bins)
        lfullkde = kde_logsum(kdes, low, high, bins=bins)

        #cdf = np.cumsum(fullkde(gwamps)) / np.sum(fullkde(gwamps))
        lpdf = lfullkde(gwamps) + np.log(10**gwamps)
        pdf = np.exp(lpdf - np.max(lpdf))
        cdf = np.cumsum(pdf) / np.sum(pdf)

        ul[ii] = gwamps[cdf > confidence][0]
    
    return gwfreqs, ul


def gw_ul_powerlaw(resdict, confidence=0.95, low=-18.0, high=-10.0, bins=100,
        gwlow=-20.0, gwhigh=-13.0, si=4.33, gwbins=50000, ngwfreqs=None):
    """
    As above, but now allow the frequencies of different pulsars to be
    different.
    """
    gwamps = np.linspace(gwlow, gwhigh, bins)
    #prior_kde = kde_gwprior(low=gwlow, high=gwhigh, bins=bins)
    lpdf = np.zeros_like(gwamps)

    for ii, gwamp in enumerate(gwamps):
        for pp, psrres in enumerate(resdict):
            gwfreqs = resdict[0]['freqs']
            if ngwfreqs is None:
                ngwfreqs = len(gwfreqs)
            else:
                ngwfreqs = min(len(gwfreqs), ngwfreqs)

            # Calculate the spectrum for this GW amplitude
            spect = np.log10(gw_pl_spectrum(gwfreqs, lh_c=gwamp, si=si))
            spect[np.where(spect < low)] = low
            spect[np.where(spect > high)] = high

            for jj, freq in enumerate(gwfreqs[:ngwfreqs]):
                kde = psrres['kdes_ul'][jj]
                lpdf[ii] += np.log(kde(spect[jj]))

        lpdf[ii] += np.log(10**gwamp)  #np.log(prior_kde(gwamp))

    # Smooth the log-pdf
    lpdf_kde = smooth_hist(lpdf, gwamps, sigma=0.75, kind='linear')
    gwamps_l = np.linspace(gwlow, gwhigh, gwbins)
    lpdf_l = lpdf_kde(gwamps_l)
    pdf = np.exp(lpdf_l - np.max(lpdf_l))
    #pdf *= gwbins / (np.sum(pdf) * (gwhigh-gwlow))
    
    # Now calculate the upper-limit
    cdf = np.cumsum(pdf) / np.sum(pdf)
    
    return gwamps_l[cdf > confidence][0], gwamps_l, pdf


def gw_ul_powerlaw_2d(resdict, confidence=0.95, low=-18.0, high=-10.0, gwlow=-17.0, gwhigh=-13.0, si=4.33, bins=100):
    """
    Using the list of result dictionaries, create the power-law gamma=4.33 upper-limits
    """
    gwfreqs = resdict[0]['freqs']      # Assume all frequencies of all pulsars are the same
    
    gwamps = np.linspace(gwlow, gwhigh, bins)
    gwsis = np.linspace(2.01, 6.99, bins)
    lpdf = np.zeros((bins, bins))

    #prior_kde = kde_gwprior(low=gwlow, high=gwhigh, bins=bins)

    for kk, gwsi in enumerate(gwsis):
        for ii, gwamp in enumerate(gwamps):
            # Calculate the spectrum for this GW amplitude
            spect = np.log10(gw_pl_spectrum(gwfreqs, lh_c=gwamp, si=gwsi))
            spect[np.where(spect < low)] = low
            spect[np.where(spect > high)] = high

            # Calculate the posterior, by multiplying the kdes of all pulsars
            for jj, freq in enumerate(gwfreqs):
                for pp, psrres in enumerate(resdict):
                    kde = psrres['kdes_ul'][jj]
                    lpdf[ii,kk] += np.log(kde(spect[jj]))

            lpdf[ii,kk] += np.log(10**gwamp) #np.log(prior_kde(gwamp))
    
    pdf = np.exp(lpdf - np.max(lpdf))

    return gwamps, gwsis, pdf


def psr_loglikelihood(psrresdict, logpsd, low=-18.0, high=-10.0, key='kdes'):
    """
    For a set of log10(PSD) values ``logpsd'', evaluate the likelihood
    for this pulsar

    @param psrresdict:  Dictionary of results for this pulsar
    @param logpsd:      Values of the PSD
    @param low:         Lowest possible value of PSD amplitude
    @param high:        Highest possible value of PSD amplitude
    @param key:         Which kde approximation to use (kdes or kdes_ul)
    """
    lpsd = np.array(logpsd).copy()
    nmaxfreqs = len(lpsd)

    lpsd[np.where(lpsd <= low)] = low
    lpsd[np.where(lpsd >= high)] = high
    loglik = 0.0

    for jj in range(nmaxfreqs):
        kde = psrresdict[key][jj]
        loglik += np.log(kde(lpsd[jj]))

    return loglik

def pta_loglikelihood(pars, resdict, stype='powerlaw',
        low=-18.0, high=-10.0, key='kdes_ul'):
    """
    For the entire PTA, calculate the log-likelihood, given parameters pars and
    the model

    @param pars:    Model parameters (amplitude, spectral-index, etc.)
    @param resdict: List of all results dictionaries
    @param stype:   Spectral/signal model ID (powerlaw/turnover)
    @param low:     Lowest possible value of PSD amplitude
    @param high:    Highest possible value of PSD amplitude
    @param key:     Which kde approximation to use (kdes or kdes_ul)
    """
    loglik = 0.0
    for pp, psrresdict in enumerate(resdict):
        freqs = psrresdict['freqs']
        Tmax = 1.0 / freqs[0]
        
        if stype=='powerlaw':
            lpsd = np.log10(gw_pl_spectrum(freqs,
                    lh_c=pars[0], si=pars[1], Tmax=Tmax))
        elif stype=='turnover':
            lpsd = np.log10(gw_turnover_spectrum(freqs,
                    Tmax=Tmax, pars=pars))
        else:
            raise NotImplementedError("Only powerlaw and turnover implemented")

        lpsd[np.where(lpsd <= low)] = low
        lpsd[np.where(lpsd >= high)] = high

        loglik += psr_loglikelihood(psrresdict, lpsd,
                low=low, high=high, key=key)

    return loglik


def logprior(pars, stype='powerlaw', model='uniform'):
    prior = 0.0
    if stype=='powerlaw' and model=='uniform':
        prior += np.log(10 ** pars[0])
    elif stype=='powerlaw' and model=='sesana':
        m = -15
        s = 0.22
        logA = pars[0]
        prior += -0.5 * (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)
    elif stype=='powerlaw' and model=='mcwilliams':
        m = np.log10(4.1e-15)
        s = 0.26
        logA = pars[0]
        prior += -0.5 * (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)
    elif stype=='turnover' and model=='uniform':
        prior += np.log(10 ** pars[0])
    elif stype=='turnover' and model=='sesana':
        m = -15
        s = 0.22
        logA = pars[0]
        prior += -0.5 * (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)
    elif stype=='turnover' and model=='mcwilliams':
        m = np.log10(4.1e-15)
        s = 0.26
        logA = pars[0]
        prior += -0.5 * (np.log(2 * np.pi * s ** 2) + (m - logA) ** 2 / s ** 2)
    return prior



################################################################################
################################################################################
################### The functions below need to be re-written ##################
################################################################################
################################################################################


def ul_loglikelihood_old(resdict, logpsd, low=-18.0, high=-10.0):
    """
    For a set of log10(PSD) values ``lpsd'', evaluate the upper-limit likelihood
    of the GW frequencies of all pulsars

    Note that this marginalized likelihood/posterior has a prior flat in
    log(gwamp)
    """
    gwfreqs = resdict[0]['freqs']      # Assume all frequencies of all pulsars are the same
    lpsd = np.array(logpsd).copy()
    loglik = 0.0
    nmaxfreqs = len(lpsd)

    # Set the practical log10(psd) for this round
    lpsd[np.where(lpsd < low)] = low
    lpsd[np.where(lpsd > high)] = high
    
    # Calculate the posterior, by multiplying the kdes of all pulsars
    for jj, freq in enumerate(gwfreqs[:nmaxfreqs]):
        for pp, psrres in enumerate(resdict):
            kde = psrres['kdes_ul'][jj]

            loglik += np.log(kde(lpsd[jj]))

    return loglik

def loglikelihood_old(pars, resdict, stype='powerlaw', model='uniform', low=-18.0,
        high=-10.0):
    """
    Logliklihood...  blah
    """
    Tmax = 1.0 / resdict[0]['freqs'][0]
    gwfreqs = resdict[0]['freqs']

    if stype=='powerlaw':
        spect = np.log10(gw_pl_spectrum(gwfreqs, lh_c=pars[0], si=pars[1],
                                        Tmax=Tmax))
    elif stype=='turnover':
        spect = np.log10(gw_turnover_spectrum(gwfreqs, Tmax=Tmax, pars=pars))
    else:
        raise NotImplementedError("Only powerlaw and turnover implemented")

    spect[np.where(spect < low)] = low
    spect[np.where(spect > high)] = high

    return ul_loglikelihood_old(resdict, spect, low=low, high=high)
