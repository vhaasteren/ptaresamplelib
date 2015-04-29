from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as sl
import scipy.stats as stats
import scipy.special as ss
import scipy.ndimage.filters as filters
import scipy.interpolate as interp
import ptmcmc
import bounded_kde
import piccard as pic
import glob, os, sys

pic_spd = 86400.0       # Seconds per day
pic_spy = 31557600.0   # Seconds per year (yr = 365.25 days, so Julian years)
pic_T0 = 53000.0        # MJD to which all HDF5 toas are referenced

class priorDraw(object):
    
    def __init__(self, pmin, pmax, signals, uniform_mask=None):
        """
        @param pmin:            Minimum bound
        @param pmax:            Maximum bound
        @param signals:         Which indices to jump in, grouped by signal
                                (2D array, [sig, ind])
        @param uniform_mask:    Which parameters have a uniform prior
                                (but are sampled in log)
        """
        self.pmin = pmin
        self.pmax = pmax
        self.signals = signals
        
        if uniform_mask is None:
            self.uniform_mask = np.zeros(len(pmin), dtype=np.bool)
        else:
            self.uniform_mask = uniform_mask

    def drawFromPrior(self, parameters, iter, beta):
        # post-jump parameters
        q = parameters.copy()

        # transition probability
        qxy = 0
        
        # Which parameter to jump in
        signum = np.unique(np.random.randint(0, len(self.signals), 1))
        
        # draw params from prior
        for ss in signum:
            for ii in self.signals[ss]:
                if self.uniform_mask[ii]:
                    q[ii] = np.log10(np.random.uniform(
                                     10 ** self.pmin[ii], 10 ** self.pmax[ii]))
                    qxy += np.log(10 ** parameters[ii] / 10 ** q[ii])
                else:
                    q[ii] = np.random.uniform(self.pmin[ii], self.pmax[ii])
                    qxy += 0.0

        return q, qxy


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

def nr_model_pars(stype='powerlaw'):
    """Return the number of parameters for this model
    """
    np=2
    if stype=='powerlaw':
        np=2
    elif stype=='turnover':
        np=4
    else:
        raise NotImplementedError("Only powerlaw and turnover implemented")
    return np


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

def create_psr_1D_kdes(psrdir, burnin=5000, low=-18.0, high=-10.0, bins=250,
        numfreqs=None, kind='cubic', interpolate=True, marginalize=True):
    """
    For a single pulsar (chain directory), create the kernel density estimate distributions
    of the per-frequency posteriors. Also create the upper-limit equivalent

    @param psrdir:      The directory of mcmc chains to parse
    @param burnin:      The burnin samples to use for all chains
    @param low:         Lowest allowed PSD value
    @param high:        Highest allowed PSD value
    @param bins:        Number of bins in the UL integration
    @param numfreqs:    Number of frequencies to process (None = all)
    @param kind:        Kind of interpolator to use ['cubic']
    @param interpolate: Whether or not to substitute the KDEs with interpolators
    @param marginalize: Whether or not to marginalize over the noise
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
        if interpolate:
            xx = np.linspace(low-0.1, high + np.log10(np.sqrt(2))+0.1, 2*bins)
            yy = kde_b(xx)
            kde = interp.interp1d(xx, yy, kind=kind)
        else:
            kde = kde_b

        kdes.append(kde)
        if marginalize:
            kde_ul = make_ul_kde(kde, low, high, bins=bins, kind=kind)
        else:
            kde_ul = None
        kdes_ul.append(kde_ul)
    
    return freqs, kdes, kdes_ul

def process_psrdirs(psrdirs, low=-18.0, high=-10.0, bins=100, numfreqs=None,
        burnin=5000, verbose=False, kind='cubic', interpolate=True,
        marginalize=True):
    """
    Create the result list of dicts of the psr dictionaries
    """
    resdict = []
    for psrdir in psrdirs:
        psrname = os.path.basename(psrdir.rstrip('/'))
        fr, kd, kdu = create_psr_1D_kdes(psrdir, burnin=burnin,
                                       low=low, high=high, bins=bins,
                                       numfreqs=numfreqs, kind=kind,
                                       interpolate=interpolate,
                                       marginalize=marginalize)
        resdict.append({'freqs':fr, 'kdes':kd, 'kdes_ul':kdu, 'psrname':psrname})
        if verbose:
            print("Done with {0}".format(psrdir))

    return resdict

def process_pl_psrdirs(psrdirs, burnin=5000, verbose=False, kind='cubic',
        interpolate=True, mcmcprior='uniform', gwblow=-18.0, gwbhigh=-14.0,
        bins=250):
    """
    Assume the list of directories in psrdirs contains MCMC files with three
    parameters: GWB amplitude, PL amplitude, PL spectral index.

    Create a results dictionary with the kdes for the GWB amplitude

    @param psrdirs:     List of MCMC directories
    @param burnin:      The burnin to use per pulsar
    @param verbose:     Whether or not to output progress
    @param kind:        What kind of interpolation to use
    @param interpolate: Whether or not to interpolate at all
    @param mcmcprior:   What prior was on the GWB amplitude
    @param gwblow:      Lower-bound on GWB amplitude
    @param gwbhigh:     Higher-bound on GWB amplitude
    @param bins:        How many bins to use for interpolation

    @return:    Results dictionary with: 'psrname', 'kde'
    """
    plresdict = []
    for psrdir in psrdirs:
        psrname = os.path.basename(psrdir.rstrip('/'))

        chain = np.loadtxt(os.path.join(psrdir, 'chain_1.txt'))
        samples = chain[burnin:,0]

        kde_b = bounded_kde.Bounded_kde(10**samples, low=10**gwblow, high=10**gwbhigh)

        if interpolate:
            xx = np.linspace(10**gwblow, 10**gwbhigh, bins)
            yy = kde_b(xx)
            kde = interp.interp1d(xx, yy, kind=kind)
        else:
            kde = kde_b

        plresdict.append({'psrname':psrname, 'kde':kde, 'prior':mcmcprior})

        if verbose:
            print("Done with {0}".format(psrdir))

    return plresdict


def gw_ul_powerlaw_from_plmcmc(plresdict, confidence=0.95, gwblow=-18.0,
        gwbhigh=-14.0, bins=250, limbins=50000):
    """
    Using the plmcmc runs, calculate the GWB upper-limit for the array by
    multiplying the per-pulsar GWB amplitude kdes

    @param plresdict:   Results dictionary of the power-law kdes
    @param confidence:  Confidence level of the upper-limit
    @param gwblow:      Lower-bound of the GWB amplitude
    @param gwbhigh:     Higher-bound of the GWB amplitude
    @param bins:        Number of bins in the multiplication chain
    @param limbins:     Number of bins to use in the limit calculation
    """
    gwbamps = np.linspace(gwblow, gwbhigh, bins)
    lpdf = np.zeros_like(gwbamps)
    nzmask = np.ones(len(lpdf), dtype=np.bool)

    for ii, gwamp in enumerate(gwbamps):
        for pp, psrres in enumerate(plresdict):
            kde = psrres['kde']
            pdfval = kde(10**gwamp)
            if pdfval > 0.0:
                lpdf[ii] += np.log(pdfval)
            else:
                lpdf[ii] = -np.inf
                nzmask[ii] = False

            if psrres['prior'] != 'uniform':
                lpdf[ii] -= np.log(10)*gwamp

        #lpdf[ii] += np.log(10)*gwamp

    minlpdf = np.min(lpdf[nzmask])
    lpdf[np.logical_not(nzmask)] = minlpdf
    
    #lpdf_kde = smooth_hist(lpdf, gwbamps, sigma=0.75, kind='linear')
    lpdf_kde = interp.interp1d(gwbamps, lpdf, kind='linear')
    gwamps_l = np.linspace(gwblow, gwbhigh, limbins)
    lpdf_l = lpdf_kde(gwamps_l)

    pdf = np.exp(lpdf_l - np.max(lpdf_l) + np.log(10)*gwamps_l)
    cdf = np.cumsum(pdf) / np.sum(pdf)
    
    return gwamps_l[cdf > confidence][0], gwamps_l, pdf


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

def read_resdict(dirname, psrlist=None, low=-18.0, high=-10.0, niter=1000,
        kind='linear'):
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
    tmp_psrlist = []
    for pfile in pfiles:
        psrname = os.path.basename(pfile.rstrip('/'))[:-15]
        if psrlist is None or psrname in psrlist:
            tmp_psrlist.append(psrname)
        elif psrlist is not None:
            pass

    # Re-sort the tmp_psrlist
    new_psrlist = []
    if psrlist is None:
        new_psrlist = tmp_psrlist
    else:
        for pname in psrlist:
            if pname in tmp_psrlist:
                new_psrlist.append(pname)

    if psrlist is not None:
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
            kde_sin = smooth_hist(kde_sin_pdf[:,ii], amp, kind=kind)
            kde_ul = smooth_hist(kde_ul_pdf[:,ii], amp, kind=kind)
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
    Calculate the GWB upper-limit, by multiplying the per-pulsar marginalized
    kdes per frequency
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


def psr_sp_loglikelihood(psrresdict, logpsd, finds=None,
        low=-18.0, high=-10.0, key='kdes'):
    """
    For a set of log10(PSD) values ``logpsd'', evaluate the likelihood
    for this pulsar

    @param psrresdict:  Dictionary of results for this pulsar
    @param logpsd:      Values of the PSD
    @param finds:       Indices of frequencies to use (None = all)
    @param low:         Lowest possible value of PSD amplitude
    @param high:        Highest possible value of PSD amplitude
    @param key:         Which kde approximation to use (kdes or kdes_ul)
    """
    lpsd = np.array(logpsd).copy()
    if finds is None:
        finds = range(len(lpsd))

    lpsd[np.where(lpsd <= low)] = low
    lpsd[np.where(lpsd >= high)] = high
    loglik = 0.0

    for jj in finds:
        kde = psrresdict[key][jj]
        loglik += np.log(kde(lpsd[jj]))

    return loglik

def pta_pl_loglikelihood(pars, resdict, stype='powerlaw',
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

        loglik += psr_sp_loglikelihood(psrresdict, lpsd,
                low=low, high=high, key=key)

    return loglik

def pta_pl_full_loglikelihood(pars, resdict, stype='powerlaw',
        transfreq=5.28e-8, low=-18.0, high=-10.0):
    """
    For the full PTA, use a more sophisticated noise model:
    - powerlaw for frequencies with f<transfreq.
    - GWB for lowest nplfreq frequencies
    - Full spectrum for the higher frequencies.
    - No marginalization over GWB individual frequencies

    @param pars:        Model parameters (amplitude, spectral-index, etc.)
    @param resdict:     List of all results dictionaries
    @param stype:       Spectral/signal model ID (powerlaw/turnover)
    @param transfreq:   Transition frequency -- below we do powerlaw
    @param low:         Lowest possible value of PSD amplitude
    @param high:        Highest possible value of PSD amplitude
    """
    loglik = 0.0
    for pp, psrresdict in enumerate(resdict):
        freqs = psrresdict['freqs']
        Tmax = 1.0 / freqs[0]
        fmask = (freqs < transfreq)
        finds = np.where(fmask)[0]
        if np.sum(fmask) == 0:
            # If there are no power-law modeled frequencies, skip this pulsar
            continue

        # Gravitational-wave model
        ngwpars = nr_model_pars(stype)
        if stype=='powerlaw':
            gwpsd = gw_pl_spectrum(freqs[finds],
                    lh_c=pars[0], si=pars[1], Tmax=Tmax)
        elif stype=='turnover':
            gwpsd = gw_turnover_spectrum(freqs[finds],
                    Tmax=Tmax, pars=pars)
        else:
            raise NotImplementedError("Only powerlaw and turnover implemented")

        # Red-noise model (pure powerlaw)
        psrpars = pars[2*pp+ngwpars:2*(pp+1)+ngwpars]
        noisepsd = gw_pl_spectrum(freqs[finds],
                    lh_c=psrpars[0], si=psrpars[1], Tmax=Tmax)

        lpsd = np.log10(gwpsd+noisepsd)

        lpsd[np.where(lpsd <= low)] = low
        lpsd[np.where(lpsd >= high)] = high

        loglik += psr_sp_loglikelihood(psrresdict, lpsd, finds=finds,
                low=low, high=high, key='kdes')

    return loglik

def pta_nanograv_loglikelihood(pars, resdict, stype='powerlaw', model='uniform',
           transfreq=5.28e-8, low=-18.0, high=-10.0, si=4.33):
    """
    Calculate the likelihood for the full 'new' NANOGrav model

    @param pars:        Model parameters
    @param resdict:     Results dictionary (form Gibbs sampler)
    @param stype:       Signal type (powerlaw/turnover, etcl.)
    @param model:       Prior model
    @param transfreq:   Model transition frequency
    @param low:         Lowest bound PSD
    @param high:        Highest bound PSD
    @param si:          Spectral index for a fixed-si GWB
    """
    if stype=='turnover_fixedsi':
        newgwpars = np.zeros(4)
        newgwpars[0] = pars[0]
        newgwpars[1] = si
        newgwpars[2:4] = pars[1:3]
        newpars = np.append(newgwpars, pars[3:])
        newstype = 'turnover'
    elif stype=='powerlaw_fixedsi':
        newgwpars = np.zeros(2)
        newgwpars[0] = pars[0]
        newgwpars[1] = si
        newpars = np.append(newgwpars, pars[1:])
        newstype = 'powerlaw'
    else:
        newpars = pars
        newstype = stype
    return pta_pl_full_loglikelihood(newpars, resdict, stype=newstype,
                                    transfreq=transfreq, low=low, high=high)

def pta_nanograv_loglik_const(pars, resdict, stype='powerlaw', model='uniform',
                 transfreq=5.28e-8, low=-18.0, high=-10.0):
    """
    Same as pta_nanograv_loglikelihood, but constant
    """
    return 0.0

def pta_nanograv_logprior(pars, pmin, pmax, stype='powerlaw',
        model='uniform', si=4.33):
    """
    Calculate the prior for the full 'new' NANOGrav model

    @param pars:        Model parameters
    @param resdict:     Results dictionary (form Gibbs sampler)
    @param stype:       Signal type (powerlaw/turnover, etcl.)
    @param model:       Prior model
    @param transfreq:   Model transition frequency
    @param low:         Lowest bound PSD
    @param high:        Highest bound PSD
    @param si:          Spectral index for a fixed-si GWB
    """
    rv = -np.inf
    if np.all(pars <= pmax) and np.all(pars >= pmin):
        if stype=='turnover_fixedsi':
            newstype = 'turnover'
            newgwpars = np.zeros(4)
            newgwpars[0] = pars[0]
            newgwpars[1] = si
            newgwpars[2:4] = pars[1:3]
            newpars = np.append(newgwpars, pars[3:])
        elif stype=='powerlaw_fixedsi':
            newstype = 'powerlaw'
            newgwpars = np.zeros(2)
            newgwpars[0] = pars[0]
            newgwpars[1] = si
            newpars = np.append(newgwpars, pars[1:])
        else:
            newstype = stype
            newpars = pars

        rv = pta_pl_full_logprior(newpars, stype=newstype, model=model)
    return rv


def pta_pl_full_logprior(pars, stype='powerlaw', model='uniform'):
    """
    Assume we have to place a prior on per-pulsar red noise, and on the GWB
    """
    lp = model_logprior(pars, stype=stype, model=model)

    ngwpars = nr_model_pars(stype)
    nnoisepars = nr_model_pars('powerlaw')

    npsrs = int((len(pars)-ngwpars) / nnoisepars)

    for ii in range(npsrs):
        psrpars = pars[ngwpars+nnoisepars*ii:ngwpars+nnoisepars*(ii+1)]
        lp += model_logprior(psrpars, stype='powerlaw', model='uniform')

    return lp


def model_logprior(pars, stype='powerlaw', model='uniform'):
    """
    Hyper-prior, using the Astrophysical model.
    """
    prior = 0.0
    if stype=='powerlaw' and model=='uniform':
        prior += np.log(10 ** pars[0])
    elif stype=='powerlaw' and model=='flatlog':
        pass
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
    elif stype=='turnover' and model=='flatlog':
        pass
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

def setup_nanograv_model(resdict, stype='powerlaw', model='uniform',
        chaindir='chains', transfreq=5.28e-8, low=-18.0, high=-10.0,
        verbose=True):
    """
    Given the parameters quoted, set up a nanograv analysis, including samplers
    and shit.
    """
    if stype == 'turnover':
        gwpmin = np.array([-18.0, 1.02, -9, 0.01])
        gwpmax = np.array([-11.0, 6.98, -7, 6.98])
        gwpstart = np.array([-15.0, 2.01, -8, 2.01])
        gwpwidth = np.array([0.1, 0.1, 0.1, 0.1])
        gwlabels = np.array(['gwamp', 'gwgamma', 'gwf0', 'gwkappa'])
        signals = [np.array([0, 1, 2, 3], dtype=np.int)]
        unimask = np.array([False]*4, dtype=np.bool)
        gwndim = 4
    elif stype == 'turnover_fixedsi':
        gwpmin = np.array([-18.0, -9, 0.01])
        gwpmax = np.array([-11.0, -7, 6.98])
        gwpstart = np.array([-15.0, -8, 2.01])
        gwpwidth = np.array([0.1, 0.1, 0.1])
        gwlabels = np.array(['gwamp', 'gwf0', 'gwkappa'])
        signals = [np.array([0, 1, 2], dtype=np.int)]
        unimask = np.array([False]*3, dtype=np.bool)
        gwndim = 3
    elif stype == 'powerlaw':
        gwpmin = np.array([-18.0, 1.02])
        gwpmax = np.array([np.log10(4e-12), 6.98])
        gwpstart = np.array([-15.0, 2.01])
        gwpwidth = np.array([0.1, 0.1])
        gwlabels = np.array(['gwamp', 'gwgamma'])
        signals = [np.array([0, 1], dtype=np.int)]
        unimask = np.array([False, False], dtype=np.bool)
        gwndim = 2
    elif stype == 'powerlaw_fixedsi':
        gwpmin = np.array([-18.0])
        gwpmax = np.array([np.log10(4e-12)])
        gwpstart = np.array([-15.0])
        gwpwidth = np.array([0.1])
        gwlabels = np.array(['gwamp'])
        signals = [np.array([0], dtype=np.int)]
        unimask = np.array([False], dtype=np.bool)
        gwndim = 1

    pmin, pmax, pstart, pwidth, labels, ndim = gwpmin.copy(), gwpmax.copy(), \
                            gwpstart.copy(), gwpwidth.copy(), gwlabels.copy(), gwndim

    for ii, rd in enumerate(resdict):
        jj = len(pmin)
        psrname = rd['psrname']
        signals.append(np.array([jj, jj+1], dtype=np.int))
        pmin = np.append(pmin, np.array([-18.0, 1.02]))
        pmax = np.append(pmax, np.array([np.log10(4e-12), 6.98]))
        pstart = np.append(pstart, np.array([-14.8, 3.01]))
        pwidth = np.append(pwidth, np.array([0.1, 0.1]))
        labels = np.append(labels, np.array(['amp-'+psrname, 'gamma-'+psrname]))
        unimask = np.append(unimask, np.array([False, False], dtype=np.bool))
        ndim += 2

    cov = np.diag(np.array(pwidth)**2)

    pdraw = priorDraw(pmin, pmax, signals[0:], uniform_mask=unimask)

    sampler = ptmcmc.PTSampler(ndim, pta_nanograv_loglikelihood,
            pta_nanograv_logprior, cov=cov, outDir=chaindir, verbose=verbose,
            loglargs=(resdict, stype, model, transfreq, low, high),
            logpargs=(pmin, pmax, stype, model))

    sampler.addProposalToCycle(pdraw.drawFromPrior, 100)

    return (pmin, pmax, pstart, pwidth, labels, sampler)



def resample_psr_loglikelihood(pars, samples, freqs, stype='powerlaw', low=-18.0, high=-10.0):
    """
    For a single pulsar, calculate the log-likelihood, given parameters pars and
    the spectrum samples (mode & power)

    @param pars:    Model parameters (amplitude, spectral-index, etc.)
    @param resdict: List of all results dictionaries
    @param stype:   Spectral/signal model ID (powerlaw/turnover)
    @param low:     Lowest possible value of PSD amplitude
    @param high:    Highest possible value of PSD amplitude
    @param key:     Which kde approximation to use (kdes or kdes_ul)
    """
    Tmax = 1.0 / freqs[0]
    nfreqs = int(samples.shape[1] / 3)
        
    if stype=='powerlaw':
        pl_lpsd = np.log10(gw_pl_spectrum(freqs,
                lh_c=pars[0], si=pars[1], Tmax=Tmax))
    elif stype=='turnover':
        pl_lpsd = np.log10(gw_turnover_spectrum(freqs,
                Tmax=Tmax, pars=pars))
    else:
        raise NotImplementedError("Only powerlaw and turnover implemented")

    # The numerator (power-law) PSD
    pl_lpsd[np.where(pl_lpsd <= low)] = low
    pl_lpsd[np.where(pl_lpsd >= high)] = high

    # The denominator (spectrum) PSD
    sp_lpsd = samples[:,-nfreqs:]

    # The sample modes
    a_cos = samples[:,:-nfreqs:2]
    a_sin = samples[:,1:-nfreqs:2]

    # Numerator xi^2 and det
    pl_lxi = np.log(np.sum(a_cos**2 / 10**pl_lpsd, axis=1) \
            + np.sum(a_sin**2 / 10**pl_lpsd, axis=1))
    pl_ldet = 2 * np.sum(np.log(10) * pl_lpsd)

    # Denominator xi^2 and det
    sp_lxi = np.log(np.sum(a_cos**2 / 10**sp_lpsd, axis=1) \
            + np.sum(a_sin**2 / 10**sp_lpsd, axis=1))
    sp_ldet = 2 * np.sum(np.log(10) * sp_lpsd, axis=1)

    return 0.5 * (sp_lxi-pl_lxi) + 0.5 * (sp_ldet-pl_ldet)

def fullkde_psr_loglikelihood(pars, freqs, kde, stype='powerlaw', low=-18.0, high=-10.0):
    """
    For a single pulsar, calculate the log-likelihood, given parameters pars and
    the spectrum samples (mode & power)

    @param pars:    Model parameters (amplitude, spectral-index, etc.)
    @param stype:   Spectral/signal model ID (powerlaw/turnover)
    @param low:     Lowest possible value of PSD amplitude
    @param high:    Highest possible value of PSD amplitude
    """
    Tmax = 1.0 / freqs[0]

    if stype=='powerlaw':
        pl_lpsd = np.log10(gw_pl_spectrum(freqs,
                lh_c=pars[0], si=pars[1], Tmax=Tmax))
    elif stype=='turnover':
        pl_lpsd = np.log10(gw_turnover_spectrum(freqs,
                Tmax=Tmax, pars=pars))
    else:
        raise NotImplementedError("Only powerlaw and turnover implemented")

    # The power-law 
    pl_lpsd[np.where(pl_lpsd <= low)] = low
    pl_lpsd[np.where(pl_lpsd >= high)] = high

    return np.log(kde(pl_lpsd))
