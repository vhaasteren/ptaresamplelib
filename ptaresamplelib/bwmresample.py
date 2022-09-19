#!/usr/bin/env python
# encoding: utf-8
# vim: tabstop=4:softtabstop=4:shiftwidth=4:expandtab

from __future__ import division

import numpy as np
import math
import scipy.linalg as sl, scipy.special as ss
import matplotlib.pyplot as plt
import os, glob
import sys
import json
import bounded_kde
from sklearn import mixture
import piccard as pic
import corner
import pickle

from cgmm import compile_gmm

# Class to do interval transforms
class intervalTransform(object):
    """Class that performs interval transformations"""

    def __init__(self, a, b):
        self._a = a
        self._b = b

    def forward(self, x):
        """Forward transformation"""
        if x <= self._a:
            p = -np.inf
        elif x >= self._b:
            p = np.inf
        else:
            p = np.log((x - self._a) / (self._b - x))

        return p

    def backward(self, p):
        """Backward transformation"""
        return (self._b - self._a) * np.exp(p) / (1 + np.exp(p)) + self._a

    def logjacobian(self, p):
        """Log-jacobian, and it's gradient"""
        return np.sum( np.log(self._b-self._a) + p - 2*np.log(1.0+np.exp(p)) )

    def logjacobian_grad(self, p):
        """Log-jacobian, and it's gradient"""
        lj = self.logjacobian(p)

        lj_grad = np.zeros_like(p)
        lj_grad = (1 - np.exp(p)) / (1 + np.exp(p))
        return lj, lj_grad

    def dxdp(self, p):
        """Derivative of x wrt p (jacobian for chain-rule) - diagonal"""
        return (self._b-self._a)*np.exp(p)/(1+np.exp(pp))**2


class BwmPsrOne(object):
    """An approximation to a single-pulsar BWM/glitch full-noise marginalized
    posterior distribution, based on a single-pulsar MCMC run"""
    
    def __init__(self, chaindir, burnin=10000,
                low=None, high=None, gmmcomponents=18,
                keepchains=False, method='gmm', check_saved=True):
        """Initialize the bwm Single pulsar approximation

        :param chaindir:
            Directory where the MCMC chains, and the approximation
            pickelizations, are stored

        :param burnin:
            When reading in the MCMC chains, use this many samples as burn-in

        :param low:
            Lower bound for the parameters. DO NOT USE WITH GMM

        :param high:
            Higher bound for the parameters. DO NOT USE WITH GMM

        :param gmmcomponents:
            How many Gaussian components to include in the Gaussian Mixture
            Model

        :param keepchains:
            Whether to keep the chains stored in memory

        :param method:
            Which approximation method to use. gmm/kde

        :param check_saved:
            Whether to check whether we have saved results saved as a pickle,
            instead of re-reading and creating the approximations at start-up
        """

        # Initialize a bwm pulsar, with two chains
        if not os.path.isdir(chaindir):
            raise IOError("Not a valid directory:", chaindir)

        self._low = low
        self._high = high
        self._norm = 0.0
        self.chain_pos = None
        self.chain_neg = None
        self.trans_epoch = None
        self.trans_amp = None

        # Method and stuff
        self._method = method
        self._chaindir = chaindir
        self._burnin = burnin

        # Get the pulsar positions, and the BWM bounds
        psrname = os.path.basename(chaindir.rstrip('/'))
        chainbase = os.path.dirname(chaindir)
        self.set_psrpos(chainbase, psrname)

        if self._method == 'kde':
            self.init_kdes(chaindir, burnin=burnin, keepchains=keepchains)
        elif self._method == 'gmm':
            self.init_gmms(chaindir, burnin=burnin, keepchains=keepchains,
                    check_saved=check_saved, gmmcomponents=gmmcomponents)

    def get_gmm_aicbic(self, range=np.arange(1, 20), check_saved=True):
        aic_pos = np.zeros_like(range)
        bic_pos = np.zeros_like(range)
        aic_neg = np.zeros_like(range)
        bic_neg = np.zeros_like(range)

        # We really do need the MCMC chains for this
        if self.chain_pos is None or self.chain_neg is None:
                self.set_bwmchain(self._chaindir, burnin=self._burnin)

        # Get the transformed chains
        trans_chain_pos = np.zeros_like(self.chain_pos)
        trans_chain_neg = np.zeros_like(self.chain_neg)
        for ii, x in enumerate(self.chain_pos):
            tepoch = self.trans_epoch.forward(x[0])
            tamp = self.trans_amp.forward(x[1])
            trans_chain_pos[ii,:] = np.array([tepoch, tamp])
        for ii, x in enumerate(self.chain_neg):
            tepoch = self.trans_epoch.forward(x[0])
            tamp = self.trans_amp.forward(x[1])
            trans_chain_neg[ii,:] = np.array([tepoch, tamp])

        for ii, components in enumerate(range):
            # Create the Gaussian Mixture models
            if check_saved and self.read_gmm_pickle(self._chaindir, components):
                pass
            else:
                self.write_gmm_pickle(self._chaindir, components,
                        overwrite=True)
            aic_pos[ii] = self.clf_pos.aic(trans_chain_pos)
            bic_pos[ii] = self.clf_pos.bic(trans_chain_pos)
            aic_neg[ii] = self.clf_neg.aic(trans_chain_neg)
            bic_neg[ii] = self.clf_neg.bic(trans_chain_neg)

        return aic_pos, bic_pos, aic_neg, bic_neg

    def get_bwminds(self, stype, labels):
        bwminds = np.where(np.array(stype) == 'psrbwm')[0]
        ampid = bwminds[np.array(labels)[bwminds] == 'amplitude'][0]
        epid = bwminds[np.array(labels)[bwminds] == 'burst-arrival'][0]
        signid = bwminds[np.array(labels)[bwminds] == 'sign'][0]

        return np.array([epid, ampid, signid])

    def init_kdes(self, chaindir, burnin=10000, keepchains=False):
        # Set positive and negative BWM chains (and normalize)
        self.set_bwmchain(chaindir, burnin=burnin)

        self.kde_neg = bounded_kde.Bounded_kde_md(self.chain_neg.T,
                        low=self._low, high=self._high)
        self.kde_pos = bounded_kde.Bounded_kde_md(self.chain_pos.T,
                        low=self._low, high=self._high)

        if not keepchains:
            self.del_bwmchains()

    def init_gmms(self, chaindir, burnin=10000, keepchains=False,
            check_saved=True, gmmcomponents=18):
        if check_saved and self.read_gmm_pickle(chaindir, gmmcomponents):
            pass
        else:
            # Set positive and negative BWM chains (and normalize)
            self.set_bwmchain(chaindir, burnin=burnin)
            self.write_gmm_pickle(chaindir, gmmcomponents, overwrite=True)

            if not keepchains:
                self.del_bwmchains()

    def get_gmm_picklenames(self, chaindir, gmmcomponents=18):
        gmm_pos_name = 'bwm-gmm-pos-{0}.pickle'.format(str(gmmcomponents))
        gmm_neg_name = 'bwm-gmm-neg-{0}.pickle'.format(str(gmmcomponents))
        norm_name = 'bwm-norm.pickle'

        gmm_pos_fullname = os.path.join(chaindir, gmm_pos_name)
        gmm_neg_fullname = os.path.join(chaindir, gmm_neg_name)
        norm_fullname = os.path.join(chaindir, norm_name)
        
        return gmm_pos_fullname, gmm_neg_fullname, norm_fullname

    def read_gmm_pickle(self, chaindir, gmmcomponents=18):
        """Check whether we have pickle files, and a normalization"""
        gmm_pos_name, gmm_neg_name, norm_name = \
            self.get_gmm_picklenames(chaindir, gmmcomponents)

        if os.path.isfile(gmm_pos_name) and \
                os.path.isfile(gmm_neg_name) and \
                os.path.isfile(norm_name):
            with open(gmm_pos_name, 'r') as fil:
                self.clf_pos = pickle.load(fil)

            with open(gmm_neg_name, 'r') as fil:
                self.clf_neg = pickle.load(fil)

            with open(norm_name, 'r') as fil:
                self._norm = pickle.load(fil)
            
            compile_gmm(self.clf_pos)
            compile_gmm(self.clf_neg)
        else:
            return False
        return True

    def write_gmm_pickle(self, chaindir, gmmcomponents=18, overwrite=True):
        """Check whether we have pickle files, and a normalization"""
        gmm_pos_name, gmm_neg_name, norm_name = \
            self.get_gmm_picklenames(chaindir, gmmcomponents)

        if (not os.path.isfile(gmm_pos_name)) or overwrite:
            self.clf_pos = self.calc_gmm(self.chain_pos, ncomponents=gmmcomponents)
            with open(gmm_pos_name, 'w') as fil:
                #self.clf_pos = pickle.load(fil)
                pickle.dump(self.clf_pos, fil)

            # Use the Cython version of the scorer instead of the built-in one
            # Much, much faster
            compile_gmm(self.clf_pos)

        if (not os.path.isfile(gmm_neg_name)) or overwrite:
            self.clf_neg = self.calc_gmm(self.chain_neg, ncomponents=gmmcomponents)
            with open(gmm_neg_name, 'w') as fil:
                #self.clf_pos = pickle.load(fil)
                pickle.dump(self.clf_neg, fil)

            # Use the Cython version of the scorer instead of the built-in one
            # Much, much faster
            compile_gmm(self.clf_neg)

        if (not os.path.isfile(norm_name)) or overwrite:
            with open(norm_name, 'w') as fil:
                pickle.dump(self._norm, fil)


    def set_bwmchain(self, chaindir, burnin=10000):
        (ll, lp, chain, labels, pulsarid, pulsarname, stype, mlpso, mlpsopars) = \
            pic.ReadMCMCFile(chaindir, incextra=True)

        inds = self.get_bwminds(stype, labels)
        #return chain[burnin:, inds], lp[burnin:]

        samples = chain[burnin:, inds]
        lnprob = lp[burnin:]

        # sign == 0.0 has no signal (but we won't ever be there btw)
        inds_pos = samples[:,2] > 0.0
        inds_neg = samples[:,2] < 0.0

        self.chain_pos = samples[inds_pos,:2]
        self.lnprob_pos = lnprob[inds_pos]
        self.chain_neg = samples[inds_neg,:2]
        self.lnprob_neg = lnprob[inds_neg]

        self._norm = np.log(np.sum(inds_pos)) - np.log(np.sum(inds_neg))

    def del_bwmchains(self):
        """To save memory, delete the chains"""
        del self.chain_pos
        del self.chain_neg
        self.chain_pos = None
        self.chain_neg = None
    
    def set_psrpos(self, chainbase, psrname):
        psrnames = np.loadtxt(os.path.join(chainbase, 'psrpos.txt'), dtype=str, usecols=[0])
        positions = np.loadtxt(os.path.join(chainbase, 'psrpos.txt'), usecols=[1,2,3,4,5,6])
        
        # Find the psr position in the list
        try:
            ind = np.atleast_1d(np.where(psrnames == psrname)[0])
            
            if len(ind) < 1:
                raise ValueError()

            self.name = psrnames[ind[0]]
        except:
            raise ValueError("Could not find {0} in psrpos file".format(psrname))

        self.raj = positions[ind[0],0]
        self.decj = positions[ind[0],1]

        if self._low is None:
            self._low = np.array([positions[ind[0],2], positions[ind[0],4]])
        else:
            self._low = np.array(self._low)

        if self._high is None:
            self._high = np.array([positions[ind[0],3], positions[ind[0],5]])
        else:
            self._high = np.array(self._high)

        self.trans_epoch = intervalTransform(self._low[0], self._high[0])
        self.trans_amp = intervalTransform(self._low[1], self._high[1])

    def calc_gmm(self, chain, ncomponents=18):
        """Calculate the Gaussian Mixture Model approximation, given the chain
        and the bounds"""
        self.trans_epoch = intervalTransform(self._low[0], self._high[0])
        self.trans_amp = intervalTransform(self._low[1], self._high[1])

        # Calculate the transformed chain
        # TODO: vectorize this!!!
        trans_chain = np.zeros_like(chain)
        for ii, x in enumerate(chain):
            tepoch = self.trans_epoch.forward(x[0])
            tamp = self.trans_amp.forward(x[1])
            trans_chain[ii,:] = np.array([tepoch, tamp])

        # Calculate the GMM
        clf = mixture.GMM(n_components=ncomponents, covariance_type='full')
        clf.fit(trans_chain)

        # clf.aic(trans_chain)
        # clf.bic(trans_chain)
        return clf

    def boundpars(self, pars, pad=0.0):
        pars = np.array(pars).copy()
        padding = pad * (np.array(self._high)-np.array(self._low))

        pars[0] = max(self._low[0] + padding[0], pars[0])
        pars[1] = max(self._low[1] + padding[1], pars[1])
        pars[0] = min(self._high[0] - padding[0], pars[0])
        pars[1] = min(self._high[1] - padding[1], pars[1])
        return pars
                
    def pdf_kde(self, pars, pos=True):
        pars = self.boundpars(pars)
        return self.kde_pos(pars)[0] if pos else self.kde_neg(pars)[0] * np.exp(self._norm)
    
    def logpdf_kde(self, pars, pos=True):
        pars = self.boundpars(pars)
        return np.log(self.kde_pos(pars)[0]) if pos else np.log(self.kde_neg(pars)[0]) + self._norm

    def logpdf_gmm(self, pars, pos=True, pad=5.0e-2):
        pars = self.boundpars(pars, pad=pad)

        # Get the transformed coordinates
        p = np.zeros_like(pars)
        p[0] = self.trans_epoch.forward(pars[0])
        p[1] = self.trans_amp.forward(pars[1])

        # Get the Jacobian of the transformation log|dx/dp|
        lj = 0.0
        lj += self.trans_epoch.logjacobian(p[0])
        lj += self.trans_amp.logjacobian(p[1])

        # Get the transformed density
        if pos:
            ll = self.clf_pos.score(np.atleast_2d(p))[0]
        else:
            ll = self.clf_neg.score(np.atleast_2d(p))[0]

        return ll - lj
        
    def pdf_gmm(self, pars, pos=True, pad=5.0e-3):
        return np.exp(self.logpdf_gmm(pars, pos=pos, pad=pad))

    def logpdf(self, pars, pos=True):
        #return self.logpdf_kde(pars, pos=pos)
        return self.logpdf_gmm(pars, pos=pos)

    def pdf(self, pars, pos=True):
        if self._method == 'kde':
            return self.pdf_kde(pars, pos=pos)
        elif self._method == 'gmm':
            return self.pdf_gmm(pars, pos=pos)
        return self.pdf_gmm(pars, pos=pos)

    def prior_transform(self, x):
        return self._low + x * (self._high - self._low)


class BwmArrayMar(object):
    """An approximation to a full Array of pulsars' likelihood for a BWM signal,
    fully noise-marginalized, based on a single-pulsar MCMC run. If glitches per
    pulsar are allowed, there need to be 2 glitch events in the single-psr MCMC
    runs"""
    
    def __init__(self, resultsdir=None, psrdirs=None,
            burnin=10000, psrlist=None,
            low=[53000.0, -18.0], high=[55000.0, -10.0],
            gmmcomponents=18, check_saved=True,
            incMonopole=False, incDipole=False, incQuadrupole=True,
            incAbsQuadrupole=False, verbose=True):
        """
        :param resultsdir:
            Top-level directory, where all subdirectories contain single-pulsar
            MCMC chains. Used if psrdirs is not set

        :param psrlist:
            When using 'resultsdir', if this parameter is set, only read in the
            pulsar if it is in psrlist

        :param psrdirs:
            List of directories containing single-pulsar MCMC chains. Useful if
            the MCMC chains are in various places

        :param ...:

        """
        self.bwmPsrs = []

        if psrdirs is None:
            self.readFromTopLevel(resultsdir, psrlist=psrlist, burnin=burnin,
                    check_saved=check_saved, gmmcomponents=gmmcomponents,
                    verbose=verbose)
        else:
            for pd in psrdirs:
                self.readPsrChain(pd, burnin=burnin, check_saved=check_saved,
                        gmmcomponents=gmmcomponents)

        self.model = []
        if incMonopole:
            self.model.append('monopole')

        if incDipole:
            self.model.append('dipole')

        if incQuadrupole:
            self.model.append('quadrupole')

        if incAbsQuadrupole:
            self.model.append('absquadrupole')

        self.setPriors(low, high)

    def readFromTopLevel(self, resultsdir, psrlist=None, burnin=10000, check_saved=True,
            gmmcomponents=18, verbose=True):
        for infile in glob.glob(os.path.join(resultsdir, '[BJ]*')):
            psrname = os.path.basename(infile)

            if psrlist is None or psrname in psrlist:
                if verbose:
                    print("Reading in {0}...".format(psrname))
                chaindir = os.path.join(resultsdir, psrname)

                self.readPsrChain(chaindir, burnin=burnin,
                        check_saved=check_saved, gmmcomponents=gmmcomponents)

    def readPsrChain(self, chaindir, burnin=10000, check_saved=True,
            gmmcomponents=18):
        self.bwmPsrs.append(BwmPsrOne(chaindir, burnin=burnin,
                keepchains=False, method='gmm', check_saved=check_saved,
                gmmcomponents=gmmcomponents))

    def setPriors(self, low, high):
        """
        Set the prior bounds on all the model parameters
        """
        startepoch = 0.5*(low[0]+high[0])
        startamp = 0.5*(low[1]+high[1])
        startraj = 0.1
        startdecj = 0.1     # Actually sin(decj)
        startpol = 0.1
        labels = ['epoch']

        # All signals share the burst/glitch epoch
        pmin = [low[0]]
        pmax = [high[0]]
        pstart = [startepoch]
        pwidth = [10.0]

        if 'monopole' in self.model:
            pmin += [low[1]]
            pmax += [high[1]]
            pstart += [startamp]
            pwidth += [0.1]
            labels += ['mono-lamp']

        if 'dipole' in self.model:
            pmin += [low[1], 0.0, -1.0, 0.0]
            pmax += [high[1], 2*np.pi, 1.0, 2*np.pi]
            pstart += [startamp, startraj, startdecj, startpol]
            pwidth += [0.1, 0.1, 0.1, 0.1]
            labels += ['dip-lamp', 'dip-raj', 'sin(dip-decj)', 'dip-pol']

        if 'quadrupole' in self.model:
            pmin += [low[1], 0.0, -1.0, 0.0]
            pmax += [high[1], 2*np.pi, 1.0, np.pi]
            pstart += [startamp, startraj, startdecj, startpol]
            pwidth += [0.1, 0.1, 0.1, 0.1]
            labels += ['bwm-lamp', 'bwm-raj', 'sin(bwm-decj)', 'bwm-pol']

        if 'absquadrupole' in self.model:
            pmin += [low[1], 0.0, -1.0, 0.0]
            pmax += [high[1], 2*np.pi, 1.0, np.pi]
            pstart += [startamp, startraj, startdecj, startpol]
            pwidth += [0.1, 0.1, 0.1, 0.1]
            labels += ['absquad-lamp', 'absquad-raj', 'sin(absquad-decj)', 'absquad-pol']

        self.pmin = np.array(pmin)
        self.pmax = np.array(pmax)
        self.pstart = np.array(pstart)
        self.pwidth = np.array(pwidth)
        self.labels = labels

    def MonopoleAntennaPattern(self):
        """Return the antenna pattern of a MonoPole
        """
        return 1.0
    
    def DipoleAntennaPattern(self, rajp, decjp, raj, decj, pol):
        """Return the dipole antenna pattern for a given source position and
        pulsar position

        :param rajp:    Right ascension pulsar (rad) [0,2pi]
        :param decj:    Declination pulsar (rad) [-pi/2,pi/2]
        :param raj:     Right ascension source (rad) [0,2pi]
        :param dec:     Declination source (rad) [-pi/2,pi/2]
        :param pol:     Polarization angle (rad) [0,2pi]
        """
        Omega = np.array([-np.cos(decj)*np.cos(raj), \
                          -np.cos(decj)*np.sin(raj), \
                          -np.sin(decj)])

        mhat = np.array([-np.sin(raj), np.cos(raj), 0])
        nhat = np.array([-np.cos(raj)*np.sin(decj), \
                         -np.sin(decj)*np.sin(raj), \
                         np.cos(decj)])

        p = np.array([np.cos(rajp)*np.cos(decjp), \
                      np.sin(rajp)*np.cos(decjp), \
                      np.sin(decjp)])

        return np.cos(pol) * np.dot(nhat, p) + \
                np.sin(pol) * np.dot(mhat, p)

    def QuadrupoleAntennaPattern(self, rajp, decjp, raj, decj, pol):
        """Return the antenna pattern for a given source position and
        pulsar position

        :param rajp:    Right ascension pulsar (rad) [0,2pi]
        :param decj:    Declination pulsar (rad) [-pi/2,pi/2]
        :param raj:     Right ascension source (rad) [0,2pi]
        :param dec:     Declination source (rad) [-pi/2,pi/2]
        :param pol:     Polarization angle (rad) [0,pi]
        """
        Omega = np.array([-np.cos(decj)*np.cos(raj), \
                          -np.cos(decj)*np.sin(raj), \
                          -np.sin(decj)])

        mhat = np.array([-np.sin(raj), np.cos(raj), 0])
        nhat = np.array([-np.cos(raj)*np.sin(decj), \
                         -np.sin(decj)*np.sin(raj), \
                         np.cos(decj)])

        p = np.array([np.cos(rajp)*np.cos(decjp), \
                      np.sin(rajp)*np.cos(decjp), \
                      np.sin(decjp)])


        Fp = 0.5 * (np.dot(nhat, p)**2 - np.dot(mhat, p)**2) / (1 + np.dot(Omega, p))
        Fc = np.dot(mhat, p) * np.dot(nhat, p) / (1 + np.dot(Omega, p))

        return np.cos(2*pol)*Fp + np.sin(2*pol)*Fc

    def loglik(self, pars):
        """
        Evaluate the full-array bwm pdf
        
        :param pars:
            Parameters of mono-, dipole-, and quadrupole signals
        """
        ll = 0.0

        epoch = pars[0]

        for bwmPsr in self.bwmPsrs:
            index = 1   # Reset the index counter (need per-pulsar)
            monoamp = 0
            dipamp = 0.0
            quadamp = 0.0
            absquadamp = 0.0

            if 'monopole' in self.model:
                mono_ap = self.MonopoleAntennaPattern()
                monoamp = mono_ap * 10**pars[index]
                index += 1
            else:
                monoamp = 0.0

            if 'dipole' in self.model:
                dip_ap = self.DipoleAntennaPattern(bwmPsr.raj, bwmPsr.decj,
                        pars[index+1], np.arcsin(pars[index+2]), pars[index+3])
                dipamp = dip_ap * 10**pars[index]
                index += 4
            else:
                dipamp = 0.0

            if 'quadrupole' in self.model:
                quad_ap = self.QuadrupoleAntennaPattern(bwmPsr.raj, bwmPsr.decj,
                        pars[index+1], np.arcsin(pars[index+2]), pars[index+3])
                quadamp = quad_ap * 10**pars[index]
                index += 4
            else:
                quadamp = 0.0

            if 'absquadrupole' in self.model:
                absquad_ap = self.QuadrupoleAntennaPattern(bwmPsr.raj, bwmPsr.decj,
                        pars[index+1], np.arcsin(pars[index+2]), pars[index+3])
                absquadamp = np.abs(absquad_ap) * 10**pars[index]
                index += 4
            else:
                absquadamp = 0.0

            amp = monoamp + dipamp + quadamp + absquadamp
            s = np.sign(amp)
            lamp = np.log10(s*amp)
            ll += bwmPsr.logpdf([epoch, lamp], s==1.0)

        return ll
    
    def logprior(self, pars):
        lp = 0.0
        if np.any(pars < self.pmin) or np.any(pars > self.pmax):
            lp = -np.inf
        
        return lp

    def prior_transform(self, x):
        return self.pmin + x * (self.pmax - self.pmin)
    
    def logposterior(self, pars):
        return self.logprior(pars) + self.loglik(pars)



