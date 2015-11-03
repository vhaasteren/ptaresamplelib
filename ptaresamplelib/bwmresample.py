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
import piccard as pic


class bwmPsrResult(object):
    
    def __init__(self, chaindir, burnin=10000,
                low=None, high=None):
        # Initialize a bwm pulsar, with two chains
        if not os.path.isdir(chaindir):
            raise IOError("Not a valid directory:", chaindir)

        self._low = low
        self._high = high
        self._norm = 0.0

        # Set positive and negative BWM chains
        self.set_bwmchain(chaindir, burnin=burnin)

        # Get the pulsar positions, and the BWM bounds
        psrname = os.path.basename(chaindir.rstrip('/'))
        chainbase = os.path.dirname(chaindir)
        self.set_psrpos(chainbase, psrname)

        # Create the kdes
        self.kde_neg = bounded_kde.Bounded_kde_md(self.chain_neg.T,
                        low=self._low, high=self._high)
        self.kde_pos = bounded_kde.Bounded_kde_md(self.chain_pos.T,
                        low=self._low, high=self._high)

        # Normalize the two halfs
        #self.normalize(Ntrial=100)

    def get_bwminds(self, stype, labels):
        bwminds = np.where(np.array(stype) == 'psrbwm')[0]
        ampid = bwminds[np.array(labels)[bwminds] == 'amplitude'][0]
        epid = bwminds[np.array(labels)[bwminds] == 'burst-arrival'][0]
        signid = bwminds[np.array(labels)[bwminds] == 'sign'][0]

        return np.array([epid, ampid, signid])
    
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
    
    def set_psrpos(self, chainbase, psrname):
        psrnames = np.loadtxt('results/psrpos.txt', dtype=str, usecols=[0])
        positions = np.loadtxt('results/psrpos.txt', usecols=[1,2,3,4,5,6])
        
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

        if self._high is None:
            self._high = np.array([positions[ind[0],3], positions[ind[0],5]])

    def boundpars(self, pars):
        pars = np.array(pars).copy()
        pars[0] = max(self._low[0], pars[0])
        pars[1] = max(self._low[1], pars[1])
        pars[0] = min(self._high[0], pars[0])
        pars[1] = min(self._high[1], pars[1])
        return pars
                
    def pdf(self, pars, pos=True):
        pars = self.boundpars(pars)
        return self.kde_pos(pars)[0] if pos else self.kde_neg(pars)[0] * np.exp(self._norm)
    
    def logpdf(self, pars, pos=True):
        pars = self.boundpars(pars)
        return np.log(self.kde_pos(pars)[0]) if pos else np.log(self.kde_neg(pars)[0]) + self._norm

class bwmArray(object):
    
    def __init__(self, resultsdir, burnin=10000, psrlist=None,
            low=[53000.0, -18.0], high=[55000.0, -10.0],
            incMonopole=False, incDipole=False, incQuadrupole=True):
        self.bwmPsrs = []
        
        for infile in glob.glob(os.path.join(resultsdir, '[BJ]*')):
            psrname = os.path.basename(infile)

            if psrlist is None or psrname in psrlist:
                chaindir = os.path.join(resultsdir, psrname)
                
                self.bwmPsrs.append(bwmPsrResult(chaindir, burnin=burnin))

        self.model = []
        if incMonopole:
            self.model.append('monopole')

        if incDipole:
            self.model.append('dipole')

        if incQuadrupole:
            self.model.append('quadrupole')

        self.setPriors(low, high)
            

    def setPriors(self, low, high):
        """
        Set the prior bounds on all the model parameters
        """
        startepoch = 0.5*(low[0]+high[0])
        startamp = 0.5*(low[1]+high[1])
        startraj = 0.1
        startdecj = 0.1
        startpol = 0.1

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

        if 'dipole' in self.model:
            pmin += [low[1], 0.0, -0.5*np.pi, 0.0]
            pmax += [high[1], 2*np.pi, 0.5*np.pi, 2*np.pi]
            pstart += [startamp, startraj, startdecj, startpol]
            pwidth += [0.1, 0.1, 0.1, 0.1]

        if 'quadrupole' in self.model:
            pmin += [low[1], 0.0, -0.5*np.pi, 0.0]
            pmax += [high[1], 2*np.pi, 0.5*np.pi, np.pi]
            pstart += [startamp, startraj, startdecj, startpol]
            pwidth += [0.1, 0.1, 0.1, 0.1]

        self.pmin = np.array(pmin)
        self.pmax = np.array(pmax)
        self.pstart = np.array(pstart)
        self.pwidth = np.array(pwidth)

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

            if 'monopole' in self.model:
                mono_ap = self.MonopoleAntennaPattern()
                monoamp = mono_ap * 10**pars[index]
                index += 1
            else:
                monoamp = 0.0

            if 'dipole' in self.model:
                dip_ap = self.DipoleAntennaPattern(bwmPsr.raj, bwmPsr.decj,
                        pars[index+1], pars[index+2], pars[index+3])
                dipamp = dip_ap * 10**pars[index]
                index += 4
            else:
                dipamp = 0.0

            if 'quadrupole' in self.model:
                quad_ap = self.QuadrupoleAntennaPattern(bwmPsr.raj, bwmPsr.decj,
                        pars[index+1], pars[index+2], pars[index+3])
                quadamp = quad_ap * 10**pars[index]
                index += 4
            else:
                quadamp = 0.0

            amp = monoamp + dipamp + quadamp
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
        return self._low + x * (self._high - self._low)
    
    def logposterior(self, pars):
        return self.logprior(pars) + self.loglik(pars)




class bwmPsrResult2S(object):
    
    def __init__(self, chaindir_pos, chaindir_neg, burnin=10000,
                low=None, high=None):
        # Initialize a bwm pulsar, with two chains
        if not os.path.isdir(chaindir_pos):
            raise IOError("Not a valid directory:", chaindir_pos)
        if not os.path.isdir(chaindir_neg):
            raise IOError("Not a valid directory:", chaindir_neg)

        self._low = low
        self._high = high
        self._norm = 0.0

        # Positive and negative BWM chains
        self.chain_pos, self.lnprob_pos = self.get_bwmchain(chaindir_pos, burnin=burnin)
        self.chain_neg, self.lnprob_neg = self.get_bwmchain(chaindir_neg, burnin=burnin)

        # Get the pulsar positions, and the BWM bounds
        psrname = os.path.basename(chaindir_pos.rstrip('/'))[:-4]
        chainbase = os.path.dirname(chaindir_pos)
        self.set_psrpos(chainbase, psrname)

        # Create the kdes
        self.kde_neg = bounded_kde.Bounded_kde_md(self.chain_neg.T,
                        low=self._low, high=self._high)
        self.kde_pos = bounded_kde.Bounded_kde_md(self.chain_pos.T,
                        low=self._low, high=self._high)

        # Normalize the two halfs
        self.normalize(Ntrial=100)

    def get_bwminds(self, stype, labels):
        bwminds = np.where(np.array(stype) == 'psrbwm')[0]
        ampid = bwminds[np.array(labels)[bwminds] == 'amplitude'][0]
        epid = bwminds[np.array(labels)[bwminds] == 'burst-arrival'][0]

        return np.array([epid, ampid])
    
    def get_bwmchain(self, chaindir, burnin=10000):
        (ll, lp, chain, labels, pulsarid, pulsarname, stype, mlpso, mlpsopars) = \
            pic.ReadMCMCFile(chaindir, incextra=True)

        inds = self.get_bwminds(stype, labels)
        return chain[burnin:, inds], lp[burnin:]
    
    def set_psrpos(self, chainbase, psrname):
        psrnames = np.loadtxt('results/psrpos.txt', dtype=str, usecols=[0])
        positions = np.loadtxt('results/psrpos.txt', usecols=[1,2,3,4,5,6])
        
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

        if self._high is None:
            self._high = np.array([positions[ind[0],3], positions[ind[0],5]])

    def normalize(self, Ntrial=100):
        """Now that we have the kde, we need to normalize the two halfs to match each other
        
        NOTE: This is a pretty dirty hack. What we actually need to do, of
              course, is to use the evidence to weight the two halves. But since
              we don't have that, this will have to do.
        """
        # Epoch, amplitude
        x = np.linspace(self._low[0], self._high[0], Ntrial)
        amp = self._low[1]
        p_neg = np.zeros_like(x)
        p_pos = np.zeros_like(x)

        for ii, epoch in enumerate(x):
            p_neg[ii] = self.kde_neg([epoch, amp])
            p_pos[ii] = self.kde_pos([epoch, amp])

        # Have to add _norm to the negative kde value
        self._norm = np.log(np.sum(p_pos)) - np.log(np.sum(p_neg))

    def boundpars(self, pars):
        pars = np.array(pars).copy()
        pars[0] = max(self._low[0], pars[0])
        pars[1] = max(self._low[1], pars[1])
        pars[0] = min(self._high[0], pars[0])
        pars[1] = min(self._high[1], pars[1])
        return pars
                
    def pdf(self, pars, pos=True):
        pars = self.boundpars(pars)
        return self.kde_pos(pars)[0] if pos else self.kde_neg(pars)[0] * np.exp(self._norm)
    
    def logpdf(self, pars, pos=True):
        pars = self.boundpars(pars)
        return np.log(self.kde_pos(pars)[0]) if pos else np.log(self.kde_neg(pars)[0]) - self._norm

class bwmArray2S(object):
    
    def __init__(self, resultsdir, burnin=10000, psrlist=None,
            low=[53000.0, -18.0], high=[55000.0, -10.0]):
        self.bwmPsrs = []
        
        for infile in glob.glob(os.path.join(resultsdir, '*-pos')):
            psrname = os.path.basename(infile)[:-4]

            if psrlist is None or psrname in psrlist:
                posdir = os.path.join(resultsdir, psrname+'-pos')
                negdir = os.path.join(resultsdir, psrname+'-neg')
                
                self.bwmPsrs.append(bwmPsrResult2S(posdir, negdir, burnin=burnin))
            
        # Set the prior bounds
        self._low = np.array([53000.0, -18.0, 0.0, -np.pi/2, 0.0])
        self._high = np.array([53000.0, -18.0, 2*np.pi, np.pi/2, np.pi])
        self._low[:2] = low
        self._high[:2] = high
    
    def DipoleAntennaPattern(self, rajp, decjp, raj, decj, pol):
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

        return np.cos(pol) * np.dot(nhat, p) + \
                np.sin(pol) * np.dot(mhat, p)

    def AntennaPattern(self, rajp, decjp, raj, decj, pol):
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
            Parameters of the BWM signal
            0) BWM epoch [mjd]
            1) BWM amplitude [log10(amp)]
            2) BWM raj [rad]
            3) BWM decj [rad]
            4) BWM pol [rad]
        """
        ll = 0.0
        for bwmPsr in self.bwmPsrs:
            ap = self.AntennaPattern(bwmPsr.raj, bwmPsr.decj, pars[2], pars[3], pars[4])

            s = np.sign(ap)
            lamp = max(np.log10(s*ap) + pars[1], self._low[1])

            ll += bwmPsr.logpdf([pars[0], lamp], s==1.0)
        
        return ll
    
    def logprior(self, pars):
        lp = 0.0
        if np.any(pars < self._low) or np.any(pars > self._high):
            lp = -np.inf
        
        return lp

    def prior_transform(self, x):
        return self._low + x * (self._high - self._low)
    
    def logposterior(self, pars):
        return self.logprior(pars) + self.loglik(pars)
