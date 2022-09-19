import numpy, numpy as np
cimport numpy, numpy as np
import cython

import math

from libc.math cimport exp, log

cdef class scorer:
    cdef int gsize
    cdef int fsize
    
    cdef double [:] ws
    cdef double [:,:] ms
    cdef double [:,:,:] invcs
    cdef double [:] norms 

    cdef double [:] y
    
    def __cinit__(self,gmm):
        cdef double [:,:] cinv
        
        self.gsize = gmm.means_.shape[0]
        self.fsize = gmm.means_.shape[1]
        
        self.ws = gmm.weights_
        self.ms = gmm.means_

        self.invcs = np.zeros(gmm.covars_.shape,'d')
        for i,c in enumerate(gmm.covars_):
            cinv = np.linalg.inv(c)
            self.invcs[i,:,:] = cinv
        
        self.norms = np.zeros(gmm.weights_.shape,'d')
        for i,c in enumerate(gmm.covars_):
            self.norms[i] = 2 * math.pi * math.sqrt(np.linalg.det(c))
        
        # just a buffer
        self.y = np.zeros(self.fsize)
        
    def __call__(self,xs):        
        cdef int i, j, k, l
        cdef int size = len(xs)
        cdef double p, nquad

        cdef double [:] x = np.zeros(self.fsize,'d')
        cdef double [:] res = np.zeros(size,'d')
        
        for i in range(size):
            p = 0.0
            x = np.asarray(xs[i])
            
            for j in range(self.gsize):
                nquad = 0.0
                for k in range(self.fsize):    
                    for l in range(self.fsize):
                        nquad = nquad + (x[k] - self.ms[j,k]) * self.invcs[j,k,l] * (x[l] - self.ms[j,l]) 
                
                p = p + self.ws[j] * exp(-0.5 * nquad) / self.norms[j]
            
            res[i] = log(p)
        
        return np.asarray(res)

cdef class scorer2:
    cdef int gsize
    cdef int fsize

    cdef double [:] ws
    cdef double [:,:] ms
    cdef double [:,:,:] invcs
    cdef double [:] norms

    cdef double [:] z

    def __cinit__(self,gmm):
        cdef double [:,:] cinv
        cdef int i

        self.gsize = gmm.means_.shape[0]
        self.fsize = gmm.means_.shape[1]

        self.ws = gmm.weights_
        self.ms = gmm.means_

        self.invcs = np.zeros(gmm.covars_.shape,'d')
        for i,c in enumerate(gmm.covars_):
            cinv = np.linalg.inv(c)
            self.invcs[i,:,:] = cinv

        self.norms = np.zeros(gmm.weights_.shape,'d')
        for i,c in enumerate(gmm.covars_):
            self.norms[i] = 2 * math.pi * math.sqrt(np.linalg.det(c))

        # just buffers
        self.z = np.zeros(self.fsize,'d')

    @cython.boundscheck(False)
    def __call__(self,numpy.ndarray[double,ndim=2] xs):
        cdef int i, j, k, l
        cdef int size = len(xs)
        cdef double p, nquad

        cdef numpy.ndarray[double,ndim=1] res = np.zeros(size,'d')

        for i in range(size):
            p = 0.0

            for j in range(self.gsize):
                for k in range(self.fsize):
                    self.z[k] = xs[i,k] - self.ms[j,k]

                nquad = 0.0
                for k in range(self.fsize):
                    nquad = nquad + self.z[k] * self.invcs[j,k,k] * self.z[k]
                    for l in range(k+1,self.fsize):
                        nquad = nquad + 2 * self.z[k] * self.invcs[j,k,l] * self.z[l]

                p = p + self.ws[j] * exp(-0.5 * nquad) / self.norms[j]

            res[i] = log(p)

        return res

def compile_gmm_v1(gmm):
    #gmm.cscore2 = scorer(gmm)
    gmm.score = scorer(gmm)

def compile_gmm(gmm):
    #gmm.cscore2 = scorer2(gmm)
    gmm.score = scorer2(gmm)
