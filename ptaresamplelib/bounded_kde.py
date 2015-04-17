import numpy as np
from scipy.special import erf
from scipy.stats import gaussian_kde

class Bounded_kde(gaussian_kde):
    r"""Represents a one-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, low=None, high=None, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param low: The lower domain boundary.

        :param high: The upper domain boundary."""
        pts = np.atleast_1d(pts)

        assert pts.ndim == 1, 'Bounded_kde can only be one-dimensional'
        
        super(Bounded_kde, self).__init__(pts, *args, **kwargs)

        self._low = low
        self._high = high

    @property
    def low(self):
        """The lower bound of the domain."""
        return self._low

    @property
    def high(self):
        """The upper bound of the domain."""
        return self._high

    def evaluate(self, xs):
        """Return an estimate of the density evaluated at the given
        points."""
        xs = np.atleast_1d(xs)
        assert xs.ndim == 1, 'points must be one-dimensional'

        pdf = super(Bounded_kde, self).evaluate(xs)

        if self.low is not None:
            pdf += super(Bounded_kde, self).evaluate(2.0*self.low - xs)

        if self.high is not None:
            pdf += super(Bounded_kde, self).evaluate(2.0*self.high - xs)

        return pdf

    __call__ = evaluate


class Bounded_kde_md(gaussian_kde):
    r"""Represents a multi-dimensional Gaussian kernel density estimator
    for a probability distribution function that exists on a bounded
    domain."""

    def __init__(self, pts, low=None, high=None, truncate=True, *args, **kwargs):
        """Initialize with the given bounds.  Either ``low`` or
        ``high`` may be ``None`` if the bounds are one-sided.  Extra
        parameters are passed to :class:`gaussian_kde`.

        :param low: The lower domain boundary.

        :param high: The upper domain boundary.
        
        :param truncate: Truncate outside of low/high bounds"""
        pts = np.atleast_1d(pts)

        assert pts.ndim >= 1, 'Bounded_kde_md requires at least 1D data'
        
        super(Bounded_kde_md, self).__init__(pts, *args, **kwargs)

        self._low = np.atleast_1d(low)
        self._high = np.atleast_1d(high)
        self._low_val = np.zeros(pts.ndim)
        self._high_val = np.zeros(pts.ndim)
        self._truncate = truncate

        self._add_reflections(pts.ndim)

    def unique_rows(self, arr):
        """ Given array arr, return the unique rows of arr. Ugly, but fast"""
        b = np.ascontiguousarray(arr).view(np.dtype((np.void, arr.dtype.itemsize
                * arr.shape[1])))
        temp, idx = np.unique(b, return_index=True)
        return np.unique(b).view(arr.dtype).reshape(-1, arr.shape[1])


    def _add_reflections(self, ndim):
        """ Initialize all the reflections. _reflections is a 2D array of all the
        reflections we will be performing on the dataset. Note that this array
        grows exponentially with the number of 'mirrors'.

        Number of reflections is number of rows. Number of columns is number of
        dimensions

        Meaning of reflection: -1 (low-bound), 0 (nothing), 1 (high-bound)
        """
        self._reflections = np.zeros((1, ndim), dtype=np.int)

        for ii, xlow in enumerate(self._low):
            if xlow is not None:
                new_reflections = self._reflections.copy()
                new_reflections[:,ii] = -1
                self._reflections = np.append(self._reflections,
                        self.unique_rows(new_reflections), axis=0)
                self._low_val[ii] = xlow
            else:
                self._low_val[ii] = -np.inf

        for ii, xhigh in enumerate(self._high):
            if xhigh is not None:
                new_reflections = self._reflections.copy()
                new_reflections[:,ii] = 1
                self._reflections = np.append(self._reflections,
                        self.unique_rows(new_reflections), axis=0)
                self._high_val[ii] = xhigh
            else:
                self._high_val[ii] = np.inf

    @property
    def low(self):
        """The lower bound of the domain."""
        return self._low

    @property
    def high(self):
        """The upper bound of the domain."""
        return self._high

    def _m_mid(self, x):
        """ Selector: 1 if x == 0, 0 if x in [-1, 1] """
        return (1-x)*(x+1)

    def _m_low(self, x):
        """ Selector: 1 if x == -1, 0 if x in [0, 1] """
        return x*(x-1)/2

    def _m_high(self, x):
        """ Selector: 1 if x == 1, 0 if x in [-1, 0] """
        return x*(x+1)/2

    def evaluate(self, xs):
        """Return an estimate of the density evaluated at the given
        points."""
        xs = np.atleast_1d(xs)
        assert xs.ndim >= 1, 'points must be at least one-dimensional'

        pdf = super(Bounded_kde_md, self).evaluate(xs)
        for reflection in self._reflections[1:]:
            pdf += super(Bounded_kde_md, self).evaluate((
                    self._m_low(reflection) * (2.0*self._low_val - xs.T) +
                    self._m_mid(reflection) * xs.T +
                    self._m_high(reflection) * (2.0*self._high_val - xs.T)
                    ).T)

        # Truncate the pdf outside of low/high
        mult = np.ones_like(pdf)
        if self._truncate:
            # If 1D, do not transpose the data, since Python auto-casts arrays
            if len(xs.shape) == 1 and not len(self._low_val) > 1:
                mult = np.logical_and(xs >= self._low_val, xs <= self._high_val)
            else:
                mult = np.logical_and(np.all((xs.T >= self._low_val).T, axis=0),
                        np.all((xs.T <= self._high_val).T, axis=0))

        return pdf * mult

    __call__ = evaluate
