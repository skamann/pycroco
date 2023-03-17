import numpy as np


# Gaussian function:

class Gauss(object):

    def __init__(self):
        pass

    @staticmethod
    def initials(peak, x, x2):
        return peak, x, np.max([x2, 2.])

    @staticmethod
    def profile(x, peak, x0, var, continuum=0):
        """
        Define a 1dim. Gaussian profile with peak intensity peak, position x0,
        and standard deviation sigma. A continuum level may also be provided.

        Parameters
        ----------
        x : nd_array, 1dim.
            The x values for which the Gaussian should be calculated.
        peak : float
            The peak intensity of the Gaussian profile.
        x0 : float
            The line centre of the Gaussian in units of pixels.
        var: float
            The variance of the Gaussian in units of pixels.
        continuum : float, optional
            The continuum level on top of which the Gaussian is defined.

        Returns
        -------
        y : nd_array
            The values of the Gaussian profile for the provided pixels 'x'.
        """
        return continuum + peak*np.exp(-0.5*((x - x0)**2/var))

    @staticmethod
    def fwhm(x0, peak, var, **kwargs):
        # if kwargs:
        #     raise IOError("Unknown parameter(s) provide: {}".format(kwargs))
        if var <= 0 or np.isnan(var):
            return np.nan
        return 2.355 * np.sqrt(var)

    @staticmethod
    def vrad(x0, peak, var, **kwargs):
        # if kwargs:
        #     raise IOError("Unknown parameter(s) provide: {}".format(kwargs))
        return x0


class Moffat(object):

    def __init__(self):
        pass

    @staticmethod
    def initials(peak, x, x2):
        return peak, x, np.sqrt(max(2., x2)), 2.5

    @staticmethod
    def profile(x, peak, x0, r0, beta=2.5, continuum=0):
        """
        Define a 1dim. Moffat profile with peak flux peak, position x0, and
        effective radius r0. A continuum level and/or the kurtosis beta may
        also be provided.

        Parameters
        ----------
        x : nd_array, 1dim.
            The x values for which the Gaussian should be calculated.
        x0 : float
            The line centre of the Moffat.
        peak : float
            The peak flux of the Moffat profile.
        r0 : float
            The effective radius of the Moffat.
        beta : float, optional
            The kurtosis parameter of the Moffat profile.
        continuum : float, optional
            The continuum level on top of which the Moffat is defined.

        Returns
        -------
        y : nd_array
            The values of the Moffat profile for the provided pixels 'x'.
        """
        return continuum + peak*(1. + ((x - x0)/r0)**2)**(-beta)

    @staticmethod
    def fwhm(**kwargs):
        # if kwargs:
        #     raise IOError("Unknown parameter(s) provide: {}".format(kwargs))
        return 2. * np.sqrt(2 ** (1. / kwargs["beta"]) - 1.) * kwargs["r0"]

    @staticmethod
    def vrad(**kwargs):
        # if kwargs:
        #     raise IOError("Unknown parameter(s) provide: {}".format(kwargs))
        return kwargs["x0"]
