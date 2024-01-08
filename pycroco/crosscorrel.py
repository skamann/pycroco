import collections
import logging
import warnings
import pandas as pd
from collections import OrderedDict
from enum import Enum, unique
from inspect import signature
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.signal import argrelextrema
from spexxy.utils.continuum import MaximumBin, SigmaClipping

from .spexxy.pycrocospectrum import PyCrocoSpectrum
from .filters import ramp
from .peak_functions import *


class CrossCorrel(object):
    """
    The class measures the radial velocities in a set of spectra by cross-
    correlating them against one or more templates.
    """

    @unique
    class ContinuumMode(Enum):
        Constant = 0
        MaximumBin = 1
        SigmaClipping = 2

    @unique
    class FitFunction(Enum):
        Gauss = 0
        Moffat = 1
        DoubleGauss = 2

    @unique
    class Filter(Enum):
        Fourier = 0
        Gauss = 1

    def __init__(self, logger=None):
        """
        Initialize a new instance of the CrossCorrel class.

        Parameters
        ----------

        """
        # logger
        self.logger = logger
        if self.logger is None:
            self.logger = logging.getLogger()
        self.logger.info("Initialising cross-correlation routine...")

        self.templates = collections.OrderedDict()
        self.spectra = collections.OrderedDict()

        self.results = pd.DataFrame(columns=['vrad', 'vrad_err', 'peak', 'fwhm', 'r_cc', 'success'],
                                    index=pd.MultiIndex(levels=[[], []], codes=[[], []],
                                                        names=['template_id', 'spectrum_id']))

        self.cc_functions = {}
        self.cc_inputs = {}
        self.cc_template = None
        self.cc_spectrum = None

    def add(self, spectrum, template=False, id=None, continuum_mode=ContinuumMode.SigmaClipping, continuum_kwargs=None):
        """
        Add a spectrum or a template.

        Parameters
        ----------
        spectrum: Instance of spexxy.data.Spectrum
            The spectrum to be added.
        template : bool, optional
            Flag indicating if the provided spectrum should be used as a
            template.
        id : int, optional
            The ID assigned to the spectrum. If None is provided, IDs are
            assigned consecutively starting from 0 for both templates and
            spectra.
        continuum_mode : ContinuumMode, optional
            The method used to determine the continuum of the spectrum.
        continuum_kwargs : dict, optional
            A set of keywords to pass to the continuum determination.
        """
        # check if input valid
        spectrum = PyCrocoSpectrum(spectrum, copy_flux=True)

        if id is None:
            if template:
                id = 1 if len(self.templates) == 0 else max(self.templates.keys()) + 1
                self.logger.info("Adding template with ID #{0} ...".format(id))
            else:
                id = 1 if len(self.spectra) == 0 else max(self.spectra.keys()) + 1
                self.logger.info("Adding spectrum with ID #{0} ...".format(id))

        # check if full range is used
        if spectrum.valid is not None:
            if spectrum.valid.all():
                self.logger.error("... data does not contain valid pixels.")
                return None

            # remove masked leading or trailing pixels (if any)
            if spectrum.valid[0] or spectrum.valid[-1]:
                i_min, i_max = np.flatnonzero(~spectrum.valid)[[0, -1]]
                spectrum = spectrum.extract_index(i_min, i_max)

        # fit continuum
        if continuum_kwargs is None:
            continuum_kwargs = {}
        if continuum_mode not in CrossCorrel.ContinuumMode:
            raise IOError("Unsupported value for 'continuum_mode': {}".format(continuum_mode))
        elif continuum_mode == CrossCorrel.ContinuumMode.SigmaClipping:
            continuum_fit = SigmaClipping(**continuum_kwargs)
            continuum = continuum_fit(spectrum.wave, spectrum.flux, valid=~spectrum.valid)
        elif continuum_mode == CrossCorrel.ContinuumMode.MaximumBin:
            continuum_fit = MaximumBin(**continuum_kwargs)
            continuum = continuum_fit(spectrum.wave, spectrum.flux, valid=~spectrum.valid)
        else:
            continuum_fit = lambda x, y: continuum_kwargs['c'] * np.ones_like(y)
            continuum = continuum_fit(spectrum.wave, spectrum.flux)

        # still no continuum?
        if continuum is None:
            self.logger.error('... could not calculate continuum.')
            return None
        # add to dictionary of spectra/templates
        spectrum.flux = spectrum.flux / continuum - 1.
        if template:
            self.templates[id] = spectrum
        else:
            self.spectra[id] = spectrum

    @staticmethod
    def apodize_edges(data, mask, apodize=0.2, threshold=3):
        """
        Mask out regions in a 1dim. array and apodize the edges around the
        masked regions.

        Parameters
        ----------
        data : nd_array
            The data on which the masking should be performed.
        mask : nd_array
            A boolean array of the same size as 'data' set to one for pixels
            that are invalid.
        apodize : float, optional
            The fraction at the edge of each valid range that is apodized with
            a cosine filter.
        threshold : int, optional
            The minimum size of a gap (in pixels) around which the data is
            apodized.

        Returns
        -------
        out_data : nd_array
            The masked and apodized data.
        """
        # get pixels where a valid region starts or ends
        # even indices contain pixels where a valid range stars, odd indices those where an invalid range starts
        indices = np.flatnonzero(np.bitwise_xor(np.r_[~mask, False], np.r_[False, ~mask]))

        # if start/end pixels are masked, set to zero
        out_data = np.copy(data)
        out_data[:indices[0]] = out_data[indices[-1]:] = 0

        # set masked ranges inside array to zero
        for i in range(2, indices.size, 2):
            start = indices[i - 1]
            stop = indices[i]
            out_data[start:stop] = 0

        # prior to apodization, remove masked ranges with length below threshold
        above_threshold = np.ones_like(indices, dtype=bool)
        for i in range(1, indices.size - 1, 2):
            if (indices[i + 1] - indices[i]) < threshold:
                above_threshold[i] = above_threshold[i + 1] = False
        indices = indices[above_threshold]

        # apodize beginning/end of each valid range using cosine function
        for i in range(0, indices.size - 1, 2):

            # pixel range affected by apodization scales with of valid data window
            n = int(apodize * (indices[i + 1] - indices[i]))

            if n > 0:
                # apodize beginning of range
                start = indices[i]
                stop = start + n
                out_data[start:stop] *= 0.5 * (1 + np.cos(np.pi * np.linspace(-1, 0, n)))

                # apodize end of range
                stop = min(indices[i + 1], data.size)
                start = stop - n
                out_data[start:stop] *= 0.5 * (1 + np.cos(np.pi * np.linspace(0, 1, n)))

        return out_data

    def fourier_filter(self, spectrum, ramp_on=4, full_on=8, full_off=None, ramp_off=None, apodize=0.2, **kwargs):

        if len(kwargs) > 0:
            raise IOError('Unknown parameters provided to CrossCorrel.fourier_filter: {}'.format(kwargs))

        tmp = self.apodize_edges(spectrum.flux, spectrum.valid, apodize=apodize)

        if full_off is None:
            full_off = len(tmp) // 3
        if ramp_off is None:
            ramp_off = len(tmp) // 2

        #  Use default values for Fourier filter parameters is none provided
        self.logger.info('... FFT-filtering with parameters {}, {}, {}, {} ...'.format(
            ramp_on, full_on, full_off, ramp_off))
        try:
            _tmp = ramp(spectrum.flux, ramp_on, full_on, full_off, ramp_off)
        except ValueError:
            self.logger.error('... FFT-filtering failed')
        else:
            tmp = _tmp

        spectrum.flux = tmp

    def gaussian_filter(self, spectrum, fwhm=None, **kwargs):

        if len(kwargs) > 0:
            raise IOError('Unknown parameters provided to CrossCorrel.fourier_filter: {}'.format(kwargs))

        if fwhm is None:
            fwhm = 3. * spectrum.wave_step

        spectrum.smooth(fwhm=fwhm)

    @staticmethod
    def _resample_valid(valid, inwave, outwave):
        assert ~valid[0]
        assert ~valid[-1]

        outvalid = np.zeros(len(outwave), dtype=bool)

        block_indices = np.argwhere(np.diff(valid)).squeeze()
        n_blocks = len(block_indices) // 2
        for i in range(n_blocks):
            j_min, j_max = block_indices[2 * i], block_indices[2 * i + 1]
            outvalid[(outwave >= inwave[j_min]) & (outwave <= inwave[j_max])] = True

        return outvalid

    @staticmethod
    def correlate(spectrum, template, stepsize=1, offset=0, velocity=False):
        """
        Correlate a spectrum with a template.

        It is assumed that both the spectrum and the template are sampled
        homogeneously in log-space and have the same stepsize.  However, the
        two arrays do not need to have the same size or start at the same
        (log x)-value .

        Parameters
        ----------
        spectrum : array_like
            The spectrum used in the cross-correlation.
        template : array_like
            The template used in the cross-correlation.
        stepsize : float, optional
            The common log-sampling of spectrum and template. Only required
            when converting to velocity space (see below).
        offset : float, optional
            The difference between in (log x) starting values between
            spectrum and template. Only required when converting to velocity
            space (see below).
        velocity : bool, optional
            Flag indicating if the cross-correlation function should be
            converted to velocity space. To do so, the `stepsize` must be set
            to the sampling of spectrum and template in log10(dlambda/lambda)
            space and the `offset` must be provided as
            log10(lambda_spectrum[0]) - log10(lambda_template[0])

        Returns
        -------
        corr : pd.Series
            The returned pandas Series will contain the normalized cross-
            correlation function as values while the abscissa (in pixel or
            velocity space) is used as index.
        """

        # normalization of cross-correlation following suggestion under
        # http://stackoverflow.com/questions/5639280/why-numpy-correlate-and-corrcoef-return-different-values-\
        # and-how-to-normalize
        # a = (a - np.mean(a)) / (np.std(a) * len(a))
        # v = (v - np.mean(v)) /  np.std(v)
        # (spectra are already continuum subtracted, so omit subtraction of mean)
        a = template / (np.std(template) * len(template))
        v = spectrum / np.std(spectrum)

        # pad template with zeros on each side, using half-length of spectrum. This is to ensure that the zero-shift
        # of the cross-correlation always corresponds to the same pixel in the output array
        pad = len(v)//2
        _a = np.pad(a, (pad, pad))

        # perform correlation, resulting array has length max(len(a), len(v)) # len(a)+len(v)-1
        corr12 = np.correlate(_a, v, "valid")

        # get abscissa for correlation function (incl. right zero point)
        xdata = np.arange(corr12.size, dtype=np.float64) - pad
        xdata *= -1
        # xdata -= max(a.size, v.size) // 2  # when self-correlating an array, the peak is at the central array pixel
        xdata *= stepsize  # change to log-wavelength space

        # account for different starting wavelengths of template and spectrum
        xdata += offset

        # change to velocity space
        if velocity:
            xdata = 3e5 * (10 ** xdata - 1.)

        return pd.Series(corr12, index=xdata)

    # def fit_peak_zucker(self, signal):
    #     from numpy.polynomial import Polynomial
    #
    #     x_cc = signal.index
    #     y_cc = signal.values
    #     center_pixels = [1,2,3]
    #     p = Polynomial.fit(x_cc[center_pixels], y_cc[center_pixels], deg=2)
    #     pi = p.deriv(m=1)
    #     i_peak = pi.roots()[0]
    #     pii = p.deriv(m=2)
    #     sig2 = (-len(signal)*pii(i_peak)/p(i_peak)*p(i_peak)**2/(1.-p(i_peak)**2))**-1
    #     print(sig2)

    def fit_peak(self, signal, search_center=None, search_width=None, fit_width=None, function=FitFunction.Gauss):

        x_cc = signal.index
        y_cc = signal.values

        if search_center is None:
            search_center = x_cc[y_cc.argmax()]

        if search_width is None:
            # find nearest minima around cross-correlation peak that are <0. They define the fitting region
            cc_minima = argrelextrema(y_cc, np.less)[0]
            cc_minima = cc_minima[y_cc[cc_minima] < 0]
            i_min = cc_minima[x_cc[cc_minima] < search_center].max()
            i_max = cc_minima[x_cc[cc_minima] > search_center].min()
            search_pixels = np.zeros(len(x_cc), dtype=bool)
            search_pixels[i_min:i_max] = True
        else:
            # select pixels in search window
            search_pixels = (x_cc > (search_center - search_width)) & (x_cc < (search_center + search_width))

        if fit_width is None:
            fit_width = (x_cc[search_pixels].max() - x_cc[search_pixels].min()) / 2.

        # find index of pixel that contains maximum of signal in search window
        i_peak = np.flatnonzero(search_pixels)[0] + y_cc[search_pixels].argmax()

        # define window around this pixel for fitting cross-correlation peak
        center_pixels = (x_cc >= (x_cc[i_peak] - fit_width)) & (x_cc <= (x_cc[i_peak] + fit_width))

        # perform moment calculation for initial guesses
        total = float(y_cc[center_pixels].sum())
        x = np.sum(x_cc[center_pixels] * y_cc[center_pixels]) / total
        x2 = np.sum(y_cc[center_pixels] * (x_cc[center_pixels] - x) ** 2) / total
        if x2 < (0.2*np.diff(x_cc[center_pixels]).min())**2:
            x2 = (0.2*np.diff(x_cc[center_pixels]).min())**2

        if function == CrossCorrel.FitFunction.Gauss:
            fct = Gauss
        elif function == CrossCorrel.FitFunction.Moffat:
            fct = Moffat
        else:
            raise IOError(f'Unknown fitting function: {function}.')

#         guesses = fct.initials(y_cc[center_pixels].max(), x, x2)
        guesses = fct.initials(y_cc[center_pixels].max(), x_cc[center_pixels][y_cc[center_pixels].argmax()], x2)

        with warnings.catch_warnings():
            warnings.simplefilter('error', OptimizeWarning)
            try:
                popt, pcov = curve_fit(fct.profile, x_cc[center_pixels], y_cc[center_pixels], p0=guesses)
            except (RuntimeError, ValueError, OptimizeWarning):
                self.logger.error(f"Peak fit failed on data: {y_cc[center_pixels]} with initial guesses {guesses}")
                popt = [np.nan] * len(guesses)
                pcov = None
                success = False
            else:
                success = True

        parameters = list(signature(fct.profile).parameters.keys())
        result = OrderedDict(zip(parameters[1:1 + len(popt)], popt))

        # only accept velocities if the peak of the Gaussian fit is inside the fit window
        if 'x0' in result.keys() and not x_cc[center_pixels].min() < result['x0'] < x_cc[center_pixels].max():
            success = False

        result['fwhm'] = fct.fwhm(**result)
        result['vrad'] = fct.vrad(**result)

        return success, result

    def calculate_rcc(self, signal, x_peak, y_peak, window_size=None):

        x_cc = signal.index
        y_cc = signal.values

        # i_c - index of pixel closest to peak
        # window_size - half-length of log-template or -spectrum, depending on which one is shorter
        i_c = abs(x_cc - x_peak).argmin()
        if window_size is None:
            window_size = min(i_c, y_cc.size - i_c)

        sigma_a2 = np.sum((y_cc[i_c:i_c + window_size] - y_cc[i_c:i_c - window_size:-1]) ** 2) / (2 * window_size)

        # Cross correlation statistics r based on Tonry & Davis(1979): ADS: 1979AJ.....84.1511T
        r_cc = max(y_peak / np.sqrt(2 * sigma_a2), 0.)
        return r_cc

    def __call__(self, filtertype=None, filterpars=None, search_width=1000., search_center=0,
                 fit_peak=True, fit_width=200., fit_function=FitFunction.Gauss, full_output=False, **kwargs):
        """
        Cross correlate the spectra loaded into the instance against the
        templates in the instance.

        Each template is resampled to log-space unless it has been provided
        log-sampled. To prepare each spectrum for the cross correlation, it is
        resampled to the same log binning as the template (same step size and
        an integer offset between the sampling points). Both arrays are
        normalized before the correlation is carried out. The position of the
        cross-correlation peak is translated into a velocity offset
        afterwards. The confidence interval around the measured velocity is
        determined using the r_cc statistics of Tonry & Davis(1979).

        Literature:
        Tonry & Davis(1979), ADS: 1979AJ.....84.1511T
        Kurtz & Mink (1998), ADS: 1998PASP..110..934K

        Parameters
        ----------
        filtertype : instance of CrossCorrel.Filter, optional
            If set to true, the continuum subtracted spectra are multiplied
            with a ramp-filter in fourier space.
        filterpars : dict, optional
            The parameters of filtering the spectra: For a Fourier filter,
            the filter shape is defined by `rampon`, `fullon`, `fulloff`, and
            `rampoff`. Also, `apodize` can be used to control the fraction at
            the edge of each valid range that is apodized with a cosine filter
            prior to Fourier filtering.
            See https://arxiv.org/pdf/0912.4755.pdf for meaning of parameters.
        search_width : float, optional
            The half width of the window in which the peak of the cross-
            correlation function is searched, given in km/s.
        search_center : float,  optional
            The center of the window in which the peak of the cross-
            correlation function is searched, given in km/s.
        fit_peak : boolean, optional
            Flag indicating if the peak of the cross-correlation curve should
            be fitted with an analytic function.
        fit_width : float or array like, optional
            The half width of the window around the detected correlation peak
            that is fitted with an analytic function to precisely determine
            the position of the peak.
        fit_function : instance of CrossCorrel.FitFunction, optional
            The analytical function used to fit the correlation peak.
            Currently supported are 'gaussian' and 'moffat'.
        full_output : boolean, optional
            Flag indicating if the cross-correlation functions should be
            returned together with the analysis results.

        Returns
        -------
        results : pandas.DataFrame
            All the results from the correlation are stored in a single data
            frame containing the following columns.
            * template_id - The ID of the template
            * spectrum_id - The ID of the spectrum
            * v_rad - The measured radial velocity in km/s
            * v_rad err - The uncertainty of the measured radial velocity as
                          defined by Kurtz & Mink (1998)
            * peak - The height of the correlation peak
            * fwhm -  The full width at half maximum of the correlation peak,
                      measured in km/s
            * r_cc - The r_cc statistics as defined by Tonry & Davis (1979)
            * success - Boolean flag indicating if the cross-correlation peak
                        has been detected.
        cc_functions : dictionary
            Only if the parameter full_output is set to True, a dictionary
            containing a pandas Series per template_spectrum pair with the
            cross-correlation signal (and the velocity of each pixel as Index)
            is returned.
        cc_inputs : dictionary
            Only if the parameter full_output is set to True, a dictionary
            containing a tuple per template_spectrum pair with the spectra
            as they entered the correlation is returned.
        """
        if kwargs:
            raise IOError('Unknown parameters provided to call function: {}'.format(kwargs))

        self.logger.info("Starting correlation with following parameters:")
        self.logger.info("  Center of search window [searchcenter]: {0} km/s".format(search_center))
        self.logger.info("  Half width of search window [searchwidth]: {0} km/s".format(search_width))
        self.logger.info("  Half width of fit window [fitwidth]: {0} km/s".format(fit_width))

        if not isinstance(fit_function, CrossCorrel.FitFunction):
            raise IOError("  Type '{}' not supported for parameter fit_function".format(type(fit_function)))
        self.logger.info("  Peak fitting function: {0}".format(fit_function))

        if not self.templates:
            self.logger.error("Can't do. No template(s) provided.")
            return None, None

        if not self.spectra:
            self.logger.error("Can't do. No spectra provided.")
            return None, None

        # if no parameters are provided for Fourier filtering, set to default values
        if filterpars is None:
            filterpars = {}
        if filtertype is None:
            filterfunc = None
        elif filtertype == CrossCorrel.Filter.Fourier:
            filterfunc = lambda x: self.fourier_filter(x, **filterpars)
        elif filtertype == CrossCorrel.Filter.Gauss:
            filterfunc = lambda x: self.gaussian_filter(x, **filterpars)
        else:
            raise IOError("Unknown filtertype provided: {}".format(filtertype))

        for template_id, _template in self.templates.items():
            self.logger.info("Working with template #{0} ...".format(template_id))

            # check if template already log-binned
            if not _template.wave_mode == PyCrocoSpectrum.Mode.LOG10LAMBDA:
                _template.mode(PyCrocoSpectrum.Mode.LOG10LAMBDA)
            # use constant sampling in log-space
            if _template.wave_step == 0.:
                template = _template.resample_const(step=(_template.wave[1:] - _template.wave[:-1]).min())
                template.valid = CrossCorrel._resample_valid(_template.valid, _template.wave, template.wave)
                template.valid = template.valid | np.isnan(template.flux)
            else:
                template = _template

            self.logger.info("... using sampling of {} in log10-space.".format(template.wave_step))

            # apply filter if requested
            if filterfunc is not None:
                filterfunc(template)

            final_template_flux = np.where(template.valid, 0, template.flux)
            self.cc_template = pd.Series(final_template_flux, index=10 ** template.wave)

            for spectrum_id, _spectrum in self.spectra.items():
                self.logger.info("Working on spectrum with ID #{0} ...".format(spectrum_id))

                # need to resample spectrum on same wavelength grid as template:
                self.logger.debug("... rebinning spectrum to match sampling of template ...")

                # use same sampling as for template in log-space
                _spectrum.mode(PyCrocoSpectrum.Mode.LOG10LAMBDA)
                # define number of pixels - note that applying .min()/.max() to pd.Series skips NaNs by default
                npix = int((np.nanmax(_spectrum.wave) - np.nanmin(_spectrum.wave)) / template.wave_step)
                # make sure difference in starting wavelengths is integer of selected sampling, i.e. template
                # and spectrum are shifted by an integer number of pixels
                n = int((np.nanmin(_spectrum.wave) - template.wave_start) / template.wave_step)
                wave_start = template.wave_start + (n + 1) * template.wave_step

                # perform interpolation
                try:
                    spectrum = _spectrum.resample(wave_start=wave_start, wave_step=template.wave_step, wave_count=npix,
                                                  fill_value=0.)
                except TypeError:
                    # if interpolation fails, it usually returns a TypeError, caused by Spectrum.indices_of_wave_range()
                    # returning None. In that case,an empty spectrum is used.
                    logging.error('Cannot resample spectrum. Check input data.')
                    spectrum = PyCrocoSpectrum(wave_start=wave_start, wave_step=template.wave_step, wave_count=npix)
                else:
                    spectrum.valid = CrossCorrel._resample_valid(_spectrum.valid, _spectrum.wave, spectrum.wave)

                # apply FFT filter if requested
                if filterfunc is not None:
                    # mask out invalid regions and apodize edges
                    filterfunc(spectrum)
                spectrum.valid = spectrum.valid | np.isnan(spectrum.flux)

                final_spectrum_flux = np.where(spectrum.valid, 0, spectrum.flux)
                self.cc_spectrum = pd.Series(final_spectrum_flux, index=10 ** spectrum.wave)

                # if spectrum_id == 1.:
                # import matplotlib.pyplot as plt
                # plt.plot(template.wave, final_template_flux, 'g-')
                # plt.plot(spectrum.wave, final_spectrum_flux, 'r-')
                # plt.show()

                # perform cross correlation
                self.logger.debug("... calling correlation routine ...")
                corr = CrossCorrel.correlate(final_spectrum_flux, final_template_flux, stepsize=template.wave_step,
                                             offset=spectrum.wave_start - template.wave_start, velocity=True)

                if full_output:
                    self.cc_functions[(template_id, spectrum_id)] = corr
                    self.cc_inputs[(template_id, spectrum_id)] = (
                        pd.Series(final_template_flux, index=template.wave),
                        pd.Series(final_spectrum_flux, index=spectrum.wave))

                if fit_peak:
                    # get maximum by fitting max. peak within search window with a Gaussian
                    self.logger.debug("... locating correlation peak ...")

                    success, bestfit = self.fit_peak(corr, search_center=search_center, search_width=search_width,
                                                     fit_width=fit_width, function=fit_function)
                    self.results.loc[(template_id, spectrum_id), 'success'] = success
                    for key, value in bestfit.items():
                        self.results.loc[(template_id, spectrum_id), key] = value

                    if success:
                        self.logger.info("... Fit successful.")

                        window_size = min(len(final_spectrum_flux), len(final_template_flux)) >> 2
                        r_cc = self.calculate_rcc(signal=corr, x_peak=bestfit['x0'], y_peak=bestfit['peak'],
                                                  window_size=window_size)
                        self.results.loc[(template_id, spectrum_id), 'r_cc'] = r_cc

                        # error estimation based on Kurtz & Mink (1998), ADS: 1998PASP..110..934K
                        fwhm_vrad = self.results.loc[(template_id, spectrum_id), 'fwhm']
                        self.results.loc[(template_id, spectrum_id), 'vrad_err'] = (3. * fwhm_vrad) / (8. * (1. + r_cc))

                        self.logger.info("... Height of correlation peak: {0:.2f}".format(bestfit["peak"]))
                        self.logger.info("... FWHM of correlation peak: {0:.2f}".format(fwhm_vrad))
                        self.logger.info("... Measured radial velocity: {0:.1f} +/- {1:.1f} km/s".format(
                            bestfit["x0"], self.results.loc[(template_id, spectrum_id), "vrad_err"]))

                    else:
                        self.results.loc[(template_id, spectrum_id), 'r_cc'] = np.nan
                        self.results.loc[(template_id, spectrum_id), 'vrad_err'] = np.nan
                        self.logger.error("... correlation failed: No valid {} fit.".format(fit_function))

        if full_output:
            return self.results, self.cc_functions, self.cc_inputs
        else:
            return self.results, None, None

    def clean_spectra(self):
        """
        Resets the list of spectra.
        """
        self.spectra = {}

    def clean_templates(self):
        """
        Resets the list of templates.
        """
        self.templates = {}

    def clean_results(self):
        """
        Resets the DataFrame containing the cross-correlation results (as well
        as the dictionary containing the cross-correlation signals if they are
        saved).
        """
        self.results = pd.DataFrame(columns=self.results.columns, index=self.results.index)
        self.cc_functions = {}

    def save_results(self, filename):
        """
        Saves current results to a csv-file.

        Parameters
        ----------
        filename : str
            The name of file used to store the results.
        """

        self.logger.info("Saving results to file {0}.".format(filename))
        if self.results.size == 0:
            self.logger.error("Nothing to save, no results yet!")
            return

        self.results.to_csv(filename, index=False)
