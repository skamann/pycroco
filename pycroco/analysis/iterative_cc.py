import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from spexxy.data import FitsSpectrum, SpectrumFitsHDU

from ..crosscorrel import CrossCorrel
from ..spexxy import PyCrocoSpectrum
from ..utils.statistics import der_snr


class ProgressPlot(object):

    def __init__(self, fig=None):

        if fig is None or len(fig.axes) == 0:
            if fig is None:
                self.fig = plt.figure(figsize=(8, 8))
            else:
                self.fig = fig
            gs = gridspec.GridSpec(2, 2)
            self.ax_template = self.fig.add_subplot(gs[0, :])
            self.ax_time = self.fig.add_subplot(gs[1, 0])
            self.ax_comp = self.fig.add_subplot(gs[1, 1])
        else:
            assert len(fig.axes) >= 3
            self.fig = fig
            self.ax_template = fig.axes[0]
            self.ax_template.clear()
            self.ax_time = fig.axes[1]
            self.ax_time.clear()
            self.ax_comp = fig.axes[2]
            self.ax_comp.clear()

        self.time_series = None
        self.mean_line = None
        self.vel_text = None

        self.ax_time.set_xlabel(r'${\rm MJD-OBS}\,[{\rm d}]$', fontsize=16)
        self.ax_time.set_ylabel(r'$v_{\rm rad}\,[{\rm km\,s^{-1}}]$', fontsize=16)

        self.vel_string = r'$\bar{{v}}={0:.1f}\,{{\rm km\,s^{{-1}}}},\,\chi^2_{{\rm r}}={1:.2f}$'

    def plot_template(self, template, mask_ranges=None):

        self.ax_template.clear()
        if template.wave_mode == PyCrocoSpectrum.Mode.LAMBDA:
            wave = template.wave
        elif template.wave_mode == PyCrocoSpectrum.Mode.LOGLAMBDA:
            wave = np.e**template.wave
        else:
            wave = 10. ** template.wave
        self.ax_template.plot(wave, template.flux, ls='-', lw=1.5, c='C0', marker='None')

        if mask_ranges is not None:
            for wmin, wmax in mask_ranges:
                self.ax_template.axvspan(wmin, wmax, facecolor='C1', alpha=0.5)

        self.ax_template.set_xlabel(r'$\lambda\,[{\rm \AA}]$', fontsize=16)
        self.ax_template.set_ylabel(r'$f_\lambda\,[{\rm a.u.}]$', fontsize=16)
        self.fig.canvas.draw()
        self.fig.tight_layout()

    def plot_velocities(self, results, x='MJD-OBS'):
        results = results.dropna(subset=['vrad', 'vrad_err'])
        if len(results) == 0:
            return
        mean_v = np.average(results['vrad'], weights=1. / results['vrad_err'] ** 2)
        chi2 = np.sum(((results['vrad'] - mean_v) ** 2) / results['vrad_err'] ** 2)

        if self.mean_line is not None:
            self.mean_line.remove()
        if self.time_series is not None:
            markers, caps, bars = self.time_series
            markers.remove()
            for cap in caps:
                cap.remove()
            for bar in bars:
                bar.remove()
        if self.vel_text is not None:
            self.vel_text.remove()
        self.mean_line = self.ax_time.axhline(y=mean_v, ls='--', lw=1.5, c='C2')
        self.time_series = self.ax_time.errorbar(x=results[x], y=results['vrad'],
                                                 yerr=abs(results['vrad_err']), ls='None', lw=1.5, c='C2', marker='o',
                                                 mew=1.5, mec='C2', mfc='None', capsize=3)
        self.vel_text = self.ax_time.text(0.5, 0.05, self.vel_string.format(mean_v, chi2 / (len(results) - 1)),
                                          fontsize=16, transform=self.ax_time.transAxes, ha='center',
                                          va='baseline')
        self.ax_time.relim()
        self.ax_time.autoscale(True, True, True)
        self.fig.canvas.draw()
        self.fig.tight_layout()

    def plot_comparison(self, new_results, old_results):

        self.ax_comp.clear()
        self.ax_comp.errorbar(old_results['vrad'], new_results['vrad'], xerr=abs(old_results['vrad_err']),
                              yerr=abs(new_results['vrad_err']), ls='None', lw=1.5, c='C1', marker='D', mew=1.5,
                              mec='C1', mfc='C1', capsize=3)
        self.ax_comp.set_aspect('equal')

        self.ax_comp.set_xlabel(r'$v_{\rm Init}\,[{\rm km\,s^{-1}}]$', fontsize=16)
        self.ax_comp.set_ylabel(r'$v_{\rm Final}\,[{\rm km\,s^{-1}}]$', fontsize=16)
        self.fig.canvas.draw()
        self.fig.tight_layout()


class IterativeCC(object):

    def __init__(self, spectra, sort_by='SNRATIO', plot=False, fig=None, wave_range=None, mask_ranges=None,
                 tellurics_spectra=None, match_tellurics_by='MJD-OBS'):

        self.sort_by = sort_by
        self.plot = plot
        self.wave_range = wave_range
        self.mask_ranges = mask_ranges
        self.match_tellurics_by = match_tellurics_by

        if self.plot:
            self.progress_plot = ProgressPlot(fig=fig)
        else:
            self.progress_plot = None

        if tellurics_spectra is not None:
            self.tellurics_spectra = tellurics_spectra
        else:
            self.tellurics_spectra = None

        self.cc = CrossCorrel()

        if self.sort_by is None:
            spectra['snr'] = np.nan

        self._data = {}
        for index, row in spectra.iterrows():
            if self.tellurics_spectra is not None:
                tellurics = self.tellurics_spectra[spectra.at[index, self.match_tellurics_by]]
            else:
                tellurics = None
            self._data[index] = self._prepare_spectrum(row['Filename'], tellurics=tellurics)
            self.cc.add(self._data[index], id=index)
            if self.sort_by is None:
                spectra.at[index, 'snr'] = der_snr(self._data[index].flux)

        if self.sort_by is None:
            self.sort_by = 'snr'

        self.spectra = spectra.sort_values(by=self.sort_by, ascending=False)
        self.template_id = self.spectra[self.sort_by].idxmax()

        if self.tellurics_spectra is not None:
            tellurics = self.tellurics_spectra[self.spectra.at[self.template_id, self.match_tellurics_by]]
        else:
            tellurics = None
        self.template = self._prepare_spectrum(self.spectra.at[self.template_id, 'Filename'], tellurics=tellurics,
                                               mode=PyCrocoSpectrum.Mode.LOG10LAMBDA)
        self.template_variance = None

        if self.plot:
            self.progress_plot.plot_template(self.template, mask_ranges=self.mask_ranges)

    def _prepare_spectrum(self, filename, mode=None, remove_tellurics=True, tellurics=None):

        fs = FitsSpectrum(filename, 'r')

        if remove_tellurics:
            if tellurics is not None:
                ts = SpectrumFitsHDU(wave=tellurics[0], flux=tellurics[1], primary=False).resample(fs.spectrum)
                ts._primary = False
                fs['TELLURICS'] = ts
            if fs.tellurics is not None:
                fs.spectrum /= fs.tellurics
                del fs['TELLURICS']

        if self.wave_range is not None:
            spectrum = fs.spectrum.extract(w1=self.wave_range[0], w2=self.wave_range[1])
        else:
            spectrum = fs.spectrum

        if mode is not None:
            spectrum.mode(mode)
            if spectrum.wave_step == 0.:
                spectrum = spectrum.resample_const(step=(spectrum.wave[1:] - spectrum.wave[:-1]).min())

        spectrum.valid = np.isnan(spectrum)
        if self.mask_ranges is not None:
            self._apply_mask(spectrum)
        return PyCrocoSpectrum(spectrum)

    def _apply_mask(self, spectrum):
        if self.mask_ranges is None:
            return

        for mask_range in self.mask_ranges:
            if spectrum.wave_mode == SpectrumFitsHDU.Mode.LOG10LAMBDA:
                w1 = np.log10(mask_range[0])
                w2 = np.log10(mask_range[1])
            elif spectrum.wave_mode == SpectrumFitsHDU.Mode.LOGLAMBDA:
                w1 = np.log(mask_range[0])
                w2 = np.log(mask_range[1])
            else:
                w1, w2 = mask_range
            try:
                imin, imax = spectrum.indices_of_wave_range(w1=w1, w2=w2)
            except TypeError:  # None returned
                logging.error(f'Cannot mask range [{w1}, {w2}] as it is outside available spectral range.')
            else:
                spectrum.valid[imin:imax] = True

    def __call__(self, max_iterations=10, dv_threshold=0.1, combine_threshold=0.2, plotfilename=None, **kwargs):

        self.cc.add(self.template, template=True, id=self.template_id)
        _r = self.cc(**kwargs)[0]
        initial_results = _r.copy()
        self.spectra['vrad'] = initial_results.loc[self.template_id, 'vrad']
        self.spectra['vrad_err'] = initial_results.loc[self.template_id, 'vrad_err']
        self.spectra['success'] = initial_results.loc[self.template_id, 'success']
        if self.plot:
            self.progress_plot.plot_velocities(self.spectra, x=self.match_tellurics_by)

        current_results = initial_results
        for _ in range(max_iterations):

            spectra_to_combine = []
            weights = []
            for i, (index, row) in enumerate(self.spectra.iterrows()):
                if self.spectra.at[index, self.sort_by] < (combine_threshold*self.spectra.at[self.template_id, self.sort_by]):
                    continue
                spectrum = self._data[index]
                spectrum.mode(PyCrocoSpectrum.Mode.LOG10LAMBDA)
                spectrum = spectrum.resample(self.template)
                if row['success'] and current_results.loc[(self.template_id, index), 'r_cc'] > 5:
                    spectrum.redshift(vrad=-row['vrad'])
                spectra_to_combine.append(np.asarray(spectrum))
                weights.append(self.spectra.at[index, self.sort_by])

            self.template.flux = np.average(spectra_to_combine, weights=weights, axis=0)
            self.template.valid = np.isnan(self.template)

            # calculate weighted variance, see http://seismo.berkeley.edu/~kirchner/Toolkits/Toolkit_12.pdf
            all_nan = np.isnan(spectra_to_combine).all(axis=0)
            x = np.asarray([s[~all_nan] for s in spectra_to_combine])
            w = np.asarray(weights)
            var = (np.sum(w[:, np.newaxis]*x**2, axis=0)/w.sum() - self.template.flux[~all_nan]**2)*w.sum()**2/(
                    w.sum()**2 - (w**2).sum())
            _template_variance = np.ones(len(all_nan), dtype=np.float64) * np.nan
            _template_variance[~all_nan] = var
            self.template_variance = PyCrocoSpectrum(flux=_template_variance, wave_mode=self.template.wave_mode,
                                                     wave_start=self.template.wave_start,
                                                     wave_step=self.template.wave_step)

            if self.mask_ranges is not None:
                for mask_range in self.mask_ranges:
                    try:
                        imin, imax = self.template.indices_of_wave_range(w1=np.log10(mask_range[0]),
                                                                         w2=np.log10(mask_range[1]))
                    except TypeError:
                        logging.error(f'Mask range [{mask_range[0]}, {mask_range[1]} no covered by template.')
                    else:
                        self.template.valid[imin:imax] = True

            self.cc.clean_templates()
            self.cc.add(self.template, template=True, id=self.template_id)
            if self.plot:
                self.progress_plot.plot_template(self.template, mask_ranges=self.mask_ranges)

            current_results = self.cc(**kwargs)[0]

            median_dv = np.median(current_results.loc[self.template_id, 'vrad'] - self.spectra['vrad'])
            median_err = np.median(current_results.loc[self.template_id, 'vrad_err'])
            # print(abs(median_dv/median_err))
            for column in ['vrad', 'vrad_err', 'success']:
                self.spectra[column] = current_results.loc[self.template_id, column]
            if self.plot:
                self.progress_plot.plot_velocities(self.spectra, x=self.match_tellurics_by)

            if abs(median_dv / median_err) < dv_threshold:
                break

        if self.plot:
            self.progress_plot.plot_comparison(old_results=initial_results, new_results=current_results)
            if plotfilename is not None:
                self.progress_plot.fig.savefig(plotfilename)

        final = current_results.droplevel(level=0)
        for column in ['Filename', self.match_tellurics_by, self.sort_by]:
            final[column] = self.spectra[column]
        return final

    def save_template(self, filename, wave_mode=None, mask=False):

        out = self.template.copy(copy_flux=True)

        if wave_mode is None or wave_mode == out.wave_mode:
            pass
        elif out.wave_mode in [SpectrumFitsHDU.Mode.LOGLAMBDA, SpectrumFitsHDU.Mode.LOG10LAMBDA] and out.wave_step > 0:
            if wave_mode == SpectrumFitsHDU.Mode.LOGLAMBDA:
                conv = np.log10(np.e)
            elif wave_mode == SpectrumFitsHDU.Mode.LOG10LAMBDA:
                conv = np.log(10.)
            else:
                raise NotImplementedError
            new_wave_step = out.wave_step/conv
            out.mode(wave_mode)
            out = out.resample_const(step=new_wave_step)

        snr = der_snr(out.flux)

        with FitsSpectrum(filename, 'w') as fs:
            fs.spectrum = SpectrumFitsHDU(spec=out, primary=True)
            fs[0].header['HIERARCH SPECTRUM SNRATIO'] = snr
            if self.template_variance is not None:
                self.template_variance.mode(wave_mode)
                sigma = self.template_variance.resample(wave_start=out.wave_start,
                                                        wave_step=out.wave_step,
                                                        wave_count=len(out.flux))
                sigma.flux = np.sqrt(sigma.flux)
                fs['SIGMA'] = SpectrumFitsHDU(spec=sigma, primary=False)
            if mask:
                self._apply_mask(out)
                fs.good_pixels = ~out.valid
            else:
                fs.good_pixels = np.isfinite(out.flux)
