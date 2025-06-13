import collections
import matplotlib.pyplot as plt
import numpy as np

from ..crosscorrel import CCData
from ..crosscorrel import CrossCorrel
from ..peak_functions import Gauss, Moffat


def plot_cc_data(cc_data: CCData, v: float = None, verr: float = None, peak_function: CrossCorrel.FitFunction = None,
                 peak_kwargs: dict = None):
    """
    Plot the cross-correlation data.

    Parameters
    ----------
    cc_data : collections.namedtuple
        Named tuple containing the cross-correlation data.
    v : float, optional
        The radial velocity of the peak in km/s. If provided, a vertical line
        will be drawn at this position.
    verr : float, optional
        The error in the radial velocity in km/s. If provided, the interval
        [v - verr, v + verr] will be highlighted with a shaded area.
    peak_function : CrossCorrel.FitFunction, optional
        The peak function used for fitting the cross-correlation data. If provided,
        the fitted profile will be plotted.
    peak_kwargs : dict, optional
        Additional keyword arguments for the peak function profile. These will be
        passed to the profile function of the peak function.
    """
    fig, (ax_input, ax_signal) = plt.subplots(figsize=(6, 6), nrows=2)

    if v is not None:
        shift = v / 299792.458 * np.log10(np.e) # Convert km/s to redshift
    else:
        shift = 0.0

    ax_input.plot(cc_data.spectrum.index, cc_data.spectrum.values, ls='-', lw=1.5, c='C0', marker='None', label='Spectrum')
    ax_input.plot(cc_data.template.index + shift, cc_data.template.values, ls='-', lw=1.5, c='C4', marker='None',
                  label='Template (shifted)' if v is not None else 'Template')

    ax_input.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncols=2, mode="expand", borderaxespad=0.)

    ax_input.set_xlabel(r'$\log_{\rm 10}(\lambda/{\rm \AA})$')
    ax_input.set_ylabel('Flux')

    ax_signal.plot(cc_data.signal.index, cc_data.signal.values, ls='-', lw=1.5, c='C0', marker='None')
    ax_signal.set_xlabel(r'Velocity (${\rm km\,s^{-1}}$)')
    ax_signal.set_ylabel(r'$f_{\rm CC}$')

    if v is not None:
        ax_signal.axvline(v, ls='--', lw=1.5, c='C3', marker='None')
        if verr is not None:
            ax_signal.fill_betweenx(ax_signal.get_ylim(), v - verr, v + verr, color='C3', alpha=0.2)    

    if peak_function is not None:
        if isinstance(peak_function, CrossCorrel.FitFunction):
            fct = peak_function.value
        else:
            raise IOError(f'Unknown fitting function: {peak_function}.')

        # Plot the fitted profile
        x_fit = cc_data.signal.index
        y_fit = fct.profile(x_fit, **peak_kwargs)
        ax_signal.plot(x_fit, y_fit, ls='--', lw=1.5, c='C3', marker='None')
    
    fig.tight_layout()
    plt.show()