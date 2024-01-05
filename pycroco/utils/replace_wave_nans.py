import numpy as np


def replace_wave_nans(wave):
    """
    The function replaces NaN values in the input wavelength array, assuming
    a constant sampling.

    :param wave: input wavelength array
    :type wave: ndarray
    :return: the wavelength array with the NaN values replaced.
    :rtype: ndarray
    """
    # find NaN values
    to_replace = np.isnan(wave)

    if not to_replace.any():  # nothing to do
        return wave
    else:
        wave = wave.copy()

    if to_replace[0] or to_replace[-1]:
        raise NotImplementedError('Replacing leading/trailing NaN values is not yet implemented.')

    i_min = None
    i_max = None
    for i in range(1, len(wave) - 1):
        if i_min is None and to_replace[i]:  # start of NaN range found
            i_min = i - 1
        elif i_min is not None and i_max is None and ~to_replace[i]:  # end of NaN range found
            i_max = i
        if i_min is not None and i_max is not None:  # replace values in range using linear interpolation
            di = i_max - i_min
            dw = (wave[i_max] - wave[i_min])/float(di)
            for j in range(i_min + 1, i_max):
                wave[j] = wave[i_min] + dw*(j - i_min)
            i_min = None
            i_max = None

    return wave
