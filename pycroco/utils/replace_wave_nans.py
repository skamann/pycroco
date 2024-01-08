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
        _wave = wave.copy()

    # replace leading NaNs (if any)
    if to_replace[0]:
        i_min = None
        i_max = None
        for i in range(1, len(_wave)):
            if not to_replace[i] and i_min is None:
                i_min = i
            elif not to_replace[i] and i_max is None:
                i_max = i
                break
        di = i_max - i_min
        dw = (_wave[i_max] - _wave[i_min])/float(di)
        for i in range(i_min):
            _wave[i] = _wave[i_min] - dw*(i_min - i)

    # replace trailing NaNs (if any)
    if to_replace[-1]:
        i_min = None
        i_max = None
        for i in range(len(_wave)-2, 0, -1):
            if not to_replace[i] and i_max is None:
                i_max = i
            elif not to_replace[i] and i_min is None:
                i_min = i
                break
        di = i_max - i_min
        dw = (_wave[i_max] - _wave[i_min])/float(di)
        for i in range(i_max+1, len(_wave)):
            _wave[i] = _wave[i_max] + dw*(i - i_max)

    # replace remaining NaNs
    i_min = None
    i_max = None
    for i in range(1, len(_wave) - 1):
        if i_min is None and to_replace[i]:  # start of NaN range found
            i_min = i - 1
        elif i_min is not None and i_max is None and ~to_replace[i]:  # end of NaN range found
            i_max = i
        if i_min is not None and i_max is not None:  # replace values in range using linear interpolation
            di = i_max - i_min
            dw = (_wave[i_max] - _wave[i_min])/float(di)
            for j in range(i_min + 1, i_max):
                _wave[j] = _wave[i_min] + dw*(j - i_min)
            i_min = None
            i_max = None

    return _wave
