import logging
import numpy as np


def ramp(data, cuton, fullon, fulloff=-1, cutoff=-1):
    """
    The function filters the provided data in Fourier space by removing
    or damping the lowest and highest frequencies.

    Parameters
    ----------
    data : nd_array
        The data to be filtered.
    cuton : int
        The start of the ramp filter. Lower spatial frequencies are removed.
    fullon : int
        The lowest frequency that is not damped by the filter. Frequencies
        between cuton and fullon are damped via inverse distance weighting.
    fulloff : int, optional
        The highest frequency that is not damped by the filter. If set to -1
        (default), set to 1/3 of length of data.
    cutoff : int, optional
        The end of the ramp filter. Higher spatial frequencies are removed.
        Frequencies between fulloff and cutoff are damped using inverse
        distance weighting. If set to -1 (default), set to 1/2 of length of
        data.

    Returns
    -------
    filtered_data : nd_array
        The filtered data array. Has the same length as the input data.
    """
    fft_data = np.fft.rfft(data)

    if fulloff == -1:
        fulloff = data.shape[-1] // 3
    elif fulloff >= fft_data.shape[-1]:
        fulloff = fft_data.shape[-1] - 1

    if cutoff == -1:
        cutoff = data.shape[-1] // 2
    elif cutoff >= fft_data.shape[-1]:
        cutoff = fft_data.shape[-1] - 1

    logging.debug("Applying ramp filter to data:")
    logging.debug("    cuton={0}, fullon={1}, fulloff={2}, cutoff={3}".format(cuton, fullon, fulloff, cutoff))

    fft_filter = np.ones(fft_data.shape, dtype=np.float32)
    fft_filter[:cuton] = 0
    fft_filter[cuton:fullon] = np.linspace(0.0, 1.0, fullon - cuton)
    fft_filter[fulloff:cutoff] = np.linspace(1.0, 0.0, cutoff - fulloff)
    fft_filter[cutoff:] = 0

    return np.fft.irfft(fft_data * fft_filter, n=data.shape[-1])
