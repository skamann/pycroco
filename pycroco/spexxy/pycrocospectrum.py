import math
import numpy as np
from spexxy.data import Spectrum


class PyCrocoSpectrum(Spectrum):

    def redshift(self, vrad):

        # which mode?
        if self._wave_mode == PyCrocoSpectrum.Mode.LAMBDA:
            # wavelength in AA means a multiplication
            self._wavelength = self.wave * (1. + vrad / 299792.458)
            # and non-constant step size
            self._wave_step = 0
        elif self._wave_mode == PyCrocoSpectrum.Mode.LOGLAMBDA:
            # wavelength in log domain is simpler
            self._wavelength = self.wave + math.log(1. + vrad / 299792.458)
            self._wave_start = self._wavelength[0]
        elif self._wave_mode == Spectrum.Mode.LOG10LAMBDA:
            self._wavelength = self.wave + math.log10(1. + vrad / 299792.458)
            self._wave_start = self._wavelength[0]
        else:
            raise NotImplementedError('Unsupported wave mode: {}'.format(self._wave_mode))

    def extract_index(self, i1: int, i2: int) -> 'Spectrum':
        """Extract spectrum in given index range.

        Args:
            i1: Start index to extract.
            i2: End index (not included) to extract.

        Returns:
            Extracted spectrum
        """
        if self._wave_step != 0.:
            spec = self.__class__(spec=self, flux=np.copy(self.flux[i1:i2]), wave_start=self.wave[i1],
                                  wave_step=self._wave_step, wave_mode=self._wave_mode, valid=None)
        else:
            spec = self.__class__(spec=self, flux=np.copy(self.flux[i1:i2]), wave=np.copy(self.wave[i1:i2]),
                                  wave_mode=self._wave_mode, valid=None)

        if self.valid is not None:
            spec.valid = np.copy(self.valid[i1:i2])

        return spec
