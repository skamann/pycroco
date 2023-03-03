import matplotlib.pyplot as plt
import numpy as np
from importlib.resources import files
from spexxy.data import SpectrumFits

from pycroco.crosscorrel import CrossCorrel
from pycroco.peak_functions import Moffat
from pycroco.data import testing

cc = CrossCorrel()

spec_file = files(testing).joinpath('muse.fits')
print("Path to test spectrum: {}".format(spec_file))

spec = SpectrumFits(spec_file)
spec.valid = np.zeros(len(spec), dtype=bool)
spec.valid[(spec.wave > 6860) & (spec.wave < 6950)] = True
spec.valid[(spec.wave > 7580) & (spec.wave < 7700)] = True
cc.add(spectrum=spec, id=1, template=False)
# spec.flux[(spec.wave > 7580) & (spec.wave < 7700)] = 0 # np.nan

temp_file = files(testing).joinpath('phoenix.fits')
print("Path to test template: {}".format(temp_file))

temp = SpectrumFits(temp_file)
cc.add(spectrum=temp, id=2, template=True)

results, ccfunc = cc(fftfilter=True, fit_width=50, full_output=True, fit_function=CrossCorrel.FitFunction.Moffat)
print(results)

fig, ax = plt.subplots()
# ax.plot(cc.spectra[1].wave, cc.spectra[1].flux, ls='-', lw=1.5, c='C0', marker='None')
# ax.plot(cc.templates[2].wave, cc.templates[2].flux, ls='-', lw=1.5, c='C1', marker='None')
ax.plot(ccfunc[(2, 1)].index, ccfunc[(2, 1)].values, ls='-', lw=1.5, c='C0', marker='None')

vrad = results.loc[(2, 1), 'vrad']
vrad_err = results.loc[(2, 1), 'vrad_err']
peak = results.loc[(2, 1), 'peak']
r0 = results.loc[(2, 1), 'r0']
beta = results.loc[(2, 1), 'beta']

slc = abs(ccfunc[(2, 1)].index - vrad) < 500
x_fit = ccfunc[(2, 1)].index[slc]
ax.plot(x_fit, Moffat.profile(x_fit, peak, vrad, r0, beta), ls='--', lw=1.5, c='C3', marker='None')
ax.errorbar([vrad], [peak + 0.05], xerr=[vrad_err], ls='None', lw=1.5, c='C4', marker='D', mew=1.5, mec='C4', mfc='C4',
            capsize=3)

fig.tight_layout()
plt.show()