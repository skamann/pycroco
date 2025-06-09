# PyCroCo

PyCroCo is a Python toolkit for cross-correlation analysis of astrophysical spectra. It provides utilities for preprocessing, cross-correlation computation, and visualization of spectral data.

PyCroco uses [Spexxy](https://github.com/thusser/spexxy) for reading and manipulating spectra. Hence, a recent version of Spexxy must be available in the same python environment for PyCroco to work properly.

## Features

- Efficient cross-correlation of 1D spectra
- Preprocessing tools for normalization, masking, and Fourier filtering
- Visualization utilities for correlation results
- Scripting support for batch processing

## Installation

```bash
git clone https://github.com/skamann/pycroco.git
cd pycroco
pip install -e .
```

## Usage

```python
from pycroco.crosscorrel import CrossCorrel
from spexxy.data import FitsSpectrum

# Example: cross-correlate two spectra
cc = CrossCorrel()
spec = FitsSpectrum('spectrum.fits')
cc.add_spectrum(spec.spectrum, id=5)
# providing IDs is optional, but helps when scanning the results
temp = FitsSpectrum('template.fits')
cc.add_spectrum(temp.spectrum, template=True, id=9)
result = cc()

# To visualise the results, the plot_cc_data() function can be used
result, cc_data = cc(full_output=True)
plot_cc_data(cc_data[(9, 5)], v=result.at[(9, 5), 'vrad'], verr=result.at[(9, 5), 'vrad_err'])
```

## Documentation

See the [docs](docs/) directory for detailed documentation and examples.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the GPL 3.0 License.