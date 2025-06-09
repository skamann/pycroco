# PyCROCO

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
_spec = FitsSpectrum('spectrum.fits')
cc.add_spectrum(_spec.spectrum)
_temp = FitsSpectrum('template.fits')
cc.add_spectrum(_temp.spectrum, template=True)
result = cc()
```

## Documentation

See the [docs](docs/) directory for detailed documentation and examples.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

## License

This project is licensed under the MIT License.