from setuptools import setup, find_packages

setup(
    name='pycroco',
    version='0.2',
    packages=find_packages('./'),
    scripts=['bin/run_pycroco_test.py', 'bin/PyCroCo'],
    package_data={"": ["*.fits"]},
    url='',
    license='gpl-3.0',
    author='Sebastian Kamann',
    author_email='s.kamann@ljmu.ac.uk',
    install_requires=['matplotlib', 'numpy', 'pandas', 'scipy', 'spexxy', 'astropy'],
    description='Python tool for cross-correlating spectra'
)
