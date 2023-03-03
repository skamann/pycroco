from setuptools import setup, find_packages

setup(
    name='pycroco',
    version='0.1',
    packages=find_packages('./'),
    scripts=['bin/run_pycroco_test.py'],
    package_data={"": ["*.fits"]},
    url='',
    license='gpl-3.0',
    author='Sebastian Kamann',
    author_email='s.kamann@ljmu.ac.uk',
    install_requires=['matplotlib', 'numpy', 'pandas', 'scipy', 'spexxy'],
    description='Python tool for cross-correlating spectra'
)
