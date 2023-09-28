from setuptools import setup

setup(name='qgutils',
      version='0.5.0',
      description='QG routines',
      url='http://github.com/bderembl/qgutils',
      author='bderembl',
      author_email='bruno.deremble@ens.fr',
      license='MIT',
      packages=['qgutils'],
      install_requires=['numpy', 'scipy', 'xarray' ],
      zip_safe=False)
