from setuptools import setup

setup(name='myCROCOtools',
      version='0.0.1',
      description='CROCO routines',
      url='https://github.com/quentinjamet/myCROCOtools',
      author='qjamet',
      author_email='quentin.jamet@inria.fr',
      license='MIT',
      packages=['myCROCOtools'],
      install_requires=['numpy', 'scipy', 'xarray', 'xgcm'],
      zip_safe=False)
