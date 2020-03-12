from setuptools import setup

setup(name='ws3',
      version='0.1a1',
      description='Wood Supply Simulation System',
      url='http://github.com/gparadis/ws3',
      author='Gregory Paradis',
      author_email='greg@globaloptimality.com',
      license='MIT',
      packages=['ws3'],
      install_requires=['scipy', 'pandas', 'numpy', 'matplotlib', 'rasterio', 'fiona'],
      zip_safe=False)
