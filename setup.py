import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ws3',
    version='0.0.1-post3',
    author='Gregory Paradis',
    author_email='0@01101.io',
    description='Wood Supply Simulation System',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/gparadis/ws3',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['scipy', 'pandas', 'numpy', 'matplotlib', 'rasterio', 'fiona', 'profilehooks']
)
 
