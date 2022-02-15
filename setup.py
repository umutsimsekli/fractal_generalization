import setuptools
setuptools.setup(
    name='jhi',
    version='1.0',
    scripts=['./scripts/jhi'],
    author='Alexander Camuto',
    package_dir={'': 'lib'},
    packages=setuptools.find_packages('lib'),
    description='This runs my script which is great.',
    # packages=['lib.jhi'],
    install_requires=[
        'setuptools', 'click', 'requests', 'torchvision', 'torch', 'scipy',
        'tqdm', 'pandas'
    ],
    python_requires='>=3.7')
