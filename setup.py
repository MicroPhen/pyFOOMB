import setuptools

__packagename__ = 'pyfoomb'

def get_version():
    import os, re
    VERSIONFILE = os.path.join(__packagename__, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))

__version__ = get_version()


setuptools.setup(name = __packagename__,
        packages = setuptools.find_packages(exclude=['examples', '*test*']),
        version=__version__,
        zip_safe=False,
        description='Package for handling systems of ordinary differential equations (ODEs) with discontinuities. Relies on assimulo package for ODE integration and pygmo package for optimization.',
        author='Johannes Hemmerich',
        author_email='hemmerich@outlook.com',
        url='https://github.com/MicroPhen/pyFOOMB',
        license='MIT',
        classifiers= [
            'Programming Language :: Python :: 3 :: Only',
            'Operating System :: OS Independent',
            'Intended Audience :: Developers'
        ],
        python_requires='>=3.7',
        install_requires=[
            'numpy',
            'scipy',
            'pandas>=0.24',
            'openpyxl',
            'joblib',
            'matplotlib',
            'seaborn',
            'assimulo',
            'psutil',
        ]
)
