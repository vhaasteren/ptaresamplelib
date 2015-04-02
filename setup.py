import os
import sys
import numpy

try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension


if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    sys.exit()


setup(
    name="ptaresamplelib",
    version='2015.03',
    author="Rutger van Haasteren",
    author_email="vhaasteren@gmail.com",
    packages=["ptaresamplelib"],
    url="http://github.com/vhaasteren/ptaresamplelib/",
    license="GPLv3",
    description="PTA analysis software",
    long_description=open("README.md").read() + "\n\n"
                    + "Changelog\n"
                    + "---------\n\n"
                    + open("HISTORY.md").read(),
    package_data={"": ["README", "LICENSE", "AUTHORS.md"]},
    include_package_data=True,
    install_requires=["numpy", "scipy", "matplotlib"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ]
)
