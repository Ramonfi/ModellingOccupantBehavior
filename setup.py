import os
from setuptools import setup, find_packages

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "ModellingOccupantBehavior",
    version = "0.9",
    author = "Roman Ficht",
    author_email = "Roman.Ficht@posteo.de",
    description = ("Digitaler Anhang zur Masterarbeit ''."),
    license = "BSD",
    keywords = "TRNSYS Occupant Behavior",
    url = '',
    python_requires=">=3.6",
    packages=find_packages("src"),
    package_dir={"": "src"},
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)