import glob

from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        scripts=glob.glob("scripts/*")
    )