"""Setup file to install a package with pip"""
import setuptools


setuptools.setup(
    name="asl-signs",
    version="0.1",
    author="farid.abdalla",
    author_email="farid.abdalla.13@gmail.com",
    packages=setuptools.find_packages(),
    license="",
    description="Repository for the ASL competition on Kaggle",
    long_description=open("README.md").read(),
    install_requires=open("requirements.txt").readlines(),
    # extras_require={
    #     "dev": open("requirements_dev.txt").readlines()
    # }
)
