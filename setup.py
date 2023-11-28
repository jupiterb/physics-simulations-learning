from setuptools import setup, find_packages

requirements = ["torch"]

setup(
    name="PhyLearn",
    version="0.0.0",
    description="Physics learn with PINNs",
    packages=find_packages(),
    install_requires=requirements,
)
