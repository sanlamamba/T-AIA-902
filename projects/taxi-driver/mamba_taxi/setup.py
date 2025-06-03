from setuptools import setup, find_packages

setup(
    name="taxi_driver",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "gymnasium",
        "matplotlib",
        "pandas",
    ],
    python_requires=">=3.8",
)