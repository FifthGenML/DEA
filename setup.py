from setuptools import setup,find_packages

setup(
    name = "DEA",
    version = "0.1",
    packages = find_packages(),
    install_requires = [
        "numpy",
        "Pillow",
        "requests",
        "pandas",
        "matplotlib",
    ]
)