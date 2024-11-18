from setuptools import setup,find_packages

setup(
    name = "foxide",
    version = "0.1",
    packages = find_packages(),
    install_requires = [
        "numpy",
        "Pillow",
        "requests",
        "pandas",
        "matplotlib",
    ],
    entry_points = {
        'console_scripts':[
            "foxide=src.attack:main"
        ],
    },
)