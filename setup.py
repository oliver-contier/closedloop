from setuptools import setup, find_packages

setup(
    name="closedloop",
    version="0.1.0",
    description=(
        "Closed-loop hypothesis generation and testing for characterizing "
        "representations in visually-responsive brain regions (FFA)."
    ),
    author="Oliver Contier",
    author_email="contier@cbs.mpg.de",
    url="https://github.com/ViCCo-Group/closedloop",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
)
