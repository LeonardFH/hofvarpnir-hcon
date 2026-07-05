from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="hofvarpnir-hcon",
    version="3.18.0",
    author="Leonard Haasbroek",
    author_email="leonardfhaasbroek@gmail.com",
    description="HófvarpnirHCON - Fast dictionary-based crystal density prediction from SMILES",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        'hofvarpnirhcon': [
            'docs/*.md',
        ],
    },
    include_package_data=True,
    install_requires=[
        "rdkit>=2023.03.1",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "tqdm>=4.62.0",
        "scipy>=1.8.0",
    ],
    python_requires=">=3.8",
    license="BSD-3-Clause",
    license_files=["LICENSE"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: BSD License",
    ],
    entry_points={
        "console_scripts": [
            # Uncomment when CLI is ready
            # "crystal-density-train=crystal_density.train:main",
            # "crystal-density-predict=crystal_density.cli:main",
        ],
    },
)