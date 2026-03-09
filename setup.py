"""
setup.py
--------
Install FARM as a Python package so that internal imports work correctly
from any working directory.

Usage:
    pip install -e .          # editable install (recommended for development)
    pip install .             # regular install
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    # Strip comments and blank lines
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="farm-continual-learning",
    version="1.0.0",
    description=(
        "FARM: Forgetting-Aware Rank-Modulated Experts "
        "for Continual Instruction Tuning of Large Language Models"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(
        exclude=["tests", "tests.*", "scripts", "figures", "results"]
    ),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
