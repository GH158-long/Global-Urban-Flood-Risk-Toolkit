#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup configuration for Global Urban Flood Risk Analysis Toolkit
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="flood-risk-toolkit",
    version="1.0.0",
    author="LONG",
    author_email="",
    description="A comprehensive toolkit for analyzing global urban flood risk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/flood-risk-toolkit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "polars>=0.16.0",
        "pyarrow>=8.0.0",
        "customtkinter>=5.0.0",
        "openpyxl>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
        ],
        "monitoring": [
            "psutil>=5.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "parquet-calc=Parquet_Field_Calculator:main",
            "flood-ead=Calculate_the_EAD_and_flood_risk_index:main",
            "flood-stats=Statistical_Analysis_of_Age-Risk_Relationship:main",
        ],
    },
    keywords=[
        "flood risk",
        "EAD",
        "expected annual damage",
        "urban flood",
        "GIS",
        "parquet",
        "climate change",
        "risk assessment",
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/flood-risk-toolkit/issues",
        "Source": "https://github.com/yourusername/flood-risk-toolkit",
    },
)
