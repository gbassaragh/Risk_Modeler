#!/usr/bin/env python
"""Setup script for Risk Modeling Tool."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="risk-tool",
    version="1.0.0",
    author="T&D Risk Modeling Team",
    author_email="risk@utility.com",
    description="Monte Carlo risk modeling tool for utility T&D projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/utility/risk-tool",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Office/Business :: Financial",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "numba>=0.58.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "openpyxl>=3.1.0",
        "pydantic>=2.0.0",
        "typer>=0.9.0",
        "pytest>=7.4.0",
        "rich>=13.0.0",
        "streamlit>=1.28.0",
        "reportlab>=4.0.0",
        "kaleido>=0.2.1",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
            "pre-commit",
        ],
    },
    entry_points={
        "console_scripts": [
            "risk-tool=risk_tool.cli:app",
        ],
    },
    include_package_data=True,
    package_data={
        "risk_tool": [
            "templates/*.xlsx",
            "templates/*.yaml",
            "templates/*.json",
        ],
    },
)