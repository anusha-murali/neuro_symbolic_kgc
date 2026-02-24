"""
setup.py
Installation script for NeuroSymbolic KGC package.
"""

from setuptools import setup, find_packages

# Read the contents of README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements from requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="neuro_symbolic_kgc",
    version="1.0.0",
    author="Anusha Murali",
    author_email="anusha.murali.gr@dartmouth.edu",
    description="Neuro-symbolic framework for knowledge graph completion on biological data (BioKG)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anusha-murali/neuro_symbolic_kgc",
    project_urls={
        "Bug Tracker": "https://github.com/anusha-murali/neuro_symbolic_kgc/issues",
        "Documentation": "https://github.com/anusha-murali/neuro_symbolic_kgc/docs",
        "Source Code": "https://github.com/anusha-murali/neuro_symbolic_kgc",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
        "Environment :: GPU :: NVIDIA CUDA",
    ],
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    py_modules=["main"],  # Include main.py at root level
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "ipykernel>=6.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuro-kgc-train=main:main",  # Creates command-line tool
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml"],  # Include config files
    },
    zip_safe=False,
)
