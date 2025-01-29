from setuptools import setup, find_packages

setup(
    name="alma-classifier",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=2.0.3",
        "numpy>=1.24.4",
        "scikit-learn>=1.2.2",
        "lightgbm>=3.3.5",
        "joblib>=1.3.2",
        "click>=8.0.0",
        "openpyxl>=3.0.0"  # For Excel support
    ],
    extras_require={
        "dimension_reduction": ["pacmap>=0.7.0"]
    },
    entry_points={
        "console_scripts": [
            "alma-classifier=alma_classifier.cli:main",
        ],
    },
    author="ALMA Team",
    description="Epigenomic classification models for methylation data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
