from setuptools import setup, find_packages

setup(
    name="alma-classifier",
    version="0.2.0",  # Updated for v2 feature
    packages=find_packages(),
    package_data={
        'alma_classifier': ['models/*', 'data/*'],
    },
    install_requires=[],  # Dependencies are managed in pyproject.toml
    extras_require={
        'v2': ['torch>=1.11.0'],  # Optional PyTorch dependency for v2 models
    },
    entry_points={
        'console_scripts': [
            'alma-classifier=alma_classifier.cli:main',
            'alma-classifier-download-v2=alma_classifier.download_models:main_v2',
        ],
    },
    python_requires=">=3.8,<=3.12",
    author="Francisco Marchi",
    author_email="flourenco@ufl.edu",
    description="A Python package for applying pre-trained epigenomic classification models with transformer-based ALMA Subtype v2",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/f-marchi/ALMA-classifier",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
)