from setuptools import setup, find_packages

setup(
    name="alma-classifier",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        'alma_classifier': ['models/__init__.py'],
    },
    install_requires=[
        "pandas~=2.0.3",
        "numpy~=1.24.4",
        "scikit-learn~=1.2.2",
        "lightgbm~=3.3.5",
        "joblib~=1.3.2",
        "openpyxl>=3.0.0"  # For Excel support
    ],
    entry_points={
        'console_scripts': [
            'alma-classifier=alma_classifier.cli:main',
        ],
    },
    python_requires='>=3.8',
    author="Your Name",
    description="A Python package for applying pre-trained epigenomic classification models",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alma-classifier",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)