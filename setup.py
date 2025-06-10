from setuptools import setup, find_packages

setup(
    name="alma-classifier",
    version="0.1.4",
    packages=find_packages(),
    package_data={
        'alma_classifier': ['models/*', 'data/*'],
    },
    install_requires=[],  # Dependencies are managed in pyproject.toml
    entry_points={
        'console_scripts': [
            'alma-classifier=alma_classifier.cli:main',
        ],
    },
    python_requires=">=3.8,<=3.10",
    author="Francisco Marchi",
    author_email="flourenco@ufl.edu",
    description="A Python package for applying pre-trained epigenomic classification models",
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