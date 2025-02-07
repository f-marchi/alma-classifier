# Changelog

All notable changes to ALMA Classifier will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2025-02-06

### Fixed
- Made python version requirement less strict

## [0.1.1] - 2025-02-06

### Added
- Demo functionality with example dataset
- Added --demo flag to CLI

## [0.1.0] - 2025-02-06

### Added
- Initial release of ALMA Classifier
- ALMA Subtype prediction model for 28 subtypes/classes
- AML Epigenomic Risk prediction model
- 38CpG AML Signature model
- Command Line Interface (CLI) for predictions
- Model downloading functionality
- Data preprocessing and validation
- Support for Excel and CSV output formats

### Dependencies
- Requires Python 3.8
- Core dependencies: pandas, numpy, scikit-learn, lightgbm, joblib
- Special dependency: pacmap==0.7.0