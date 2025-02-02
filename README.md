# ALMA Classifier

A Python package for applying pre-trained epigenomic classification models to methylation data. This package provides two main predictive models:

1. **ALMA Subtype**: Predicts 28 subtypes/classes (27 WHO 2022 subtypes of acute leukemia + otherwise-normal control).
2. **AML Epigenomic Risk**: Predicts the probability of death within 5 years for AML patients.

## Installation

```bash
# first make sure you are running python3.8
python --version

# create a virtual env
python -m venv .venv
source .venv/bin/activate

# install pacmap==0.7.0 (required)
pip install pacmap==0.7.0

# Then install alma-classifier
pip install alma-classifier

# Download model files (required)
python -m alma_classifier.download_models
```

### Important Notes
1. This is a pre-release research tool. Initial versions will be picky and annoying to deal with. Future versions will be flexible and easy to use.
2. Our diagnostic model currently does not know about important cases, which we really need training data for: AML with Down Syndrome, juvenile myelomonocytic leukemia, transient abnormal myelopoiesis, bone marrow failures, low-risk MDS, lymphomas, and others.
3. Our prognostic models are currently only limited to AML cases.

## Usage

### Command Line Interface

```bash
# Basic usage
alma-classifier --input path/to/data.pkl --output path/to/predictions.xlsx
```

## Input Data Format

The input data should be a matrix of methylation beta values with:
- Rows representing samples
- Columns representing CpG sites
- Values between 0 and 1

## Citation

If you use this package in your research, please cite:

Francisco Marchi, Marieke Landwehr, Ann-Kathrin Schade et al. Long-read epigenomic diagnosis and prognosis of Acute Myeloid Leukemia, 12 December 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-5450972/v1]

## License

See license file.