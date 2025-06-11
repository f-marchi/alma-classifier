# ALMA Classifier

A Python package for applying pre-trained epigenomic classification models to methylation data. This package provides three main predictive models:

1. **ALMA Subtype**: Predicts 28 subtypes/classes (27 WHO 2022 subtypes of acute leukemia + otherwise-normal control).
2. **AML Epigenomic Risk**: Predicts the probability of death within 5 years for AML patients using ALMA.
3. **38CpG AML Signature**: Predicts same as above, but using a targeted panel of 38 CpGs.

## Installation

```bash
# first make sure you are running python3.8
python --version

# create a virtual env
python -m venv .venv
source .venv/bin/activate

# install pacmap==0.7.0 (required)
pip install pacmap==0.7.0

# MacOS users only: 
# brew install lightgbm

# Then install alma-classifier
pip install alma-classifier

# Download model files (required)
python -m alma_classifier.download_models
```

### Important Note

Our diagnostic model currently does not know about important cases: AML with Down Syndrome, juvenile myelomonocytic leukemia, transient abnormal myelopoiesis, low-risk MDS, lymphomas, and others.

## Usage

### Command Line Interface

```bash
# Run with your own data (various formats supported)
alma-classifier --input path/to/dataset.pkl --output path/to/predictions.xlsx
alma-classifier --input path/to/nanopore_sample.bed.gz --output path/to/predictions.csv

# Try the demo with example dataset
alma-classifier --demo --output predictions.xlsx
```

## Input Data Formats

- Rows should represent samples
- Columns should represent CpG sites  
- Values should be between 0 and 1 (beta values)

### BED File Format

BED files from nanopore WGS should follow the standard bedMethyl format with these key columns:

- Column 1: `chrom` - Chromosome name
- Column 2: `start_position` - 0-based start position  
- Column 4: `modified_base_code` - Single letter code for modified base
- Column 11: `fraction_modified` - Percentage of methylation (0-100)

The package automatically converts BED files to the internal methylation format using the reference CpG mapping.

## Model Outputs and Prediction Behavior

The package generates predictions with the following behavior:

1. **ALMA Subtype Classification**:
   - Outputs the predicted subtype and its probability
   - If confidence is below threshold (default 0.5), outputs "Not confident"
   - For predictions with confidence between 0.5-0.8, also outputs the second most likely subtype and its probability

2. **AML Epigenomic Risk** (only for AML/MDS samples):
   - Outputs "High" or "Low" risk prediction
   - Includes P(Death) at 5y probability
   - Outputs "Not confident" if prediction confidence is below threshold
   - Non-AML/MDS samples receive "NaN" values

3. **38CpG AML Signature** (only for AML/MDS samples):
   - Outputs continuous hazard score (38CpG-HazardScore)
   - Provides binary risk stratification (38CpG-AMLsignature: High/Low)
   - Non-AML/MDS samples receive "NaN" values

## Citation

If you use this package in your research, please cite:

Francisco Marchi, Marieke Landwehr, Ann-Kathrin Schade et al. Long-read epigenomic diagnosis and prognosis of Acute Myeloid Leukemia, 12 December 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-5450972/v1]

## License

See license file.
