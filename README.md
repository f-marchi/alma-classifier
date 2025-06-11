# ALMA Classifier

A Python package for epigenomic diagnosis and prognosis of acute myeloid leukemia.

## Models

1. **ALMA Subtype**: Classifies 28 subtypes (27 WHO 2022 acute leukemia subtypes + normal control)
2. **AML Epigenomic Risk**: Predicts 5-year mortality probability for AML patients
3. **38CpG AML Signature**: Risk stratification using targeted 38 CpG panel

## Installation

### Docker (Recommended)

```bash
docker pull fmarchi/alma-classifier:0.1.4
```

### pip

```bash
python -m venv .venv && source .venv/bin/activate
pip install pacmap==0.7.0
# MacOS users need `brew install lightgbm`
pip install alma-classifier
python -m alma_classifier.download_models
```

## Usage

### Docker

```bash
# Demo
docker run --rm -v $(pwd)/output:/output fmarchi/alma-classifier:0.1.4 --demo --output /output/demo_predictions.xlsx

# Your data
## Transfer your input data to ./data/
docker run --rm -v "$(pwd)/data":/data -v "$(pwd)/output":/output   fmarchi/alma-classifier:0.1.4 --input /data/your_file.pkl --output /output/your_results.xlsx
```

### Command Line

```bash
alma-classifier --input data.pkl --output predictions.xlsx
alma-classifier --demo --output demo_results.csv
alma-classifier --input data.pkl --output results.xlsx --confidence 0.5
```

## Input Formats

For Illumina Methylation450k or EPIC, prepare a .pkl dataset in python3.8 with the following structure:

- **Rows**: Samples
- **Columns**: CpG sites
- **Values**: Beta values (0-1)

For nanopore WGS, follow the standard bedMethyl format with these key columns:

- **Column 1**: `chrom` - Chromosome name
- **Column 2**: `start_position` - 0-based start position  
- **Column 4**: `modified_base_code` - Single letter code for modified base
- **Column 11**: `fraction_modified` - Percentage of methylation (0-100)

## Output

Results include subtype classification, risk prediction, and confidence scores. Predictions below confidence threshold (default 0.5) are marked "Not confident".

## Limitations

The diagnostic model does not recognize: AML with Down Syndrome, juvenile myelomonocytic leukemia, transient abnormal myelopoiesis, low-risk MDS, or lymphomas.

## Citation

Francisco Marchi, Marieke Landwehr, Ann-Kathrin Schade et al. Long-read epigenomic diagnosis and prognosis of Acute Myeloid Leukemia, 12 December 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-5450972/v1]
