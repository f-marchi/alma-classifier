# ALMA Classifier

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15636415.svg)](https://doi.org/10.5281/zenodo.15636415)

A Python package for epigenomic diagnosis and prognosis of acute myeloid leukemia.

## Models

1. **ALMA Subtype**: Classifies 28 subtypes (27 WHO 2022 acute leukemia subtypes + normal control)
2. **ALMA Subtype v2**: New transformer-based diagnostic model with improved accuracy
3. **AML Epigenomic Risk**: Predicts 5-year mortality probability for AML patients
4. **38CpG AML Signature**: Risk stratification using targeted 38 CpG panel

## What's New in v0.2.0

- **ALMA Subtype v2**: New transformer-based diagnostic model using autoencoder feature extraction
- Enhanced accuracy through ensemble prediction across multiple folds
- Optional PyTorch integration for advanced deep learning models
- Backward compatibility with existing v1 models

## Installation

### Docker (recommended)

```bash
docker pull fmarchi/alma-classifier:0.2.0
```

### pip (python 3.8-3.12)

```bash
python -m venv .venv && source .venv/bin/activate
pip install pacmap==0.7.0
# MacOS users need `brew install lightgbm`
pip install alma-classifier

# Download standard models
python -m alma_classifier.download_models

# Optional: Install PyTorch and download v2 models for transformer-based predictions
pip install alma-classifier[v2]
alma-classifier --download-models-v2
```

## Usage

### Docker

#### Demo

```bash
docker run --rm -v $(pwd):/output fmarchi/alma-classifier:0.2.0 \
    alma-classifier --demo --output /output/demo_results.xlsx
```

#### Your data

```bash
## Transfer your input data to ./data/
docker run --rm -v $(pwd):/data fmarchi/alma-classifier:0.2.0 \
    alma-classifier --input /data/your_methylation_data.pkl --output /data/results.xlsx
```

#### Using ALMA Subtype v2

```bash
docker run --rm -v $(pwd):/output fmarchi/alma-classifier:0.2.0 \
    alma-classifier --demo --output /output/demo_results_v2.xlsx --include-v2
```

### pip (python 3.8-3.12)

#### Demo

```bash
alma-classifier --demo --output demo_results.csv
```

#### Your data

```bash
alma-classifier --input data.pkl --output predictions.xlsx
```

#### Using ALMA Subtype v2

```bash
# First download v2 models (requires PyTorch)
alma-classifier --download-models-v2

# Run with v2 predictions
alma-classifier --input data.pkl --output predictions_v2.xlsx --include-v2
```

## Input Formats

### Illumina Methylation450k or EPIC
Prepare a .pkl dataset in python3.8 with the following structure:

- **Rows**: Samples
- **Columns**: CpG sites
- **Values**: Beta values (0-1)

Got .idat files? Use [SeSAMe](https://github.com/zwdzwd/sesame) first.

### Nanopore whole genome sequencing
Follow the standard bedMethyl format with these key columns:

- **Column 1**: `chrom` - Chromosome name
- **Column 2**: `start_position` - 0-based start position  
- **Column 4**: `modified_base_code` - Single letter code for modified base
- **Column 11**: `fraction_modified` - Percentage of methylation (0-100)

Got .bam files? Use [modkit](https://nanoporetech.github.io/modkit/intro_pileup.html) first:

```bash
modkit pileup \
"$bam_file" \
"$bed_file" \
-t $threads \
--combine-strands \
--cpg \
--ignore h \
--ref ref/hg38.fna \
--no-filtering
```

## Output

Results include subtype classification, risk prediction, and confidence scores. Predictions below confidence threshold (default 0.5) are marked "Not confident".

## Limitations

The diagnostic model does not recognize: AML with Down Syndrome, juvenile myelomonocytic leukemia, transient abnormal myelopoiesis, low-risk MDS, or lymphomas.

## Citation

Marchi, F., Shastri, V.M., Marrero, R.J. et al. Epigenomic diagnosis and prognosis of Acute Myeloid Leukemia. Nat Commun 16, 6961 (2025). <https://doi.org/10.1038/s41467-025-62005-4>
