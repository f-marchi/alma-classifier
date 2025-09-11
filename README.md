# ALMA Classifier

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15636415.svg)](https://doi.org/10.5281/zenodo.15636415)
[![Nat Commun](https://img.shields.io/badge/Nat%20Commun-2025-0a7bbc.svg)](https://www.nature.com/articles/s41467-025-62005-4)
[![Downloads](https://static.pepy.tech/personalized-badge/alma-classifier?period=total&units=international_system&left_color=grey&right_color=blue&left_text=Downloads)](https://pepy.tech/project/alma-classifier)

Epigenomic diagnosis of acute leukemia and prognosis of AML.

## Models

1. **ALMA Subtype**: Classifies 27 subtypes of acute leukemia according to WHO 2022 + healthy control
2. **AML Epigenomic Risk (v0.1.4 only)**: Predicts 5-year mortality probability for AML patients
3. **38CpG AML Signature**: Risk stratification using targeted 38 CpG panel

## What's New in v0.2.0

- **ALMA Subtype v2**:
  - New diagnostic model using autoencoder-transformer architecture
  - Near-perfect accuracy in both methylation arrays and nanopore epigenomes
  - v0.1.4 (from the publication) remains available in pip and docker.

## Installation

### Docker (recommended)

```bash
docker pull fmarchi/alma-classifier:0.2.0
```

### Python 3.11

```bash
pip install alma-classifier
```

## Usage

### Docker

Run demo:

```bash
docker run --rm -v "$(pwd)":/work -w /work fmarchi/alma-classifier:0.2.0 \
  alma-classifier --demo
```

Run using your data:

```bash
# Transfer your input data to current working directory
docker run --rm -v "$(pwd)":/work -w /work fmarchi/alma-classifier:0.2.0 \
  alma-classifier -i /work/your_data.pkl
```

### Python 3.11 (CLI)

Run demo:

```bash
alma-classifier --demo
```

Run using your data:

```bash
alma-classifier -i path/to/your_data.pkl
```

## Input Formats

### Illumina Methylation450k or EPIC

Prepare a .pkl (or csv.gz) dataset with the following structure:

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

## CLI options

```bash
usage: alma-classifier [-h] [-i INPUT_DATA] [-o OUTPUT] [--download-models] [--demo] [--all_probs]

🩸🧬 ALMA Classifier – Epigenomic diagnosis of acute leukemia (research use only) 🧬🩸

options:
  -h, --help            show this help message and exit
  -i INPUT_DATA, --input_data INPUT_DATA
                        Input file: .pkl with β‑values, .csv/.csv.gz with β‑values, or .bed/.bed.gz nanopore file
  -o OUTPUT, --output OUTPUT
                        .csv output (default: alongside input data)
  --download-models     Download model weights from GitHub release
  --demo                Run demo with example dataset
  --all_probs           Include all subtype/class probabilities as separate columns in the output
```

## Output

Results include subtype classification, risk prediction, and confidence scores.

## Important limitations

- The diagnostic model does not currently recognize: AML with Down Syndrome, juvenile myelomonocytic leukemia, transient abnormal myelopoiesis, low-risk MDS, or lymphomas. We need reference methylation data for these patient populations.
- Follow preprocessing as instructed in "Input Formats" above. Different or erroneous preprocessing may lead to poor performance. This applies to bad wet-lab handling of samples.
- The models will attempt to work with missing CpGs. Ideally, use Methylation Array 450k,EPIC or WGS Nanopore Seq with >5x coverage. Anything below that may compromise performance.

## Citation

Marchi, F., Shastri, V.M., Marrero, R.J. et al. Epigenomic diagnosis and prognosis of Acute Myeloid Leukemia. Nat Commun 16, 6961 (2025). <https://doi.org/10.1038/s41467-025-62005-4>
