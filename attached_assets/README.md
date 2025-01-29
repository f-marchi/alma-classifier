# ALMA Classifier

A Python package for applying pre-trained epigenomic classification models to methylation data. This package provides two main predictive models:

1. **AL Epigenomic Subtype**: Predicts 28 subtypes/classes (27 WHO 2022 subtypes of acute leukemia + otherwise-normal control).
2. **AML Epigenomic Risk**: Predicts the probability of death within 5 years for AML patients.

## Installation

```bash
pip install alma-classifier
```

### Requirements

- Python 3.8
- Dependencies:
  - pandas ~= 2.0.3
  - numpy ~= 1.24.4
  - scikit-learn ~= 1.2.2
  - lightgbm ~= 3.3.5
  - pacmap ~= 0.7.0
  - joblib ~= 1.3.2

## Quick Start

```python
from alma_classifier import ALMAPredictor

# Initialize predictor
predictor = ALMAPredictor(confidence_threshold=0.5)

# Load your methylation data
# Supported formats: .pkl, .csv, .xlsx
data_path = "methylation_data.csv"

# Generate predictions
results = predictor.predict(
    data=data_path,
    sample_type="AML"  # Optional sample type info
)

# Export results
results.to_excel("alma_predictions.xlsx")
```

## Input Data Format

The input data should be a matrix of methylation beta values with:
- Rows representing samples
- Columns representing CpG sites
- Values between 0 and 1

## Output Format

The predictor returns a DataFrame with the following columns:

- `AL Epigenomic Subtype`: Predicted leukemia subtype
- `AML Epigenomic Risk`: Risk level (High/Low) for AML samples
- `P(Subtype)`: Probability scores for each subtype
- `P(Remission) at 5y`: Probability of 5-year remission
- `P(Death) at 5y`: Probability of death within 5 years
- Confidence indicators for predictions

## Citation

If you use this package in your research, please cite:
[Citation information to be added]

## License

[License information to be added]
