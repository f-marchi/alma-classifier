"""
38‑CpG AML Signature
Returns a continuous hazard score and Low/High risk category.
"""
import numpy as np, pandas as pd

_EPS = 1e-6

# Parse the coefficient data by splitting all elements and then grouping in pairs
_coef_data = """
cg17099306 0.074340 cg14978242 0.070943 cg10089193 0.069453 cg02678414 0.068424
cg14882966 0.060078 cg09890699 0.059517 cg14458815 0.055677 cg05800336 0.046168
cg00151914 0.046075 cg19706516 0.044311 cg11817631 0.039837 cg00532502 0.027137
cg05348324 0.017771 cg04663203 0.013881 cg10591771 0.013116 cg19357999 0.012883
cg16721321 0.011104 cg01543603 0.007179 cg04713531 -0.008017 cg06748884 -0.010514
cg08900363 -0.011278 cg03762237 -0.011526 cg00059652 -0.014232 cg09041251 -0.014263
cg17632028 -0.016400 cg01052291 -0.016661 cg08480739 -0.020440 cg05480169 -0.020608
cg18964582 -0.021158 cg10280339 -0.030177 cg07080653 -0.035893 cg04839706 -0.049787
cg14928764 -0.056552 cg24355048 -0.059278 cg02312559 -0.068133 cg06339275 -0.071282
cg02905663 -0.076509 cg00521620 -0.095595
""".split()

_COEF = pd.Series({_coef_data[i]: float(_coef_data[i+1]) for i in range(0, len(_coef_data), 2)})

def _beta2m(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(arr, _EPS, 1 - _EPS)
    return np.log2(arr / (1 - arr))

def hazard(df: pd.DataFrame, cutoff: float = -2.0431) -> pd.DataFrame:
    """Return continuous score and High/Low category."""
    missing = _COEF.index.difference(df.columns)
    if not missing.empty:
        raise ValueError(f"Missing CpGs: {missing.tolist()}")
    m = _beta2m(df[_COEF.index].values)
    score = (m * _COEF.values).sum(axis=1)
    risk = pd.cut(
        score,
        bins=[-np.inf, cutoff, np.inf],
        labels=["Low Risk", "High Risk"]
    )
    return pd.DataFrame({
        "38CpG-HazardScore": score,
        "38CpG-AMLsignature": risk
    }, index=df.index)
