
"""AML signature model utilities."""
import numpy as np
import pandas as pd
import math
from typing import Dict

def beta2m(val: float, epsilon: float = 1e-6) -> float:
    """
    Transform beta-values into m-values with robust handling of edge cases.
    
    Args:
        val: Beta value to transform
        epsilon: Small value to prevent log(0)
    """
    # Clamp values to valid range
    val = max(epsilon, min(1 - epsilon, val))
    return math.log2(val/(1-val))

def get_38cpg_coefficients() -> pd.Series:
    """Get coefficients for 38CpG AML signature."""
    coef_data = """cg17099306    0.074340
cg14978242    0.070943
cg10089193    0.069453
cg02678414    0.068424
cg14882966    0.060078
cg09890699    0.059517
cg14458815    0.055677
cg05800336    0.046168
cg00151914    0.046075
cg19706516    0.044311
cg11817631    0.039837
cg00532502    0.027137
cg05348324    0.017771
cg04663203    0.013881
cg10591771    0.013116
cg19357999    0.012883
cg16721321    0.011104
cg01543603    0.007179
cg04713531   -0.008017
cg06748884   -0.010514
cg08900363   -0.011278
cg03762237   -0.011526
cg00059652   -0.014232
cg09041251   -0.014263
cg17632028   -0.016400
cg01052291   -0.016661
cg08480739   -0.020440
cg05480169   -0.020608
cg18964582   -0.021158
cg10280339   -0.030177
cg07080653   -0.035893
cg04839706   -0.049787
cg14928764   -0.056552
cg24355048   -0.059278
cg02312559   -0.068133
cg06339275   -0.071282
cg02905663   -0.076509
cg00521620   -0.095595"""
    return pd.Series({k: float(v) for k, v in 
                     (line.split() for line in coef_data.strip().split('\n'))})

def generate_coxph_score(methyl_data: pd.DataFrame, 
                        cont_model_name: str = '38CpG-HazardScore',
                        cat_model_name: str = '38CpG-AMLsignature',
                        cutoff: float = -2.0431) -> pd.DataFrame:
    """Generate CoxPH score predictions."""
    coef_mean = get_38cpg_coefficients()
    
    # Prepare data
    x = methyl_data.copy()
    x = x.clip(0, 1)  # Clamp values to [0,1] range
    x = x.apply(np.vectorize(beta2m))
    
    # Calculate coefficients
    df2 = pd.DataFrame()
    for i, (cpg, coef) in enumerate(coef_mean.items()):
        df2[f'coef_{i}'] = x[cpg].apply(lambda x: x * coef)
    
    # Calculate scores
    df = pd.DataFrame(index=methyl_data.index)
    df[cont_model_name] = df2.sum(axis=1)
    
    # Binarize score
    df[cat_model_name] = pd.cut(df[cont_model_name],
                               bins=[-np.inf, cutoff, np.inf],
                               labels=['Low', 'High'])
    
    return df
