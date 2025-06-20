import numpy as np
import pandas as pd

def generate_sample_data():
    """Generate simple dataset"""
    np.random.seed(42)
    n_samples = 50
    data = {
        'feature1': np.random.uniform(0, 10, n_samples),
        'feature2': np.random.uniform(0, 10, n_samples),
        'target': np.where(np.random.uniform(0, 10, n_samples) > 5, 1, 0)
    }
    return pd.DataFrame(data)
