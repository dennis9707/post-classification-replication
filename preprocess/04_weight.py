import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
df = pd.read_csv("../data/30_combined_text.csv")
classes = np.array([1, 0])
print(df['API_CHANGE'])
weight = compute_class_weight('balanced', np.array([0, 1]), df['REVIEW'].to_numpy())
print(weight)
