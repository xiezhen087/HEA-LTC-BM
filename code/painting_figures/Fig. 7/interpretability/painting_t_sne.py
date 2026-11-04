# -*- coding: utf-8 -*-
"""
Created on Fri Oct 24 01:34:17 2025

@author: Realunknown
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


data_path = "./features_normalized_bulk_modulus_log.csv"
features = pd.read_csv(data_path)
features = features.drop("id",axis = 1)
labels = features['target']
labels = (labels - min(labels))/(max(labels) - min(labels))

tsne = TSNE(
    n_components=2,  # reduce to 2 dimensions for visualization
    perplexity=30,   # recommended value between 5 and 50; strongly affects the result
    random_state=42, # ensure reproducible results
    n_iter=1000     # number of iterations
)

tsne_results = tsne.fit_transform(features)

result_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])

# If labels are available, add a label column for coloring
if labels is not None:
    result_df['target'] = labels.values

# Plot the visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(result_df['TSNE1'], result_df['TSNE2'], 
                     c=result_df['target'] if 'target' in result_df.columns else 'blue',
                     cmap='viridis', alpha=0.7)

if 'label' in result_df.columns:
    plt.colorbar(scatter)
plt.title('t-SNE Visualization of Dataset')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.show()