import numpy as np 
import colorsys
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os 

def tsne(features,label,destination_dir):
    def generate_n_colors(n):
        hues = np.linspace(0, 1, n, endpoint=False)
        colors = [colorsys.hsv_to_rgb(h, 0.65, 0.9) for h in hues]  # fix saturation & value
        return colors


    tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10).fit_transform(features.squeeze())
    unique_labels = np.unique(label)
    colors = generate_n_colors(len(unique_labels))

    plt.figure(figsize=(8, 6))
    for i, l in enumerate([0,1]):
        idx = label == l
        plt.scatter(tsne[idx, 0], tsne[idx, 1],
                    c=[colors[i]], label=str(unique_labels[i]), alpha=0.7, edgecolors='k')



    plt.title("t-SNE Visualization")
    plt.legend(title="Class",loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(destination_dir,"TSNE"))