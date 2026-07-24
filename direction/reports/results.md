# Direction-clustering results

## Attempt counts per direction

| direction | n attempts |
|---|---|
| positive_z | 21 |
| negative_x | 17 |
| negative_y | 16 |
| positive_x | 14 |
| negative_z | 13 |
| positive_y | 13 |

Total attempts: 94 across 6 trials (2 per axis x 3 axes).

## Segmentation notes

- Segmentation warnings: none
- Trials using a per-trial rest_frac/active_frac override: none

## t-SNE / PCA parameters

- Primary t-SNE perplexity: 10 (capped below the smallest class size)
- Perplexity grid checked: [5, 10, 15, 30]
- random_state: 0
- PCA explained variance: PC1 26.2%, PC2 18.8%

## Interpretation

See fig_tsne_main.png, fig_tsne_perplexity_grid.png, and fig_pca.png. Compare whether the 6 direction classes (color=axis, marker=sign) separate consistently across perplexities AND agree with the deterministic PCA projection -- agreement between the two is stronger evidence of real clustering than either alone, given the small per-class sample sizes here.
