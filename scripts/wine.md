
###

README for the wine datasets (c.f. https://archive.ics.uci.edu/dataset/109/wine)

###

Optimization routes to decrease variance in small datasets:

1) Cross-validation
- StratifiedKFold

2) Regularization (weights)
- L2 weight decay
- Early stopping

3) Normalization (inputs)
- Input standardization
- Layer normalization

4) Optimization parameters
- Lower learning rate (1e-4)
- Fewer epochs + patience

5) Model averaging
- Ensembling

6) Feature selection
- PCA (with 95% variance retained)
- 
