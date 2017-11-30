# Excellent Solutions:
## 1. [2nd, 三个臭皮匠's approach(s)](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44558)

## 2. [18th, Place Solution - Careful Ensembling + Resampling Diversity](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44579)
### (1)Frame:
- **16 base models, stacked with LR**（过于花哨的stacking可能会带来比较严重的过拟合）
- **model diversity**：1) different models（or different hyper-parameter）, 2) different features 3) different resampling of the training data
- **resampling strategies**: Snow Dog.
