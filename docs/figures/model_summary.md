# Model comparison on the shared validation set

| model                  | config                              |   val_acc |   macro_f1 |
|:-----------------------|:------------------------------------|----------:|-----------:|
| CompactSignalCNN (OOF) | ~0.9M params, single-channel 128x64 |     57.32 |      56.04 |
| RandomForest           | n_estimators=400                    |     37.26 |      34.79 |
| LogReg                 | C=1.0                               |     28.48 |      24.28 |
| SVM-RBF                | C=1.0, gamma=0.01 (n=8000)          |     28.26 |      25.83 |
| Linear-SVM             | C=1.0                               |     28.06 |      22.04 |
| k-NN                   | k=25, weights=uniform               |     24.48 |      21.93 |

## k-NN hyperparameter grid

|   k | weights   |   val_acc |
|----:|:----------|----------:|
|   1 | uniform   |     22.35 |
|   1 | distance  |     22.35 |
|   3 | uniform   |     22.97 |
|   3 | distance  |     23    |
|   5 | uniform   |     22.94 |
|   5 | distance  |     22.71 |
|   9 | uniform   |     22.23 |
|   9 | distance  |     22.23 |
|  15 | uniform   |     23.23 |
|  15 | distance  |     23.32 |
|  25 | uniform   |     24.48 |
|  25 | distance  |     24.26 |
|  49 | uniform   |     23.16 |
|  49 | distance  |     23.58 |

## SVM-RBF hyperparameter grid

|    C | gamma   |   val_acc |
|-----:|:--------|----------:|
|  0.5 | scale   |     27.87 |
|  0.5 | 0.01    |     27.9  |
|  0.5 | 0.1     |     22.42 |
|  1   | scale   |     28.03 |
|  1   | 0.01    |     28.26 |
|  1   | 0.1     |     23    |
|  5   | scale   |     28.1  |
|  5   | 0.01    |     28.16 |
|  5   | 0.1     |     25.48 |
| 10   | scale   |     27.9  |
| 10   | 0.01    |     27.87 |
| 10   | 0.1     |     25.39 |
