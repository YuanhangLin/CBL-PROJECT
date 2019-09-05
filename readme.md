Linear mapping with test cases
- PixMat
- RandMat(sampling by shuffling/uniform distribution)

CBL utilities with test cases
- Graph Convolution
- Polygon Aggregation
	- Linear, Non-linear, Weighted, USER-DEFINED
- some other helper functions

CBL Missing data
- Exp1 : replace missing data(NaN/0) with sensor mean, do not update specific layer if its corresponding sensor data is missing

Some experiments for Ron's workshop paper
- Effect of CDA, PixMat on balance/unbalance training set
- CDA is the most important factor
- PixMat/RandMat without CDA using KNN, ~40% accuracy
- PixMat/RandMat with CDA using KNN, ~70% accuracy

Two kinds of Graph Convolution Network
(1) Spectral-based Graph Convolution Network
(2) Spatial-based Graph Convolution Network