from sklearn import preprocessing
import numpy as np
X_train = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
X_scaled = preprocessing.scale(X_train)

X_scaled

X_scaled.mean(axis=0)
X_scaled.std(axis=0)

scaler = preprocessing.StandardScaler().fit(X_train)
scaler
scaler.mean_
scaler.scale_
scaler.transform(X_train)

X_test = [[-1., 1., 0.]]
scaler.transform(X_test)       
