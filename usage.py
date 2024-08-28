"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
# Test case 1
# Real Input and Real Output

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_hat, y))
    print("MAE: ", mae(y_hat, y))

# Test case 2
# Real Input and Discrete Output

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))


# Test case 3
# Discrete Input and Discrete Output

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size=N), dtype="category")

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("Accuracy: ", accuracy(y_hat, y))
    for cls in y.unique():
        print("Precision: ", precision(y_hat, y, cls))
        print("Recall: ", recall(y_hat, y, cls))

# Test case 4
# Discrete Input and Real Output

N = 30
P = 5
X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

for criteria in ["information_gain", "gini_index"]:
    tree = DecisionTree(criterion=criteria)  # Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print("Criteria :", criteria)
    print("RMSE: ", rmse(y_hat, y))
    print("MAE: ", mae(y_hat, y))


"""
OUTPUT:
real real
?(X[1] <= -1.1939721441501923)
  Y:
    Prediction: 2.720169166589619
  N:
    ?(X[4] <= 0.3865751136808898)
      Y:
        ?(X[0] <= 1.5077915869695468)
          Y:
            ?(X[2] <= 1.03032756205547)
              Y:
                ?(X[3] <= 0.4176142278414079)
                  Y:
                    Prediction: -0.3818098280800404
                  N:
                    Prediction: 0.28744229051350034
              N:
                ?(X[3] <= -0.08732596435259299)
                  Y:
                    Prediction: 0.82206015999449
                  N:
                    Prediction: 0.6565536086338297
          N:
            ?(X[2] <= -0.44417962290422375)
              Y:
                Prediction: 0.8271832490360238
              N:
                Prediction: 1.4535340771573169
      N:
        ?(X[0] <= -0.35665559739723524)
          Y:
            ?(X[2] <= -0.5599160993719811)
              Y:
                Prediction: 1.158595579007404
              N:
                ?(X[3] <= -0.7937355663614694)
                  Y:
                    Prediction: 1.8657745111447566
                  N:
                    Prediction: 1.8967929826539474
          N:
            ?(X[2] <= -0.4930671880785866)
              Y:
                ?(X[3] <= 0.8772495516779442)
                  Y:
                    Prediction: -0.9746816702273214
                  N:
                    Prediction: 0.3411519748166435
              N:
                ?(X[3] <= -0.06910547726569408)
                  Y:
                    Prediction: 0.31156950441349746
                  N:
                    Prediction: 0.787084603742452
Criteria : information_gain
RMSE:  0.38854961933943377
MAE:  0.2602377720907791
real real
?(X[1] <= -1.1939721441501923)
  Y:
    Prediction: 2.720169166589619
  N:
    ?(X[4] <= 0.3865751136808898)
      Y:
        ?(X[0] <= 1.5077915869695468)
          Y:
            ?(X[2] <= 1.03032756205547)
              Y:
                ?(X[3] <= 0.4176142278414079)
                  Y:
                    Prediction: -0.3818098280800404
                  N:
                    Prediction: 0.28744229051350034
              N:
                ?(X[3] <= -0.08732596435259299)
                  Y:
                    Prediction: 0.82206015999449
                  N:
                    Prediction: 0.6565536086338297
          N:
            ?(X[2] <= -0.44417962290422375)
              Y:
                Prediction: 0.8271832490360238
              N:
                Prediction: 1.4535340771573169
      N:
        ?(X[0] <= -0.35665559739723524)
          Y:
            ?(X[2] <= -0.5599160993719811)
              Y:
                Prediction: 1.158595579007404
              N:
                ?(X[3] <= -0.7937355663614694)
                  Y:
                    Prediction: 1.8657745111447566
                  N:
                    Prediction: 1.8967929826539474
          N:
            ?(X[2] <= -0.4930671880785866)
              Y:
                ?(X[3] <= 0.8772495516779442)
                  Y:
                    Prediction: -0.9746816702273214
                  N:
                    Prediction: 0.3411519748166435
              N:
                ?(X[3] <= -0.06910547726569408)
                  Y:
                    Prediction: 0.31156950441349746
                  N:
                    Prediction: 0.787084603742452
Criteria : gini_index
RMSE:  0.38854961933943377
MAE:  0.2602377720907791
real discrete
?(X[0] <= 0.5164969924782189)
  Y:
    ?(X[1] <= 1.0083193995209832)
      Y:
        ?(X[2] <= -0.7357749579035922)
          Y:
            ?(X[3] <= 0.5339998171755852)
              Y:
                Prediction: 0
              N:
                Prediction: 2
          N:
            ?(X[4] <= 2.998337789990023)
              Y:
                ?(X[3] <= -0.9447079232163329)
                  Y:
                    Prediction: 0
                  N:
                    Prediction: 1
              N:
                Prediction: 4
      N:
        ?(X[2] <= -0.6496204272273054)
          Y:
            ?(X[3] <= 0.0018874707246223366)
              Y:
                Prediction: 4
              N:
                Prediction: 3
          N:
            Prediction: 4
  N:
    ?(X[1] <= -1.9462038896246776)
      Y:
        Prediction: 3
      N:
        ?(X[4] <= -0.7267202584186923)
          Y:
            Prediction: 4
          N:
            ?(X[2] <= 0.22472079181725235)
              Y:
                ?(X[3] <= -0.6003336285445359)
                  Y:
                    Prediction: 2
                  N:
                    Prediction: 4
              N:
                Prediction: 2
Criteria : information_gain
Accuracy:  0.8666666666666667
Precision:  100.0
Recall:  90.0
Precision:  75.0
Recall:  90.0
Precision:  100.0
Recall:  80.0
Precision:  100.0
Recall:  100.0
Precision:  66.66666666666666
Recall:  66.66666666666666
real discrete
?(X[0] <= 0.5164969924782189)
  Y:
    ?(X[1] <= 1.0083193995209832)
      Y:
        ?(X[2] <= -0.7357749579035922)
          Y:
            ?(X[3] <= 0.5339998171755852)
              Y:
                Prediction: 0
              N:
                Prediction: 2
          N:
            ?(X[4] <= 2.998337789990023)
              Y:
                ?(X[3] <= -0.9447079232163329)
                  Y:
                    Prediction: 0
                  N:
                    Prediction: 1
              N:
                Prediction: 4
      N:
        ?(X[2] <= -0.6496204272273054)
          Y:
            ?(X[3] <= 0.0018874707246223366)
              Y:
                Prediction: 4
              N:
                Prediction: 3
          N:
            Prediction: 4
  N:
    ?(X[1] <= -1.9462038896246776)
      Y:
        Prediction: 3
      N:
        ?(X[4] <= -0.7267202584186923)
          Y:
            Prediction: 4
          N:
            ?(X[2] <= 0.22472079181725235)
              Y:
                ?(X[3] <= -0.6003336285445359)
                  Y:
                    Prediction: 2
                  N:
                    Prediction: 4
              N:
                Prediction: 2
Criteria : gini_index
Accuracy:  0.8666666666666667
Precision:  100.0
Recall:  90.0
Precision:  75.0
Recall:  90.0
Precision:  100.0
Recall:  80.0
Precision:  100.0
Recall:  100.0
Precision:  66.66666666666666
Recall:  66.66666666666666
One hot encoding the features!
discrete discrete
?(X[1] <= 4)
  Value: 0
    ?(X[4] <= 4)
      Value: 0
        Prediction: 0
      Value: 1
        ?(X[0] <= 3)
          Value: 0
            Prediction: 0
          Value: 1
            Prediction: 0
          Value: 2
            Prediction: 0
          Value: 3
            ?(X[2] <= 4)
              Value: 0
                Prediction: 0
              Value: 1
                Prediction: 0
              Value: 2
                Prediction: 0
              Value: 3
                Prediction: 0
              Value: 4
                ?(X[3] <= 0)
                  Value: 0
                    Prediction: 0
                  Value: 1
                    Prediction: 0
                  Value: 2
                    Prediction: 0
                  Value: 3
                    Prediction: 0
                  Value: 4
                    Prediction: 0
          Value: 4
            Prediction: 0
      Value: 2
        Prediction: 4
      Value: 3
        Prediction: 4
      Value: 4
        ?(X[0] <= 3)
          Value: 0
            Prediction: 4
          Value: 1
            Prediction: 4
          Value: 2
            Prediction: 4
          Value: 3
            Prediction: 0
          Value: 4
            Prediction: 4
  Value: 1
    ?(X[0] <= 3)
      Value: 0
        Prediction: 3
      Value: 1
        Prediction: 2
      Value: 2
        Prediction: 2
      Value: 3
        Prediction: 2
      Value: 4
        Prediction: 2
  Value: 2
    ?(X[2] <= 4)
      Value: 0
        Prediction: 0
      Value: 1
        Prediction: 1
      Value: 2
        Prediction: 1
      Value: 3
        Prediction: 0
      Value: 4
        Prediction: 2
  Value: 3
    ?(X[4] <= 4)
      Value: 0
        Prediction: 3
      Value: 1
        Prediction: 0
      Value: 2
        Prediction: 2
      Value: 3
        Prediction: 3
      Value: 4
        ?(X[0] <= 3)
          Value: 0
            Prediction: 0
          Value: 1
            Prediction: 2
          Value: 2
            Prediction: 0
          Value: 3
            Prediction: 0
          Value: 4
            Prediction: 0
  Value: 4
    ?(X[0] <= 4)
      Value: 0
        Prediction: 3
      Value: 1
        Prediction: 1
      Value: 2
        Prediction: 1
      Value: 3
        ?(X[2] <= 3)
          Value: 0
            Prediction: 1
          Value: 1
            Prediction: 3
          Value: 2
            Prediction: 1
          Value: 3
            Prediction: 1
          Value: 4
            Prediction: 1
      Value: 4
        Prediction: 1
Criteria : information_gain
Accuracy:  0.9666666666666667
Precision:  87.5
Recall:  100.0
Precision:  100.0
Recall:  100.0
Precision:  100.0
Recall:  88.88888888888889
Precision:  100.0
Recall:  100.0
Precision:  100.0
Recall:  100.0
One hot encoding the features!
discrete discrete
?(X[1] <= 4)
  Value: 0
    ?(X[4] <= 4)
      Value: 0
        Prediction: 0
      Value: 1
        ?(X[0] <= 3)
          Value: 0
            Prediction: 0
          Value: 1
            Prediction: 0
          Value: 2
            Prediction: 0
          Value: 3
            ?(X[2] <= 4)
              Value: 0
                Prediction: 0
              Value: 1
                Prediction: 0
              Value: 2
                Prediction: 0
              Value: 3
                Prediction: 0
              Value: 4
                ?(X[3] <= 0)
                  Value: 0
                    Prediction: 0
                  Value: 1
                    Prediction: 0
                  Value: 2
                    Prediction: 0
                  Value: 3
                    Prediction: 0
                  Value: 4
                    Prediction: 0
          Value: 4
            Prediction: 0
      Value: 2
        Prediction: 4
      Value: 3
        Prediction: 4
      Value: 4
        ?(X[0] <= 3)
          Value: 0
            Prediction: 4
          Value: 1
            Prediction: 4
          Value: 2
            Prediction: 4
          Value: 3
            Prediction: 0
          Value: 4
            Prediction: 4
  Value: 1
    ?(X[0] <= 3)
      Value: 0
        Prediction: 3
      Value: 1
        Prediction: 2
      Value: 2
        Prediction: 2
      Value: 3
        Prediction: 2
      Value: 4
        Prediction: 2
  Value: 2
    ?(X[2] <= 4)
      Value: 0
        Prediction: 0
      Value: 1
        Prediction: 1
      Value: 2
        Prediction: 1
      Value: 3
        Prediction: 0
      Value: 4
        Prediction: 2
  Value: 3
    ?(X[4] <= 4)
      Value: 0
        Prediction: 3
      Value: 1
        Prediction: 0
      Value: 2
        Prediction: 2
      Value: 3
        Prediction: 3
      Value: 4
        ?(X[0] <= 3)
          Value: 0
            Prediction: 0
          Value: 1
            Prediction: 2
          Value: 2
            Prediction: 0
          Value: 3
            Prediction: 0
          Value: 4
            Prediction: 0
  Value: 4
    ?(X[0] <= 4)
      Value: 0
        Prediction: 3
      Value: 1
        Prediction: 1
      Value: 2
        Prediction: 1
      Value: 3
        ?(X[2] <= 3)
          Value: 0
            Prediction: 1
          Value: 1
            Prediction: 3
          Value: 2
            Prediction: 1
          Value: 3
            Prediction: 1
          Value: 4
            Prediction: 1
      Value: 4
        Prediction: 1
Criteria : gini_index
Accuracy:  0.9666666666666667
Precision:  87.5
Recall:  100.0
Precision:  100.0
Recall:  100.0
Precision:  100.0
Recall:  88.88888888888889
Precision:  100.0
Recall:  100.0
Precision:  100.0
Recall:  100.0
One hot encoding the features!
discrete real
?(X[1] <= 4)
  Value: 0
    ?(X[0] <= 4)
      Value: 0
        Prediction: 1.1677820616598074
      Value: 1
        Prediction: -0.42098448082026296
      Value: 2
        ?(X[2] <= 4)
          Value: 0
            Prediction: -0.6042524694362549
          Value: 1
            Prediction: -0.7968952554704768
          Value: 2
            Prediction: -0.6042524694362549
          Value: 3
            Prediction: -0.4118769661224674
          Value: 4
            Prediction: -0.6039851867158206
      Value: 3
        Prediction: -0.15567723539207948
      Value: 4
        ?(X[4] <= 4)
          Value: 0
            Prediction: -0.43738219393415667
          Value: 1
            Prediction: -1.129706854657618
          Value: 2
            Prediction: -0.5768918695231487
          Value: 3
            Prediction: -0.43738219393415667
          Value: 4
            Prediction: 0.39445214237829684
  Value: 1
    ?(X[2] <= 4)
      Value: 0
        Prediction: -0.48760622407249354
      Value: 1
        Prediction: -2.4716445001272893
      Value: 2
        ?(X[0] <= 4)
          Value: 0
            Prediction: 0.12552429750634944
          Value: 1
            Prediction: 0.12552429750634944
          Value: 2
            Prediction: 0.57707212718054
          Value: 3
            Prediction: 0.12552429750634944
          Value: 4
            Prediction: -0.32602353216784113
      Value: 3
        ?(X[0] <= 4)
          Value: 0
            Prediction: -0.4607482617222521
          Value: 1
            Prediction: -0.4607482617222521
          Value: 2
            Prediction: -0.4607482617222521
          Value: 3
            Prediction: 0.08658978747289992
          Value: 4
            Prediction: -1.008086310917404
      Value: 4
        ?(X[0] <= 4)
          Value: 0
            Prediction: 0.5167466238181745
          Value: 1
            Prediction: 0.5167466238181745
          Value: 2
            Prediction: 0.5167466238181745
          Value: 3
            Prediction: -0.4080753730215514
          Value: 4
            Prediction: 1.4415686206579004
  Value: 2
    ?(X[0] <= 2)
      Value: 0
        Prediction: 0.37114587337130883
      Value: 1
        Prediction: 0.30511385785631157
      Value: 2
        ?(X[3] <= 2)
          Value: 0
            Prediction: 0.2544208433012131
          Value: 1
            Prediction: 0.272097850098813
          Value: 2
            Prediction: 0.28977485689641286
          Value: 3
            Prediction: 0.272097850098813
          Value: 4
            Prediction: 0.272097850098813
      Value: 3
        Prediction: 0.30511385785631157
      Value: 4
        Prediction: 0.30511385785631157
  Value: 3
    ?(X[2] <= 3)
      Value: 0
        ?(X[3] <= 3)
          Value: 0
            Prediction: 0.38432786717216194
          Value: 1
            Prediction: 0.38432786717216194
          Value: 2
            Prediction: -0.4325581878196209
          Value: 3
            Prediction: 1.2012139221639448
          Value: 4
            Prediction: 0.38432786717216194
      Value: 1
        Prediction: 0.8711247034316923
      Value: 2
        Prediction: 0.5423961539228248
      Value: 3
        Prediction: 0.5298041779152828
      Value: 4
        Prediction: 0.5423961539228248
  Value: 4
    ?(X[3] <= 4)
      Value: 0
        Prediction: 0.33563641012989603
      Value: 1
        ?(X[0] <= 3)
          Value: 0
            Prediction: 0.4016237115857239
          Value: 1
            Prediction: 0.4016237115857239
          Value: 2
            Prediction: -0.2030453860429927
          Value: 3
            Prediction: 1.0062928092144405
          Value: 4
            Prediction: 0.4016237115857239
      Value: 2
        Prediction: -2.038124535177854
      Value: 3
        Prediction: 0.8356921120651418
      Value: 4
        ?(X[2] <= 3)
          Value: 0
            Prediction: 1.2065017303603203
          Value: 1
            Prediction: 0.3376026620752022
          Value: 2
            Prediction: 1.2065017303603203
          Value: 3
            Prediction: 2.0754007986454384
          Value: 4
            Prediction: 1.2065017303603203
Criteria : information_gain
RMSE:  0.0
MAE:  0.0
One hot encoding the features!
discrete real
?(X[1] <= 4)
  Value: 0
    ?(X[0] <= 4)
      Value: 0
        Prediction: 1.1677820616598074
      Value: 1
        Prediction: -0.42098448082026296
      Value: 2
        ?(X[2] <= 4)
          Value: 0
            Prediction: -0.6042524694362549
          Value: 1
            Prediction: -0.7968952554704768
          Value: 2
            Prediction: -0.6042524694362549
          Value: 3
            Prediction: -0.4118769661224674
          Value: 4
            Prediction: -0.6039851867158206
      Value: 3
        Prediction: -0.15567723539207948
      Value: 4
        ?(X[4] <= 4)
          Value: 0
            Prediction: -0.43738219393415667
          Value: 1
            Prediction: -1.129706854657618
          Value: 2
            Prediction: -0.5768918695231487
          Value: 3
            Prediction: -0.43738219393415667
          Value: 4
            Prediction: 0.39445214237829684
  Value: 1
    ?(X[2] <= 4)
      Value: 0
        Prediction: -0.48760622407249354
      Value: 1
        Prediction: -2.4716445001272893
      Value: 2
        ?(X[0] <= 4)
          Value: 0
            Prediction: 0.12552429750634944
          Value: 1
            Prediction: 0.12552429750634944
          Value: 2
            Prediction: 0.57707212718054
          Value: 3
            Prediction: 0.12552429750634944
          Value: 4
            Prediction: -0.32602353216784113
      Value: 3
        ?(X[0] <= 4)
          Value: 0
            Prediction: -0.4607482617222521
          Value: 1
            Prediction: -0.4607482617222521
          Value: 2
            Prediction: -0.4607482617222521
          Value: 3
            Prediction: 0.08658978747289992
          Value: 4
            Prediction: -1.008086310917404
      Value: 4
        ?(X[0] <= 4)
          Value: 0
            Prediction: 0.5167466238181745
          Value: 1
            Prediction: 0.5167466238181745
          Value: 2
            Prediction: 0.5167466238181745
          Value: 3
            Prediction: -0.4080753730215514
          Value: 4
            Prediction: 1.4415686206579004
  Value: 2
    ?(X[0] <= 2)
      Value: 0
        Prediction: 0.37114587337130883
      Value: 1
        Prediction: 0.30511385785631157
      Value: 2
        ?(X[3] <= 2)
          Value: 0
            Prediction: 0.2544208433012131
          Value: 1
            Prediction: 0.272097850098813
          Value: 2
            Prediction: 0.28977485689641286
          Value: 3
            Prediction: 0.272097850098813
          Value: 4
            Prediction: 0.272097850098813
      Value: 3
        Prediction: 0.30511385785631157
      Value: 4
        Prediction: 0.30511385785631157
  Value: 3
    ?(X[2] <= 3)
      Value: 0
        ?(X[3] <= 3)
          Value: 0
            Prediction: 0.38432786717216194
          Value: 1
            Prediction: 0.38432786717216194
          Value: 2
            Prediction: -0.4325581878196209
          Value: 3
            Prediction: 1.2012139221639448
          Value: 4
            Prediction: 0.38432786717216194
      Value: 1
        Prediction: 0.8711247034316923
      Value: 2
        Prediction: 0.5423961539228248
      Value: 3
        Prediction: 0.5298041779152828
      Value: 4
        Prediction: 0.5423961539228248
  Value: 4
    ?(X[3] <= 4)
      Value: 0
        Prediction: 0.33563641012989603
      Value: 1
        ?(X[0] <= 3)
          Value: 0
            Prediction: 0.4016237115857239
          Value: 1
            Prediction: 0.4016237115857239
          Value: 2
            Prediction: -0.2030453860429927
          Value: 3
            Prediction: 1.0062928092144405
          Value: 4
            Prediction: 0.4016237115857239
      Value: 2
        Prediction: -2.038124535177854
      Value: 3
        Prediction: 0.8356921120651418
      Value: 4
        ?(X[2] <= 3)
          Value: 0
            Prediction: 1.2065017303603203
          Value: 1
            Prediction: 0.3376026620752022
          Value: 2
            Prediction: 1.2065017303603203
          Value: 3
            Prediction: 2.0754007986454384
          Value: 4
            Prediction: 1.2065017303603203
Criteria : gini_index
RMSE:  0.0
MAE:  0.0
"""