import pandas as pd
import numpy as np
from utils import DoF

testLabelsFile = 'H:/Documents/Datasets/Hands2017/trackingLabels.csv'
dfLabels = pd.read_csv(testLabelsFile)

prev = DoF(np.reshape(np.asarray(dfLabels.iloc[0, 1:].values, dtype='float32'), (21, 3)))
sum = np.zeros((21))
for i in range(1, 3000):
    label = np.asarray(dfLabels.iloc[i, 1:].values, dtype='float32')
    label = np.reshape(label, (21, 3))
    current = DoF(label)
    print(np.max(np.abs(current - prev)/(1/15.0)))
    prev = current
    # if (i%100 ==0):
    #     print(i)

# sum = sum / 3000
# print(sum)
