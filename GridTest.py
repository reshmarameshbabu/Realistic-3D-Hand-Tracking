import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

testLabelsFile = 'H:/Documents/Datasets/Hands2017/trackingLabels.csv'
dfLabels = pd.read_csv(testLabelsFile)
dfPred = pd.read_csv("outputTrack.csv")
nb_samples = 30000
errors = np.zeros(11)

for i in range(0, nb_samples):
    label = np.asarray(dfLabels.iloc[i, 1:].values, dtype='float32')
    label = np.reshape(label, (21, 3))
    preds = np.asarray(dfPred.iloc[i].values, dtype='float32')
    for j in range(0, 11):
        pred = preds[(j * 63):(j + 1) * 63]
        pred = np.reshape(pred, (21, 3))
        sumOfError = 0.
        for k in range(0, 21):
            sumOfError = sumOfError + np.linalg.norm(pred[k] - label[k])

        errors[j] += sumOfError / 21.0
    if i % 100 == 0:
        print(i)

errors = errors / nb_samples
plt.plot(errors)
plt.ylabel('Error')
plt.show()
