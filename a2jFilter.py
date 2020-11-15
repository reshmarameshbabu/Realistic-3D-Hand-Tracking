import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import correction, minmaxCheck, correctionDebug, correctionCheckCombinedDebug, DoF

# fo = open("result_HANDS2017.txt", "r")
fo = open("result_ensemble.txt", "r")
testLabelsFile = 'H:/Documents/Datasets/Hands2017/trackingLabels.csv'
testFrames = 'H:/Documents/Datasets/Hands2017/TestFrameLabels.csv'
dfLabels = pd.read_csv(testFrames)
line = fo.readlines()

firstFrame = True
nb_samples = 100  # len(line)
# errors = np.zeros(20)
# rerrors = np.zeros(20)
errors = 0.0
# rerror = 0.0
previousPose = np.zeros(22)
numofhighs = np.zeros(11)
for i in range(0, nb_samples):
    label = np.asarray(dfLabels.iloc[i, 1:].values, dtype='float32')
    label = np.reshape(label, (21, 3))
    lst = line[i].split()
    pred = np.asarray([float(j) for j in lst[1:]])
    pred = np.reshape(pred, (21, 3))
    # for val in range(0, 11):
    #     alpha = val * 0.1
    # # alpha = 1.0
    #     predJoints, rerror = correctionCheckCombinedDebug(pred, alpha)
    #     rerrors[val] += rerror
    #     sumOfError = 0.
    #     for k in range(0, 21):
    #         sumOfError = sumOfError + np.linalg.norm(predJoints[k] - label[k])
    #     errors[val] += sumOfError / 21.0
    # l = 0
    # for val in np.arange(0, 1.1, 0.1):
    # alpha = val
    # alpha = 1.0
    # predJoints, rerror = correctionCheckCombinedDebug(pred, val)
    # rerrors[k] += rerror
    # if rerror > 100:
    #     numofhighs[l] += 1
    sumOfError = 0.
    p = DoF(pred)
    t = DoF(label)
    for k in range(0, 22):
        sumOfError = sumOfError + np.abs(p[k] - t[k])

    errors += sumOfError / 22.0
    # l += 1
    # errors += sumOfError / 21.0
    if i % 10 == 0:
        print(i)

# totalError = errors / nb_samples
# print(totalError)
# totalError = rerrors / nb_samples
# print(totalError)

# print(numofhighs)  # a2j 591 v2v 600
errors = errors / nb_samples
print(errors)
# plt.plot(errors)
# plt.ylabel('Error')
# plt.show() 9.6944
