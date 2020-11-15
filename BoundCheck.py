import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import correction, minmaxCheck, correctionDebug, correctionCheckCombinedDebug

fo = open("result_HANDS2017.txt", "r")
# fo = open("result_ensemble.txt", "r")
line = fo.readlines()

dfPred = pd.read_csv("outputJ.csv")
testFrames = 'H:/Documents/Datasets/Hands2017/TestFrameLabels.csv'
dfLabels = pd.read_csv(testFrames)


firstFrame = True
nb_samples = 1000 #len(line)
errors = np.zeros(21)
rerrors = np.zeros(21)
# errors = 0.0
# rerror = 0.0
previousPose = np.zeros(22)
numofhighs = 0
numofmed = 0
numoflow = 0

for i in range(0, nb_samples):

    label = np.asarray(dfLabels.iloc[i, 1:].values, dtype='float32')
    # label = np.reshape(label, (21, 3))
    # lst = line[i].split()
    # pred = np.asarray([float(j) for j in lst[1:]])
    # pred = np.reshape(pred, (21, 3))


    label = np.asarray(dfLabels.iloc[i, 1:].values, dtype='float32')
    label = np.reshape(label, (21, 3))
    preds = np.asarray(dfPred.iloc[i].values, dtype='float32')
    pred = preds[0:63]
    pred = np.reshape(pred, (21, 3))

    l = 0
    # for val in np.arange(0, 1.1, 0.1):
    #     # alpha = val
    #     # alpha = 1.0
    predJoints, rerror = correctionCheckCombinedDebug(pred, 0)
        # rerrors[k] += rerror
    if rerror < 50:
        numoflow += 1
    elif rerror < 100:
        numofmed += 1
    else:
        numofhighs += 1
        # sumOfError = 0.
        # for k in range(0, 21):
        #     sumOfError = sumOfError + np.linalg.norm(predJoints[k] - label[k])
        # errors[l] += sumOfError / 21.0
    # l += 1
    if i % 5 == 0:
        print(i)

# totalError = errors / nb_samples
# print(totalError)
# totalError = rerrors / nb_samples
# print(totalError)
print(numofhighs) #38% 360
print(numofmed)
print(numoflow)
