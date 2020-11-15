import csv

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from ModelDesign import GetModelPre
from smallestenclosingcircle import make_circle

from utils import world2pixel, fx, fy, u0, v0, cropSize, depth_thres, pixel2world, correction

# testLabelsFile = 'H:/Documents/Datasets/Hands2017/Cropped/testLabelCropped.csv'
# testImagesFolder = 'H:/Documents/Datasets/Hands2017/Cropped/test/'
testImagesFolder = 'H:/Documents/Datasets/Hands2017/frame/images/'
testLabelsFile = 'H:/Documents/Datasets/Hands2017/TestFrameLabels.csv'
testCentersFile = 'H:/Documents/Datasets/Hands2017/Cropped/centersTest.csv'

print('Loading Files')
model = GetModelPre('weightsBoneG2.h5')
dfLabels = pd.read_csv(testLabelsFile)
imageList = dfLabels.iloc[:, 0].values
nb_samples = len(imageList)

print('Testing Model')

error = 0.0
csvfile = "outputu.csv"

#16.44908

def preprocess(image):
    mark = image[0, 0].copy()
    image[0, 0] = (image[0, 1] + image[1, 0]) / 2.
    image = image - mark
    image = image / 255.
    return image



with open(csvfile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(['X', 'Y', 'R'])
    for i in range(0, nb_samples):
        image = np.array(Image.open(testImagesFolder + imageList[i]))
        label = np.asarray(dfLabels.iloc[i, 1:].values, dtype='float32')

        labelP = world2pixel(label.reshape((21, 3)).copy(), fx, fy, u0, v0)

        circx, circy, radius = make_circle(labelP[:, 0:2].copy())

        center = np.asarray([circx, circy])
        radius = radius + 16
        lefttop_pixel = center - radius
        rightbottom_pixel = center + radius

        new_Xmin = max(lefttop_pixel[0], 0)
        new_Ymin = max(lefttop_pixel[1], 0)
        new_Xmax = min(rightbottom_pixel[0], image.shape[1] - 1)
        new_Ymax = min(rightbottom_pixel[1], image.shape[0] - 1)
        # print([new_Xmin, new_Xmax, new_Ymin, new_Ymax])
        if new_Xmin > 640 or abs(new_Xmin - new_Xmax) < 20:
            continue

        imCrop = image.copy()[int(new_Ymin):int(new_Ymax), int(new_Xmin):int(new_Xmax)]
        imgResize = np.asarray(cv2.resize(imCrop, (cropSize, cropSize), interpolation=cv2.INTER_NEAREST),
                               dtype='float32')
        c = sum(labelP.reshape((21, 3))) / 21.
        centerDepth = c[2]
        imgResize[np.where(imgResize >= centerDepth + depth_thres)] = centerDepth
        imgResize[np.where(imgResize <= centerDepth - depth_thres)] = centerDepth
        maxPixel = imgResize.max()

        mark = centerDepth / maxPixel
        imgResize = (imgResize / maxPixel) - mark
        trueJoints = np.reshape(label, (21, 3))
        # imgResize = preprocess(image)
        img = np.squeeze(np.stack((imgResize,) * 3, -1))
        img = np.reshape(img, (1, 176, 176, 3))
        pred = model.predict(img)

        predJoints = np.reshape(pred, (21, 3)) * 10

        predJoints[:, 0] = (predJoints[:, 0] * (new_Xmax - new_Xmin) / cropSize) + new_Xmin
        predJoints[:, 1] = (predJoints[:, 1] * (new_Ymax - new_Ymin) / cropSize) + new_Ymin
        predJoints[:, 2] = predJoints[:, 2] + centerDepth
        predJoints = pixel2world(predJoints, fx, fy, u0, v0)

        fullList = []
        for val in range(0, 11):
            alpha = val * 0.1
            predJoints = correction(predJoints, alpha)
            temp = np.reshape(predJoints, (63,))
            fullList.append(temp)
            # writer.writerow(temp)
            # sumOfError = 0.
            # for j in range(0, 21):
            #     sumOfError = sumOfError + np.linalg.norm(predJoints[j] - trueJoints[j])
            #     # print(np.linalg.norm(predJoints[j] - trueJoints[j]))
            # error += sumOfError / 21.0
        writer.writerow(x for x in np.nditer(np.asarray(fullList)))
        if i % 100 == 0:
            print(i)

# overall = error / nb_samples
# print(overall)

