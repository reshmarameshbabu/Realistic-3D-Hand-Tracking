import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
batch_size = 16

# trainLabelsFile = 'H:/Documents/Datasets/Hands2017/trialSet/train.csv'
# testLabelsFile = 'H:/Documents/Datasets/Hands2017/trialSet/test.csv'
# trainImagesFolder = 'H:/Documents/Datasets/Hands2017/trialSet/train'
# testImagesFolder = 'H:/Documents/Datasets/Hands2017/trialSet/test'

# trainLabelsFile = 'H:/Documents/Datasets/Hands2017/trialSet/trainLabelCropped.csv'
# trainImagesFolder = 'H:/Documents/Datasets/Hands2017/trialSet/trainCropped/'
trainLabelsFile = 'H:/Documents/Datasets/Hands2017/Cropped/trainLabelCropped.csv'
trainImagesFolder = 'H:/Documents/Datasets/Hands2017/Cropped/train/'
testLabelsFile = 'H:/Documents/Datasets/Hands2017/Cropped/testLabelCropped.csv'
testImagesFolder = 'H:/Documents/Datasets/Hands2017/Cropped/test/'
cropSize = 176


def preprocess(image):
    mark = image[0, 0].copy()
    image[0, 0] = (image[0, 1] + image[1, 0]) / 2.
    image = image - mark
    image = image / 255.
    return image


datagen = ImageDataGenerator(preprocessing_function=preprocess)
                             # validation_split=0.10)
features = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13", "F14", "F15", "F16",
            "F17", "F18", "F19", "F20", "F21", "F22", "F23", "F24", "F25", "F26", "F27", "F28", "F29", "F30", "F31",
            "F32", "F33", "F34", "F35", "F36", "F37", "F38", "F39", "F40", "F41", "F42", "F43", "F44", "F45", "F46",
            "F47", "F48", "F49", "F50", "F51", "F52", "F53", "F54", "F55", "F56", "F57", "F58", "F59", "F60", "F61",
            "F62", "F63"]


def GetGenerators():
    traindf = pd.read_csv(trainLabelsFile, dtype=str)
    # load and iterate training dataset
    train_generator = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=trainImagesFolder,
        x_col="ID",
        y_col=features,
        subset="training",
        batch_size=batch_size,  # 32,
        # seed=97,
        shuffle=True,
        class_mode='raw',
        target_size=(cropSize, cropSize))

    # load and iterate validation dataset
    validation_generator = datagen.flow_from_dataframe(
        dataframe=traindf,
        directory=trainImagesFolder,
        x_col="ID",
        y_col=features,
        subset="validation",
        batch_size=batch_size,  # 32,
        shuffle=True,
        class_mode='raw',
        target_size=(cropSize, cropSize))
    return train_generator, validation_generator




def GetTestGenerator():
    # load and iterate test dataset
    testdf = pd.read_csv(testLabelsFile, dtype=str)
    test_generator = datagen.flow_from_dataframe(
        dataframe=testdf,
        directory=testImagesFolder,
        x_col="ID",
        y_col=features,
        batch_size=16,
        class_mode='raw',
        target_size=(cropSize, cropSize))
    return test_generator

