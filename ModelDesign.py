import tensorflow as tf
from keras import Model, optimizers
from keras.layers import Conv2D, Dropout, Input, GlobalAveragePooling2D, BatchNormalization, Activation, add, \
    ZeroPadding2D, MaxPooling2D, Lambda, MaxPool2D, concatenate, AveragePooling2D, Flatten, Dense
from keras.applications import resnet50
from keras.models import load_model


cropSize = 176
resnet_model = resnet50.ResNet50(weights='imagenet', input_shape=(cropSize, cropSize, 3), include_top=False)

WEIGHTS_PATH = ('https://github.com/fchollet/deep-learning-models/'
                'releases/download/v0.2/'
                'resnet50_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.2/'
                       'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')


sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
sgd2 = optimizers.SGD(lr=1e-6, decay=1e-8, momentum=0.9, nesterov=True)
sgd3 = optimizers.SGD(lr=0.003, decay=1e-6, momentum=0.9, nesterov=True)

adam = optimizers.adam(learning_rate=0.00035, amsgrad=True)  # 0.00035
nadam = optimizers.Nadam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

opt = adam


def BoneFilter(tj, pj):
    def boneError(t, p, i, j):
        bonep = tf.norm(p[i] - p[j], ord='euclidean')
        bonet = tf.norm(t[i] - t[j], ord='euclidean')
        return tf.abs(bonep - bonet)

    cost = tf.constant(0, dtype=tf.float32)
    cost = cost + boneError(tj, pj, 0, 1)
    cost = cost + boneError(tj, pj, 0, 2)
    cost = cost + boneError(tj, pj, 0, 3)
    cost = cost + boneError(tj, pj, 0, 4)
    cost = cost + boneError(tj, pj, 0, 5)
    cost = cost + boneError(tj, pj, 1, 6)
    cost = cost + boneError(tj, pj, 6, 7)
    cost = cost + boneError(tj, pj, 7, 8)
    cost = cost + boneError(tj, pj, 2, 9)
    cost = cost + boneError(tj, pj, 9, 10)
    cost = cost + boneError(tj, pj, 10, 11)
    cost = cost + boneError(tj, pj, 3, 12)
    cost = cost + boneError(tj, pj, 12, 13)
    cost = cost + boneError(tj, pj, 13, 14)
    cost = cost + boneError(tj, pj, 4, 15)
    cost = cost + boneError(tj, pj, 15, 16)
    cost = cost + boneError(tj, pj, 16, 17)
    cost = cost + boneError(tj, pj, 5, 18)
    cost = cost + boneError(tj, pj, 18, 19)
    cost = cost + boneError(tj, pj, 19, 20)

    return cost


def HandCostFunction(y_true, y_pred):
    batchSize = tf.shape(y_true)
    trueJoints = tf.reshape(y_true, (batchSize[0], 21, 3))
    predJoints = tf.multiply(tf.reshape(y_pred, (batchSize[0], 21, 3)), 10)

    def error(tj, pj):
        sumOfError = tf.constant(0, dtype=tf.float32)
        for i in range(0, 21):
            sumOfError = sumOfError + tf.norm(pj[i] - tj[i], ord='euclidean')
        errorPerJoint = sumOfError / tf.constant(21.0, dtype=tf.float32)
        return errorPerJoint #+ (tf.constant(0.1, dtype=tf.float32) * BoneFilter(tj, pj))
        # return errorPerJoint

    elems = tf.range(0, batchSize[0], dtype=tf.int32)
    mapFn = lambda i: error(trueJoints[i], predJoints[i])
    costs = tf.map_fn(mapFn, elems, dtype=tf.float32)
    cost2 = tf.reduce_sum(costs) / tf.cast(batchSize[0], dtype=tf.float32)
    return cost2


def GetModel():
    inputImage = Input(shape=(cropSize, cropSize, 3), name='image_input')
    x = resnet_model(inputImage)
    x = Conv2D(filters=512, kernel_size=6, strides=1, padding='valid', activation='relu')(x)
    # x = Flatten()(x)
    # x = Dense(258, input_dim=512, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(258, input_dim=258, activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(63, input_dim=258, activation='linear')(x)

    x = Flatten()(x)
    x = Dense(256, input_dim=512, activation='relu')(x)
    x = Dense(256, input_dim=256, activation='relu')(x)
    x = Dense(128, input_dim=256, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(63, input_dim=128, activation='linear')(x)

    model = Model(inputs=inputImage, outputs=x)
    return model


def Compile(model):
    model.compile(loss=HandCostFunction, optimizer=opt)
    return model


def GetModelPre(file):
    model = load_model(file, custom_objects={'HandCostFunction': HandCostFunction, 'tf': tf})
    # model.compile(loss=HandCostFunction, optimizer=opt)
    return model