from keras.callbacks import ModelCheckpoint, EarlyStopping
from DataGenerators import GetGenerators, GetGenerators2
from ModelDesign import GetModel, Compile, GetModelPre

checkpointFilePath = "weightsBoneG2.h5"
epochs = 40
nworkers = 11

print('Making Model')
# model = GetModel()
# model = Compile(model)
model = GetModelPre('weightsBoneG.h5')

# model.summary()

checkpoint = ModelCheckpoint(checkpointFilePath, monitor='val_loss', verbose=1, save_weights_only=False,
                             save_best_only=True, mode='min')
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
callbacks_list = [es, checkpoint]

print('Making Generators')
train_generator, validation_generator = GetGenerators2()

print('Training Model')
model.fit_generator(
    train_generator,
    # steps_per_epoch=5000,
    epochs=6,#epochs,
    # initial_epoch=17,
    verbose=1,
    validation_data=train_generator,
    # validation_steps=5000,
    workers=nworkers,
    max_queue_size=10,
    callbacks=callbacks_list,
    use_multiprocessing=False)


