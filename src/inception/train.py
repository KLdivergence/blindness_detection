import os
import pickle as pk

import tensorflow as tf
from tensorflow.keras.backend import set_session
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

from model import create_model


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)

### Training stage
traingen = ImageDataGenerator(rescale=1./255,
                              shear_range=0.2,
                              zoom_range=0.2,
                              # featurewise_center=True,
                              # featurewise_std_normalization=True,
                              rotation_range=20,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True,
                              validation_split=0.1)

train_df = pd.read_csv("../../data/train.csv")
cols = train_df.columns
# add extensions to filenames
train_df[cols[0]] = train_df[cols[0]].apply("{}.png".format)

train_generator = traingen.flow_from_dataframe(
        train_df,
        "../../data/train_images",
        cols[0],
        cols[1],
        class_mode="categorical",
        target_size=(299, 299),
        batch_size=32,
        subset="training")

val_generator = traingen.flow_from_dataframe(
        train_df,
        "../../data/train_images",
        cols[0],
        cols[1],
        class_mode="categorical",
        target_size=(299, 299),
        batch_size=32,
        subset="validation")

model = create_model()
STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VAL = val_generator.n // val_generator.batch_size
model.fit_generator(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=50,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VAL)

### Testing stage
testgen = ImageDataGenerator(rescale=1./255)
test_generator = testgen.flow_from_directory(
        "../../data/test_images",
        target_size=(299, 299),
        shuffle = False,
        class_mode='categorical',
        batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_generator, steps = nb_samples)
predict = dict(zip(filenames, predict))

### Save prediction data
save_path = "../../results/inception"
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, "predict.pk"), "wb") as f:
    pk.dump(predict, f)

### Save predictions to csv
test_csv = "../../data/sample_submission.csv"
test_df = pd.read_csv(test_csv)
test_df["diagnosis"] = test_df["id_code"].map(
    lambda id: np.argmax(predict[r"0\\"+str(id)+r".png"]))
test_df.to_csv("../../results/submit.csv", index=False)
