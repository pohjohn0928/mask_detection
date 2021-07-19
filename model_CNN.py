import cv2
import glob
import numpy as np
import tensorflow.keras as keras
from tensorflow.python.keras.layers import Conv2D,Dense,MaxPool2D,Flatten
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint

class MaskModel:
    def GetTrainData(self):
        num = 500

        bad_wear_dataset = []
        for i in range(5):
            bad_wear_images_path = glob.glob(f"mask_dataset/bad_wear/{i}/*.jpg")
            for i,image in enumerate(bad_wear_images_path):
                if i == num:
                    break
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (200, 200)) / 255
                bad_wear_dataset.append(img)


        good_wear_dataset = []
        for i in range(5):
            good_wear_images_path = glob.glob(f"mask_dataset/good_wear/{i}/*.jpg")
            for i,image in enumerate(good_wear_images_path):
                if i == num:
                    break
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (200, 200)) / 255
                good_wear_dataset.append(img)

        return bad_wear_dataset,good_wear_dataset


    def createModel(self):
        model = keras.models.Sequential()
        model.add(Conv2D(filters=16,
                         kernel_size=(5, 5),
                         padding='same',
                         input_shape=(200,200,1),
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=26,
                         kernel_size=(5, 5),
                         padding='same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        return model

    def fit(self,images,labels):
        model = self.createModel()
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1234,
                                                            shuffle=True)
        model.compile(loss=keras.losses.binary_crossentropy , optimizer='adam', metrics=['accuracy'])

        checkpoint_path = "mask_model/mask_model.hdf5"
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1)

        model.fit(x=x_train,
                  y=y_train,
                  validation_data=(x_test,y_test),
                  epochs=10,
                  batch_size=32,
                  verbose=1,
                  callbacks=[model_checkpoint_callback])

    def predict(self,images):
        model = self.createModel()
        model.load_weights("mask_model/mask_model.hdf5")
        return model.predict(images)


    def predict_pic(self,image):
        img = cv2.resize(image, (200, 200)) / 255
        img = np.array([img])
        img = img.reshape(img.shape[0], 200, 200, 1)
        pre = self.predict([img])
        print(pre)
        return pre[0][0]


model = MaskModel()
bad_wear_dataset,good_wear_dataset = model.GetTrainData()
print(f'good wear data: {len(good_wear_dataset)}')
print(f'bad wear data: {len(bad_wear_dataset)}')

bad_wear_labels = len(bad_wear_dataset) * [0]
good_wear_labels = len(good_wear_dataset) * [1]

dataset = bad_wear_dataset + good_wear_dataset
labels = bad_wear_labels + good_wear_labels

dataset = np.array(dataset)
labels = np.array(labels)

dataset = dataset.reshape(dataset.shape[0],200,200,1)
dataset,labels = shuffle(dataset,labels)
model.fit(dataset,labels)

# path = 'mask_dataset/good_wear/0/00003_Mask.jpg'
# img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (200, 200)) / 255
# img = np.array([img])
# img = img.reshape(img.shape[0],200,200,1)
# pre = model.predict([img])
# print(pre)
# print(round(pre[0][0]))
# def show(image):
#     cv2.imshow('image', image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
#
# image = 'myself1.jpg'
# img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
# img = cv2.resize(img, (200, 200)) / 255
# img = np.array([img])
# print(img.shape)
# img = img.reshape(img.shape[0],200,200,1)
#
# model = MaskModel()
# pre = model.predict(img)
# print(pre)


