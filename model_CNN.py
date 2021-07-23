import cv2
import glob
import numpy as np
import tensorflow.keras as keras
from tensorflow.python.keras.layers import Conv2D,Dense,MaxPool2D,Flatten,Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.callbacks import ModelCheckpoint,EarlyStopping

class MaskModel:
    def __init__(self):
        self.pic_size = 200
        self.error = 50

    def GetTrainData(self):
        num = 500

        bad_wear_dataset = []
        for i in range(5):
            bad_wear_images_path = glob.glob(f"mask_dataset/bad_wear/{i}/*.jpg")
            for i,image in enumerate(bad_wear_images_path):
                if i == num:
                    break
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.pic_size, self.pic_size)) / 255
                bad_wear_dataset.append(img)


        good_wear_dataset = []
        for i in range(5):
            good_wear_images_path = glob.glob(f"mask_dataset/good_wear/{i}/*.jpg")
            for i,image in enumerate(good_wear_images_path):
                if i == num:
                    break
                img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.pic_size, self.pic_size)) / 255
                good_wear_dataset.append(img)

        return bad_wear_dataset,good_wear_dataset


    def createModel(self):
        model = keras.models.Sequential()
        model.add(Conv2D(filters=16,
                         kernel_size=(5, 5),
                         padding='same',
                         input_shape=(self.pic_size,self.pic_size,1),
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=26,
                         kernel_size=(5, 5),
                         padding='same',
                         activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation='sigmoid'))

        return model

    def fit(self,images,labels):
        model = self.createModel()
        x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=1234,
                                                            shuffle=True)
        model.compile(loss=keras.losses.binary_crossentropy , optimizer='adam', metrics=['accuracy'])

        checkpoint_path = "mask_model/mask_model.hdf5"
        model_checkpoint_callback = ModelCheckpoint(
            checkpoint_path,
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True,
            verbose=1)

        earlystop_callback = EarlyStopping(
            monitor='val_accuracy',
            min_delta=0.0001,  # 精確度至少提高0.0001
            patience=3)

        model.fit(x=x_train,
                  y=y_train,
                  validation_data=(x_test,y_test),
                  epochs=1,
                  batch_size=32,
                  verbose=1,
                  callbacks=[model_checkpoint_callback,earlystop_callback])

    def predict(self,images):
        model = self.createModel()
        model.load_weights("mask_model/mask_model.hdf5")
        return model.predict(images)


    def predict_single_pic(self,image):
        # image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(image, (self.pic_size, self.pic_size)) / 255
        img = img.reshape(self.pic_size, self.pic_size, 1)
        img = np.array([img])
        pre = self.predict([img])
        print(pre)
        return pre[0][0]


def show_img(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def train():
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

    dataset = dataset.reshape(dataset.shape[0],model.pic_size,model.pic_size,1)
    dataset,labels = shuffle(dataset,labels)
    model.fit(dataset,labels)

# train()

# model = MaskModel()
# img = 'myself/myself.png'
# print(model.predict_single_pic(img))
