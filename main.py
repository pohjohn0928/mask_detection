import cv2
from model_CNN import MaskModel

cars_cascade = cv2.CascadeClassifier('cars.xml')

def detect_cars(frame):
    cars = cars_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=5)
    return frame

def Simulator():
    CarVideo = cv2.VideoCapture('dataset_video1.avi')
    while CarVideo.isOpened():
        ret, frame = CarVideo.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        cars = cars_cascade.detectMultiScale(gray, 1.1, 1)
        for (x, y, w, h) in cars:
            cv2.rectangle(gray, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=5)
        cv2.imshow('video2',gray)
        if cv2.waitKey(100) == 27:
            break
    CarVideo.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Simulator()
    model = MaskModel()
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    counter = 1

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        hit = 0

        input = []
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            if roi_color.shape[0] >= 200:
                hit = 1
                input = roi_gray

        if hit == 1:
            pre = model.predict_pic(input)
            cv2.putText(img, f'confidence:{pre}', (x, y), font, 0.5, (11, 255, 255), 2, cv2.LINE_AA)
            pre = round(pre)
            if pre == 1:
                print('Good')
                cv2.putText(img, f'good', (x, y + h), font, 0.5, (11, 255, 255), 2, cv2.LINE_AA)
            if pre == 0:
                print('bad')
                cv2.putText(img, f'bad', (x, y + h), font, 0.5, (11, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('img', img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def show_img():
    cv2.imshow('img',input)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def save_img(filename,img):
    cv2.imwrite(f'{filename}.jpg', img)
