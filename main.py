import cv2
from model_CNN import MaskModel

cars_cascade = cv2.CascadeClassifier('cars.xml')

def detect_cars(frame):
    cars = cars_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=5)
    return frame

if __name__ == '__main__':
    def show_img():
        cv2.imshow('img', input)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def save_img(filename, img):
        cv2.imwrite(f'myself/{filename}.jpg', img)

    model = MaskModel()
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
    font = cv2.FONT_HERSHEY_SIMPLEX
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3) # ,5
        hit = 0
        input = []
        for (x, y, w, h) in faces:
            x = x - model.error
            y = y - model.error
            w = w + model.error
            h = h + model.error
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            if roi_color.shape[0] >= model.pic_size:
                hit = 1
                input = roi_gray

        if hit == 1:
            pre = model.predict_single_pic(input)
            label = round(pre)
            if label == 1:
                print('Good')
                cv2.putText(img, f'confidence:{pre}', (x, y), font, 0.5, (11, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f'good wear', (x, y + h), font, 2, (0, 255, 0), 2, cv2.LINE_AA)
                save_img('good', img)
            if label == 0:
                print('bad')
                cv2.putText(img, f'confidence:{1 - pre}', (x, y), font, 0.5, (11, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(img, f'bad wear', (x, y + h), font, 2, (0, 0, 255), 2, cv2.LINE_AA)
                save_img('bad', img)

        cv2.imshow('img', img)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

