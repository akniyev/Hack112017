import numpy as np
import cv2
import dlib

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def getLandmarks(opencv_img, x, y, w, h) -> dlib.full_object_detection:
    rect = dlib.rectangle(int(x), int(y), int(w), int(h))

    shape = predictor(img, rect)

    return shape

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.1, 2)

    dets = detector(img, 1)

    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        for i in range(shape.num_parts):
            x = shape.part(i).x
            y = shape.part(i).y

            cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

    # for (x, y, w, h) in faces:
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = img[y:y + h, x:x + w]
    #
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #
    #     shape = getLandmarks(img, x, y, x + w, y + h)







    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()







