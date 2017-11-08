import numpy as np
import cv2
import dlib

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 2)


    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]

        face = False
        eyes = eye_cascade.detectMultiScale(roi_gray, 2, 5)
        if len(eyes) > 0:
            face = True

        if face:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        eye_img = 0
        for (ex, ey, ew, eh) in eyes:
            if max(ew, eh) > h // 4:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                eye_img = roi_gray[ey:ey+eh, ex:ex+ew]

                alpha = 5
                beta = -700
                eye_contrast = cv2.addWeighted(eye_img, alpha, np.zeros(eye_img.shape, eye_img.dtype), 0, beta)
                # img[0:ew, 0:eh] = cv2.cvtColor(eye_contrast, cv2.COLOR_GRAY2BGR)
                # eye_canny = cv2.Canny(eye_contrast, 10, 200, 7)

                # circles = cv2.HoughCircles(eye_contrast, cv2.HOUGH_GRADIENT, 1, 20,
                #                            param1=25, param2=17, minRadius=0, maxRadius=0)

                # if circles is not None:
                #     circles = np.uint16(np.around(circles))
                #     for i in circles[0, :]:
                #         # draw the outer circle
                #         # cv2.circle(img, (x + ex + i[0], y + ey + i[1]), i[2], (0, 255, 0), 2)
                #         # draw the center of the circle
                #         cv2.circle(img, (x + ex + i[0], y + ey + i[1]), 2, (0, 0, 255), 3)

                xs = 0
                ys = 0
                count = 0

                for i in range(ew):
                    for j in range(eh):
                        if eye_contrast[i,j] < 10:
                            xs += i
                            ys += j
                            count += 1

                if count > 0:
                    xc = xs // count
                    yc = ys // count
                    cv2.circle(img, (x + ex + xc, y + ey + yc), 2, (0, 0, 255), 3)

                img[0:ew, 0:eh] = cv2.cvtColor(eye_contrast, cv2.COLOR_GRAY2BGR)


    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()