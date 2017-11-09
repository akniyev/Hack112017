import numpy as np
import cv2
import dlib
import datetime
import pickle

with open('saved_model.pickle', 'rb') as handle:
    classifier = pickle.load(handle)
print(classifier)

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_smile.xml
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def getLandmarks(opencv_img, x, y, w, h) -> dlib.full_object_detection:
    rect = dlib.rectangle(int(x), int(y), int(w), int(h))

    shape = predictor(img, rect)

    return shape




def size_that_fits(w, h, dw, dh):
    ratio1 = dw / w
    ratio2 = dh / h

    if ratio1 * h <= dh:
        return (int(w * ratio1), int(h * ratio1))
    else:
        return (int(w * ratio2), int(h * ratio2))

generate_dataset = False
counter = 0
while 1:
    ret, img = cap.read()
    orig_img = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.1, 2)

    #smiles
    # smiles = smile_cascade.detectMultiScale(gray, 9.3, 4)
    # for (x, y, w, h) in smiles:
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)




    dets = detector(img, 0)

    if generate_dataset:
        counter = counter + 1
        if counter % 10 != 0:
            continue

    for k, d in enumerate(dets):

        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        # for i in [30, 8, 36, 45, 48, 54]: #range(36, 48):
        # #for i in range(shape.num_parts):
        #     x = shape.part(i).x
        #     y = shape.part(i).y
        #
        #     cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),  # Nose tip
            (shape.part(8).x, shape.part(8).y),  # Chin
            (shape.part(36).x, shape.part(36).y),  # Left eye left corner
            (shape.part(45).x, shape.part(45).y),  # Right eye right corne
            (shape.part(48).x, shape.part(48).y),  # Left Mouth corner
            (shape.part(54).x, shape.part(54).y)  # Right mouth corner
        ], dtype="double")

        # 3D model points.
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner

        ])

        # Camera internals
        size = img.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                      dist_coeffs,
                                                                      flags=cv2.SOLVEPNP_ITERATIVE)

        # Project a 3D point (0, 0, 1000.0) onto the image plane.
        # We use this to draw a line sticking out of the nose


        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                         translation_vector,
                                                         camera_matrix, dist_coeffs)

        for p in image_points:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

        p1 = (int(image_points[0][0]), int(image_points[0][1]))
        p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

        cv2.line(img, p1, p2, (255, 0, 0), 2)

        # finding pupils

        roi_corners1_big = np.array([[
            (shape.part(36).x, shape.part(36).y),
            (shape.part(37).x, shape.part(37).y),
            (shape.part(38).x, shape.part(38).y),
            (shape.part(39).x, shape.part(39).y),
            (shape.part(40).x, shape.part(40).y),
            (shape.part(41).x, shape.part(41).y)
        ]], dtype=np.int32)

        y1 = min(map(lambda a: a[0], roi_corners1_big[0]))
        y2 = max(map(lambda a: a[0], roi_corners1_big[0]))
        x1 = min(map(lambda a: a[1], roi_corners1_big[0]))
        x2 = max(map(lambda a: a[1], roi_corners1_big[0]))

        roi_corners1 = np.array([[
            (shape.part(36).x-y1, shape.part(36).y-x1),
            (shape.part(37).x-y1, shape.part(37).y-x1),
            (shape.part(38).x-y1, shape.part(38).y-x1),
            (shape.part(39).x-y1, shape.part(39).y-x1),
            (shape.part(40).x-y1, shape.part(40).y-x1),
            (shape.part(41).x-y1, shape.part(41).y-x1)
        ]], dtype=np.int32)

        left_eye_img = orig_img[x1:x2, y1:y2]

        mask1 = np.zeros(left_eye_img.shape, dtype=np.uint8)

        channel_count = left_eye_img.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask1, roi_corners1, ignore_mask_color)

        masked_image1 = cv2.bitwise_and(left_eye_img, mask1)



        # Second eye

        roi_corners2_big = np.array([[
            (shape.part(42).x, shape.part(42).y),
            (shape.part(43).x, shape.part(43).y),
            (shape.part(44).x, shape.part(44).y),
            (shape.part(45).x, shape.part(45).y),
            (shape.part(46).x, shape.part(46).y),
            (shape.part(47).x, shape.part(47).y)
        ]], dtype=np.int32)

        y1 = min(map(lambda a: a[0], roi_corners2_big[0]))
        y2 = max(map(lambda a: a[0], roi_corners2_big[0]))
        x1 = min(map(lambda a: a[1], roi_corners2_big[0]))
        x2 = max(map(lambda a: a[1], roi_corners2_big[0]))

        roi_corners2 = np.array([[
            (shape.part(42).x - y1, shape.part(42).y - x1),
            (shape.part(43).x - y1, shape.part(43).y - x1),
            (shape.part(44).x - y1, shape.part(44).y - x1),
            (shape.part(45).x - y1, shape.part(45).y - x1),
            (shape.part(46).x - y1, shape.part(46).y - x1),
            (shape.part(47).x - y1, shape.part(47).y - x1)
        ]], dtype=np.int32)

        right_eye_img = orig_img[x1:x2, y1:y2]

        mask2 = np.zeros(right_eye_img.shape, dtype=np.uint8)

        channel_count = right_eye_img.shape[2]
        ignore_mask_color = (255,) * channel_count
        cv2.fillPoly(mask2, roi_corners2, ignore_mask_color)

        masked_image2 = cv2.bitwise_and(right_eye_img, mask2)


        # nw1, nh1 = size_that_fits(left_eye_img.shape[1], left_eye_img.shape[0], 40, 20)
        resized_image1 = np.zeros((20, 10, 3), dtype=np.uint8)
        resized_image2 = np.zeros((20, 10, 3), dtype=np.uint8)

        if resized_image1.shape[0] * resized_image1.shape[1] * resized_image2.shape[0] * resized_image2.shape[1] <= 0:
            continue

        resized_image1 = cv2.cvtColor(cv2.resize(masked_image1, (20, 10)), cv2.COLOR_BGR2GRAY)
        resized_image2 = cv2.cvtColor(cv2.resize(masked_image2, (20, 10)), cv2.COLOR_BGR2GRAY)


        time = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S') + str(counter)

        nose_vector = [p2[0] - p1[0], p2[1] - p1[1]]

        if generate_dataset:
            cv2.imwrite('data/'+time+'.jpg', orig_img)
            np.savetxt('data/'+time+'_eye1.txt', resized_image1, '%d')
            #cv2.imwrite('data/eye2_' + time + '.jpg', right_eye_img)
            np.savetxt('data/' + time + '_eye2.txt', resized_image2, '%d')
            np.savetxt('data/' + time + '_nose.txt', nose_vector, '%d')

        #gray_eye1 = cv2.cvtColor(masked_image1, cv2.COLOR_BGR2GRAY)
        #gray_eye2 = cv2.cvtColor(masked_image2, cv2.COLOR_BGR2GRAY)


        cv2.imshow('eye', resized_image1)
        cv2.imshow('eye2', resized_image2)

        data = dlib.vector(np.concatenate((resized_image1.flatten(), resized_image2.flatten(), nose_vector)).tolist())
        np.savetxt('r.txt', data)

        c = classifier(data)

        if c < 0:
            cv2.rectangle(img, (5, 5), (100, 100), (0, 0, 255), 5)
        print(c)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()







