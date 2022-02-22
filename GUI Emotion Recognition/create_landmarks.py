import cv2
import os
import dlib

def start_capture(name):
        path = "./data/" + name
        num_of_images = 0
        try:
            os.makedirs(path)
        except:
            print('Directory Already Created')
        vid = cv2.VideoCapture(0)
        while True:
            ret, img = vid.read()
            cv2.imshow("FaceDetection", img)
            key = cv2.waitKey(1) & 0xFF
            try:
                cv2.imwrite((path + "/" + name + ".jpg"), img)
            except:
                pass
            if key == ord("q") or key == 13:
                break

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        img =cv2.imread(path + "/"  + name + ".jpg")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        path_w = path + "/" + name + ".pts"
        with open(path_w, mode='a') as f:
            f.write("version: 1 " + "\n" + "n_points:  68" + "\n" + "{" + "\n")
        for face in faces:
            x1 = face.left()
            y1 = face.top()
            x2 = face.right()
            y2 = face.bottom()
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)
            landmarks = predictor(gray, face)
            for n in range(0, 68):
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
                print(x,y)

                with open(path_w, mode='a') as f:
                 f.write(str(x) +" " )
                 f.write(str(y))
                 f.write("\n")

                 f.close()
        with open(path_w, mode='a') as f:
            f.write("}")

        cv2.destroyAllWindows()
        return num_of_images

def start_capture1(name):
    path = "./data/" + name
    num_of_images = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    try:
        os.makedirs(path)
    except:
        print('Directory Already Created')
    vid = cv2.VideoCapture(0)
    while True:
        ret, img = vid.read()
        cv2.imshow("FaceDetection", img)
        key = cv2.waitKey(1) & 0xFF

        try:
            cv2.imwrite((path + "/" + name + "1" + ".jpg"), img)
        except:

            pass
        if key == ord("q") or key == 13:
            break

    img = cv2.imread(path + "/" + name + "1" + ".jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    path_w = path + "/" + name + "1" + ".pts"
    with open(path_w, mode='a') as f:
        f.write("version: 1 " + "\n" + "n_points:  68" + "\n" + "{" + "\n")
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 3)

        landmarks = predictor(gray, face)

        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
            print(x, y)

            with open(path_w, mode='a') as f:
                f.write(str(x) + " ")
                f.write(str(y))
                f.write("\n")
                f.close()
    with open(path_w, mode='a') as f:
        f.write("}")

    cv2.destroyAllWindows()
    return








