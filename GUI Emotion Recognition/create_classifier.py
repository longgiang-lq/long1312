import math
import dlib
import menpo.io as mio
import pickle

def rolate(vector, angle):
    rx = vector[0] * (math.cos(angle)) - vector[1] * (math.sin(angle))
    ry = vector[0] * (math.sin(angle)) + vector[1] * (math.cos(angle))
    return [rx, ry]

def normalize_perk_landmark(landmark_perk, landmark_neutral):
    neutral_center = landmark_neutral[30]
    perk_center = landmark_perk[30]
    move_vector = [neutral_center[0] - perk_center[0], neutral_center[1] - perk_center[1]]
    for lm in landmark_perk:
        lm[0] += move_vector[0]
        lm[1] += move_vector[1]

    scale_neutral = [landmark_neutral[30][0] - landmark_neutral[27][0],
                     landmark_neutral[30][1] - landmark_neutral[27][1]]
    scale_perk = [landmark_perk[30][0] - landmark_perk[27][0], landmark_perk[30][1] - landmark_perk[27][1]]
    ratio = math.sqrt(scale_neutral[0] * scale_neutral[0] + scale_neutral[1] * scale_neutral[1]) / math.sqrt(
        scale_perk[0] * scale_perk[0] + scale_perk[1] * scale_perk[1])

    for lm in landmark_perk:
        lm[0] = (perk_center[0] - lm[0]) * (1 - ratio) + lm[0]
        lm[1] = (perk_center[1] - lm[1]) * (1 - ratio) + lm[1]

    sign_y = scale_perk[0] * scale_neutral[1] - scale_perk[1] * scale_neutral[0]
    sign_x = scale_perk[0] * scale_neutral[0] + scale_perk[1] * scale_neutral[1]
    angle = math.atan2(sign_y, sign_x)
    for lm in landmark_perk:
        tmp_vector = [lm[0] - landmark_perk[30][0], lm[1] - landmark_perk[30][1]]
        new_vector = rolate(tmp_vector, angle)
        lm[0] = new_vector[0] + landmark_perk[30][0]
        lm[1] = new_vector[1] + landmark_perk[30][1]

    return landmark_perk


def process_input_image(image, crop_proportion=0.2, max_diagonal=400):
    if image.n_channels == 3:
        image = image.as_greyscale()
    image = image.crop_to_landmarks_proportion(crop_proportion)
    d = image.diagonal()
    if d > max_diagonal:
        image = image.rescale(float(max_diagonal) / d)
    return image

Model_PATH = "shape_predictor_68_face_landmarks.dat"
frontalFaceDetector = dlib.get_frontal_face_detector()
faceLandmarkDetector = dlib.shape_predictor(Model_PATH)


def svm_classifer(name):
    filename = '/Users/longg/PycharmProjects/59TH3_175A071330_TruongGiangLong/Hệ thống nhận diện cảm xúc/models_SVM(o-v-r).sav'
    models = pickle.load(open(filename, 'rb'))
    path_to_svm_test_database = "/Users/longg/PycharmProjects/59TH3_175A071330_TruongGiangLong/Hệ thống nhận diện cảm xúc/data/" \
                                + name +"/**/*"
    test_images = mio.import_images(path_to_svm_test_database, verbose=True)
    test_images = test_images.map(process_input_image)

    count = 0
    landmark =[]
    landmark_neutral1 = test_images[count].landmarks['PTS'].lms.points
    landmark_perk1 = test_images[count + 1].landmarks['PTS'].lms.points
    landmark_perk1 = normalize_perk_landmark(landmark_perk1, landmark_neutral1)
    for i in range(0, 68):
        landmark.append(
            [landmark_perk1[i][0] - landmark_neutral1[i][0], landmark_perk1[i][1] - landmark_neutral1[i][1]])

    x_test = []
    tmp = []
    for vector in landmark:
        tmp.append(vector[0])
        tmp.append(vector[1])
    x_test.append(tmp)

    for model in models:
        label = model.predict(x_test)
    def emotion_decode(argument):
        emotion = {
            1: "Anger",
            2: "Contempt",
            3: "Disgust",
            4: "Fear",
            5: "Happy",
            6: "Sadness",
            7: "Surprise",
        }
        return emotion.get(argument)

    emotion_predict = emotion_decode(label[0])
    return emotion_predict
def rf_classifer(name):

    filename = '/Users/longg/PycharmProjects/59TH3_175A071330_TruongGiangLong/Hệ thống nhận diện cảm xúc/models_RF.sav'
    models = pickle.load(open(filename, 'rb'))
    path_to_svm_test_database = "/Users/longg/PycharmProjects/59TH3_175A071330_TruongGiangLong/Hệ thống nhận diện cảm xúc/data/" + name +"/**/*"
    test_images = mio.import_images(path_to_svm_test_database, verbose=True)
    test_images = test_images.map(process_input_image)

    count = 0
    landmark =[]
    landmark_neutral1 = test_images[count].landmarks['PTS'].lms.points
    landmark_perk1 = test_images[count + 1].landmarks['PTS'].lms.points
    landmark_perk1 = normalize_perk_landmark(landmark_perk1, landmark_neutral1)

    for i in range(0, 68):
        landmark.append(
            [landmark_perk1[i][0] - landmark_neutral1[i][0], landmark_perk1[i][1] - landmark_neutral1[i][1]])
    x_test = []
    tmp = []
    for vector in landmark:
        tmp.append(vector[0])
        tmp.append(vector[1])
    x_test.append(tmp)

    for model in models:
        r = model.predict(x_test)
    print(" Emorion_predict :")


    def emotion_decode(argument):
        emotion = {
            1: "Anger",
            2: "Contempt",
            3: "Disgust",
            4: "Fear",
            5: "Happy",
            6: "Sadness",
            7: "Surprise",
        }
        return emotion.get(argument)

    emotion_predict = emotion_decode(r[0])
    print(emotion_predict)
    return emotion_predict

