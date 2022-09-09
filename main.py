import sys
from math import sqrt
import cv2
import numpy as np
import mediapipe as mp
import eye_measurement
import utils


#shoulder points
def get_left_shoulder_point(results):
    return results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_SHOULDER]


def get_right_shoulder_points(results):
    return results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_SHOULDER]

#knee points
def get_left_knee(results):
    return results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_KNEE]


def get_right_knee(results):
    return results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_KNEE]


#chest points
def get_left_chest_points_x(results):
    return (((((get_left_shoulder_point(results).x + get_left_hip_points(results).x) / 2) + get_left_shoulder_point(
        results).x) / 2) + get_left_shoulder_point(results).x) / 2

def get_left_chest_points_y(results):
    return (((((get_left_shoulder_point(results).y + get_left_hip_points(results).y) / 2) + get_left_shoulder_point(
        results).y) / 2) + get_left_shoulder_point(results).y) / 2

def get_right_chest_points_y(results):
    return (((((get_right_shoulder_points(results).y + get_right_hip_points(
        results).y) / 2) + get_right_shoulder_points(
        results).y) / 2) + get_right_shoulder_points(results).y) / 2


def get_right_chest_points_x(results):
    return (((((get_right_shoulder_points(results).x + get_right_hip_points(
        results).x) / 2) + get_right_shoulder_points(
        results).x) / 2) + get_right_shoulder_points(results).x) / 2

#hips points
def get_left_hip_points(results):
    return results.pose_landmarks.landmark[mpPose.PoseLandmark.LEFT_HIP]


def get_right_hip_points(results):
    return results.pose_landmarks.landmark[mpPose.PoseLandmark.RIGHT_HIP]


def calc_distance(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def draw_line(x1, y1, x2, y2):
    cv2.line(img, (x1, y1), (x2, y2), utils.GREEN, thickness=1, lineType=8)


def get_cm_distance(x1, y1, x2, y2):
    pixel_cm_ratio = eye_measurement.eye_measurements(s)
    return calc_distance(x1, y1, x2, y2) / pixel_cm_ratio


def chest_distance():
    left_x = int(get_left_chest_points_x(results) * w)
    left_y = int(get_left_chest_points_y(results) * h)
    right_x = int(get_right_chest_points_x(results) * w)
    right_y = int(get_right_chest_points_y(results) * h)
    draw_line(left_x, left_y, right_x, right_y)
    ans_cm = get_cm_distance(left_x, left_y, right_x, right_y)
    cv2.putText(img, "{} cm".format(round(ans_cm, 1)),
                (int((left_x + right_x) / 2), int((left_y + right_y) / 2 - 10)), cv2.FONT_HERSHEY_PLAIN, 2,
                (100, 200, 0), 2)


def cal_shoulder_distance(contours, results, img):
    x = sys.maxsize
    y = -sys.maxsize - 1

    h, w, c = img.shape
    left_shoulder = int(get_left_shoulder_point(results).y * h)
    right_shoulder = int(get_right_shoulder_points(results).y * h)
    contours = np.vstack(contours).squeeze()
    for i in contours:
        if left_shoulder + 10 >= i[1] >= left_shoulder - 10:
            x = min(x, i[0])
        if right_shoulder + 10 >= i[1] >= right_shoulder - 10:
            y = max(y, i[0])

    draw_line(x, left_shoulder, y, right_shoulder)
    ans_cm = get_cm_distance(x, left_shoulder, y, right_shoulder)
    cv2.putText(img, "{} cm".format(round(ans_cm, 1)),
                (int((x + y) / 2), int((left_shoulder + right_shoulder) / 2 - 10)), cv2.FONT_HERSHEY_PLAIN, 2,
                (100, 200, 0), 2)


def knee_distance(contours, results, img):
    x = sys.maxsize
    y = -sys.maxsize - 1
    h, w, c = img.shape
    left_knee = int(get_left_knee(results).y * h)
    left_knee_x = int(get_left_knee(results).x * w)
    right_knee = int(get_right_knee(results).y * h)
    right_knee_x = int(get_right_knee(results).x * w)

    contours = np.vstack(contours).squeeze()
    for i in contours:
        if left_knee + 10 >= i[1] >= left_knee - 10:
            x = min(x, i[0])
        if right_knee + 10 >= i[1] >= right_knee - 10:
            y = max(y, i[0])

    # cv2.circle(img, (x, right_knee), 8, (255, 0, 0),
    #            cv2.FILLED)
    # cv2.circle(img, (2*right_knee_x - x, right_knee), 8, (255, 0, 0),
    #            cv2.FILLED)
    draw_line(x, right_knee, 2 * right_knee_x - x, right_knee)
    ans_cm = get_cm_distance(x, right_knee, 2 * right_knee_x - x, right_knee)
    cv2.putText(img, "{} cm".format(round(ans_cm, 1)),
                (int((x + 2 * right_knee_x - x) / 2), int((right_knee + right_knee) / 2 - 10)), cv2.FONT_HERSHEY_PLAIN,
                2,
                (100, 200, 0), 2)

    draw_line(y, left_knee, 2 * left_knee_x - y, left_knee)
    ans_cm = get_cm_distance(y, left_knee, 2 * left_knee_x - y, left_knee)
    cv2.putText(img, "{} cm".format(round(ans_cm, 1)),
                (int((y + 2 * left_knee_x - y) / 2), int((left_knee + left_knee) / 2 - 10)), cv2.FONT_HERSHEY_PLAIN, 2,
                (100, 200, 0), 2)


def waist_distance(contours, results, img):
    x = sys.maxsize
    y = -sys.maxsize - 1

    h, w, c = img.shape
    left_hip = int((((get_left_hip_points(results).y * h + get_left_shoulder_point(
        results).y * h) / 2) + get_left_hip_points(results).y * h) / 2)
    right_hip = int((((get_right_hip_points(results).y * h + get_right_shoulder_points(
        results).y * h) / 2) + get_right_hip_points(results).y * h) / 2)

    contours = np.vstack(contours).squeeze()
    for i in contours:
        if left_hip + 10 >= i[1] >= left_hip - 10:
            x = min(x, i[0])

    for i in contours:
        if right_hip + 10 >= i[1] >= right_hip - 10:
            y = max(y, i[0])

    draw_line(x, left_hip, y, right_hip)
    ans_cm = get_cm_distance(x, left_hip, y, right_hip)
    cv2.putText(img, "{} cm".format(round(ans_cm, 1)), (int(x + y / 2), int((left_hip + right_hip) / 2 - 10)),
                cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)


def draw_landmarks(results):
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, ln in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            print(id, ln)
            cx, cy = int(ln.x * w), int(ln.y * h)
            print(cx)
            print(cy)
            cv2.circle(img, (cx, cy), 1, (255, 0, 0), cv2.FILLED)
            cv2.putText(img, "{}".format(round(id, 1)), (cx, cy),
                        cv2.FONT_HERSHEY_PLAIN, 1, (100, 200, 0), 6)


mpPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mpPose.Pose()

s = 'images/person38.png'
img = cv2.imread(s)
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
results = pose.process(imgRGB)



ret, thresh = cv2.threshold(imgray, 10, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key=cv2.contourArea, reverse=True)

# print(contours)

h, w, c = img.shape

cal_shoulder_distance(contours, results, img)
knee_distance(contours, results, img)
waist_distance(contours, results, img)
chest_distance()
# draw_landmarks(results)

# cv2.circle(img, (shoulder_edge_point[0], get_left_shoulder_point(results).y * h), 8, (255, 0, 0), cv2.FILLED)
# cv2.circle(img, (shoulder_edge_point[1], get_right_shoulder_points(results).y * h), 8, (255, 0, 0), cv2.FILLED)

perimeters = [cv2.arcLength(contours[i], True) for i in range(len(contours))]
listindex = [i for i in range(len(perimeters)) if perimeters[i] > perimeters[0] / 2]
numcards = len(listindex)

card_number = -1  # just so happened that this is the worst case
stencil = np.zeros(img.shape).astype(img.dtype)

cv2.drawContours(stencil, [contours[listindex[card_number]]], 0, (255, 255, 255), cv2.FILLED)
res = cv2.bitwise_and(img, stencil)
canny = cv2.Canny(res, 100, 200)

# cv2.drawContours(img,contours,-1,(0,255,0),1)
scale_percent = 20  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imshow('mediapipe', img)

cv2.imwrite('output.jpg', canny)

cv2.waitKey(0)
cv2.destroyAllWindows()