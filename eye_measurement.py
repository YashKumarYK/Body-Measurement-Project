import argparse

import cv2
import mediapipe as mp
import numpy

import utils

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# For static images:
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    # if draw:
    #     [cv2.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]

    lefteye = [mesh_coord[p] for p in RIGHT_EYE]
    i = 0
    for p in lefteye:
        cv2.circle(img, p, 2, utils.GREEN, -1)
        cv2.putText(img, "{}".format(round(LEFT_EYE[i], 1)), p,
                    cv2.FONT_HERSHEY_PLAIN, 0, (100, 200, 0), 0)
        i+=1

    # returning the list of tuples for each landmarks
    return mesh_coord


def eye_measurements(s):
    IMAGE_FILES = [s]

    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            # Convert the BGR image to RGB before processing.
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # Print and draw face mesh landmarks on the image.
            if not results.multi_face_landmarks:
                continue


            annotated_image = image.copy()
            mesh_coords = landmarksDetection(image, results, True)
            # image = utils.fillPolyTrans(image, [mesh_coords[p] for p in LEFT_EYE], utils.GREEN, opacity=0.4)
            # image = utils.fillPolyTrans(image, [mesh_coords[p] for p in RIGHT_EYE], utils.GREEN, opacity=0.4)

            # cv2.imshow('image',image)
            lefteye = [mesh_coords[p] for p in LEFT_EYE]
            righteye = [mesh_coords[p] for p in RIGHT_EYE]

            lefteye = numpy.array(lefteye)
            righteye = numpy.array(righteye)
            peri = cv2.arcLength(lefteye, True)
            perir = cv2.arcLength(righteye, True)
            print("Left peri : " + str(peri / 7.524) + "Right peri :" + str(perir / 7.524))

            # cv2.imshow('image', image)
            # cv2.imwrite('output1.jpg', image)

            return ((peri / 7) + (perir / 7)) / 2



            # for face_landmarks in results.multi_face_landmarks:
            #   print('face_landmarks:', face_landmarks)
            #   mp_drawing.draw_landmarks(
            #       image=annotated_image,
            #       landmark_list=face_landmarks,
            #       connections=mp_face_mesh.FACEMESH_TESSELATION,
            #       landmark_drawing_spec=None,
            #       connection_drawing_spec=mp_drawing_styles
            #       .get_default_face_mesh_tesselation_style())
            #   mp_drawing.draw_landmarks(
            #       image=annotated_image,
            #       landmark_list=face_landmarks,
            #       connections=mp_face_mesh.FACEMESH_CONTOURS,
            #       landmark_drawing_spec=None,
            #       connection_drawing_spec=mp_drawing_styles
            #       .get_default_face_mesh_contours_style())
