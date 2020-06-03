import A2_
import cv2
import numpy as np
import math

plane_img = np.zeros((801, 801), np.uint8) + 255
smile = cv2.imread('smile.png', cv2.IMREAD_GRAYSCALE)
sh, sw = np.shape(smile)
half_sh = int(sh / 2)
half_sw = int(sw / 2)

plane_img[400 - half_sh:400 + half_sh + 1, 400 - half_sw:400 + half_sw + 1] = smile

aM = np.array([[1, 0, 1],
               [0, 1, 0],
               [0, 0, 1]])

result = plane_img

while True:
    cv2.arrowedLine(result, (400, 800), (400, 0), 0, thickness=1, tipLength=0.01)
    cv2.arrowedLine(result, (0, 400), (800, 400), 0, thickness=1, tipLength=0.01)
    cv2.imshow("Part #1. 2D Transformations", result)
    ch = cv2.waitKey()
    if ch == 81:  # Q
        cv2.destroyAllWindows()
        break
    elif ch == 72:  # H restore to the initial state
        aM = np.array([[1, 0, 1],
                       [0, 1, 0],
                       [0, 0, 1]])
        result = plane_img
    elif ch == 102:  # f y flip
        aM = np.array([[1, 0, 0],
                       [0, -1, 0],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
    elif ch == 70:  # F x flip
        aM = np.array([[-1, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
    elif ch == 97:  # a
        aM = np.array([[1, 0, 0],
                       [0, 1, -5],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
    elif ch == 100:  # d
        aM = np.array([[1, 0, 0],
                       [0, 1, 5],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
    elif ch == 119:  # w
        aM = np.array([[1, 0, -5],
                       [0, 1, 0],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
    elif ch == 115:  # s
        aM = np.array([[1, 0, 5],
                       [0, 1, 0],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
    elif ch == 82:  # R 시계방향으로 5도 회전
        p = -1 * math.pi / 36
        aM = np.array([[math.cos(p), -1 * math.sin(p), 0],
                       [math.sin(p), math.cos(p), 0],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
    elif ch == 114:  # r 반시계방향으로 5도 회전
        p = math.pi / 36
        aM = np.array([[math.cos(p), -1 * math.sin(p), 0],
                       [math.sin(p), math.cos(p), 0],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
    elif ch == 88:  # X x 5% 늘리기
        aM = np.array([[1, 0, 0],
                       [0, 1.05, 0],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
    elif ch == 120:  # x x 5% 줄이기
        aM = np.array([[1, 0, 0],
                       [0, 0.95, 0],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
    elif ch == 89:  # Y y 5% 늘리기
        aM = np.array([[1.05, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
    elif ch == 121:  # y y 5% 줄이기
        aM = np.array([[0.95, 0, 0],
                       [0, 1, 0],
                       [0, 0, 1]]) @ aM
        result = A2_.get_transformed_img(smile, aM)
