import numpy as np
import math
import cv2
import sys

def img_padding(type, padn, cor_pixel):
    h, w = np.shape(cor_pixel)
    if type == 2: # 2d 정사각형 패딩
        for x in range(padn-1, -1, -1):
            cor_pixel[x] = cor_pixel[x+1]
            cor_pixel[h-1-x] = cor_pixel[h-2-x]
            cor_pixel[:,x] = cor_pixel[:,x+1]
            cor_pixel[:,w-1-x] = cor_pixel[:,w-2-x]
    elif type == 1: #세로 커널, 위아래 패딩
        for x in range(padn-1, -1, -1):
            cor_pixel[x] = cor_pixel[x+1]
            cor_pixel[h-1-x] = cor_pixel[h-2-x]
    elif type == 0: #가로커널, 좌우패딩
        for x in range(padn-1, -1, -1):
            cor_pixel[:,x] = cor_pixel[:,x+1]
            cor_pixel[:,w-1-x] = cor_pixel[:,w-2-x]
    return cor_pixel

def cross_correlation_1d(img, kernel):
    ih, iw = img.shape
    kh, kw = np.shape(kernel)
    if kh >= kw: #세로커널 (위아래패딩)
        padn=int(kh/2)
        cor_pixel = np.zeros((ih+2*padn, iw))
        result_pixel = np.zeros((ih, iw))
        cor_pixel[padn:ih+padn] = img
        cor_pixel = img_padding(1, padn, cor_pixel)
        for x in range(padn,ih+padn):
            for y in range(iw):
                temp = np.array(cor_pixel[x-padn:x+padn+1, y:y+1])
                result_pixel[x-padn][y] = np.sum(temp*kernel)
    else: #가로커널 (좌우패딩)
        padn = int(kw/2)
        cor_pixel = np.zeros((ih, iw+2*padn))
        result_pixel = np.zeros((ih, iw))
        cor_pixel[:, padn:iw+padn] = img ###############
        cor_pixel = img_padding(0, padn, cor_pixel)
        for x in range(ih):
            for y in range(padn, iw+padn):
                temp = np.array(cor_pixel[x:x+1, y-padn:y+padn+1])
                result_pixel[x][y-padn] = np.sum(temp*kernel)
    return result_pixel

def cross_correlation_2d(img, kernel):
    ih, iw = img.shape
    k_sum = np.sum(kernel)
    if k_sum != 0:
        kernel=kernel/k_sum
    kh, kw = np.shape(kernel)
    padn = int(kh/2)
    cor_pixel = np.zeros((ih+2*padn, iw+2*padn))
    result_pixel = np.zeros((ih, iw))
    cor_pixel[padn:ih+padn, padn:iw+padn] = img
    cor_pixel = img_padding(2, padn, cor_pixel)
    for x in range(padn, ih+padn):
        for y in range(padn, iw+padn):
            temp = np.array(cor_pixel[x-padn:x+padn+1, y-padn:y+padn+1])
            result_pixel[x-padn][y-padn] = np.sum(temp * kernel)
    return result_pixel


def get_gaussian_filter_1d(size, sigma):
    kernel = np.zeros((size, 1))  #세로커널만 만들기...!!
    padn = int(size/2)
    a = math.sqrt(2*math.pi*(sigma**2))
    b = 2*(sigma**2)
    for i in range ((-1)*padn, padn+1, 1):
        kernel[i+padn] = (1/a)*math.e**(((-1)*(i**2))/b)
    k_sum = np.sum(kernel)
    if k_sum != 0:
        kernel=kernel/k_sum
    return kernel

def get_gaussian_filter_2d(size, sigma):
    kernel = np.zeros((size,size))
    padn = int(size/2)
    a = 2*math.pi*(sigma**2)
    b = 2*(sigma**2)
    for i in range((-1)*padn, padn+1, 1):
        for j in range((-1)*padn, padn+1, 1):
            ij = i**2 + j**2
            kernel[i+padn][j+padn] = (1/a)*math.e**((-1)*ij/b)
    k_sum = np.sum(kernel)
    if k_sum != 0:
        kernel=kernel/k_sum
    return kernel



def compute_image_gradient(timg):
    img = timg.copy()
    ih, iw = img.shape
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Sy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    x_gradient = cross_correlation_2d(img, Sx)
    y_gradient = cross_correlation_2d(img, Sy)
    magnitude = np.sqrt(x_gradient*x_gradient + y_gradient*y_gradient)
    magnitude = magnitude * 255 / np.max(magnitude)
    direction = np.arctan2(y_gradient, x_gradient)
    return magnitude.astype('uint8'), direction

def non_maximum_suppression_dir(mag, dir):
    ih, iw = np.shape(mag)
    nms = np.zeros((ih, iw), dtype = np.int32)
    dir = dir * 180 / np.pi
    for x in range(1, ih-1):
        for y in range(1, iw-1):
            cur = mag[x][y]
            temp = round(dir[x][y]/45)%4
            x_t = -1
            y_t = 1
            if temp == 0:
                x_t = 0
            elif temp == 2:
                y_t = 0
            elif temp == 3:
                y_t = -1
            if cur>=mag[x+x_t][y+y_t] and cur>=mag[x-x_t][y-y_t]:
                nms[x][y] = cur
            else:
                nms[x][y] = 0
    return nms.astype('uint8')

def compute_corner_response(timg):
    img = timg.copy()
    Sx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
    Sy = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])
    x_gradient = cross_correlation_2d(img, Sx)
    y_gradient = cross_correlation_2d(img, Sy)
    Ixx = x_gradient * x_gradient
    Ixy = x_gradient * y_gradient
    Iyy = y_gradient * y_gradient

    window = np.array([[1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1],
                       [1, 1, 1, 1, 1]])

    Ixx = cross_correlation_2d(Ixx, window)
    Ixy = cross_correlation_2d(Ixy, window)
    Iyy = cross_correlation_2d(Iyy, window)

    trace = Ixx + Iyy
    det = Ixx*Iyy - Ixy*Ixy

    R = det - 0.04*(trace*trace)
    #R[R<0] = 0
    R = R.clip(min=0)
    if np.max(R) != 0:
        R = R / np.max(R)

    return R

def green_dot(name, R):
    np.set_printoptions(threshold=sys.maxsize)
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    greendotimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    g_index = np.where(R > 0.1)
    l = len(g_index[0])
    for x in range(l):
        greendotimg.itemset((g_index[0][x], g_index[1][x], 1), 255)
    return greendotimg

def green_circle(name, R):
    np.set_printoptions(threshold=sys.maxsize)
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    greencircleimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    g_index = np.where(R > 0.1)
    l = len(g_index[0])
    for x in range(l):
        cv2.circle(greencircleimg, ( g_index[1][x], g_index[0][x]), 6, (0, 255, 0), 2)
    return greencircleimg

def non_maximum_suppression_win (R_, winSize):
    padn = int(winSize/2)
    h, w = np.shape(R_)
    R = np.zeros((h+2*padn, w+2*padn))
    R[padn:h+padn, padn:w+padn] = R_
    h, w = np.shape(R)
    for x in range(padn, h-padn):
        for y in range(padn, w-padn):
            if R[x][y] != np.max(R[x-padn:x+padn+1, y-padn:y+padn+1]):
                R[x][y] = 0
            else:
                if R[x][y] <= 0.1:
                    R[x][y] = 0
    R = R[padn:h-padn, padn:w-padn]
    return R