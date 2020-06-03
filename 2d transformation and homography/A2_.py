import numpy as np
import cv2
import time

def get_transformed_img(img, M):
    h, w = np.shape(img)
    half_h = int(h / 2)
    half_w = int(w / 2)
    plane = np.zeros((801, 801)) + 255
    for i in range(h):
        for j in range(w):
            t1, t2, one = M @ np.array([[i - half_h, j - half_w, 1]]).T
            x = int(t1[0] / one[0])
            y = int(t2[0] / one[0])
            plane[x + 400, y + 400] = img[i, j]
    return plane

def return_Dmatch(cover_des, desk_des):
    c_len = len(cover_des)
    d_len = len(desk_des)
    dist = np.zeros((c_len, d_len))
    result = []
    for j in range(d_len):
        min = 100000
        mini = -1
        for i in range(c_len):
            if min > cv2.norm(cover_des[i], desk_des[j], cv2.NORM_HAMMING):
                min = cv2.norm(cover_des[i], desk_des[j], cv2.NORM_HAMMING)
                mini = i
        temp = cv2.DMatch()
        temp.imgIdx = 0
        temp.queryIdx = j
        temp.trainIdx = mini
        temp.distance = min
        result.append(temp)
    result = sorted(result, key=lambda x: x.distance)
    '''th = 1
    print(len(result))
    for i in range(len(result)-1):
        if abs(result[i].distance - result[i+1].distance) < th:
           result.remove(result[i])'''
    return result


def return_T(P):
    #np.seterr(divide='ignore', invalid='ignore')
    P_size = len(P)

    P_centroid = (sum(P[:, 0]) / P_size, sum(P[:, 1]) / P_size)
    P_mean = P - P_centroid
    P_dist = []
    for kp in P_mean:
        P_dist.append((kp[0] ** 2 + kp[1] ** 2) ** 0.5)
    if max(P_dist) == 0:
        print(max(P_dist))
        print(P)
        print(P_centroid)
        print(P_mean)
        print(P_dist)
    Pp = (2 ** 0.5) / max(P_dist)

    T = np.array([[Pp, 0, (-1) * Pp * P_centroid[0]],
                  [0, Pp, (-1) * Pp * P_centroid[1]],
                  [0, 0, 1]])
    return T


def compute_homography(srcP, destP):
    slen = len(srcP)
    dlen = len(destP)
    Td = return_T(destP)
    Ts = return_T(srcP)
    srcP = np.hstack((srcP, np.ones(slen).reshape(-1, 1)))
    destP = np.hstack((destP, np.ones(dlen).reshape(-1, 1)))
    norm_srcP = np.ones((slen, 3))
    norm_destP = np.ones((dlen, 3))
    for i in range(slen):
        norm_srcP[i] = (Ts @ srcP[i].T).T
        norm_destP[i] = (Td @ destP[i].T).T
    A = np.empty((0, 9))
    for i in range(slen):
        x = norm_srcP[i, 0]
        y = norm_srcP[i, 1]
        x_ = norm_destP[i, 0]
        y_ = norm_destP[i, 1]
        A = np.append(A, np.array([[-1 * x, -1 * y, -1, 0, 0, 0, x * x_, y * x_, x_],
                                   [0, 0, 0, -1 * x, -1 * y, -1, x * y_, y * y_, y_]]), axis=0)
    u, s, v = np.linalg.svd(A, full_matrices=True)
    H = v.T[:, -1].reshape(3, 3)
    H = H /H[-1, -1]
    return H

def count_inlier(srcP, destP, th, H):
    slen = len(srcP)
    Td = return_T(destP)
    Ts = return_T(srcP)
    srcP = np.hstack((srcP, np.ones(slen).reshape(-1, 1)))
    computed_destP = np.zeros((slen, 3))
    dist = np.zeros(slen)
    for i in range(slen):
        computed_destP[i] = (np.matmul(np.matmul(np.matmul(np.linalg.inv(Td), H), Ts), srcP[i].T)).T
        computed_destP[i] = computed_destP[i]/computed_destP[i, -1]
        dist[i] = np.linalg.norm(computed_destP[i, 0:2] - destP[i])
    idx = np.array(np.where(dist<=th)).flatten()
    return len(idx)



def compute_homography_ransac(srcP, destP, th):
    slen = len(srcP)
    inlier_num = -1
    k = 20
    t = time.time()
    #for i in range(1000):
    while time.time()-t < 2.98:
        idx = np.arange(slen)
        np.random.shuffle(idx)
        selected_srcP = np.zeros((k, 2))
        selected_destP = np.zeros((k, 2))
        for j in range(k):  # 점 고르기
            selected_srcP[j] = np.array([[srcP[idx[j], 0], srcP[idx[j], 1]]])
            selected_destP[j] = np.array([[destP[idx[j], 0], destP[idx[j], 1]]])
        tempH = compute_homography(selected_srcP, selected_destP)
        ## inlier 개수 세기!
        cur_num = count_inlier(srcP, destP, th, tempH)
        if cur_num > inlier_num:
            inlier_num = cur_num
            H = tempH
    return H