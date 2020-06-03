import numpy as np
import cv2
import time

####################################################
###################### Part 1 ######################
####################################################

def compute_avg_reproj_error(_M, _F):
    N = _M.shape[0]

    X = np.c_[ _M[:,0:2] , np.ones( (N,1) ) ].transpose()
    L = np.matmul( _F , X ).transpose()
    norms = np.sqrt( L[:,0]**2 + L[:,1]**2 )
    L = np.divide( L , np.kron( np.ones( (3,1) ) , norms ).transpose() )
    L = ( np.multiply( L , np.c_[ _M[:,2:4] , np.ones( (N,1) ) ] ) ).sum(axis=1)
    error = (np.fabs(L)).sum()

    X = np.c_[_M[:, 2:4], np.ones((N, 1))].transpose()
    L = np.matmul(_F.transpose(), X).transpose()
    norms = np.sqrt(L[:, 0] ** 2 + L[:, 1] ** 2)
    L = np.divide(L, np.kron(np.ones((3, 1)), norms).transpose())
    L = ( np.multiply( L , np.c_[ _M[:,0:2] , np.ones( (N,1) ) ] ) ).sum(axis=1)
    error += (np.fabs(L)).sum()

    return error/(N*2)



def compute_F_raw ( M ):
    numM = len(M)
    Lx = M[:, 0]
    Ly = M[:, 1]
    Rx = M[:, 2]
    Ry = M[:, 3]
    A = np.zeros((numM, 9))
    for i in range(numM):
        A[i] = np.array([Lx[i]*Rx[i], Lx[i]*Ry[i], Lx[i], Ly[i]*Rx[i], Ly[i]*Ry[i], Ly[i], Rx[i], Ry[i], 1])

    u, s, v = np.linalg.svd(A, full_matrices=True)
    F = v.T[:, -1].reshape(3, 3)
    F = F / F[-1, -1]
    return F

def normalize_M(h, w, M):
    Left = M[:, 0:2]
    Right = M[:, 2:4]
    numM = len(M)
    T = np.array([[1, 0, -h / 2],
                  [0, 1, -w / 2],
                  [0, 0, 1]])
    aT = np.array([[2 / h, 0, 0],
                   [0, 2 / w, 0],
                   [0, 0, 1]])
    FT = aT @ T
    Left = np.hstack((Left, np.ones(numM).reshape(-1, 1)))
    Right = np.hstack((Right, np.ones(numM).reshape(-1, 1)))
    norm_left = np.ones((numM, 3))
    norm_right = np.ones((numM, 3))
    for i in range(numM):
        norm_left[i] = (FT @ Left[i].T).T
        norm_right[i] = (FT @ Right[i].T).T
    norm_M = np.ones((numM, 4))
    norm_M[:, 0:2] = norm_left[:, 0:2]
    norm_M[:, 2:4] = norm_right[:, 0:2]
    return FT, norm_M

def compute_F_norm (h, w, M):
    # input h : height of input image
    # input w : width of input image
    # M : image matches
    T, norm_M = normalize_M(h, w, M)
    F = compute_F_raw(norm_M)
    u, s, v = np.linalg.svd(F, full_matrices=True)
    s[2] = 0
    F = np.dot(u, np.dot(np.diag(s), v))
    F = F / F[-1, -1]
    F = T.T @ F @ T

    return F

def compute_epipole(F):
    #F 쓰면 오른쪽 epipole
    #F.T 쓰면 왼쪽 epipole....
    u, s, v = np.linalg.svd(F)
    e = v[-1]
    e = e/e[2]
    return e


def until_q(limg, rimg, F, M):
    numM = len(M)
    idx = np.arange(numM)
    np.random.shuffle(idx)
    ## 무작위로 점 세 개 선택...!
    left = np.array([[M[idx[0],0], M[idx[0], 1], 1],
                     [M[idx[1], 0], M[idx[1], 1], 1],
                     [M[idx[2], 0], M[idx[2], 1], 1]])
    right = np.array([[M[idx[0], 2], M[idx[0], 3], 1],
                     [M[idx[1], 2], M[idx[1], 3], 1],
                     [M[idx[2], 2], M[idx[2], 3], 1]])
    el = compute_epipole(F.T)
    er = compute_epipole(F)
    h, w, d = np.shape(limg)
    limg_c = limg.copy()
    rimg_c = rimg.copy()
    for i in range(3):
        ## left x값으로 right line 만들고
        ## right x'값으로 left line 만들기
        color_l = [0, 0, 0]
        color_l[i] = 255
        color = tuple(color_l)
        line_right = np.dot(F, left[i])     ## l' = F * x
        line_left = np.dot(F.T, right[i])   ## l = F.T * x'

        t_r = (np.linspace(0, w, 2)).astype('int')
        lt_r = (np.array([(line_right[2]+line_right[0]*tt)/(-line_right[1]) for tt in t_r])).astype('int')
        t_l = (np.linspace(0, w, 2)).astype('int')
        lt_l = (np.array([(line_left[2]+line_left[0]*tt)/(-line_left[1]) for tt in t_l])).astype('int')

        limg_c = cv2.line(limg_c, (t_l[0], lt_l[0]), (t_l[1], lt_l[1]), color)
        limg_c = cv2.circle(limg_c, (int(left[i, 0]), int(left[i, 1])), 4, color)

        limg_c = cv2.circle(limg_c, (int(el[0]), int(el[1])), 10, color, 10)

        rimg_c = cv2.line(rimg_c, (t_r[0], lt_r[0]), (t_r[1], lt_r[1]), color)
        rimg_c = cv2.circle(rimg_c, (int(right[i, 0]), int(right[i, 1])), 4, color)

    limg_c = cv2.circle(limg_c, (int(el[0]), int(el[1])), 10, (255, 255, 255), 10)
    rimg_c = cv2.circle(rimg_c, (int(er[0]), int(er[1])), 10, (255, 255, 255), 10)
    final_img = cv2.hconcat([limg_c, rimg_c])
    return final_img

def count_inlier(M, F):
    result_sum = 0
    numM = len(M)
    for i in range(numM):
        x_x = M[i, 2]
        x_y = M[i, 3]
        xx = M[i, 0]
        xy = M[i, 1]
        A = np.array([[x_x, x_y, 1]]) @ F @ np.array([[xx, xy, 1]]).T
        result_sum += A[0]**2
    return result_sum

def compute_F_mine (h, w, M, seed):
    numM = len(M)
    inlier_num = 1000000
    k = 15
    T, norm_M = normalize_M(h, w, M)
    for i in range(100):
        np.random.seed(seed)
        idx = np.arange(numM)
        np.random.shuffle(idx)
        tempM = np.zeros((k, 4))
        for j in range(k):
            tempM[j] = np.array([M[idx[j], 0], M[idx[j], 1], M[idx[j], 2], M[idx[j], 3]])
        tempF = compute_F_norm(h, w, tempM)
        ## inlier 개수 세기
        #cur_num = compute_avg_reproj_error(M, tempF)
        cur_num = count_inlier(M, tempF)
        if cur_num < inlier_num:
            inlier_num = cur_num
            F = tempF
    return F

####################################################
###################### Part 2 ######################
####################################################

def initial_center(sift_list, cluster_num):
    idx = np.arange(999)
    np.random.shuffle(idx)
    IC = np.zeros((cluster_num, 128))
    for i in range(cluster_num):
        index = np.random.randint(len(sift_list))
        l = len(sift_list[index])
        IC[i] = sift_list[index][np.random.randint(l)]
    # IC는 클러스터개수 * 128 짜리 어레이
    return IC


def k_means(k, IC, sift_ele):
    same_check = 0
    # sift_ele는 트레이닝 시킬거 array로!
    # 예를들어 100000개 feature로 만든다고치면
    # 100000 * 128짜리 어레이
    n = len(sift_ele)
    old_IC = IC
    new_IC = np.zeros(np.shape(IC))
    label = np.zeros(n, dtype=int)
    prev_error = -1
    z=0
    while(not np.allclose(old_IC, new_IC)):
    #   for z in range(20):
        print("------------ iter "+str(z)+" ------------")
        t = time.time()
        old_IC = IC
        A = sift_ele[:, None, :]
        B = IC[None, :, :]
        dist = np.sqrt(np.sum((A - B) ** 2, axis=2))
        for i in range(n):
            label[i] = np.argmin(dist[i])
        print(sum(label))
        IC = MeanCenter(n, k, IC, label, sift_ele)
        new_IC = IC
        z += 1
        error = 0
        for i in range(n):
            error += np.linalg.norm(sift_ele[i] - new_IC[label[i]])
        print("prev error = "+str(prev_error))
        print("error = "+str(error))
        print("time : "+str(time.time()-t))
        if prev_error <= error:
          same_check += 1
          if same_check == 3:
            return new_IC
        else:
          same_check = 0
        prev_error = error
    return new_IC


def MeanCenter(n, k, IC, label, sift_ele):
    count = np.zeros(k)
    center = np.zeros(np.shape(IC))
    label = label.astype('int')
    for i in range(n):
        IC[label[i]] = IC[label[i]] + sift_ele[i]
        count[label[i]] += 1
    for i in range(k):
        if(count[i] != 0):
            center[i] = IC[i]/count[i]
    return center

def make_histogram(k_IC, sift_list, k):
  img_num = len(sift_list)
  H = np.zeros((img_num, k))
  for i in range(img_num): # sift_list # n = 10000 (training data 개수)
    if i%100 == 0:
      print("i = "+str(i))
    tmp_sift = sift_list[i]
    cur_sift_len = len(tmp_sift)
    dist = np.sqrt(np.sum((tmp_sift[:, None, :] - k_IC[None, :, :])**2, axis=2))
    for j in range(cur_sift_len):
      H[i][np.argmin(dist[j])] += 1
    H[i] = H[i]/sum(H[i])
  return H