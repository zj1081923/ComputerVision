import cv2
import numpy as np
import A2_

cover = cv2.imread('cv_cover.jpg', cv2.IMREAD_GRAYSCALE)
desk = cv2.imread('cv_desk.png', cv2.IMREAD_GRAYSCALE)
harry_ = cv2.imread('hp_cover.jpg', cv2.IMREAD_GRAYSCALE)
harry = cv2.resize(harry_, (cover.shape[1], cover.shape[0]), interpolation=cv2.INTER_CUBIC)
# initiate SIFT detector
orb = cv2.ORB_create()
# find the keypoints and descriptors with SIFT
cover_kp, cover_des = orb.detectAndCompute(cover, None)
desk_kp, desk_des = orb.detectAndCompute(desk, None)

D_ = A2_.return_Dmatch(cover_des, desk_des)
desk_P = np.float32([desk_kp[m.queryIdx].pt for m in D_[:18]]).reshape((-1, 2))
cover_P = np.float32([cover_kp[m.trainIdx].pt for m in D_[:18]]).reshape((-1, 2))

match_img = cv2.drawMatches(desk, desk_kp, cover, cover_kp, D_[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow("2-1. Feature detection, description, and matching", match_img)

###################################################################
norm_H = A2_.compute_homography(cover_P, desk_P)
Td = A2_.return_T(desk_P)
Ts = A2_.return_T(cover_P)
h, w = desk.shape[:2]
norm_cover_img = cv2.warpPerspective(cover, np.matmul(np.matmul(np.linalg.inv(Td), norm_H), Ts), (w, h))
norm_desk_img = desk.copy()

for i in range(h):
    for j in range(w):
        if norm_cover_img[i, j] != 0:
            norm_desk_img[i, j] = norm_cover_img[i, j]
cv2.imshow("2-4. Homography with normalization (cover)", norm_cover_img)
cv2.imshow("2-4. Homography with normalization (desk + cover)", norm_desk_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
###################################################################

desk_P = np.float32([desk_kp[m.queryIdx].pt for m in D_[:25]]).reshape((-1, 2))
cover_P = np.float32([cover_kp[m.trainIdx].pt for m in D_[:25]]).reshape((-1, 2))
ransac_H = A2_.compute_homography_ransac(cover_P, desk_P, 4)
ransac_cover_img = cv2.warpPerspective(cover, np.matmul(np.matmul(np.linalg.inv(Td), ransac_H), Ts), (w, h))
ransac_desk_img = desk.copy()
for i in range(h):
    for j in range(w):
        if ransac_cover_img[i, j] != 0:
            ransac_desk_img[i, j] = ransac_cover_img[i, j]
cv2.imshow("2-4. Homography with ransac (cover)", ransac_cover_img)
cv2.imshow("2-4. Homography with ransac (cover + desk)", ransac_desk_img)
#################################################################
ransac_harry_img = cv2.warpPerspective(harry, np.matmul(np.matmul(np.linalg.inv(Td), ransac_H), Ts), (w, h))
harry_desk_img = desk.copy()
for i in range(h):
    for j in range(w):
        if ransac_cover_img[i, j] != 0:
            harry_desk_img[i, j] = ransac_harry_img[i, j]
cv2.imshow("2-4. Homography with ransac (harry potter)", ransac_harry_img)
cv2.imshow("2-4. Homography with ransac (harry potter + desk)", harry_desk_img)

#########################################################################

img10 = cv2.imread('diamondhead-10.png', cv2.IMREAD_GRAYSCALE)
img11 = cv2.imread('diamondhead-11.png', cv2.IMREAD_GRAYSCALE)
kp10, des10 = orb.detectAndCompute(img10, None)
kp11, des11 = orb.detectAndCompute(img11, None)
matches = A2_.return_Dmatch(des11, des10)
img10_P = np.float32([kp10[m.queryIdx].pt for m in matches[:25]]).reshape((-1, 2))
img11_P = np.float32([kp11[m.trainIdx].pt for m in matches[:25]]).reshape((-1, 2))
stH = A2_.compute_homography_ransac(img10_P, img11_P, 5)
h, w =  img11.shape
Td = A2_.return_T(img10_P)
Ts = A2_.return_T(img11_P)
dst = cv2.warpPerspective(img11, np.matmul(np.matmul(np.linalg.inv(Td), stH), Ts), (img11.shape[1]+img10.shape[1], img11.shape[0]))
right = dst.copy()
x1 = np.array([0, 0, 1])
x2 = np.array([0, h-1, 1])
x3 = np.array([w-1, h-1, 1])
x4 = np.array([w-1, 0, 1])
x1 = np.matmul(np.matmul(np.matmul(np.linalg.inv(Td), stH), Ts), x1.T).T
x1 = x1/x1[2]
x2 = np.matmul(np.matmul(np.matmul(np.linalg.inv(Td), stH), Ts), x2.T).T
x2 = x2/x2[2]
x3 = np.matmul(np.matmul(np.matmul(np.linalg.inv(Td), stH), Ts), x3.T).T
x3 = x3/x3[2]
x4 = np.matmul(np.matmul(np.matmul(np.linalg.inv(Td), stH), Ts), x4.T).T
x4 = x4/x4[2]

dst[0:img10.shape[0], 0:img10.shape[1]] = img10
croped_img = dst[int(max(x1[1], x4[1])):int(min(x2[1], x3[1])), 0:int(min(x3[0], x4[0]))]
cv2.imshow("2-5. Image stitching, (a)",croped_img)

############# blending
h11, blending_pos = img10.shape
h, w = croped_img.shape
length = 100
left = img10[:, blending_pos-length:blending_pos]
right = right[:, blending_pos-length:blending_pos]
r = np.linspace(0, 1, length)
l = np.linspace(1, 0, length)
new = right.copy()
for i in range(h):
    new[i] = r*right[i] + l*left[i]
new_ = new[int(max(x1[1], x4[1])):int(min(x2[1], x3[1])),:]
croped_img[:, blending_pos-length:blending_pos] = new_

cv2.imshow("2-5. Image stitching, (b)",croped_img)

cv2.waitKey(0)
cv2.destroyAllWindows()