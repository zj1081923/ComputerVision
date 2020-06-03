import numpy as np
import A3_functions
import cv2

## temple case

M = np.loadtxt('temple_matches.txt')
numM = len(M)
raw_F = A3_functions.compute_F_raw(M)
temple_img_color = cv2.imread('temple1.png')
temple_img2_color = cv2.imread('temple2.png')

h, w, d = np.shape(temple_img_color)
norm_F = A3_functions.compute_F_norm(h, w, M)
mine_F = A3_functions.compute_F_mine(h, w, M, 4183)

print("Average Reprojection Errors (temple1.png and temple2.png)")
error = A3_functions.compute_avg_reproj_error(M, raw_F)
print("Raw = ", end=" ")
print(error)
error = A3_functions.compute_avg_reproj_error(M, norm_F)
print("Norm = ", end=" ")
print(error)
error = A3_functions.compute_avg_reproj_error(M, mine_F)
print("Mine = ", end=" ")
print(error)


while True:
    img = A3_functions.until_q(temple_img_color, temple_img2_color, mine_F, M)
    cv2.imshow("Part #1-2. Visualization of epipolar lines", img)
    ch = cv2.waitKey()
    cv2.destroyAllWindows()
    if ch == 113:
        break




## house1.jpg
M = np.loadtxt('house_matches.txt')
numM = len(M)
raw_F = A3_functions.compute_F_raw(M)
house_img_color = cv2.imread('house1.jpg')
house_img2_color = cv2.imread('house2.jpg')
h, w, d = np.shape(house_img_color)

raw_F = A3_functions.compute_F_raw(M)
norm_F = A3_functions.compute_F_norm(h, w, M)
mine_F = A3_functions.compute_F_mine(h, w, M, 18726)

print("Average Reprojection Errors (house1.jpg and house2.jpg)")
error = A3_functions.compute_avg_reproj_error(M, raw_F)
print("Raw = ", end=" ")
print(error)
error = A3_functions.compute_avg_reproj_error(M, norm_F)
print("Norm = ", end=" ")
print(error)
error = A3_functions.compute_avg_reproj_error(M, mine_F)
print("Mine = ", end=" ")
print(error)


while True:
    img = A3_functions.until_q(house_img_color, house_img2_color, mine_F, M)
    cv2.imshow("Part #1-2. Visualization of epipolar lines", img)
    ch = cv2.waitKey()
    cv2.destroyAllWindows()
    if ch == 113:
        break


## library1.jpg

M = np.loadtxt('library_matches.txt')
numM = len(M)
raw_F = A3_functions.compute_F_raw(M)
library_img_color = cv2.imread('library1.jpg')
library_img2_color = cv2.imread('library2.jpg')
h, w, d = np.shape(library_img_color)

raw_F = A3_functions.compute_F_raw(M)
norm_F = A3_functions.compute_F_norm(h, w, M)
mine_F = A3_functions.compute_F_mine(h, w, M, 3978)

print("Average Reprojection Errors (library1.jpg and library2.jpg)")
error = A3_functions.compute_avg_reproj_error(M, raw_F)
print("Raw = ", end=" ")
print(error)
error = A3_functions.compute_avg_reproj_error(M, norm_F)
print("Norm = ", end=" ")
print(error)
error = A3_functions.compute_avg_reproj_error(M, mine_F)
print("Mine = ", end=" ")
print(error)

while True:
    img = A3_functions.until_q(library_img_color, library_img2_color, mine_F, M)
    cv2.imshow("Part #1-2. Visualization of epipolar lines", img)
    ch = cv2.waitKey()
    cv2.destroyAllWindows()
    if ch == 113:
        break