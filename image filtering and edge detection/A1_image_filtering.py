import cv2
import A1_function1
import time
import numpy
import A1_total_process

ori = cv2.imread("shapes.png", cv2.IMREAD_GRAYSCALE)
h, w = ori.shape
sub = numpy.zeros((h, w), dtype='uint8')
## print kernel!!!!!!!!!!!
print("------get_gaussian_filter_1d(5,1)(vertical)------")
print(A1_function1.get_gaussian_filter_1d(5,1))
print("-----get_gaussian_filter_1d(5,1)(horizontal)-----")
print((A1_function1.get_gaussian_filter_1d(5,1)).T)
print("-----------get_gaussian_filter_2d(5,1)-----------")
print(A1_function1.get_gaussian_filter_2d(5,1))


## gaussian filter!!!!!!
filtering_list = ['shapes.png', 'lenna.png']
time_start = time.clock()
A1_total_process.image_filtering_process(filtering_list)

## computational time, kernel (5,1)
print("-------------Computational Time(5,1)-------------")
kernel1d = A1_function1.get_gaussian_filter_1d(5, 1)
time_start = time.clock()
ori1 = A1_function1.cross_correlation_1d(ori, kernel1d)
img = A1_function1.cross_correlation_1d(ori1, kernel1d.T)
time_elapsed = (time.clock() - time_start)
print("1D filtering time :", end=' ')
print(time_elapsed)

time_start = time.clock()
kernel2d = A1_function1.get_gaussian_filter_2d(5, 1)
time_start = time.clock()
img2 = A1_function1.cross_correlation_2d(ori, kernel2d)
time_elapsed = (time.clock() - time_start)
print("2D filtering time :", end=' ')
print(time_elapsed)


sub = (numpy.abs(img2-img)).astype('uint8')
print("sum of difference :", end=' ')
print(numpy.sum(sub))
cv2.imshow("difference map", sub)

cv2.waitKey(0)
cv2.destroyAllWindows()