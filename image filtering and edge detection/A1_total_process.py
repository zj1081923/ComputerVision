import A1_function1
import cv2
import time

def image_filtering_process(name_list):
    for s in name_list:
        timg = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
        img_h=[]
        img_v=[]
        for x in range(5, 18, 6): #size
            for y in range(1, 12, 5): #sigma
                img = timg.copy()
                kernel = A1_function1.get_gaussian_filter_2d(x, y)
                img = A1_function1.cross_correlation_2d(img, kernel).astype('uint8')
                string = str(x) + 'x' + str(x) + ' s=' + str(y)
                img = cv2.putText(img, string, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                img_h.append(img)
            img_v.append(cv2.hconcat(img_h))
            img_h = []
        cv2.imwrite('result/part_1_gaussian_filtered_'+s, cv2.vconcat(img_v))
        img = cv2.imread("result/part_1_gaussian_filtered_"+s)
        cv2.imshow('part_1_gaussian_filtered_'+s, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def edge_detection_process(name_list):
    kernel = A1_function1.get_gaussian_filter_2d(7, 1.5)
    for s in name_list:
        img = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
        img = A1_function1.cross_correlation_2d(img, kernel)
        time_start = time.clock()
        mag, dir = A1_function1.compute_image_gradient(img)
        time_elapsed = (time.clock() - time_start)
        print('compute_image_gradient computational time (' + s + ') :', end=' ')
        print(time_elapsed)
        cv2.imwrite('result/part_2_edge_raw_' + s, mag)

        time_start = time.clock()
        nms_img = A1_function1.non_maximum_suppression_dir(mag, dir)
        time_elapsed = (time.clock() - time_start)
        print('non_maximum_suppression_dir computational time (' + s + ') :', end=' ')
        print(time_elapsed)
        cv2.imwrite('result/part_2_edge_sup_' + s, nms_img)

        img = cv2.imread("result/part_2_edge_raw_"+s)
        cv2.imshow('part_2_edge_raw_'+s, img)
        img = cv2.imread("result/part_2_edge_sup_"+s)
        cv2.imshow('part_2_edge_sup_'+s, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def win_show_p2(name_list):
    for s in name_list:
        img = cv2.imread("result/part_2_edge_raw_"+s)
        cv2.imshow('part_2_edge_raw_'+s, img)
        img = cv2.imread("result/part_2_edge_sup_"+s)
        cv2.imshow('part_2_edge_sup_'+s, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def corner_detection_process(name_list):
    kernel = A1_function1.get_gaussian_filter_2d(7, 1.5)
    for s in name_list:
        img = cv2.imread(s, cv2.IMREAD_GRAYSCALE)
        img = A1_function1.cross_correlation_2d(img, kernel)
        time_start = time.clock()
        R = A1_function1.compute_corner_response(img)
        time_elasped = (time.clock() - time_start)
        print('compute_corner_response computational time (' + s +') :', end=' ')
        print(time_elasped)
        cv2.imwrite('result/part_3_corner_raw_' + s, R*255)
        greendotimg = A1_function1.green_dot(s, R)
        cv2.imwrite('result/part_3_corner_bin_' + s, greendotimg)
        time_start = time.clock()
        R = A1_function1.non_maximum_suppression_win(R, 11)
        time_elasped = (time.clock() - time_start)
        print('non_maximum_suppression_win computational (' + s +') :', end=' ')
        print(time_elasped)
        greencircleimg = A1_function1.green_circle(s, R)
        cv2.imwrite('result/part_3_corner_sup_' + s, greencircleimg)

        img = cv2.imread("result/part_3_corner_raw_"+s)
        cv2.imshow('part_3_corner_raw_'+s, img)
        img = cv2.imread("result/part_3_corner_bin_"+s)
        cv2.imshow('part_3_corner_bin_'+s, img)
        img = cv2.imread("result/part_3_corner_sup_"+s)
        cv2.imshow('part_3_corner_sup_'+s, img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()


def win_show_p3(name_list):
    for s in name_list:
        img = cv2.imread("result/part_3_corner_raw_"+s)
        cv2.imshow('part_3_corner_raw_'+s, img)
        img = cv2.imread("result/part_3_corner_bin_"+s)
        cv2.imshow('part_3_corner_bin_'+s, img)
        img = cv2.imread("result/part_3_corner_sup_"+s)
        cv2.imshow('part_3_corner_sup_'+s, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
