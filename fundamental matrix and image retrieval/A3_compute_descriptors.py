import A3_functions
import numpy as np
import pickle
import struct
import sys
np.set_printoptions(threshold=sys.maxsize)

sift_list = []
k = 1024 # cluster 개수
n = 100000 # training data 개수
for i in range(100000, 101000):
    s = np.fromfile('part2_data/sift'+str(i), dtype=np.ubyte)
    s = s.reshape(int(len(s)/128), 128)
    sift_list.append(s)


# Initialize each cluster center
k_IC = A3_functions.initial_center(sift_list, k)

# randomly select the training data
sift_ele = np.zeros((n, 128)) # 10만개
for i in range(n):# 10만개
    t = sift_list[np.random.randint(len(sift_list))]
    sift_ele[i] = t[np.random.randint(len(t))]


# train the data with k_means
'''k_IC = A3_functions.k_means(k, IC, sift_ele[0:10000, :])
with open('k_IC'+str(1)+'.txt', 'wb') as f:
    pickle.dump(k_IC, f)
k_IC = A3_functions.k_means(k, k_IC, sift_ele[10000:20000, :])
with open('k_IC'+str(2)+'.txt', 'wb') as f:
    pickle.dump(k_IC, f)
k_IC = A3_functions.k_means(k, k_IC, sift_ele[20000:30000, :])
with open('k_IC'+str(3)+'.txt', 'wb') as f:
    pickle.dump(k_IC, f)
k_IC = A3_functions.k_means(k, k_IC, sift_ele[30000:40000, :])
with open('k_IC'+str(4)+'.txt', 'wb') as f:
    pickle.dump(k_IC, f)
k_IC = A3_functions.k_means(k, k_IC, sift_ele[40000:50000, :])
with open('k_IC'+str(5)+'.txt', 'wb') as f:
    pickle.dump(k_IC, f)
k_IC = A3_functions.k_means(k, k_IC, sift_ele[50000:60000, :])
with open('k_IC'+str(6)+'.txt', 'wb') as f:
    pickle.dump(k_IC, f)
k_IC = A3_functions.k_means(k, k_IC, sift_ele[60000:70000, :])
with open('k_IC'+str(7)+'.txt', 'wb') as f:
    pickle.dump(k_IC, f)
k_IC = A3_functions.k_means(k, k_IC, sift_ele[70000:80000, :])
with open('k_IC'+str(8)+'.txt', 'wb') as f:
    pickle.dump(k_IC, f)
k_IC = A3_functions.k_means(k, k_IC, sift_ele[80000:90000, :])
with open('k_IC'+str(9)+'.txt', 'wb') as f:
    pickle.dump(k_IC, f)
k_IC = A3_functions.k_means(k, k_IC, sift_ele[90000:100000, :])
with open('k_IC'+str(10)+'.txt', 'wb') as f:
    pickle.dump(k_IC, f)'''

# open center information after k-means clustering
with open('k_IC10.txt', 'rb') as f:
    k_IC = pickle.load(f)

# make histogram with center information
#H = A3_functions.make_histogram(k_IC, sift_list, k)
#with open('histogram_norm.txt', 'wb') as f:
#   pickle.dump(H, f)

# open histogram
with open('histogram_norm.txt', 'rb') as f:
    H = pickle.load(f)


# make descriptor file with histogram information
f = open('A3_2016313024.des', 'wb')
f.write(struct.pack('i', 1000))
f.write(struct.pack('i', k))
for i in range(1000):
    for j in range(1024):
        f.write(struct.pack('f', H[i, j]))
