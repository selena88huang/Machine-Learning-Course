from unittest import result
import numpy as np
import matplotlib.pyplot as plt
import random
import time

def rand_samples(m, b, n_points, rand_param):
    x_coors, y_coors, labels = np.array([]), np.array([]), np.array([])
    c = 1 if m >= 0 else -1

    pos_num = int(n_points / 2)
    neg_num = n_points - pos_num

    for state, n_points in [['pos', pos_num], ['neg', neg_num]]:
        x = np.random.randint(0, rand_param, n_points)
        r = np.random.randint(1, rand_param, n_points)

        if state == 'pos':
            y = m * x + b - (r * c)
            labels = np.append(labels, np.ones(n_points, dtype=int))
        else:
            y = m * x + b + (r * c)
            labels = np.append(labels, -1*np.ones(n_points, dtype=int))

        x_coors = np.append(x_coors, x)
        y_coors = np.append(y_coors, y)

    return x_coors ,y_coors, labels

def array_to_matrix(dataset,labels):
    c = []
    for i in range(len(dataset)):
        c.append( ((dataset[i][0],dataset[i][1]),labels[i]) )
    return c

def check_error(w, dataset):
    result = None
    error = 0
    for x, s in dataset:
        x = np.array(x)
        if np.sign(w.T.dot(x)) != s:
            result =  x, s
            error += 1
    # print  ("error = %s" % error)
    return result , error

#PLA演算法實作
def pla(dataset):
    w = np.zeros(2)
    iteration = 0
    while True:
        result, error = check_error(w, dataset)
        if error != 0:
            iteration += 1
            x, s = result
            w += s * x
        if error == 0: break
    # print("iteration = %s" % iteration)
    return w, iteration

#Pocket演算法實作
def pocket(dataset):
    w = np.zeros(2)
    present_weight = np.zeros(2)
    present_error = 0
    iteration = 0
    min_error = n_points
    max_iteration = 1000

    for i in range(0, max_iteration):
        while True:
            result, error = check_error(w, dataset)
            if error != 0:
                iteration += 1
                x, s = result
                present_weight += s * x
                result, present_error = check_error(present_weight, dataset)
                break
 
        if present_error < min_error:
            w = present_weight
            min_error = present_error

        # print  ("present_weight = %s" % present_weight)
        # print  ("min_error = %s" % min_error)
 
        if min_error == 0:
            break
    return (w, min_error, iteration)

if __name__ == '__main__':
    m, b = 2, 1
    n_points = 2000
    rand_param = 2000
    pos_num = int(n_points / 2)

    # randomly generate points
    x_coors ,y_coors, labels = rand_samples(m, b, n_points, rand_param)
    
    # dataset = np.array([x_coors[:],y_coors[:]]).T
    # labels = labels.reshape(30,1)
    # dataset = np.hstack((dataset,labels))
    # print(dataset)

    dataset = np.array([x_coors[:],y_coors[:]]).T
    labels = labels.reshape(n_points,1)
    dataset = array_to_matrix(dataset,labels)
    
    # 執行PLA
    start_time = time.time()
    w, iteration = pla(dataset)
    execution_time = time.time() - start_time
    print("\nPLA's result:")
    print(" w = %s \n iteration = %s \n execution_time = %s" % (w,iteration,execution_time))

    plt.subplot(1, 2, 1)
    plt.title('PLA')
    # plot random points. Blue: positive, red: negative
    plt.plot(x_coors[:pos_num], y_coors[:pos_num], 'o', color='blue', markersize = '2')   # positive
    plt.plot(x_coors[pos_num:], y_coors[pos_num:], 'o', color='red', markersize = '2')    # negative
    # 利用找到的最佳w畫出預測線(預測結果)
    x = np.arange(rand_param + 1)
    y_pred =  [-w[0] / w[1]] * x - 1 / w[1]
    plt.plot(x, y_pred , color='green')
    plt.text(800, -1500, 'iteration = %s execution_time = %s' % (iteration,execution_time)) 
    # 原本生成random points的線(標準答案)
    y = m * x + b
    plt.plot(x, y , color='purple')

    # 執行Pocket
    start_time = time.time()
    w, min_error, iteration = pocket(dataset)
    execution_time = time.time() - start_time
    print("\nPocket's result:")
    print(" w = %s \n iteration = %s \n min_error = %s \n execution_time = %s" % (w,iteration,min_error,execution_time))
    
    plt.subplot(1, 2, 2)
    plt.title('Pocket')
    # plot random points. Blue: positive, red: negative
    plt.plot(x_coors[:pos_num], y_coors[:pos_num], 'o', color='blue', markersize = '2')   # positive
    plt.plot(x_coors[pos_num:], y_coors[pos_num:], 'o', color='red', markersize = '2')    # negative
    # 利用找到的最佳w畫出預測線(預測結果)
    x = np.arange(rand_param + 1)
    y_pred =  [-w[0] / w[1]] * x - 1 / w[1]
    plt.plot(x, y_pred , color='green')
    plt.text(800, -1500, 'iteration = %s execution_time = %s' % (iteration,execution_time)) 
    # 原本生成random points的線(標準答案)
    y = m * x + b
    plt.plot(x, y , color='purple')
    
    plt.show()

