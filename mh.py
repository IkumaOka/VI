# メトロポリス・ヘイスティングス法
# 参考URL: https://qiita.com/muka1206/items/513760ac77d283530350

import copy
import numpy as np
import matplotlib.pyplot as plt
import math

total_lefty = 13
male = 52
female = 48

#真の値の計算
def combinations_count(n, r):
    return math.factorial(n) / (math.factorial(n - r) * math.factorial(r))

def pi(x):
    sum = 0
    for i in range(total_lefty + 1):
        sum+=combinations_count(male, i) * combinations_count(female, total_lefty - i)
    return combinations_count(male, x) * combinations_count(female, total_lefty - x) / sum



# Q(x,y) : ranfom walk
# ランダムにxから1ずれるようなyを出力
def Q(x):
    if(x == 0):
        if(np.random.uniform(0, 1) < 0.5):
            y = 1
        else:
            y = x
    elif(x == total_lefty):
        if(np.random.uniform(0, 1) < 0.5):
            y = total_lefty - 1
        else:
            y = x
    else:
        if(np.random.uniform(0, 1) < 0.5):
            y = x - 1
        else:
            y = x + 1
    return y


def metropolis(N):
    current = 0
    sample = []
    sample.append(current)
    accept_ratio = 0

    for i in range(N):
        candidate = Q(current)
        x = current
        y = candidate
        if(y == x + 1):
            a = ((total_lefty - x) * (male - x)) / (y * (female - total_lefty + y))
        elif(y == x - 1):
            a = (x * (female - total_lefty + x)) / ((total_lefty - y) * (male  - y))
        else:
            a = 1
        #確率aによって新しいcandidateを出力
        if(a > np.random.uniform(0, 1)):
            # Update state
            current = candidate
            sample.append(current)
            accept_ratio+=1

    print('Accept ratio:', float(accept_ratio / N))
    return np.array(sample)

def main():
    #MH法
    N = 100000
    sample = metropolis(N)
    target_dist = plt.hist(sample, bins=total_lefty + 1,range=(0,total_lefty),normed=True)
    plt.title('x(MH method)')
    plt.show()
    #真の値を表示
    ax_x = range(0,total_lefty + 1)
    answer = []
    for i in ax_x:
        answer.append(pi(i))
    plt.title('x(target distribution)')
    plt.bar(ax_x, answer)
    plt.show()

if __name__ == '__main__':
    main()