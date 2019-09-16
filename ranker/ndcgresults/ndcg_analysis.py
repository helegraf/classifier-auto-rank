import numpy as np
import matplotlib.pyplot as plt
import math
import random

random.seed(123456789)

def log2plus1(x):
    return math.log(x + 1)/math.log(2)

def pow2minus1(x):
    return math.pow(2,x) - 1

def val(i,j):
    return pow2minus1(i)/log2plus1(j)

def ndcg_from_values(perfect, predicted, cutoff):
    x = [y for y in range(1, cutoff+1)] 
    return  np.sum([val(predicted[i-1],i) for i in x]) / np.sum([val(perfect[i-1],i) for i in x])

def plotndcg():
    plt.figure(figsize=(8,8), num="0 correct, no cutoff, increasing len")

    x = [a for a in range(1,23)]
    y = [ndcg_from_values([a for a in range(1,n+1)],n) for n in x]
    plt.subplot(211)
    plt.scatter(x,y)
    plt.title("0 correct, no cutoff, increasing len")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1.1))

    y2 = [ndcg_from_values([a for a in range(1,23)],n) for n in x]
    plt.subplot(212)
    plt.scatter(x,y2)
    plt.title("0 correct, increasing cutoff, len 22")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1.1))

    plt.tight_layout()
    plt.show()

def swap_last_n_to_first(arr, n):
    newarr = arr.copy()
    if n > len(arr):
        n = len(arr)

    for i in range(n):
        newarr[i] = arr[-(i+1)]

    for i in range(n,len(arr)):
        newarr[i] = arr[i-n]
    return newarr

def plotndcgrelevancies():
    plt.figure(figsize=(8,8), num="0 correct, no cutoff, increasing len")

    x = [a for a in range(1,23)]
    perfect = [random.randrange(0,100,1)/100.0 for a in x]
    perfect.sort(reverse=True)
    #perfect = [22-x for x in range(22)]

    plt.subplot(421)
    y1 = [ndcg_from_values(perfect, [perfect[n-a] for a in range(1,n+1)], n) for n in x]
    plt.scatter(x,y1)
    plt.title("0 correct, no cutoff, increasing len")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(422)
    y2 = [ndcg_from_values(perfect, swap_last_n_to_first([perfect[n-a] for a in range(1,n+1)],1), n) for n in x]
    plt.scatter(x,y2)
    plt.title("1 correct, no cutoff, increasing len")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(423)
    y2 = [ndcg_from_values(perfect, swap_last_n_to_first([perfect[n-a] for a in range(1,n+1)],3), n) for n in x]
    plt.scatter(x,y2)
    plt.title("3 correct, no cutoff, increasing len")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(424)
    y2 = [ndcg_from_values(perfect, swap_last_n_to_first([perfect[n-a] for a in range(1,n+1)],5), n) for n in x]
    plt.scatter(x,y2)
    plt.title("5 correct, no cutoff, increasing len")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(425)
    y2 = [ndcg_from_values(perfect, [perfect[22-a] for a in range(1,23)], n) for n in x]
    plt.scatter(x,y2)
    plt.title("0 correct, increasing cutoff, len 22")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(426)
    y2 = [ndcg_from_values(perfect, swap_last_n_to_first([perfect[22-a] for a in range(1,23)],1), n) for n in x]
    plt.scatter(x,y2)
    plt.title("1 correct, increasing cutoff, len 22")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(427)
    y2 = [ndcg_from_values(perfect, swap_last_n_to_first([perfect[22-a] for a in range(1,23)],3), n) for n in x]
    plt.scatter(x,y2)
    plt.title("3 correct, increasing cutoff, len 22")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(428)
    y2 = [ndcg_from_values(perfect, swap_last_n_to_first([perfect[22-a] for a in range(1,23)],5), n) for n in x]
    plt.scatter(x,y2)
    plt.title("5 correct, increasing cutoff, len 22")
    x1,x2,y1,y2 = plt.axis()
    plt.axis((x1,x2,-0.1,1.1))

    plt.tight_layout()
    plt.show()

def plotsfromjava():
    file = open("results.txt")
    data = np.fromfile(file, sep=" ")
    data = np.reshape(data,(-1,4))

    fig = plt.figure(figsize=(8,8), num="Scaled Exponent, log 2")

    plt.subplot(421)
    plt.scatter(data[0:22,0],data[0:22,3])
    plt.title('0 correct, no cutoff, increasing len')
    x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(422)
    plt.scatter(data[22:44,0],data[22:44,3])
    plt.title('1 correct, no cutoff, increasing len')
    x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(423)
    plt.scatter(data[44:66,0],data[44:66,3])
    plt.title('3 correct, no cutoff, increasing len')
    x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(424)
    plt.scatter(data[66:88,0],data[66:88,3])
    plt.title('5 correct, no cutoff, increasing len')
    x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(425)
    plt.scatter(data[88:110,2],data[88:110,3])
    plt.title('0 correct, increasing cutoff, len 22')
    x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(426)
    plt.scatter(data[110:132,2],data[110:132,3])
    plt.title('1 correct, increasing cutoff, len 22')
    x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(427)
    plt.scatter(data[132:154,2],data[132:154,3])
    plt.title('3 correct, increasing cutoff, len 22')
    x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,-0.1,1.1))

    plt.subplot(428)
    plt.scatter(data[154:176,2],data[154:176,3])
    plt.title('5 correct, increasing cutoff, len 22')
    x1,x2,y1,y2 = plt.axis()
    #plt.axis((x1,x2,-0.1,1.1))

    plt.tight_layout()
    plt.show()

plotndcgrelevancies()