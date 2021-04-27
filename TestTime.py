import torch
import time
import torch.nn as nn
from torch.nn import init
from numpy import *


def generate_matrix():
    m1 = torch.FloatTensor(2708, 1433)
    m2 = torch.FloatTensor(1433, 1024)
    init.xavier_uniform_(m1)
    init.xavier_uniform_(m2)
    return m1, m2


def generate_random(a, b):
    rm = torch.FloatTensor(a, b)
    init.xavier_uniform_(rm)
    return rm


def mm_cpu(m1, m2):
    start = time.time()
    for i in range(20):
        out = torch.mm(m1, m2)
    end = time.time()
    # print(out)
    # print(end - start)
    return (end - start)/20


def mm_gpu(m1, m2):
    start = time.time()
    r = generate_random(2708, 1433)
    u = torch.mm(r, m2)
    # store = m1
    m1h = m1 + r
    device = torch.device("cuda")
    m1h = m1h.to(device)
    m2 = m2.to(device)
    outh = torch.mm(m1h, m2)
    outh = outh.to('cpu')
    out = outh - u
    end = time.time()
    # print(out)
    # print(end - start)
    return (end - start)/20


def mm_gpu_no_privacy(m1, m2):
    start = time.time()
    device = torch.device("cuda")
    m1 = m1.to(device)
    m2 = m2.to(device)
    for i in range(20):
        out = torch.mm(m1, m2)
    end = time.time()
    return (end - start)/20


if __name__ == '__main__':
    c_mm = []
    g_mm = []
    g_mm_no_privacy = []

    for i in range(100):
        print(i)
        mat1, mat2 = generate_matrix()
        t_cpu = mm_cpu(mat1, mat2)
        # t_gpu = mm_gpu(mat1, mat2)
        t_gpu_no_privacy = mm_gpu_no_privacy(mat1, mat2)
        c_mm.append(t_cpu)
        # g_mm.append(t_gpu)
        g_mm_no_privacy.append(t_gpu_no_privacy)

    print(mean(c_mm))
    # print(mean(g_mm))
    print(mean(g_mm_no_privacy))




