import random; #random.seed(1)
import matplotlib.pyplot as plt
import numpy as np

b = 1
h = 0.5

def get_w():
    return random.random()

def tran(x,u,w):
    """transition: x_k+1 = f(x_k, u_k, w_k)"""
    return max(0, x + u - w)

def get_c(x,u,w):
    """cost function"""
    return max(0, x + u - w)*h + max(0,-x - u + w)*b

def pi(x,theta):
    """policy: state --> action; theta: model parameter"""
    return max(0,theta-x)

def SGD(theta,data, alpha = 0.3, beta = 0.5):
    w,t = data
    eta = lambda t: alpha/t**beta
    return theta - eta(t)*(h if theta > w else -b)

def SAA(data): # data is emperical distribution
    data.sort()
#   import pdb; pdb.set_trace()
    return data[int((b/(h+b)*len(data)))]

def total_cost(T = 1000, method = "SGD", plot = False):
    x = 0
    theta = 1
    t_cost = 0
    data = []
    for t in range(1,T):
        w = get_w()
        u = pi(x,theta)
        c = get_c(x,u,w)
        t_cost += c
        x = tran(x,u,w)
        if method == "SGD":
            data2 = [w,t]    
            theta = SGD(theta,data2)
        elif method == "SAA":
            data = data + [w]
            theta = SAA(data)
        else:
            print("Unimplemented method")
            break
        if plot == True:
            plt.plot(t, total_cost - 0.166*t,'ro')
    return t_cost

def long_run_average_cost(T = 10000,theta = 0.1666): # 0.166
    x = theta
    total_cost = 0
    for t in range(1,T):
        w = get_w()
        u = pi(x,theta)
        c = get_c(x,u,w)
        x = tran(x,u,w)
        total_cost += c
    return total_cost/T

def find_opt_target(T = 10000,method = "SGD"): # 0.6666
    tar = simulation(T = T,method )    
    return tar
