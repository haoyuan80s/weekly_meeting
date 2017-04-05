from tools import *

def compare(n = 1):
    random.seed(n)
    c1 = total_cost(method= "SGD")    
    random.seed(n)
    c2 = total_cost(method= "SAA")
    print (c1 > c2)

count = 0
for i in range(20):
    compare(i)

