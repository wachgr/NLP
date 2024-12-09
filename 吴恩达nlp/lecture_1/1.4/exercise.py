import numpy as np
p=np.array([1,1])
v=np.array([1,-1])
dotproduct = np.dot(p,v)
#用来确定点积的符号,正数就是1，负数就是-1
sign_of_dot_product = np.sign(dotproduct)
print(dotproduct)
print(sign_of_dot_product)