import numpy as np
nparray = np.array([1,2,3,4])
norm1 = np.linalg.norm(nparray)
print(nparray)
print(norm1)
nparray2 = np.array([[1,2],[3,4]])
norm2 = np.linalg.norm(nparray2)
print(nparray2)
print(norm2)
nparray3 = np.array([[1,2,3],[4,5,6]])
norm3 = np.linalg.norm(nparray3,axis=0)
norm4 = np.linalg.norm(nparray3,axis=1)
print(norm3)
print(norm4)

n1 = np.array([1,2,3,4])
n2 = np.array([2,3,4,5]).T
result1 = np.dot(n1,n2)
print(result1)

result2 = np.sum(n1*n2)
print(result2)

result3=0
for a,b in zip(n1,n2):
    result3 += a*b
print(result3)

result4 = n1@n2
print(result4)

n4 = np.array([[1,2],[3,4],[5,6]])
n4mean = np.mean(n4,axis=1)
print(n4mean)
n5 = n4.T-n4mean
print(n4.T)
print(n5)
print(n5.T)
print(n1-n2)
x=np.array([[1,2],[1,3],[2,3]])
y=np.array([[1,2],[1,2]])
z=np.dot(x,y)
print(z)