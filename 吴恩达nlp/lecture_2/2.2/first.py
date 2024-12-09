import numpy as np
import pandas as pd
tags=['RB','NN','TO']
transition_counts = {
    ('NN','NN'):16241,
    ('RB','RB'):2263,
    ('TO','TO'):2,
    ('NN','TO'):5256,
    ('RB','TO'):855,
    ('TO','NN'):734,
    ('NN','RB'):2431,
    ('RB','NN'):358,
    ('TO','RB'):200
}
num_tags=len(tags)
transition_matrix=np.zeros((num_tags,num_tags))
print(transition_matrix)
sorted_tags = sorted(tags)
for i in range(num_tags):
    for j in range(num_tags):
        tag_tuple = (sorted_tags[i],sorted_tags[j])
        transition_matrix[i,j] = transition_counts.get(tag_tuple)


print(transition_matrix)
def print_matrix(matrix):
    print(pd.DataFrame(matrix,index=sorted_tags,columns=sorted_tags))
print_matrix(transition_matrix)
rows_sum = transition_matrix.sum(axis=1, keepdims=True)
print(rows_sum)
transition_matrix = transition_matrix / rows_sum
print_matrix(transition_matrix)
import math
t_matrix_for = np.copy(transition_matrix)
t_matrix_np = np.copy(transition_matrix)
for i in range(num_tags):
    t_matrix_for[i, i] =  t_matrix_for[i, i] + math.log(rows_sum[i])
print_matrix(t_matrix_for)
d = np.diag(t_matrix_np)
print(d)
d = np.reshape(d, (3,1))
print(d)
#d与rows_sum的对数相加
d = d + np.vectorize(math.log)(rows_sum)
np.fill_diagonal(t_matrix_np, d)
print_matrix(t_matrix_np)
