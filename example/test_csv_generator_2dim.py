import numpy as np

number_rows=7
number_sample_points=5000
A=5*np.random.rand(number_rows,number_sample_points)

for i in range(0,np.size(A,1)):
    # 2-dim output function
    A[0,i]=A[3,i]*2
    A[1,i]=np.exp(A[3,i])

    # dependent input featuers
    A[4,i]=A[2,i]*0.01 + 5
    A[5,i]=A[2,i]*A[3,i]**2
    A[6,i]=np.exp(-A[3,i])

np.savetxt("test_PFA_2dim_cont_y.csv", A, delimiter=",")