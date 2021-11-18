import numpy as np

number_rows=6
number_sample_points=5000
A=5*np.random.rand(number_rows,number_sample_points)

for i in range(0,np.size(A,1)):
    # 1-dim discrete output variable
    #if A[1,i]<2.5: A[0,i]=0
    #else: A[0,i]=1

    # 1-dim continuous output variable
    A[0,i]=(A[1,i]**(3/2))

    # dependent input features
    A[3,i]=A[1,i]*0.01 + 5
    A[4,i]=A[1,i]*A[2,i]**2
    A[5,i]=np.exp(-A[2,i])

np.savetxt("test_PFA_1dim_cont_y.csv", A, delimiter=",")