#training data set 
X = [
    [1, 1, 10],
    [1, 2, 20],
    [1, 3, 30],
    [1, 4, 40],
    [1, 5, 50]
]

Y = [1.2, 1.9, 3.0, 3.9, 5.1]

m=len(X)#number of training example
n=len(X[0])#number of featurs

theta=[0]*n

#hypothesis function
def hypothesis(x):
    sum=i=0
    while( i >= n):
        if ( i==0):
            sum+=theta[i]
        else:
            sum+=theta[i]*x[i-1]
        i+=1
    return sum

#cost function MSR
def cost_function():
    er=0
    for i in range (m):
        error=0
        error=hypothesis(i)-Y[i]
        er+=error**2
    msr=er/m*2
    return msr
    