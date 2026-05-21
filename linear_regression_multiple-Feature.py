#training data set 
X = [
    [3.75, 9.51, 7.32], [5.99, 1.56, 1.56], [0.58, 8.66, 6.01], 
    [7.08, 0.21, 9.7], [8.32, 2.12, 1.82], [1.83, 3.04, 5.25], 
    [4.32, 2.91, 6.12], [1.39, 2.92, 3.66], [4.56, 7.85, 2.0], 
    [5.14, 5.92, 0.46]
]


Y = [15.07, 15.7, 7.98, 47.86, 18.01, 20.89, 26.07, 12.0, 2.96, 0.26]


m=len(X)#number of training example
n=(len(X[0]))+1#number of featurs plus bias

theta=[0]*n
learningrate=0.001

#hypothesis function
def hypothesis(x):
    sum=i=0
    while( i < n):
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
        error=hypothesis(X[i])-Y[i]
        er+=error**2
    msr=er/(m*2)
    return msr

#graident descent (batch)
def bgd():
    global theta
    temp_theta=theta.copy()
    for i in range (n):
        partial=0
        for j in range (m):
            if (i==0):
                partial+=(hypothesis(X[j])-Y[j])*1
            else:
                partial+=(hypothesis(X[j])-Y[j])*X[j][i-1]
        error=(partial/m)*learningrate
        temp_theta[i]-=error
    theta=temp_theta
    return theta


#treainig start for 1000 iteration
for i in range(1000):
    bgd()
    if(not((i+1)%100)):
        msr=cost_function()
        print(f"\n mean square error at {i} iteration is {msr}")

for i in range(n):
    print(f"theta[{i}]-{theta[i]}\n")





