#treainig data sets
X=[1,2,3,4,5]
Y=[1.2,1.9,3.0,3.9,5.1]

#global variables
theta=[0,0]#for bias and one feature
m=len(X)
li=0.01

#we are going to use batch gradient descent 

#hypothesis function
def hypo(x):
    return theta[0]+theta[1]*x

#mean squared error(MSE)
def MSE():
    esum=0
    for i in range(m):
        esum+=(hypo(X[i]) - Y[i])**2
    return esum/(2*m)

def gradientD():
    global theta
    temp_theta=theta.copy()
    for i in range(len(temp_theta)):
        partial=0
        for j in range(m):
            error=hypo(X[j])- Y[j]
            if i==0:
                partial+=error*1
            else:
                partial+=error*X[j]
        temp_theta[i]-=li*(partial/m)
    theta=temp_theta
    return

#treainig start for 1000 iteration
for i in range(1000):
    gradientD()
    if((i%100)==0):
        mse=MSE()
        print("mean squared error is :",mse)
    
print("theta[0] is :-",theta[0])
print("theta[1] is :-",theta[1])

                

