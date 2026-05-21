def cost_function():
    er=0
    for i in range (m):
        er+=hypothesis(X[i])-Y[i]
        er=er**2
    msr=er/(m*2)
    return msr