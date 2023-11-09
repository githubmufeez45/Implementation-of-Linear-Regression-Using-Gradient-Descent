# Implementation-of-Linear-Regression-Using-Gradient-Descent

## Aim:
To write a program to implement the linear regression using gradient descent.

## Equipment's Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm:
1. Use the standard libraries in python for Gradient Design.
2. Upload the dataset and check any null value using .isnull() function.
3. Declare the default values for linear regression.
4. Calculate the loss usinng Mean Square Error.
5. Predict the value of y.
6. Plot the graph respect to hours and scores using scatter plot function.

## Program:
~~~
# Program to implement the linear regression using gradient descent.
# Developed by: SHAIK MUFEEZUR RAHAMAN
# RegisterNumber:  212221043007
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("student_scores.csv")
data.head()
data.isnull().sum()
x=data.Hours
y=data.Scores
y.head()
n=len(x)
m=0
c=0
L=0.001
loss=[]
for i in range(10000):
    ypred=m*x+c
    MSE=(1/n)*sum((ypred-y)*2)
    dm=(2/n)*sum(x*(ypred-y))
    dc=(2-n)*sum(ypred-y)
    c=c-L*dc
    m=m-L*dm
    loss.append(MSE)
    #print(m)
print(m,c)
y_pred=m*x+c
plt.scatter(x,y,color="black")
plt.plot(x,y_pred,color="red")
plt.xlabel("Study hours")
plt.ylabel("Scores")
plt.title("Study hours vs Scores")
plt.plot(loss)
plt.xlabel("iteration")
plt.ylabel("loss")
~~~

## Output:

![267841425-578f8c8e-5ac8-48ac-a2b5-3ca8793fbaa5](https://github.com/githubmufeez45/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/134826568/294800ab-3ba9-4a8d-97d6-8a8407e8f997)

![267841491-6ab4660c-cc5a-492c-aa20-1c59f65f0a99](https://github.com/githubmufeez45/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/134826568/903a5b2a-36ed-4008-8e0a-5ad286e4cba7)

![267841598-e7fe26b6-6b8e-4a19-8bd7-b26b19200ce5](https://github.com/githubmufeez45/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/134826568/2de6d152-10c7-47e0-a4b2-6389d47ad5bd)

![267841680-d6b40a4f-3d4f-4140-9df6-1875df9a835f](https://github.com/githubmufeez45/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/134826568/7450908a-6947-4032-b0f8-c71817f4a9f9)

![267841803-715fc47f-7b3f-471d-a56c-dcc2058eb18d](https://github.com/githubmufeez45/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/134826568/4370109f-78cf-402f-9b5d-b0fc2fedaa5a)






## Result:
Thus, the program to implement the linear regression using gradient descent is written and verified using python programming.
