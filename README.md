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
# Developed by: shaik mufeezur rahaman
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
![1 (1)](https://github.com/githubmufeez45/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/134826568/e71285e4-6026-460b-8534-5a232d46cd54)

![2](https://github.com/githubmufeez45/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/134826568/6d0640ef-7807-4c3d-a564-6132dfb599ea)




## Result:
Thus, the program to implement the linear regression using gradient descent is written and verified using python programming.
