#importing libraries 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df=pd.read_csv(r"C:\Users\Nadeem Anwar\Desktop\Sadaf\ml\simple linear regression\placement.csv")
df



plt.figure(figsize=(5,4))
plt.scatter(df["cgpa"],df["package"])
plt.xlabel("cgpa")
plt.ylabel("package(LPA)")


X = df.iloc[:,0:1]
y = df.iloc[:,-1]
y
X


#!pip install sklearn

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


from sklearn.linear_model import LinearRegression

lr1 = LinearRegression()
lr1.fit(X_train,y_train)



plt.scatter(df['cgpa'],df['package'])
plt.plot(X_train,lr1.predict(X_train),color='red')
plt.xlabel('CGPA')
plt.ylabel('Package(in lpa)')

# basically linear regression drws a line(best fit line) with equation ,y=mx+b(proove)


m=lr1.coef_#slope
m



b=lr1.intercept_#y intercept
b

#mo we have slope and y intercept we will just provide one inpur(x) and we will get one ouput(y)
#simple linear regression class



class MyLr:
    def __inti__(self):
        self.m=None#slope
        self.b=None#y intercept
    def fit(self,X_train,Y_train):
        num=0
        den=0
        for i in range (X_train.shape[0]):
            num=num+((X_train[i]-(X_train.mean()))*(Y_train[i]-(Y_train.mean())))
            den=den+(X_train[i]-(X_train.mean()))**2
        self.m=num/den#slope
        self.b=(Y_train.mean())-(self.m*(X_train.mean()))

    def predict(self,X_test):
        y=self.m*X_test+self.b
        return y
    


df1=pd.read_csv(r"C:\Users\Nadeem Anwar\Desktop\Sadaf\ml\simple linear regression\placement.csv")
df1.head()
df1.shape


x=df1.iloc[:,0].values#cgpa
y=df1.iloc[:,1].values#package 
x
y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=2)


lr=MyLr()
lr.fit(X_train,y_train)
lr.m
y_train.mean()-lr.m*X_train.mean()
lr.b

lr.predict(6.89)