#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#just to find path of csv file

import os
cwd = os.getcwd()
print(cwd)


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model


# In[5]:


import pandas as pd
Price_data = {
    'Area':['2600','3000','3200','3600','4000'],
    'Price':['550000','565000','610000','650000','725000']
  
}
df = pd.DataFrame(Price_data)
df


# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area and Price Data")
plt.scatter(df.Area,df.Price, color='black', )


# In[20]:


#to predict data
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(df[['Area']],df.Price)


# In[22]:


reg.predict([[3300]])


# In[35]:


#Now I am supply more data to get their future value
#Here the Area must same a previous 
#look at word capital small properly
Price_Data = {
    'Area':['2300','3700','3230','3140','4180']
}
d = pd.DataFrame(Price_Data)   
d


# In[36]:


p=reg.predict(d)
d['Price']=p
d


# In[37]:


d.to_csv("Forecast_Value.csv")


# In[51]:


#Let make it on our original graph only

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("Area and Price Data")
plt.scatter(df.Area,df.Price, color='black')
plt.plot(df.Area,reg.predict(df[['Area']]),color='blue')


# In[52]:


#Multiple variables
#y = m1*x1 + m2*x2 + m3*x3 + c

import pandas as pd
import numpy as np
from sklearn import  linear_model


# In[56]:


Price_Data={
   'Area':['2300','3700','3230','3140','4180'],
    'Bedrooms':[2,3,5,4,4],
    'Age':[20,25,31,29,42],
    'Price':['40000','45000','55000','49500','52500']
}
df = pd.DataFrame(Price_Data)
df


# In[60]:


reg=linear_model.LinearRegression()
reg.fit(df[['Area','Bedrooms','Age']],df.Price)


# In[61]:


reg.coef_


# In[62]:


reg.intercept_


# In[63]:


reg.predict([[3000,3,40]])


# In[2]:


#How to convert words into number
import pandas as pd
import word2number
from word2number import w2n
New_Data={
    "experience":['five', 'two', 'seven','four','eight'],
    'Bedrooms':[2,3,5,4,4],
    'Age':[20,25,31,29,42],
    'Price':['40000','45000','55000','49500','52500']

    }
df = pd.DataFrame(New_Data)
df


# In[89]:


df = pd.DataFrame({"experience":['five', 'two', 'seven','four','eight']}) 
df.experience = df.experience.apply(w2n.word_to_num)
df


# In[96]:


#Saving Model

import pickle
with open('model_Pickle','wb') as f:
    pickle.dump('model',f)


# In[98]:


#Loading or Reading the file

with open('model_Pickle','rb') as f:
    pickle.load(f)


# In[93]:


#Gradient Descent

import numpy as np

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 10000
    n = len(x)
    learning_rate = 0.08

    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print ("m {}, b {}, cost {} iteration {}".format(m_curr,b_curr,cost, i))

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)


# In[99]:


Core Industries	Combined index
Refinery Production	28%
Electricity Generation	19.90%
Steel Production	17.90%
Coal Production	10.30%
Crude Oil Production 	8.90%
Natural Gas Production	6.90%
Cement Production	5.40%
Fertilizers Production	2.60%


# In[113]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model
import pandas as pd
PMI_Data = {
    'Month':['Jan 02,2023 (Dec)','Feb 01,2023 (Jan)','Mar 01,2023 (Feb)','Apr 03,2023 Mar)',
             'May 01,2023 (Apr)','Jun 01,2023 (May)'],
    'Actual_Value':[57.8,55.4, 55.3, 56.4, 57.2, 58.7],
    'Forcast_Value':['54.3','57.4','54.3','55.0','55.8','56.5']
  
}
df = pd.DataFrame(PMI_Data)
df


# In[121]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.xlabel("Month",size='15')
plt.ylabel("Actual_Value",size='15')
plt.title("Actual value in each Month",size='15')
plt.scatter(df.Month,df.Actual_Value, )


# In[126]:


reg=linear_model.LinearRegression()
reg.fit(df[['Forcast_Value']],df.Actual_Value)
reg.predict([[58]])


# In[125]:


reg.predict([[58]])


# In[135]:


import pandas as pd
Price_Data = {
    'Country':['India','India','India','India','India','India','India','America','America',
               'America','America','America','America','Germany','Germany','Germany','Germany','Germany'],
    'Area':[5500,4500,3400,6500,5700,5450,4650,3789,4357,4879,3967,4924,4539,4531,3789,4532,5123,4777],
    'Price':['80000',  '85432 ' ,'90000  ' ,'54000','60000 ', '65432' , '70000' , '75678' , ' 95678','78970',
  '55000','75000','48572',' 55555 ',' 60000','56780','75870','57800']
}
               
df = pd.DataFrame(Price_Data)
df


# In[ ]:




