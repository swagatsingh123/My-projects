# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 15:18:12 2024

@author: KIIT
"""
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
#examing dataset
df=pd.read_csv(r'C:/Users/KIIT/Downloads/DsResearch/Banking/banking_data.csv')
unique_counts=df.nunique()
print(unique_counts)
age_unique_values=df.age.unique()
print(age_unique_values)
age_value_count=df.age.value_counts()
print(age_value_count)
age=age_value_count.index
print(age)
no_of_people=age_value_count.values
print(no_of_people)
plt.bar(age,no_of_people)
plt.xlabel('Age')
plt.ylabel('No. of clients')
plt.title('Age Distribution')
plt.show()
job_value_counts=df.job.value_counts()
print(job_value_counts)
job_unique_values=df.job.unique()
print(job_unique_values)
job_frequency=job_value_counts.values
print(job_frequency)
plt.bar(job_unique_values,job_frequency)
plt.xlabel('job')
plt.ylabel('No. of clients')
plt.title('Job distribution')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()
marital_status_value_counts=df.marital_status.value_counts()
marital_status_unique_values=df.marital_status.unique()
marital_status_frequency=marital_status_value_counts.values
plt.bar(marital_status_unique_values[0:3],marital_status_frequency)
plt.xlabel('Marital Status')
plt.ylabel('No. of clients')
plt.title('Marital status distribution')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()
credit_value_counts=df.default.value_counts()
plt.bar(credit_value_counts.index,credit_value_counts.values)
plt.xlabel('credit status')
plt.ylabel('No. of clients')
plt.title('credit status distribution')
plt.xticks(rotation=45,ha='right')
plt.tight_layout()
plt.show()
balance_value_counts=df.balance.value_counts()
plt.plot(balance_value_counts.index,balance_value_counts.values)
plt.xlabel('credit status')
plt.ylabel('No. of clients')
plt.title('credit status distribution')
plt.show()
housing_value_counts=df.housing.value_counts()
plt.bar(housing_value_counts.index,housing_value_counts.values)
plt.xlabel('housing loan status')
plt.ylabel('No. of clients')
plt.title('housing loan  status distribution')
plt.show()
sns.countplot(x='loan',data=df)
plt.show()
communication_types=df.contact.unique()
print(communication_types)
sns.countplot(x='day',data=df)
plt.show()
sns.countplot(x='month',data=df)
plt.show()
duration_value_counts=df.duration.value_counts()
plt.plot(duration_value_counts.index,duration_value_counts.values)
plt.xlabel('duration')
plt.ylabel('No. of clients')
plt.title('duration distribution')
plt.show()
sns.countplot(y='campaign',data=df)
plt.show()
print(df.campaign)
pdays_value_counts=df.pdays.value_counts()
plt.plot(pdays_value_counts.index,pdays_value_counts.values)
plt.xlabel('pdays')
plt.ylabel('No. of clients')
plt.title('pdays distribution')
plt.show()
print(df.previous)
sns.countplot(x='poutcome',data=df)
plt.show()
sns.countplot(x='y',data=df)
plt.show()
sns.set(rc={'figure.figsize':(10,6)})
sns.countplot(x='job',hue='y',data=df)
plt.xticks(rotation=60)
plt.show()
ct=pd.crosstab(df.y,df.job,normalize='index',margins=True)
print(ct)
df2=df[['y','balance']]
df3=df2.groupby(['y']).mean()
print(df3)
df5=pd.crosstab(df.y,df.housing,normalize='index',margins=True)
df4=pd.crosstab(df.y,df.education,normalize='index',margins=True)
df=df.drop(['day','month','day_month','contact'],axis=1)
bank_df_dummies = pd.get_dummies(df, columns=['job','marital','default','marital_status','housing','loan','education','poutcome'], drop_first=True)
bank_df_dummies.replace({'no': 0, 'yes': 1}, inplace=True)

# Ensure all data is numeric
bank_df_dummies = bank_df_dummies.apply(pd.to_numeric, errors='coerce')

# Drop rows with NaN values
bank_df_dummies.dropna(inplace=True)

# Calculate correlations
correlations = bank_df_dummies.drop('y', axis=1).apply(lambda x: x.corr(bank_df_dummies['y'], method='spearman'))
sorted_correlations = correlations.sort_values(ascending=False)

# Print sorted correlations
print(sorted_correlations)

























#providing information of the dataset
#print(df.info())

#print(df.columns)-provides information regarding columnms
#print(df.head())_shows first 5 rows of the dataset
#print(df.shape) -providing dat on number of rows and columns
#unique_counts=df.nunique()-count of unique values in each column
#print(unique_counts)





    
    
    


    
    
    
    
    
        

  

        
        


    
    