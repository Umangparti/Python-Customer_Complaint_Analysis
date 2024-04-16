#!/usr/bin/env python
# coding: utf-8

# # Mini Project 2–ConsumerComplaintResolutionAnalysis

# Submitted to: Edureka 

# Data Science and Machine Learning Internship Program

# Submitted by: Umang Parti

# Scenario: Product review is the most basic function/factor in resolvingcustomerissues and increasingthe sales growth of any product.Wecan understandtheir mindset toward our service withoutasking each customer.When consumers are unhappywith some aspect of a business, they reach out to customer service and might raise a complaint. Companiestry their best to resolve the complaints that they receive. However, it might not always be possible to appease every customer.So Here, we will analyze data, and with the help ofdifferent algorithms, we are finding the best classification ofcustomercategoryso thatwe can predict our test data

# Objective: Use Python libraries such as Pandas for data operations, Seaborn and Matplotlib for data visualization and EDA tasks, Sklearn for model building and performance visualization, andbased on the best model,make a prediction for the test file and save the output.The mainobjective is to predict whetherour customer is disputedor not with the help ofgiven data.

# Customers faced some issues and tried to report their problems to customer care.
# Dispute: This is our target variable based on train data; we have two groups, one with a dispute with the bank and another don’t have any issue with the bank.
# Date received: The day complaint was received.
# Product: different products offered by the bank (credit cards, debit cards, different types of transaction methods, accounts, locker services, and money-related)
# Sub-product: loan, insurance, other mortgage options
# Issue: Complaint of customers 
# Company public response: Company’s response to consumer complaint
# Company: Company name
# State: State where the customer lives (different state of USA)
# ZIP code: Where the customer lives 
# Submitted via: Register complaints via different platforms (online web, phone, referral, fax, post mail) 
# Date sent to company: The day complaint was registered
# Timely response?: Yes/no
# Consumer disputed?: yes/no (target variable)
# Complaint ID: unique to each consumer

# In[1]:


# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Read the Data from the Given excel file. Importing the dataset
consumer_train_df = pd.read_csv(
    r"C:\Users\umang\Desktop\edureka\mini project-2\Consumer_Complaints_train.csv")
consumer_train_df.head()


# In[3]:


consumer_test_df = pd.read_csv(r"C:\Users\umang\Desktop\edureka\mini project-2\Consumer_Complaints_test.csv")
consumer_test_df.head()


# In[4]:


# Check the data type for both data (test file and train file)
consumer_train_df.info()


# In[5]:


consumer_test_df.info()


# In[6]:


# converting date column from object to datetime
consumer_train_df['Date received'] = pd.to_datetime(consumer_train_df['Date received'])
consumer_train_df['Date sent to company'] = pd.to_datetime(consumer_train_df['Date sent to company'])
consumer_test_df['Date received'] = pd.to_datetime(consumer_test_df['Date received'])
consumer_test_df['Date sent to company'] = pd.to_datetime(consumer_test_df['Date sent to company'])


# In[7]:


# Do missing value analysis and drop columns where more than 25% of data are missing 
consumer_train_df.isnull().sum()


# In[8]:


consumer_test_df.isnull().sum()


# In[9]:


# dropping columns with more than 25% nulll values (25% of 358810 = 89702)
columns_to_drop_train = ['Sub-product','Sub-issue','Consumer complaint narrative',
                         'Company public response','Tags','Consumer consent provided?']
consumer_train_df.drop(columns_to_drop_train, axis=1,inplace=True)


# In[10]:


# dropping columns with more than 25% nulll values (25% of 119606 = 29901)
columns_to_drop_test = ['Sub-product','Sub-issue','Consumer complaint narrative',
                         'Company public response','Tags','Consumer consent provided?']
consumer_test_df.drop(columns_to_drop_test, axis=1,inplace=True)


# In[11]:


# filling null values with mode
mode_value_state = consumer_train_df['State'].mode()[0]
mode_value_zip = consumer_train_df['ZIP code'].mode()[0]
consumer_train_df['State'].fillna(mode_value_state, inplace=True)
consumer_train_df['ZIP code'].fillna(mode_value_zip, inplace=True)


# In[12]:


mode_value_state_test = consumer_test_df['State'].mode()[0]
mode_value_zip_test = consumer_test_df['ZIP code'].mode()[0]
consumer_test_df['State'].fillna(mode_value_state_test, inplace=True)
consumer_test_df['ZIP code'].fillna(mode_value_zip_test, inplace=True)


# In[13]:


# we have successfully handled all null values
print(consumer_train_df.isnull().sum())
print(consumer_test_df.isnull().sum())


# In[14]:


# Extracting Day, Month,and Year from Date Received Column and
# create new fields for a month, year,and day
consumer_train_df['year']= consumer_train_df['Date received'].dt.year
consumer_train_df['month']=consumer_train_df['Date received'].dt.month
consumer_train_df['day']=consumer_train_df['Date received'].dt.day


# In[15]:


consumer_test_df['year']=consumer_test_df['Date received'].dt.year
consumer_test_df['month']=consumer_test_df['Date received'].dt.month
consumer_test_df['day']=consumer_test_df['Date received'].dt.day


# In[16]:


# Calculate the Number of Days the Complaint was with the Company as a new field “Days held”
consumer_train_df['Days Held'] = (consumer_train_df['Date sent to company'] - 
                                  consumer_train_df['Date received']).dt.days
consumer_test_df['Days Held'] = (consumer_test_df['Date sent to company'] - 
                                 consumer_test_df['Date received']).dt.days


# In[17]:


# with the help of the days we calculated above,create a newfield 'Week_Received'where we
# calculate the week based on the day of receiving.
consumer_train_df['Week_Received'] = (consumer_train_df['Date received'].dt.week + 
                                      consumer_train_df['Days Held'] // 7)
consumer_test_df['Week_Received'] = (consumer_test_df['Date received'].dt.week + 
                                     consumer_test_df['Days Held'] // 7)


# In[18]:


# Drop "Date Received","Date Sent to Company","ZIP Code", "Complaint ID"fields 
columns = ['Date received','Date sent to company','ZIP code','Complaint ID']
consumer_train_df.drop(columns, axis=1,inplace=True)
consumer_test_df.drop(columns, axis=1,inplace=True)


# In[19]:


# store data of disputed people into the “disputed_cons” variable for future tasks
disputed_cons = consumer_train_df[consumer_train_df['Consumer disputed?'] == 'Yes']
disputed_cons.head()


# In[20]:


# Plot bar graph of thetotal no of disputes of consumers with the help ofseaborn 
plt.figure(figsize=(10, 6))
sns.countplot(x='Consumer disputed?', data=disputed_cons)
plt.xlabel('Year')
plt.ylabel('Number of Disputes')
plt.title('Total Number of Consumer Disputes Over the Years')
plt.show()


# In[21]:


# Plot bar graph of the total no of disputes products-wise with the help ofseaborn 
plt.figure(figsize=(10, 6))
sns.countplot(x='Product', data=disputed_cons, order=disputed_cons['Product'].value_counts().index)
plt.xlabel('Products')
plt.ylabel('Number of Products')
plt.title('Total Number of Product Wise ')
plt.xticks(rotation=90, ha='center')
plt.show()


# In[22]:


# Plot bar graph of the total no of disputes with Top Issues by Highest Disputes
top_issue = disputed_cons['Issue'].value_counts().nlargest(15)
top_issue_df = pd.DataFrame({'Issue': top_issue.index, 'Count': top_issue.values})

plt.figure(figsize=(10, 6))
sns.barplot(x='Issue', y='Count', data=top_issue_df)
plt.xlabel('Issue')
plt.ylabel('Number of highest disputes')
plt.title('Total Number of Product Wise ')
plt.xticks(rotation=90, ha='center')
plt.show()


# In[23]:


# Plot bar graph of the total no of disputes by State with Maximum Disputes 
state_with_max_disputes = disputed_cons['State'].value_counts().idxmax()
total_disputes_by_state = disputed_cons.groupby('State').size().reset_index(name='Total Disputes')
max_disputes_state_df = total_disputes_by_state[total_disputes_by_state['State'] == 
                                                state_with_max_disputes]

plt.figure(figsize=(12, 6))
sns.barplot(x='State', y='Total Disputes', data=max_disputes_state_df)
plt.xlabel('State')
plt.ylabel('Total Number of Disputes')
plt.title(f'Total Number of Disputes by State (Max Disputes: {state_with_max_disputes})')
plt.show()


# In[24]:


# Plotbar graph of the total no of disputes Submitted Via different source 
top_submitted = disputed_cons['Submitted via'].value_counts().nlargest(15)
top_submitted_df = pd.DataFrame({'Submitted via': top_submitted.index, 'Count': top_submitted.values})

plt.figure(figsize=(10, 6))
sns.barplot(x='Submitted via', y='Count', data=top_submitted_df)
plt.xlabel('Submitted via')
plt.ylabel('Number of disputes')
plt.title('Total Number of disputes Submitted Via different source')
plt.xticks(rotation=90, ha='center')
plt.show()


# In[25]:


# Plot bar graph of the total no of disputes where the Company's Response to the Complaints
top_response = disputed_cons['Company response to consumer'].value_counts().nlargest(15)
top_response_df = pd.DataFrame({'Company response to consumer': top_response.index,
                                'Count': top_response.values})

plt.figure(figsize=(10, 6))
sns.barplot(x='Company response to consumer', y='Count', data=top_response_df)
plt.xlabel('Company response to consumer')
plt.ylabel('Number of disputes')
plt.title('Total Number of disputes Company response to consumer')
plt.xticks(rotation=90, ha='center')
plt.show()


# In[26]:


# Plot bar graph ofthe total no of disputes.Whether there are Disputes Instead of Timely Response 
plt.figure(figsize=(10, 6))
sns.countplot(x='Timely response?', data=disputed_cons)
plt.xlabel('Company response to consumer')
plt.ylabel('Number of disputes')
plt.title('Total Number of disputes Company response to consumer')
plt.show()


# In[27]:


# Plot bar graph of the total no of disputes over Year Wise Complaints 
plt.figure(figsize=(10, 6))
sns.countplot(x='year', data=disputed_cons)
plt.xlabel('year')
plt.ylabel('Number of disputes')
plt.title('Total Number of disputes by year')
plt.show()


# In[28]:


# Plot bar graph of Top Companies with Highest Complaints 
top_company = disputed_cons['Company'].value_counts().nlargest(15)
top_company_df = pd.DataFrame({'Company': top_company.index, 'Count': top_company.values})

plt.figure(figsize=(10, 6))
sns.barplot(x='Company', y='Count', data=top_company_df)
plt.xlabel('Company')
plt.ylabel('Number of disputes')
plt.title('Total Number of disputes by Company')
plt.xticks(rotation=90, ha='center')
plt.show()


# In[29]:


# Converte all negative days held to zero(time taken by the authority that can't be negative)
disputed_cons['Days Held'] = disputed_cons['Days Held'].apply(lambda x: max(0, x))
disputed_cons['Days Held'].unique()


# In[30]:


drop_columns=['Company','State','year','Days Held','Issue']


# In[31]:


# Drop UnnecessaryColumns for the Model Buildinglike:'Company', 'State', 'Year_Received', 'Days_held'
consumer_train_df.drop(drop_columns, axis=1, inplace=True)


# In[32]:


consumer_test_df.drop(drop_columns, axis=1, inplace=True)


# In[33]:


# Change Consumer Disputed Column to 0 and 1(yes to 1, and no to 0)
consumer_train_df['Consumer disputed?'] = consumer_train_df['Consumer disputed?'].map(
    {'Yes': 1, 'No': 0})


# In[34]:


# Create Dummy Variables for categoricalfeaturesand concat with the original data frame
# like: 'Product,’'Submitted via,’'Company response to consumer,’'Timely response?' 
categorical_columns = ['Product', 'Submitted via', 'Company response to consumer', 'Timely response?']
dummy_df_train = pd.get_dummies(consumer_train_df[categorical_columns])
consumer_train_df = pd.concat([consumer_train_df, dummy_df_train], axis=1)
consumer_train_df = consumer_train_df.drop(categorical_columns, axis=1)


# In[35]:


dummy_df_test = pd.get_dummies(consumer_test_df[categorical_columns])
consumer_test_df = pd.concat([consumer_test_df, dummy_df_test], axis=1)
consumer_test_df = consumer_test_df.drop(categorical_columns, axis=1)


# In[36]:


consumer_train_df.head()


# In[37]:


consumer_test_df.head()


# In[38]:


# Scaling the Data Sets (note:discard dependent variable before doing standardization)and 
# Makefeature Selection with the help ofPCAup to 80% of the information.
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[39]:


# we take just the train dataset as it contains label ie consumer disrupted column
# Splitting the Data Sets Into X and Y by the dependent and independent variables 
# (data selected by PCA)
x = consumer_train_df.drop('Consumer disputed?', axis=1)
y = consumer_train_df['Consumer disputed?']


# In[40]:


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# In[41]:


pca = PCA(n_components=0.8)
x_pca = pca.fit_transform(x_scaled)


# In[42]:


print(f"Number of selected features: {x_pca.shape[1]}")
print(f"Percentage of retained information: {round(sum(pca.explained_variance_ratio_) * 100, 2)}%")


# In[43]:


selected_features = pd.DataFrame(x_pca, columns=[f'PC{i}' for i in range(1, x_pca.shape[1] + 1)])
final_data = pd.concat([selected_features, y], axis=1)
final_data.head()


# In[44]:


# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 21)
x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[45]:


# Build given models and measure their test and validation accuracy:
# Logistic Regression
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression(solver="liblinear")


# In[46]:


logreg.fit(x_train,y_train)
y_pred_test_logreg = logreg.predict(x_test)
y_pred_train_logreg = logreg.predict(x_train)


# In[47]:


print("Accuracy Score of Logistic Regression Model on Test",
      accuracy_score(y_test, y_pred_test_logreg))
print("Accuracy Score of Logistic Regression Model on Train",
      accuracy_score(y_train, y_pred_train_logreg))


# In[48]:


# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()


# In[49]:


dt_model.fit(x_train,y_train)
y_pred_test_dt = dt_model.predict(x_test)
y_pred_train_dt = dt_model.predict(x_train)


# In[50]:


print("Accuracy Score of Decision Tree Model on Test",
      accuracy_score(y_test, y_pred_test_dt))
print("Accuracy Score of Decision Tree Model on Train",
      accuracy_score(y_train, y_pred_train_dt))


# In[51]:


# RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc_model = RandomForestClassifier(n_estimators=50)


# In[52]:


rfc_model.fit(x_train,y_train)
y_pred_test_rfc = rfc_model.predict(x_test)
y_pred_train_rfc = rfc_model.predict(x_train)


# In[53]:


print("Accuracy Score of Random Forest Classifier Model on Test",
      accuracy_score(y_test, y_pred_test_rfc))
print("Accuracy Score of Random Forest Classifier Model on Train",
      accuracy_score(y_train, y_pred_train_rfc))


# In[54]:


# AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier(n_estimators=80)


# In[55]:


ada_model.fit(x_train,y_train)
y_pred_test_abc = ada_model.predict(x_test)
y_pred_train_abc = ada_model.predict(x_train)


# In[56]:


print("Accuracy Score of Ada Boost Classifier Model on Test",
      accuracy_score(y_test, y_pred_test_abc))
print("Accuracy Score of Ada Boost Classifier Model on Train",
      accuracy_score(y_train, y_pred_train_abc))


# In[57]:


# GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
gb_model = GradientBoostingClassifier(n_estimators=20,learning_rate=0.02)


# In[58]:


gb_model.fit(x_train,y_train)
y_pred_test_gbc = gb_model.predict(x_test)
y_pred_train_gbc = gb_model.predict(x_train)


# In[59]:


print("Accuracy Score of Gradient Boosting Classifier Model on Test",
      accuracy_score(y_test, y_pred_test_gbc))
print("Accuracy Score of Gradient Boosting Classifier Model on Train",
      accuracy_score(y_train, y_pred_train_gbc))


# In[60]:


# XGBClassifier
from xgboost.sklearn import XGBClassifier
xgb_model = XGBClassifier(n_estimators=20, objective="multi:softmax", num_class=2)


# In[61]:


xgb_model.fit(x_train,y_train)
y_pred_test_xgb = xgb_model.predict(x_test)
y_pred_train_xgb = xgb_model.predict(x_train)


# In[62]:


print("Accuracy Score of XG Boost Classifier Model on Test",
      accuracy_score(y_test, y_pred_test_xgb))
print("Accuracy Score of XG Boost Classifier Model on Train",
      accuracy_score(y_train, y_pred_train_xgb))


# In[63]:


# Whoever gives the most accurate result uses it and predicts the outcome for the test file and 
# fills its dispute column so the business team can take some action accordingly.
# our highest accuracy is 78% for both test and train dataset for Logreg, Ada boost, Gradient and 
# XG boost classifier
x_test['Consumer disputed?'] = y_pred_test_logreg
print(x_test.head())


# In[ ]:


#...................................................................................................

