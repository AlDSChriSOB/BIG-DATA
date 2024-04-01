#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#import pandas_profiling as pp
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


accident = pd.read_csv('dft-road-casualty-statistics-accident-2019.csv')




accident


# In[3]:


casualty = pd.read_csv('dft-road-casualty-statistics-casualty-2019.csv')

casualty


# In[4]:


vehicle = pd.read_csv('dft-road-casualty-statistics-vehicle-2019.csv')

vehicle


# In[5]:


similar_columns = [x for x in accident.columns if x in casualty.columns and x in vehicle.columns]
    
similar_columns 


# In[6]:


#profile = pp.ProfileReport(accident)

#profile


# In[7]:


accident.isnull().sum()


# In[8]:


accident = accident.fillna(-9999) 

accident


# In[9]:


accident.isnull().sum()


# In[10]:


casualty.isnull().sum()


# In[11]:


vehicle.isnull().sum()


# #### WEEK 2

# In[12]:


accident['converted_time'] = pd.DatetimeIndex(accident['time'])


# In[13]:


accident['hours'] = accident['converted_time'].dt.hour
accident['minutes'] = accident['converted_time'].dt.minute

accident['decimal_time'] = accident.hours + accident.minutes/60
accident


# In[14]:


sns.distplot(x=accident['decimal_time'], kde=True)


# In[15]:


accident.decimal_time.describe()


# In[16]:


sns.displot(data=accident, x='day_of_week')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### WEEK 3

# ###### Rule 1: Speed Limit = 20mph → Accident Severity = Slight

# In[17]:


#support

total = len(accident)


# In[18]:


rule1_df = accident[(accident['speed_limit']== 20) & (accident['accident_severity']==3)]
freq_rule1 = len(rule1_df)


# In[19]:


support = freq_rule1/total
support


# In[20]:


# Confidence
freq_speedlimit20 = len(accident[accident['speed_limit']==20])
confidence = freq_rule1/freq_speedlimit20
confidence


# In[21]:


# Completeness
freq_slight = len(accident[accident['accident_severity']==3])
completeness = freq_rule1/freq_slight
completeness


# In[22]:


# Lift
support_speedlimit20 = freq_speedlimit20/total
support_slight = freq_slight/total
lift = support/(support_speedlimit20 * support_slight)
lift


# In[23]:


# Conviction
conviction = (1 - support_slight)/(1 - confidence)
conviction


# In[24]:


# Piatetsky-Shapiro Rule Interestingness
ri = freq_rule1 - (freq_speedlimit20 * freq_slight / total)
ri


# ###### Rule 2: Speed Limit = 70mph → Accident Severity = Fatal

# In[25]:


#support

total = len(accident)
rule2_df = accident[(accident['speed_limit']== 70) & (accident['accident_severity']==1)]
freq_rule2 = len(rule2_df)


# In[26]:


support2 = freq_rule2/total
support2


# In[27]:


# Confidence
freq_speedlimit70 = len(accident[accident['speed_limit']==70])
confidence2 = freq_rule2/freq_speedlimit70
confidence2


# In[28]:


# Completeness
freq_fatal = len(accident[accident['accident_severity']==1])
completeness2 = freq_rule2/freq_fatal
completeness2


# In[29]:


# Lift
support_speedlimit70 = freq_speedlimit70/total
support_fatal = freq_fatal/total
lift2 = support2/(support_speedlimit70 * support_fatal)
lift2


# In[30]:


# Conviction
conviction2 = (1 - support_fatal)/(1 - confidence2)
conviction2


# In[31]:


# Piatetsky-Shapiro Rule Interestingness
ri2 = freq_rule2 - (freq_speedlimit70 * freq_fatal / total)
ri2


# ###### Rule 3: Pedestrian Crossing Code = Zebra → Casualty Type = Pedestrian

# In[32]:


accident_casualty = pd.merge(accident, casualty, how='right', on='accident_reference')
accident_casualty


# In[33]:


#support

total = len(accident_casualty)
rule3_df = accident[(accident_casualty['pedestrian_crossing_physical_facilities']== 1) & (accident_casualty['casualty_type']==0)]
freq_rule3 = len(rule3_df)


# In[34]:


support3 = freq_rule3/total
support3


# In[35]:


# Confidence
freq_zebra = len(accident_casualty[accident_casualty['pedestrian_crossing_physical_facilities']==1])
confidence3 = freq_rule3/freq_zebra
confidence3


# In[36]:


# Completeness
freq_pedes = len(accident_casualty[accident_casualty['casualty_type']==0])
completeness3 = freq_rule3/freq_pedes
completeness3


# In[37]:


# Lift
support_zebra = freq_zebra/total
support_pedes = freq_pedes/total
lift3 = support3/(support_zebra * support_pedes)
lift3


# In[38]:


# Conviction
conviction3 = (1 - support_pedes)/(1 - confidence3)
conviction3


# In[39]:


# Piatetsky-Shapiro Rule Interestingness
ri3 = freq_rule3 - (freq_zebra * freq_pedes / total)
ri3


# In[ ]:





# ###### Exercise 3

# In[40]:


speed_limits = [20, 30, 40, 50, 60, 70, -1]
severity = [1, 2, 3]
total = len(accident)

for i in speed_limits:
    #support
    freq = len(accident[(accident['speed_limit']== i) & (accident['accident_severity']==1)])
    support = freq/total
    print('speed limit', i, ', accident severity fatal')
    print('support:', support)
    
    
    # Confidence
    freq_speedlimit = len(accident[accident['speed_limit']==i])
    confidence = freq/freq_speedlimit
    print('confidence:', confidence)
    print('')
    


# In[41]:


speed_limits = [20, 30, 40, 50, 60, 70, -1]
total = len(accident)

for i in speed_limits:
    #support
    freq = len(accident[(accident['speed_limit']== i) & (accident['accident_severity']==2)])
    support = freq/total
    print('speed limit', i, ', accident severity serious')
    print('support:', support)
    
    
    # Confidence
    freq_speedlimit = len(accident[accident['speed_limit']==i])
    confidence = freq/freq_speedlimit
    print('confidence:', confidence)
    print('')
    


# In[42]:


speed_limits = [20, 30, 40, 50, 60, 70, -1]
total = len(accident)

for i in speed_limits:
    #support
    freq = len(accident[(accident['speed_limit']== i) & (accident['accident_severity']==3)])
    support = freq/total
    print('speed limit', i, ', accident severity slight')
    print('support:', support)
    
    
    # Confidence
    freq_speedlimit = len(accident[accident['speed_limit']==i])
    confidence = freq/freq_speedlimit
    print('confidence:', confidence)
    print('')
    


# In[ ]:





# #### Geographic clustering

# In[128]:


df = pd.read_csv('dft-road-casualty-statistics-accident-2019.csv')


# In[129]:


df.isnull().sum()


# In[130]:


df = df.dropna()


# In[131]:


df.head()


# In[132]:


from sklearn.cluster import KMeans

df_long_lat= df[['longitude', 'latitude']]
df_long_lat.head()


# In[133]:


kmeans = KMeans(n_clusters=25, random_state=0)
kmeans.fit(df_long_lat)


# In[134]:


labels = kmeans.predict(df_long_lat)
centriods = kmeans.cluster_centers_


# In[135]:


labels


# In[136]:


centriods


# In[137]:


fig = plt.figure(figsize= (5,7))
colors = map(lambda x: colmap[x+1], labels)

plt.scatter(df_long_lat['longitude'], df_long_lat['latitude'], color='blue', s=0.05)
plt.ylim(45, 65)
plt.xlim(-10, 4)

plt.ylabel("Latitude")
plt.xlabel("Longitude")

plt.scatter(centriods[:, 0], centriods[:,1], color='orange')
plt.show()


# In[138]:


# Inertia

inertia = kmeans.inertia_
inertia


# In[139]:


for k in range(25, 31):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(df_long_lat)
    print(k, ':', kmeans.inertia_)


# #### Different Clustering

# In[140]:


df_speed_weather = df[['speed_limit', 'weather_conditions']]

df_speed_weather.describe()


# In[141]:


kmeans = KMeans(n_clusters = 6, random_state=0)
kmeans.fit(df_speed_weather)


# In[142]:


labels = kmeans.predict(df_speed_weather)
centriods = kmeans.cluster_centers_


# In[143]:


labels


# In[144]:


centriods


# In[145]:


fig = plt.figure(figsize= (5,7))
colors = map(lambda x: colmap[x+1], labels)

plt.scatter(df_speed_weather['weather_conditions'], df_speed_weather['speed_limit'], color='blue', s=1)
plt.ylim(-10, 80)
plt.xlim(0, 10)

plt.xlabel("Weather condition")
plt.ylabel("Speed limit")

plt.scatter(centriods[:, 1], centriods[:,0], color='orange')
plt.show()


# In[146]:


# 12 clusters

kmeans = KMeans(n_clusters = 12, random_state=0)
kmeans.fit(df_speed_weather)


# In[147]:


labels = kmeans.predict(df_speed_weather)
centriods = kmeans.cluster_centers_


# In[148]:


labels


# In[149]:


centriods


# In[150]:


fig = plt.figure(figsize= (5,7))
colors = map(lambda x: colmap[x+1], labels)

plt.scatter(df_speed_weather['weather_conditions'], df_speed_weather['speed_limit'], color='blue', s=1)
plt.ylim(-10, 80)
plt.xlim(0, 10)

plt.xlabel("Weather condition")
plt.ylabel("Speed limit")

plt.scatter(centriods[:, 1], centriods[:,0], color='orange')
plt.show()


# In[151]:


df.columns


# In[ ]:


features = ['road_surface_conditions', 'speed_limit', 'light_conditions', 'junction_control', 'time']
x = df.loc[:, features].values
y = df.loc[:,['accident_severity']].values
x = StandardScaler().fit_transform(x)


# In[157]:


df.corr()


# In[ ]:




