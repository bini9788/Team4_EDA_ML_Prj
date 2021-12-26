#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 패키지 임포트
import numpy as np
import pandas as pd
import scipy.stats as ss
import os 
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install plotly
import plotly.offline as off
import plotly.graph_objs as go
pd.set_option('display.max_columns', 500)

plt.rc("font", family="Malgun Gothic") # matplotlib 한글깨짐 방지


# In[2]:


# 거래금액, 전용면적 데이터 컬럼명 변경
train_data=pd.read_csv('train_data.csv', encoding='CP949')
train_data.rename(columns = {'거래금액(만원)':'거래금액', '전용면적(㎡)':'전용면적'}, inplace=True)
train_data


# ## 인코딩 - 건축년도와 전용면적

# ### 전용면적
# #### 결측지 제거
# #### 값 그대로/로그변환/50단위 나눠 인코딩

# In[3]:


#거래금액 - 전용면적 분포도 확인 (12/26)
plt.figure(figsize=(20,5))
plt.subplot(1, 2, 1)
plt.scatter(train_data['전용면적'],train_data['거래금액'])

plt.subplot(1, 2, 2)
plt.scatter(train_data['전용면적'],train_data['거래금액'])
plt.xlim([300,400])#max=395.06


# In[4]:


#결측치 제거하기
train_data.drop(train_data.loc[train_data['전용면적']>300].index, inplace=True)
train_data


# In[5]:


train_data['전용면적'].describe()


# In[6]:


# 값 그대로
encoding_data=pd.DataFrame(data=train_data['전용면적'])
encoding_data.rename(columns={'전용면적':'전용면적_값'}, inplace=True)
encoding_data


# In[7]:


#로그변환
encoding_data['전용면적_log']=np.log1p(train_data['전용면적'])
encoding_data


# In[8]:


#10단위로 29구간 인코딩 (10~290)
labels=list(range(1,30))
encoding_data['전용면적_cut']=pd.cut(train_data['전용면적'],
                                bins=29, labels=labels, include_lowest=True)
encoding_data


# In[9]:


encoding_data['전용면적_cut'].value_counts()


# In[10]:


train_data['전용면적'].value_counts().sort_values(ascending=False)


# ### 건축년도
# #### 

# In[11]:


train_data['건축년도'].describe()


# In[12]:


#건축년도 변수-가장 작은 건축년도 = 값으로 인코딩
encoding_data['건축년도']=train_data['건축년도'] #drop, 단순 값 확인
encoding_data['건축년도_min_차이']=train_data['건축년도']-train_data['건축년도'].min()
encoding_data


# In[13]:


#거래금액 DESC 정렬, 건축년도 정렬 후 index 초기화...
#반대로 해야됌...
pd.DataFrame(train_data['건축년도'].loc[train_data['거래금액'].sort_values(ascending=False).index].reset_index())


# In[16]:


#5단위로 13구간 인코딩 (1961~2021)
#5단위로 12구간 인코딩 (1961~2020)시 60구간이라 더 효과적일수도 있음
labels=list(range(1,14))
encoding_data['건축년도_cut']=pd.cut(train_data['건축년도'],
                                bins=13, labels=labels, include_lowest=True)
encoding_data


# In[ ]:


#13개의 구간으로 인코딩된 건축년도_cut과
#각 구간별 평균 거래금액 기준으로 rank 부여한 컬럼 생성하여 가중치 컬럼 생성


# In[ ]:


'''
#계약년월별 건축년도 count (12/26)
#ex)2020년 1월에 거래된 아파트의 건축년도 분포도를 알고 싶다.
group_2018=train_years.groupby(['계약년월','계약월','건축년도'])
group_2018.size().reset_index(name='counts')

# construction_month_count=train_years.groupby('계약년도').get_group('2018').value_counts()
group=train_years.groupby(['계약년월','계약년도','건축년도'])
construction_month_count=group.size().reset_index(name='counts')
df_construction_month_count=pd.DataFrame(construction_month_count)
# df_construction_month_count
df_construction_month_count_2018=df_construction_month_count.groupby('계약년도').get_group('2018')
df_construction_month_count_2019=df_construction_month_count.groupby('계약년도').get_group('2019')
df_construction_month_count_2020=df_construction_month_count.groupby('계약년도').get_group('2020')
df_construction_month_count_2018

#2018~2020 월별 건축년도 분포도
#월별 분포도를 보여줘야 하는 것 아닌가. 
fig=plt.figure(figsize=(20, 10))
plt.subplot(3,1,1)
plt.scatter(df_construction_month_count_2018['건축년도'],df_construction_month_count_2018['counts'])
plt.ylabel('2018')
plt.xticks(visible=False)

plt.subplot(3,1,2)
plt.scatter(df_construction_month_count_2019['건축년도'],df_construction_month_count_2019['counts'])
plt.ylabel('2019')
plt.xticks(visible=False)

plt.subplot(3,1,3)
plt.scatter(df_construction_month_count_2020['건축년도'],df_construction_month_count_2020['counts'])
plt.ylabel('2020')

##### 구축이냐 신축이냐가 아닌 동일 아파트의 갯수가 얼마냐 더 많냐에 따라 비례적으로 계약건수 차이가 있음. 즉 건축년도와 계약 건수의 상관관계는 크지 않다.
'''

'''
#거래금액 - 전용면적 분포도 확인 (12/26)
plt.figure(figsize=(20,5))
plt.subplot(1, 2, 1)
plt.scatter(train_data['전용면적'],train_data['거래금액'])
plt.xlabel('전용면적')

plt.subplot(1, 2, 2)
plt.scatter(train_data['전용면적'],train_data['거래금액'])
plt.xlim([230,300])
plt.xlabel('전용면적')
'''

