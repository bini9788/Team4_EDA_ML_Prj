#!/usr/bin/env python
# coding: utf-8

# In[386]:


# 패키지 임포트
import numpy as np
import pandas as pd
import os 
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 500)

plt.rc("font", family="Malgun Gothic") # matplotlib 한글깨짐


# In[337]:


# 데이터 로드 
# 부동산 데이터
train_data=pd.read_csv('C:/Users/bini9/OneDrive/문서/솔트룩스/프로젝트/4차 - EDA&ML/data/train_data.csv', encoding='utf-8')


# In[338]:


# 데이터 확인
train_data


# In[339]:


train_data.shape


# In[415]:


train_data.describe(include='all')


# In[341]:


train_data.info()


# In[342]:


train_data.isna().sum()


# In[343]:


#### 전처리
## 거래금액
# 거래금액 데이터 컬럼명 변경
train_data.rename(columns = {'거래금액(만원)':'거래금액'}, inplace=True)


# In[344]:


# 거래금액 , 제거 후 숫자형 변환
train_data['거래금액'] = train_data['거래금액'].str.replace(',','').astype(int)


# In[371]:


# 거래 가격 분포 확인
f, ax = plt.subplots(figsize = (10, 6)) 
sns.distplot(train_data["거래금액"])


# fig=plt.figure(figsize=(10, 5))
# plt.hist(train_data['거래금액'],bins=50)
# plt.title('거래금액 히스토그램')
# plt.xlabel('거래금액')
# plt.show()


# In[372]:


# 거래 금액 로그변환
train_data['거래금액_로그'] = np.log1p(train_data['거래금액'])


# In[373]:


# 로그 거래 가격 분포 확인
f, ax = plt.subplots(figsize = (10, 6)) 
sns.distplot(train_data['거래금액_로그'])


# In[400]:


# 층과 거래금액의 상관계수 확인
print(round(np.corrcoef(train_data['층'], train_data['거래금액'])[0,1],5)) #0.17049

# 시각화
plt.scatter(train_data['층'], train_data['거래금액'])
plt.plot( train_data['거래금액'].groupby(train_data['층']).mean(), color='red')
plt.show()


# In[399]:


# 층과 로그 거래금액의 상관계수 확인 - 로그변환한 값이 상관관계가 더 높다
print(round(np.corrcoef(train_data['층'], train_data['거래금액_로그'])[0,1],5)) #0.18676

# 시각화
plt.scatter(train_data['층'], train_data['거래금액_로그'])
plt.plot( train_data['거래금액_로그'].groupby(train_data['층']).mean(), color='red')
plt.show()


# In[348]:


# 층 변수 분포 확인
fig=plt.figure(figsize=(10, 5))
plt.hist(train_data['층'],bins=50)
plt.title('층 수 히스토그램')
plt.xlabel('층 수')
plt.show()


# In[364]:


# 층 중 음수 개수 : 70개, 전체의 약 0.03%
sum(train_data['층']<=0)


# In[365]:


# 층 중 음수 개수 확인
train_data.loc[train_data['층']<0,'층'].value_counts()


# In[349]:


# 높은 층수 확인
train_data.loc[train_data['층']==69,'단지명']


# In[438]:


# 이상치 처리 방법에 따른 로그 거래금액과의 correlation 확인

# 음수가 아닌 층과 로그 거래금액의 상관계수 확인
df_floor_above_0 =train_data.loc[train_data['층']>0,['층', '거래금액_로그']]
delete_outlier_corr=round(np.corrcoef(df_floor_above_0['층'], df_floor_above_0['거래금액_로그'])[0,1],5) # 0.18714

# 음수 층의 -를 제거해서 양수로 변환 
floor_plus = [floor*(-1) if floor<0 else floor for floor in train_data['층'] ]
# 로그 거래금액의 상관계수 확인
replace_plus_corr=round(np.corrcoef(floor_plus, train_data['거래금액_로그'])[0,1],5)#0.18687

# 음수 층을 모두 최저층인 1로 변환 
floor_one = [1 if floor<0 else floor for floor in train_data['층'] ]
# 로그 거래금액의 상관계수 확인
replace_one_corr = round(np.corrcoef(floor_one, train_data['거래금액_로그'])[0,1],5) #0.18685

# 층수의 평균값으로 변환
floor_mean = [train_data['층'].mean() if floor<0 else floor for floor in train_data['층'] ]
replace_mean_corr = round(np.corrcoef(floor_mean, train_data['거래금액_로그'])[0,1],5)

# corr비교
pd.DataFrame({'delete_outlier':[delete_outlier_corr], 'replace_plus':[replace_plus_corr],
             'replace_one':[replace_one_corr],'replace_mean':[replace_mean_corr]}, index = ['correlation'])

# 삭제 했을때가 가장 correlation이 높다


# In[445]:


train_data[train_data['층']<0].index


# In[449]:


## 위의 결과에 따라 이상치인 값들을 제거
train_data.drop(train_data.loc[train_data['층']<0].index, inplace=True)


# In[451]:


# 삭제 후 층 변수 분포
fig=plt.figure(figsize=(10, 5))
plt.hist(train_data['층'],bins=50)
plt.title('층 수 히스토그램')
plt.xlabel('층 수')
plt.show()


# In[351]:


## 계약년월 변수 - 이상치로 판단할 값은 없는것으로 보임
train_data['계약년월'] = train_data['계약년월'].astype('str') # 시각화할때 int형으로 들어가면 숫자 간격이 동일하지 않아서 이상하게 출력됌
train_data_df_ym = pd.DataFrame(train_data['계약년월'].value_counts())

plt.rc('font', size=13)
fig=plt.figure(figsize=(15, 10))
plt.barh(train_data_df_ym.sort_index().index,train_data_df_ym.sort_index().iloc[:,0])
plt.title('계약년월별 거래건수')
plt.xlabel('계약년월')
plt.show()


# In[355]:


## 계약년월
# 계약년도와 계약월 컬럼 생성(EDA)
train_data['계약년도'] = train_data['계약년월'].map(lambda x : str(x)[:4])
train_data['계약월'] = train_data['계약년월'].map(lambda x : str(x)[4:])
train_data


# In[356]:


# 계약년월 별로 그룹화 후 평균 거래금액 산출 
train_data_grouped_ym_mean = train_data.groupby(train_data['계약년월'])['거래금액'].mean()

# 시각화
plt.rc('font', size=15)
fig=plt.figure(figsize=(14, 5))
# plt.boxplot(train_data['계약년월'], train_data['거래금액'])
plt.plot(train_data_grouped_ym_mean)
plt.xticks(rotation='vertical')
plt.title('2018~2020년도 계약월별 평균 거래금액')
plt.xlabel('년-월')
plt.ylabel('평균 거래금액')
plt.show()


# In[357]:


# 계약년도 별로 그룹화 후 평균 거래금액 산출 
# train_data['계약년도'] = train_data['계약년도'].astype('str')
train_data_year_mean = train_data.groupby(train_data['계약년도'])['거래금액'].mean()

# 시각화
plt.figure(figsize=(10, 5))
plt.plot(train_data_year_mean)
plt.xlabel('년도')
plt.ylabel('평균 거래금액')
plt.show()


# In[358]:


# 각 년도 마다 월 별로 그룹화 후 평균 거래금액 산출 
train_data_grouped_ym_mean_2018 = train_data[train_data['계약년도']=='2018'].groupby(train_data['계약월'])['거래금액'].mean()
train_data_grouped_ym_mean_2019 = train_data[train_data['계약년도']=='2019'].groupby(train_data['계약월'])['거래금액'].mean()
train_data_grouped_ym_mean_2020 = train_data[train_data['계약년도']=='2020'].groupby(train_data['계약월'])['거래금액'].mean()


# In[359]:


# 시각화
plt.rc('font', size=15)
fig=plt.figure(figsize=(10, 5))
plt.plot(list(range(1,13)), train_data_grouped_ym_mean_2018, label = '2018')
plt.plot(list(range(1,13)), train_data_grouped_ym_mean_2019, label = '2019')
plt.plot(list(range(1,13)), train_data_grouped_ym_mean_2020, label = '2020')

plt.legend()
plt.title('계약년도에 따른 월별 평균 거래금액')
plt.xticks(list(range(1,13)))
plt.xlabel('월')
plt.ylabel('평균 거래금액')
plt.show()
# 월별 추이가 년도별로 매우 다르고 시간에 따른 흐름이 중요하기 때문에 모형에는 '계약년월' 변수만 추가







