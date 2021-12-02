#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


data = pd.read_csv("0723_all_concat.csv", index_col = 0 )
data.head()


# In[3]:


data.columns


# In[ ]:





# In[ ]:





# In[3]:


data.shape


# In[4]:


data.columns


# In[4]:


# 필요없는 변수 삭제
train = data.drop(columns = ["playerId","name"], axis = 1)


# In[5]:


# 게임에서 튕긴 매치 제거
# 이동거리0 피해량0인 튕긴것같은 매치스텟

logout_index = train[(train["walkDistance"] == 0) & (train["damageDealt"] == 0)].index
train.drop(logout_index, inplace = True)


# ## Feature Engineering

# 1. 이동 거리 가중 평균
# 
# 평균 속도로 각 방법 별로 이동 시간 예측 → 그 비율을 계산해서 가중치!

# In[6]:


# 가중치 결과값 → 수영 0.00 / 걷기 0.78 / 자동차 0.22
train["movement_w_mean"] = 0*train["swimDistance"] + 0.78*train["walkDistance"] + 0.22*train["vehicleDestroys"]


# 2. 킬플레이스/윈플레이스
# 
# killPlace : 킬수에 따른 그 매치에서의 순위
# winPlace : 몇번째로 죽었는가에 대한 순위
# 
# => 얼마나 적게 죽이고 오래 살았나를 보면 "간디"메타와 "여포"메타를 구별지을 지표라 생각된다.

# In[7]:


train["kill_over_winPlae"] = round(train["killPlace"]/train["winPlace"], 4)


# 3. 킬수/생존시간  
# 
# "여포"메타일수록 생존시간대비 킬수가 높을 것이다.
# 

# In[8]:


train["kill_over_timeSurvived"] = round(train["kills"]/train["timeSurvived"], 4)


# 4. Label encoding of categorical variable

# In[10]:


from sklearn.preprocessing import LabelEncoder


# In[11]:


columns = ["deathType", "mapName", "matchType"]

for col in columns :
    label_encoder = LabelEncoder()
    label_encoder.fit(train[col])
    train[col] = label_encoder.transform(train[col])


# In[12]:


train.info()


# In[12]:


# 0으로 나눠져 NaN되는 것을 입실론으로 처리
# 일단 드랍하고 진행
train["longestKill_over_movement_w_mean"] = round(train["longestKill"]/(train["movement_w_mean"]+ 1e-10), 4)


# In[13]:


# 했는데 스케일링에서 infinity? 너무 큰 값 오류
train["timeSurvived_over_weaponsAcquired"] = round(train["timeSurvived"]/(train["weaponsAcquired"]+ 1e-10), 4)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Normalization
# ### MinMaxScaler

# In[14]:


from sklearn.preprocessing import MinMaxScaler


# In[15]:


scaler = MinMaxScaler()
scaler.fit(train)
train_scaled = scaler.transform(train)

train_scaled_df = pd.DataFrame(data = train_scaled, columns = train.columns )


# In[16]:


train_scaled_df.shape


# In[ ]:





# ### StandardScaler

# In[16]:


from sklearn.preprocessing import StandardScaler


# In[17]:


scaler = StandardScaler()
scaler.fit(train)
train_scaled = scaler.transform(train)

sta_train_scaled_df = pd.DataFrame(data = train_scaled, columns = train.columns )


# In[ ]:





# In[ ]:





# In[ ]:





# # 군집분석
# 
# ## GMM

# In[17]:


from sklearn.mixture import GaussianMixture


# In[19]:


# 군집갯수 : 2개


# In[64]:


# 학습
# 세팅과 피팅
gmm = GaussianMixture(n_components = 3, random_state = 1234).fit(train)
# 아웃풋
gmm_cluster_labels = gmm.predict(train)

# 쌓기
train_scaled_df['GMM_cluster'] = gmm_cluster_labels


# In[30]:


train_scaled_df


# - 평가하기

# -> 실루엣 분석을 동일하게 사용했습니다.

# In[18]:


from sklearn.metrics import silhouette_samples, silhouette_score 


# In[36]:


# 모든 개별 데이터 실루엣 계수 값 구하기
score_samples = silhouette_samples(train_scaled, sta_train_scaled_df["GMM_cluster"])
print("silhouette_samples() return 값의 shape", score_samples.shape)


# In[37]:


# train_scaled_df에 실루엣 계수 컬럼 추가
sta_train_scaled_df["silhouette_coeff"] = score_samples


# In[38]:


# 모든 데이터의 평균 실루엣 계수 값 구하기
average_score = silhouette_score(train_scaled, train_scaled_df["GMM_cluster"])
print("개인 플레이 성향 Silhouette Analysis Score: {0:3f}".format(average_score))


# In[ ]:





# ### PCA 차원축소

# In[19]:


from sklearn.decomposition import PCA

n_columns = 13
pca = PCA(n_components = 13)
pca.fit(train)


# In[20]:


plt.plot(pca.explained_variance_ratio_)


# In[21]:


n_columns = 2
pca = PCA(n_components = 2)
pca.fit(train)

np.sum(pca.explained_variance_ratio_)


# In[26]:


train_pca = pca.transform(train)

mMs = MinMaxScaler()
######################################################################################
train_pca_norm = mMs.fit_transform(train_pca)
train_pca_df = pd.DataFrame(data = train_pca_norm, columns = ['pc1', 'pc2'])


# In[27]:


# 학습
# 세팅과 피팅
gmm = GaussianMixture(n_components = 4, random_state = 1234). fit(train_pca_df)
# 아웃풋
gmm_cluster_labels = gmm.predict(train_pca_df)

# 쌓기
train_scaled_df['GMM_cluster'] = gmm_cluster_labels


# In[ ]:





# ### tsne 차원축소

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# - Silhouette

# In[29]:


# 모든 개별 데이터 실루엣 계수 값 구하기
score_samples = silhouette_samples(train_pca_norm, train_scaled_df["GMM_cluster"])
print("silhouette_samples() return 값의 shape", score_samples.shape)


# In[30]:


# train_scaled_df에 실루엣 계수 컬럼 추가
train_scaled_df["silhouette_coeff"] = score_samples


# In[31]:


# 모든 데이터의 평균 실루엣 계수 값 구하기
average_score = silhouette_score(train_pca_norm, train_scaled_df["GMM_cluster"])
print("개인 플레이 성향 Silhouette Analysis Score: {0:3f}".format(average_score))


# In[ ]:





# In[ ]:





# In[ ]:




