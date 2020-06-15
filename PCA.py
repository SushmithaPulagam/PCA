#!/usr/bin/env python
# coding: utf-8

# ### Reading the dataset

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("G:\\Pizza.csv")


# In[3]:


df.head()


# In[4]:


df = df.drop(['Brand'],axis = 1)


# In[5]:


df.head()


# ### Stadardizing the data

# In[6]:


from sklearn.preprocessing import StandardScaler
df_std = StandardScaler().fit_transform(df)
df_std


# ### Calculating covariance matrix

# In[7]:


df_cov_matrix = np.cov(df_std.T)
df_cov_matrix


# ### Calculating Eigendecomposition

# In[8]:


eig_vals, eig_vecs = np.linalg.eig(df_cov_matrix)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


# ### Sorting Eigenvalues

# In[9]:


# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])


# ### Calculating cumulative variance to select number of components

# In[11]:


total = sum(eig_vals)
var_exp = [(i / total)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("Variance captured by each component is \n",var_exp)
print("Cumulative variance captured as we travel with each component \n",cum_var_exp)


# In[12]:


df1 = pd.read_csv("G:\\Pizza.csv")


# ### Scree plot for visualization

# In[14]:


from sklearn.decomposition import PCA
pca = PCA().fit(df_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('No of components')
plt.ylabel('Cumulative explained variance')
plt.show()


# ### Creating 3 Principal components

# In[15]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pcs = pca.fit_transform(df_std)
df_new = pd.DataFrame(data=pcs, columns={'PC1','PC2','PC3'})
df_new['target'] = df1['Brand'] 
df_new.head()


# In[ ]:




