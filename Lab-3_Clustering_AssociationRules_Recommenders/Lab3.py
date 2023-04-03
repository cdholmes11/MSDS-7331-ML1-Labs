#!/usr/bin/env python
# coding: utf-8

# __Title:__ Lab 3: Clustering, Association Rules, or Recommenders  
# __Authors:__ Butler, Derner, Holmes, Traxler  
# __Date:__ 4/9/23

# ## Ruberic

# | Category                  | Available | Requirements                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
# |---------------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
# | Total Points              | 100       |                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
# | Business Understanding 1  | 10        | Describe the purpose of the data set you selected (i.e., why was this data collected in the first place?). How will you measure the effectiveness of a good algorithm? Why does your chosen validation method make sense for this specific dataset and the stakeholders needs?                                                                                                                                                                                     |
# | Data Understanding 1      | 10        | Describe the meaning and type of data (scale, values, etc.) for each attribute in the data file. Verify data quality: Are there missing values? Duplicate data? Outliers? Are those mistakes? How do you deal with these problems?                                                                                                                                                                                                                                 |
# | Data Understanding 2      | 10        | Visualize the any important attributes appropriately. Important: Provide an interpretation for any charts or graphs.                                                                                                                                                                                                                                                                                                                                               |
# | Modeling and Evaluation 1 | 10        | Train and adjust parameters                                                                                                                                                                                                                                                                                                                                                                                                                                        |
# | Modeling and Evaluation 2 | 10        | Evaluate and Compare                                                                                                                                                                                                                                                                                                                                                                                                                                               |
# | Modeling and Evaluation 3 | 10        | Visualize Results                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
# | Modeling and Evaluation 4 | 20        | Summarize the Ramifications                                                                                                                                                                                                                                                                                                                                                                                                                                        |
# | Deployment                | 10        | Be critical of your performance and tell the reader how you current model might be usable by other parties. Did you achieve your goals? If not, can you reign in the utility of your modeling? How useful is your model for interested parties (i.e., the companies or organizations that might want to use it)? How would your deploy your model for interested parties? What other data should be collected? How often would the model need to be updated, etc.? |
# | Exceptional Work          | 10        | You have free reign to provide additional analyses or combine analyses.                                                                                                                                                                                                                                                                                                                                                                                            |

# 

# In[1]:


# Import libraries
## Support Libraries
import pandas as pd
import numpy as np
import warnings

## Plotting
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as sp
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

## Preprocessing
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

## Model Selection
from sklearn.model_selection import train_test_split, GridSearchCV

## Models
from sklearn.svm import SVC, SVR
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

## Feature Selection
from sklearn.feature_selection import SelectFromModel, VarianceThreshold, SelectPercentile

## Model Performance
from sklearn.metrics import classification_report, roc_curve, auc, mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics.cluster import contingency_matrix
from sklearn.inspection import permutation_importance

# Notebook Settings
warnings.filterwarnings(action='once')
pd.set_option('display.max_columns', None)


# In[2]:


# Dataset
url = 'https://github.com/cdholmes11/MSDS-7331-ML1-Labs/blob/main/Mini-Lab_LogisticRegression_SVMs/Hotel%20Reservations.csv?raw=true'
hotel_df = pd.read_csv(url, encoding = "utf-8")


# In[3]:


# Dropping index column arrival_year
hotel_df_trim = hotel_df.drop(['Booking_ID', 'arrival_year', 'no_of_previous_bookings_not_canceled'], axis=1)
hotel_df_final = hotel_df_trim.loc[hotel_df_trim['avg_price_per_room'] < 400]

# Create data type groups
cat_features = ['type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved', 'market_segment_type',
    'repeated_guest', 'booking_status']
int_features = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'arrival_month',
    'arrival_date', 'no_of_previous_cancellations', 'no_of_special_requests']
float_features = ['lead_time', 'avg_price_per_room']
cont_features = int_features + float_features

# Enforce data types
hotel_df_trim[cat_features] = hotel_df_trim[cat_features].astype('category')
hotel_df_trim[int_features] = hotel_df_trim[int_features].astype(np.int64)
hotel_df_trim[float_features] = hotel_df_trim[float_features].astype(np.float64)

# Making indexable list suitable for pipeline
cat_features_final = list(hotel_df_final[cat_features].columns)
cont_features_final = list(hotel_df_final[cont_features].columns)


# In[4]:


# Pipeline - Classification
numeric_features = cont_features_final
numeric_transformer = Pipeline(
    steps=[("scaler", StandardScaler())]
)

categorical_features = cat_features_final
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# In[5]:


kmeans_pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('pca', PCA(n_components=2)),
    ('clustering', KMeans(n_clusters=2))
    ]
)

kmeans_pipe.fit(hotel_df_trim)
cluster_labels = kmeans_pipe.predict(hotel_df_trim)

hotel_df_trim['cluster_labels'] = cluster_labels


# In[6]:


fig = px.scatter(
    hotel_df_trim,
    y = 'avg_price_per_room',
    color='cluster_labels',
    width= 1000,
    height= 600
    )
fig.show()


# In[7]:


print(hotel_df)


# In[9]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
url = 'https://github.com/cdholmes11/MSDS-7331-ML1-Labs/blob/main/Mini-Lab_LogisticRegression_SVMs/Hotel%20Reservations.csv?raw=true'
dfrd = pd.read_csv(url)

# Extract the "length_of_stay" variable
X = dfrd[["no_of_week_nights"]]

# Standardize the variable
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 11), inertia)
plt.title("Elbow Plot")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

# Perform K-Means clustering with K=3
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)

# Add the cluster labels to the original dataframe
dfrd["cluster"] = kmeans.labels_

# Display the cluster means
print(dfrd.groupby("cluster")["no_of_week_nights"].mean())


# In[11]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd

# Load data
url = 'https://github.com/cdholmes11/MSDS-7331-ML1-Labs/blob/main/Mini-Lab_LogisticRegression_SVMs/Hotel%20Reservations.csv?raw=true'
df = pd.read_csv(url)

# Preprocessing
df = df[df['booking_status'] == 'Not_Canceled']  # only keep non-canceled bookings
X = df[['lead_time']].values  # cluster on lead time

# Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Fit model
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X)

# Visualize clustering results
plt.scatter(X[y_pred == 0, 0], [0]*sum(y_pred == 0), s=50, c='red', label='Cluster 1')
plt.scatter(X[y_pred == 1, 0], [0]*sum(y_pred == 1), s=50, c='blue', label='Cluster 2')
plt.scatter(X[y_pred == 2, 0], [0]*sum(y_pred == 2), s=50, c='green', label='Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], [0]*len(kmeans.cluster_centers_), s=200, c='black', label='Centroids')
plt.title('Lead Time Clusters')
plt.xlabel('Lead Time')
plt.legend()
plt.show()

# Silhouette score
silhouette_avg = silhouette_score(X, y_pred)
print("The average silhouette_score is :", silhouette_avg)

# Calinski-Harabasz Index
ch_score = calinski_harabasz_score(X, y_pred)
print("Calinski-Harabasz Index is:", ch_score)

# Davies-Bouldin Index
db_score = davies_bouldin_score(X, y_pred)
print("Davies-Bouldin Index is:", db_score)


# In[12]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd

# Load data
url = 'https://github.com/cdholmes11/MSDS-7331-ML1-Labs/blob/main/Mini-Lab_LogisticRegression_SVMs/Hotel%20Reservations.csv?raw=true'
df = pd.read_csv(url)

# Preprocessing
df = df[df['booking_status'] == 'Not_Canceled']  # only keep non-canceled bookings
X = df[['lead_time']].values  # cluster on lead time

# Optimize number of clusters
scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    y_pred = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, y_pred)
    scores.append(silhouette_avg)
    
# Plot silhouette scores
plt.plot(range(2, 11), scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# Fit model with optimized number of clusters
n_clusters = 4  # choose number of clusters based on silhouette score plot
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
y_pred = kmeans.fit_predict(X)

# Visualize clustering results
plt.scatter(X[y_pred == 0, 0], [0]*sum(y_pred == 0), s=50, c='red', label='Cluster 1')
plt.scatter(X[y_pred == 1, 0], [0]*sum(y_pred == 1), s=50, c='blue', label='Cluster 2')
plt.scatter(X[y_pred == 2, 0], [0]*sum(y_pred == 2), s=50, c='green', label='Cluster 3')
plt.scatter(X[y_pred == 3, 0], [0]*sum(y_pred == 3), s=50, c='orange', label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], [0]*len(kmeans.cluster_centers_), s=200, c='black', label='Centroids')
plt.title('Lead Time Clusters')
plt.xlabel('Lead Time')
plt.legend()
plt.show()


# Silhouette score
silhouette_avg = silhouette_score(X, y_pred)
print("The average silhouette_score is :", silhouette_avg)

# Calinski-Harabasz Index
ch_score = calinski_harabasz_score(X, y_pred)
print("Calinski-Harabasz Index is:", ch_score)

# Davies-Bouldin Index
db_score = davies_bouldin_score(X, y_pred)
print("Davies-Bouldin Index is:", db_score)


# For the silhouette score, a higher score (closer to 1) indicates better-defined clusters and a better overall clustering performance.
# 
# For the CH score, a higher score indicates that the clusters are more compact and well-separated, indicating a better clustering performance.
# 
# The DBI score ranges from 0 to infinity, with lower values indicating better clustering. A score of 0 indicates perfect clustering, where each cluster is well-separated and compact, while higher scores indicate that the clusters are overlapping or poorly separated.
# 
# Like the other evaluation metrics, a lower DBI score indicates better clustering. Therefore, a good clustering model should have a low DBI score, indicating that the clusters are well-separated and distinct.
# 
# It's important to note that no single metric can fully capture the quality of clustering, so it's always a good idea to use multiple evaluation metrics to get a more comprehensive view of the clustering performance.

# In[ ]:




