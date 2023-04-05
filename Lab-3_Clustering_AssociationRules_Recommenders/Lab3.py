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

# In[2]:


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


# In[3]:


# Dataset
url = 'https://github.com/cdholmes11/MSDS-7331-ML1-Labs/blob/main/Mini-Lab_LogisticRegression_SVMs/Hotel%20Reservations.csv?raw=true'
hotel_df = pd.read_csv(url, encoding = "utf-8")


# In[4]:


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


# In[5]:


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


# In[6]:


kmeans_pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('pca', PCA(n_components=2)),
    ('clustering', KMeans(n_clusters=2))
    ]
)

kmeans_pipe.fit(hotel_df_trim)
cluster_labels = kmeans_pipe.predict(hotel_df_trim)

hotel_df_trim['cluster_labels'] = cluster_labels


# In[7]:


fig = px.scatter(
    hotel_df_trim,
    y = 'avg_price_per_room',
    color='cluster_labels',
    width= 1000,
    height= 600
    )
fig.show()


# # Exceptional Work

# In the case of our Hotel Reservations dataset, we are interested in segmenting customers based on their booking behavior, this lead us to cluster based on lead time (the time between booking and arrival). We wanted to see if certain types of customers book well in advanced vs last minute. 
# 
# Since examining only customers who booked a hotel room we will remove cancelled bookings since they don't represent completed bookings. We'll standarize the data using 'StandardScaler()' so that the lead time values are on the same scale as the booking status values and are easily plottable. 
# 
# For evaluating the model we initially decided to use a visual method and decided on plotting an elbow chart to determine the optimal number of clusters. Fitting K-Means clustering models with different numbers of clusters and plot the within-cluster sum of squares (WCSS) for each model. We choose the number of clusters where the decrease in WCSS begins to level off, which in this case appears to be 3.

# In[9]:


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import pandas as pd


# Preprocessing
df = hotel_df_final[hotel_df_final['booking_status'] == 'Not_Canceled']  # only keep non-canceled bookings
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


# Examming the fit of the K-Means clustering model with 3 clusters and visualizing the clusters by plotting the lead time values against the booking status values, the resulting plot shows that there are three distinct groups of customers based on their lead time and booking status. Since we have only 3 clusters, the Silhouette score may not be very informative. If we had more clusters, the Silhouette score could be more useful for evaluating the quality of the clustering results.

# ### Optimizing the model

# In[10]:


# Preprocessing
df = hotel_df_final[hotel_df_final['booking_status'] == 'Not_Canceled']  # only keep non-canceled bookings
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


# Optimizing the number of clusters by computing the silhouette score for the different numbers of clusters (2-10 in the above case) and choosing the number of clusters that gives the highest score. We are able to fit the KMeans model with the optimized number of clusters and visualize the clustering results. 
# 
# We can observe and provide inferences based on the below clusters:
# 
# Cluster 1: This cluster represents bookings with the shortest lead time. These bookings were likely made last-minute, perhaps due to a sudden change in plans or a desire to take advantage of a last-minute deal. These customers may be price-sensitive and willing to take a risk on availability.
# 
# Cluster 2: This cluster represents bookings with a relatively short lead time, but not as short as Cluster 1. These customers may be slightly more planned, but still looking for a deal or to take advantage of a special offer.
# 
# Cluster 3: This cluster represents bookings with a moderate lead time. These customers are likely more planned and have a specific trip or event in mind. They may be willing to pay a bit more for convenience or amenities.
# 
# Cluster 4: This cluster represents bookings with a longer lead time. These customers are likely very planned and have a specific destination and/or dates in mind. They may be willing to pay a premium for a desirable location or property.
# 
# Cluster 5: This cluster represents bookings with the longest lead time. These customers are likely the most planned and have a specific destination and/or dates in mind well in advance. They may be looking for luxury accommodations or unique experiences and are likely less price-sensitive than customers in other clusters.
# 
# 
# Based on the metrics used to evaluate the clustering models, they appear to be equal. The optimized model has a slight edge in comparsion even with a lower silhouette score (0.61 compared to 0.64) and a slightly better Davies-Bouldin Index (.49 compared to .50), indicating slightly better separation between the clusters and lower overlap between them. This becomes more apprent upon examining the Calinski-Harabasz Index (87355.41 compared to original model score of 67207.82). The customer segmentation is better represented as the clusters are more compact and well separated in the optimized model vs the original model. 

# In[ ]:




