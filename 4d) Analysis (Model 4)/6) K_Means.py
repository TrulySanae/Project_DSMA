import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import shutil

# Specify the name or path of the folder you want to create
folder_name = '6b) K_Means_Results'

# Remove the folder if it exists and create a new one
if os.path.exists(folder_name):
    shutil.rmtree(folder_name)  # Removes the existing folder and its contents
    print(f"Folder '{folder_name}' has been removed.")

os.makedirs(folder_name)  # Create the folder
print(f"Folder '{folder_name}' has been created.")

# Read the data from the CSV file
df = pd.read_csv('5b) Sum_Revenue.csv')


# Define the columns for clustering
continuous_vars = ['total_revenue', 'week_hours', 'rating_stars', 'rating_review_count', 'rating_popularity', 'rating']
ordinal_vars = ['social_media']
binary_vars = [
    'ambiance_intimate', 'ambiance_touristy', 'ambiance_hipster', 'ambiance_divey', 'ambiance_classy',
    'ambiance_upscale', 'ambiance_casual', 'ambiance_trendy', 'ambiance_romantic', 'meals_breakfast',
    'meals_brunch', 'meals_lunch', 'meals_dinner', 'meals_dessert', 'attr_parking', 'attr_credit_cards',
    'attr_outdoor_seating', 'attr_tv', 'reservations', 'service_good_for_kids', 'service_good_for_groups',
    'service_caters',
    'collect_takeout', 'collect_delivery', 'alcohol',
    'wifi',
    'service_table_service',
    'meals_latenight',
]

# Scale continuous variables
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[continuous_vars])

# Perform PCA to reduce dimensionality
pca = PCA(n_components=0.95)  # Retain 95% variance
df_pca = pca.fit_transform(df_scaled)
print(f"PCA reduced the data to {df_pca.shape[1]} dimensions.")

# Elbow Method with PCA data
inertia = []
k_range = range(1, 15)

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    kmeans.fit(df_pca)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_range)
plt.grid(True)
plt.savefig(f'{folder_name}/elbow_plot_with_pca.png')
plt.close()

# Silhouette Scores to validate clusters
silhouette_scores = []

for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    labels = kmeans.fit_predict(df_pca)
    silhouette_scores.append(silhouette_score(df_pca, labels))

plt.figure(figsize=(8, 6))
plt.plot(range(2, 15), silhouette_scores, marker='o', label='Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Method For Optimal k')
plt.legend()
plt.savefig(f'{folder_name}/silhouette_plot.png')
plt.close()

# Choose the optimal number of clusters based on elbow and silhouette methods
optimal_k = 5  # Update this manually after inspecting the plots
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', n_init=20, random_state=42)
labels = kmeans.fit_predict(df_pca)
df['Cluster'] = labels

# Cluster Summary
all_vars = continuous_vars + ordinal_vars + binary_vars
cluster_means = df.groupby('Cluster')[continuous_vars + ordinal_vars].mean()
binary_vars_means = df.groupby('Cluster')[binary_vars].mean()
cluster_summary = pd.concat([cluster_means, binary_vars_means], axis=1).round(4)
output_file = f"{folder_name}/kmeans_cluster_summary.csv"
cluster_summary.to_csv(output_file)
print(f"Cluster summary has been saved to {output_file}")

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(df_pca)
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels, cmap='viridis', s=50)
plt.colorbar(label='Cluster')
plt.title('t-SNE Clustering Visualization')
plt.savefig(f'{folder_name}/tsne_plot.png')
plt.close()

# Correlation Heatmap for Feature Selection
corr_matrix = df[continuous_vars + ordinal_vars].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', cbar=True)
plt.title('Correlation Heatmap')
plt.savefig(f'{folder_name}/correlation_heatmap.png')
plt.close()
