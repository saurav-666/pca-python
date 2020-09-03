import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


data = pd.read_csv('PCA_practice_dataset.csv')

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

#creation of PCA object
pca = PCA()
pca_data = pca.fit_transform(data)

cum_var = pca.explained_variance_ratio_.cumsum()*100

threshold = [i for i in range(90,98)]

components = [np.argmax(cum_var>t) for t in threshold]

for c,t in zip(components,threshold):
    print("Threshold: {}% , Components: {}".format(t,c))
    
#plotting the dataset    
plt.plot(components,range(90,98),'ro-')
plt.xlabel("Principal components")
plt.ylabel("Threshold in percentage")
plt.show()

#dimensionality reduction
for c,t in zip(components,threshold):
    pca = PCA(n_components = c)
    transformed_data = pca.fit_transform(pca_data)
    print("Dimensionality reduction at {}% threshold".format(t))
    print("Shape of reduced dataset :",transformed_data.shape)
    