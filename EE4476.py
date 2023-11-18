from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# load data
mnist = fetch_openml('mnist_784', data_home='./data', cache=True)
X, y = mnist['data'], mnist['target']

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)

# Convert DataFrame to NumPy array
X_train_array = X_train.to_numpy()

# Reshape the data from 784 to 28x28
images = X_train_array.reshape((-1, 28, 28))

# Create a 5x5 grid of subplots
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(8,8))

# Display the train_images
for i in range(5):
    for j in range(5):
        axes[i,j].imshow(images[np.random.randint(0, len(images)-1)], cmap='gray')
        axes[i,j].axis('off')
plt.savefig('Train.png')


# 2 dimension LDA
lda_2d = LDA(n_components=2)
# Data after LDA
X_train_lda_2d = lda_2d.fit_transform(X_train, y_train)
# Plot the data
plt.clf()
plt.figure(figsize=(30,30))
plt.scatter(
    X_train_lda_2d[:,0],
    X_train_lda_2d[:,1],
    c=y_train.cat.codes,
    cmap='rainbow',
    alpha=0.7,
    s=5
)
plt.savefig('Train_LDA_2d.png')
# predict
y_pred = lda_2d.predict(X_test)
# Accuracy
accuracy = sum(y_pred == y_test) / len(y_test)
print("2D Test set accuracy: ", accuracy)
print(lda_2d.explained_variance_ratio_)
# save model
joblib.dump(lda_2d, 'lda_2d_model.pkl')


# 3 dimension LDA
lda_3d = LDA(n_components=3)
# Data after LDA
X_train_lda_3d = lda_3d.fit_transform(X_train, y_train)
# Plot the data
plt.clf()
fig=plt.figure(figsize=(30,30))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_train_lda_3d[:,0],
    X_train_lda_3d[:,1],
    X_train_lda_3d[:,2],
    c=y_train.cat.codes,
    cmap='rainbow',
    alpha=0.7,
    s=5
)
plt.savefig('Train_LDA_3d.png')
# predict
y_pred = lda_3d.predict(X_test)
# Accuracy
accuracy = sum(y_pred == y_test) / len(y_test)
print("3D Test set accuracy: ", accuracy)
print(lda_3d.explained_variance_ratio_)
# save model
joblib.dump(lda_3d, 'lda_3d_model.pkl')



# 4 dimension LDA
lda_4d = LDA(n_components=4)
# Data after LDA
X_train_lda_4d = lda_4d.fit_transform(X_train, y_train)
# Plot the data
plt.clf()
fig=plt.figure(figsize=(30,30))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    X_train_lda_4d[:,0],
    X_train_lda_4d[:,1],
    X_train_lda_4d[:,3],
    c=y_train.cat.codes,
    cmap='rainbow',
    alpha=0.7,
    s=5
)
plt.savefig('Train_LDA_4d.png')
# predict
y_pred = lda_4d.predict(X_test)
# Accuracy
accuracy = sum(y_pred == y_test) / len(y_test)
print("4D Test set accuracy: ", accuracy)
print(lda_4d.explained_variance_ratio_)
# save model
joblib.dump(lda_4d, 'lda_4d_model.pkl')

