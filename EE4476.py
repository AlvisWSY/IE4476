from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import normalize
import joblib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd

MNIST=True
POKEMON=True

if MNIST:
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
    plt.savefig('MNIST_Train.png')



    # define LDA
    MNIST_lda_2d = LDA(n_components=2)
    MNIST_lda_3d = LDA(n_components=3)
    MNIST_lda_4d = LDA(n_components=4)



    # 2 dimension LDA
    # Data after LDA
    X_train_lda_2d = MNIST_lda_2d.fit_transform(X_train, y_train)
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
    plt.savefig('MNIST_Train_LDA_2d.png')
    # predict
    y_pred = MNIST_lda_2d.predict(X_test)
    # Accuracy
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("MNIST_2D Test set accuracy: ", accuracy)
    print(MNIST_lda_2d.explained_variance_ratio_)
    # save model
    joblib.dump(MNIST_lda_2d, 'MNIST_lda_2d_model.pkl')



    # 3 dimension LDA
    # Data after LDA
    X_train_lda_3d = MNIST_lda_3d.fit_transform(X_train, y_train)
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
    plt.savefig('MNIST_Train_LDA_3d.png')
    # predict
    y_pred = MNIST_lda_3d.predict(X_test)
    # Accuracy
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("MNIST_3D Test set accuracy: ", accuracy)
    print(MNIST_lda_3d.explained_variance_ratio_)
    # save model
    joblib.dump(MNIST_lda_3d, 'MNIST_lda_3d_model.pkl')



    # 4 dimension LDA
    # Data after LDA
    X_train_lda_4d = MNIST_lda_4d.fit_transform(X_train, y_train)
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
    plt.savefig('MNIST_Train_LDA_4d.png')
    # predict
    y_pred = MNIST_lda_4d.predict(X_test)
    # Accuracy
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("4D Test set accuracy: ", accuracy)
    print(MNIST_lda_4d.explained_variance_ratio_)
    # save model
    joblib.dump(MNIST_lda_4d, 'MNIST_lda_4d_model.pkl')

if POKEMON:
    # Import Pokemon Data
    pokemon = pd.read_csv('data/pokemon.csv')

    # Clean Pokemon Data (only keep Pokemons with single type)
    df = pokemon[pokemon['type2'].isnull()].loc[
        :, ['sp_attack', 'sp_defense', 'attack', 'defense', 'speed', 'hp', 'type1']
    ]

    # Make Data and Labels
    X = df.iloc[:, :-1].values
    X=normalize(X)
    y = df.iloc[:, -1].values

    # Split Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13)
    y_train_num = pd.factorize(y_train)[0]
    print(y_train_num)

    # define LDA
    pokemon_lda_1d = LDA(n_components=1)
    pokemon_lda_2d = LDA(n_components=2)
    pokemon_lda_3d = LDA(n_components=3)

    # 1 dimension LDA
    # Data after LDA
    X_train_lda_1d = pokemon_lda_1d.fit_transform(X_train, y_train)
    # Plot the data
    plt.clf()
    plt.figure(figsize=(10,10))
    plt.scatter(
        X_train_lda_1d[:,0],
        np.zeros(len(X_train_lda_1d)),
        c=y_train_num,
        cmap='rainbow',
        alpha=0.7,
        edgecolors='b'
    )
    plt.savefig('Pokemon_Train_LDA_1D.png')
    # predict
    y_pred = pokemon_lda_1d.predict(X_test)
    # Accuracy
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("Pokemon_1D Test set accuracy: ", accuracy)
    print(pokemon_lda_1d.explained_variance_ratio_)
    # save model
    joblib.dump(pokemon_lda_1d, 'Pokemon_lda_1d_model.pkl')


    # 2 dimension LDA
    # Data after LDA
    X_train_lda_2d = pokemon_lda_2d.fit_transform(X_train, y_train)
    # Plot the data
    plt.clf()
    plt.figure(figsize=(10,10))
    plt.scatter(
        X_train_lda_2d[:,0],
        X_train_lda_2d[:,1],
        c=y_train_num,
        cmap='rainbow',
        alpha=0.7,
        edgecolors='b'
    )
    plt.savefig('Pokemon_Train_LDA_2D.png')
    # predict
    y_pred = pokemon_lda_2d.predict(X_test)
    # Accuracy
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("Pokemon_2D Test set accuracy: ", accuracy)
    print(pokemon_lda_2d.explained_variance_ratio_)
    # save model
    joblib.dump(pokemon_lda_2d, 'Pokemon_lda_2d_model.pkl')


    # 3 dimension LDA
    # Data after LDA
    X_train_lda_3d = pokemon_lda_3d.fit_transform(X_train, y_train)
    # Plot the data
    plt.clf()
    fig=plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        X_train_lda_3d[:,0],
        X_train_lda_3d[:,1],
        X_train_lda_3d[:,2],
        c=y_train_num,
        cmap='rainbow',
        alpha=0.7,
        edgecolors='b'
    )
    plt.savefig('Pokemon_Train_LDA_3D.png')
    # predict
    y_pred = pokemon_lda_3d.predict(X_test)
    # Accuracy
    accuracy = sum(y_pred == y_test) / len(y_test)
    print("Pokemon_3D Test set accuracy: ", accuracy)
    print(pokemon_lda_3d.explained_variance_ratio_)
    # save model
    joblib.dump(pokemon_lda_3d, 'Pokemon_lda_3d_model.pkl')