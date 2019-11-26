from time import time

from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.objectives import binary_crossentropy
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC

import numpy as np

import matplotlib.pyplot as plt
import os

# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae"):

    encoder, decoder = models
    x_test, y_test = data

    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

x = lfw_people.data
y = lfw_people.target

x_train_orig, x_test_orig, y_train_orig, y_test_orig = train_test_split(x, y, test_size=0.25, random_state=42)
n_samples, h, w = lfw_people.images.shape

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

original_dim = h * w

idx = np.arange(x_train_orig.shape[0])
np.random.shuffle(idx)
x_train = np.zeros(shape=(x_train_orig.shape[0], x_train_orig.shape[1]))
y_train = np.zeros(shape=(x_train_orig.shape[0],))
for i in range(0, x_train_orig.shape[0]):
    x_train[i] = x_train_orig[ idx[i] ]
    y_train[i] = y_train_orig[ idx[i] ]

idx = np.arange(x_test_orig.shape[0])
np.random.shuffle(idx)
x_test = np.zeros(shape=(x_test_orig.shape[0], x_test_orig.shape[1]))
y_test = np.zeros(shape=(x_test_orig.shape[0],))
for i in range(0, x_test_orig.shape[0]):
    x_test[i] = x_test_orig[ idx[i] ]
    y_test[i] = y_test_orig[ idx[i] ]

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

intermediate_dim = 512
batch_size = 100
latent_dim = 150
epochs = 50000000

input = Input(shape=(original_dim,))
#h = Dense(intermediate_dim, activation='relu')(input)
#z_mean = Dense(latent_dim)(h)
#z_log_sigma = Dense(latent_dim)(h)
#z = Lambda(sampling, output_shape=(latent_dim, ), name='z')([z_mean, z_log_sigma])
latent_space = Dense(latent_dim, activation='relu')(input)
# encoder, from inputs to latent space
encoder = Model(input, latent_space, name="encoder")
encoder.summary()

#encoded_input = Input(shape=(latent_dim,))
#decoder_h = Dense(intermediate_dim, activation='relu')(encoded_input)
#decoder_mean = Dense(original_dim, activation='sigmoid')(decoder_h)
output = Dense(original_dim, activation='sigmoid')(latent_space)

# create the decoder model
#decoder = Model(latent_space, output, name="decoder")
#decoder.summary()

def vae_loss(x, x_decoded_mean):
    xent_loss = binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return xent_loss + kl_loss

# end-to-end autoencoder
#output = decoder(encoder(input)[2])

vae = Model(input, output, name="vae")
vae.compile(optimizer='Adam', loss='binary_crossentropy', metrics=["accuracy"])
vae.summary()

print("Training the autoencoder to the training set")
vae.fit(x_train, x_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, x_test))


x_train_latentspace = encoder.predict(x_train, batch_size)
x_test_latentspace = encoder.predict(x_test, batch_size)
print(x_train_latentspace.shape)

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1, 5, 10, 1e2, 5e2, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.5], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5, iid=False)
clf = clf.fit(x_train_latentspace, y_train)

print("done in %0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)

print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(x_test_latentspace)
print("done in %0.3fs" % (time() - t0))

print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
