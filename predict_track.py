from collections import namedtuple
from random import choice

import os

import scipy.spatial.distance
import spotipy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pickle

TrackInfo = namedtuple('TrackInfo', ['id', 'track_href', 'danceability', 'energy', 'loudness', 'speechiness',
                                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])

PARAMETERS_DNN = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence']


train = True
training_epochs = 40000
hidden_layer_size = 150
display_times = 10
display_step = 2000
evaluation_step = 500
batch_size = 50


def load_data():
    with open('tracks.db', 'rb') as file:
        tracks = pickle.load(file)

    with open('playlists.db', 'rb') as file:
        playlists = pickle.load(file)

    return tracks, playlists


def generate_training_data(playlists, songs):
    x = []
    y = []

    for playlist in filter(lambda pl: len(pl) > 1, playlists.values()):
        for i in range(len(playlist) - 1):
            id = playlist[i]

            song = [songs[id]._asdict()[category] for category in PARAMETERS_DNN]
            next_song = [songs[id]._asdict()[category] for category in PARAMETERS_DNN]

            x.append(song)
            y.append(next_song)

    return np.array(x), np.array(y)


def get_closest_song(prediction, songs):
    nodes = np.asarray(songs)
    dist_2 = scipy.spatial.distance.cdist(np.array([prediction]), nodes, 'euclidean')
    return np.argmin(dist_2), np.min(dist_2)


input = tf.placeholder(tf.float32, shape=(None, len(PARAMETERS_DNN)))
correct_output = tf.placeholder(tf.float32, shape=(None, len(PARAMETERS_DNN)))

weight_1 = tf.Variable(tf.random_normal(shape=(len(PARAMETERS_DNN), hidden_layer_size)))
bias_1 = tf.Variable(tf.random_normal(shape=(1, hidden_layer_size)))

layer1 = tf.matmul(input, weight_1) + bias_1

h_weight_2 = tf.Variable(tf.random_normal(shape=(hidden_layer_size, hidden_layer_size)))
h_bias_2 = tf.Variable(tf.random_normal(shape=(1, hidden_layer_size)))

layer2 = tf.matmul(layer1, h_weight_2) + h_bias_2

h_weight_3 = tf.Variable(tf.random_normal(shape=(hidden_layer_size, len(PARAMETERS_DNN))))
h_bias_3 = tf.Variable(tf.random_normal(shape=(1, len(PARAMETERS_DNN))))

layer3 = tf.matmul(layer2, h_weight_3) + h_bias_3

loss = tf.div(tf.reduce_sum(tf.nn.l2_loss(layer3 - correct_output)), batch_size)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.9999).minimize(loss)

init = tf.global_variables_initializer()
tracks, playlists = load_data()
train_x, train_y = generate_training_data(playlists, tracks)

with tf.Session() as sess:
    saver = tf.train.Saver()
    best_loss = float('inf')

    sess.run(init)

    if train:
        datapoints = []

        for epoch in range(training_epochs):
            random_ids = np.random.random_integers(0, len(train_x)-1, 50)

            training_input = train_x[random_ids]
            training_output = train_y[random_ids]

            sess.run(optimizer, feed_dict={input: training_input, correct_output: training_output})

            if epoch % evaluation_step == 0:
                cost = sess.run(loss, feed_dict={input: train_x, correct_output: train_y})
                datapoints.append(cost)

                if cost < best_loss:
                    saver.save(sess, os.getcwd() + '/best_model.ckpt')
                    best_loss = cost

            if epoch % display_step == 0:
                cost = sess.run(loss, feed_dict={input: train_x, correct_output: train_y})

                print('Epoch: {0}, loss: {1:.9f}'.format(epoch, cost))

        print('Optimization finished! Best loss: {}'.format(best_loss))

        plt.semilogy(datapoints)
        plt.show()

    spotify = spotipy.Spotify()
    saver.restore(sess, os.getcwd() + '/best_model.ckpt')
    tracks_modified = [[track._asdict()[category] for category in PARAMETERS_DNN] for track in tracks.values()]

    songs = choice(list(playlists.values()))
    previous_song = tracks[choice(songs)]
    print('Starting to create playlist out of {}'.format(spotify.track(previous_song.id)['name']))

    previous_songs = []

    for i in range(display_times):
        network_input = [previous_song._asdict()[category] for category in PARAMETERS_DNN]
        previous_songs.append(network_input)

        pred = sess.run(layer3, feed_dict={input: [network_input]})
        predicted_song, cost = get_closest_song(np.squeeze(pred), [x for x in tracks_modified if x not in previous_songs])

        track_key = list(tracks.keys())[np.asscalar(predicted_song)]

        predicted = spotify.track(track_key)
        previous_song = tracks[track_key]

        print('{}: {} (loss {:.9}) - {}'.format(i+1, predicted['name'], cost, predicted['external_urls']['spotify']))
