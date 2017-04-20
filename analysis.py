import pickle
from collections import namedtuple

import matplotlib.pyplot as plt
import spotipy

TrackInfo = namedtuple('TrackInfo', ['id', 'track_href', 'danceability', 'energy', 'loudness', 'speechiness',
                                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])

spotify = spotipy.Spotify()


def top_n_of_category(db, category, n):
    top = sorted(db, key=lambda x: x._asdict()[category] or 0, reverse=True)[:n]

    for entry in top:
        track = spotify.track(entry.id)
        track_url = track['external_urls']['spotify']

        print('{0} with {1} {2}\n{3}'.format(track['name'], category, entry._asdict()[category], track_url))


def perform_analysis(database):
    top_n_of_category(database, 'energy', 10)


def plot_category(database, category):
    plt.hist([x._asdict()[category] or 0 for x in database], 100)
    plt.show()


if __name__ == '__main__':
    with open('tracks.db', 'rb') as file:
        database = pickle.load(file)

    plot_category(database, 'liveness')
