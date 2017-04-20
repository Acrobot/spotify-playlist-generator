from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import spotipy
import spotipy.util as util
import pickle

PlaylistInfo = namedtuple('PlaylistInfo', ['id', 'owner_id'])
TrackInfo = namedtuple('TrackInfo', ['id', 'track_href', 'danceability', 'energy', 'loudness', 'speechiness',
                                     'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo'])


def chunks(input, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(input), n):
        yield input[i:i + n]


def get_tracks_from_playlist(playlist: PlaylistInfo):
    tracks = []
    track_items = spotify.user_playlist_tracks(user=playlist.owner_id,
                                               playlist_id=playlist.id,
                                               fields='items.track.id')['items']

    for item in track_items:
        id = item['track']['id']

        if id is not None:
            tracks.append(item['track']['id'])

    return playlist.id, tracks


def get_metadata_from_tracks(ids):
    track_data = spotify.audio_features(ids)
    infos = {}

    for track in track_data:
        if track is None or [track[field] for field in TrackInfo._fields if track[field] is None]:
            continue

        info = TrackInfo(*(track[field] for field in TrackInfo._fields))
        infos[info.id] = info

    return infos

USER_NAME = 'your_user_name'
token = util.prompt_for_user_token(username=USER_NAME,
                                   scope='playlist-read-private user-library-read',
                                   client_id='your_client_id',
                                   client_secret='your_secret_id',
                                   redirect_uri='https://github.com/plamere/spotipy/blob/master/redirect_page.md')
spotify = spotipy.Spotify(auth=token)

playlists = []
offset = 0

while True:
    result = spotify.current_user_playlists(offset=offset)
    playlists.extend([PlaylistInfo(item['id'], item['owner']['id'])
                 for item in result['items']])

    if result['next']:
        offset += 50
    else:
        break

pool = ThreadPoolExecutor(1)
futures = []

for playlist in playlists:
    futures.append(pool.submit(get_tracks_from_playlist, playlist))

playlist_tracks = {}
completed = 0

for x in as_completed(futures):
    result = x.result()
    completed += 1

    print('Finished request for {0} ({1} / {2})'.format(result[0], completed, len(playlists)))
    playlist_tracks[result[0]] = result[1]

tracks = []
for track_list in playlist_tracks.values():
    tracks.extend(track_list)

pool = ThreadPoolExecutor(1)
futures = []

for track_chunk in chunks(tracks, 50):
    futures.append(pool.submit(get_metadata_from_tracks, track_chunk))

completed = 0
infos = {}
for x in as_completed(futures):
    result = x.result()
    completed += 50

    print('Finished request for ({0} / {1})'.format(completed, len(tracks)))
    infos.update(result)

for playlist_name, tracks in playlist_tracks.items():
    playlist_tracks[playlist_name][:] = [x for x in tracks if x in infos]

with open('playlists.db', 'wb') as file:
    pickle.dump(playlist_tracks, file, pickle.HIGHEST_PROTOCOL)

with open('tracks.db', 'wb') as file:
    pickle.dump(infos, file, pickle.HIGHEST_PROTOCOL)
