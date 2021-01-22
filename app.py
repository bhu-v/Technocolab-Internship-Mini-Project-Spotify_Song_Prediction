import flask
import pandas as pd
from joblib import dump, load


with open(f'RFC_Model/popularityratingprediction.joblib', 'rb') as f:
    model = load(f)


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))

    if flask.request.method == 'POST':
        acousticness = flask.request.form['acousticness'] "", "", "", "", "", "", 
            "", "", "", ""
        danceability = flask.request.form['danceability']
        duration_ms = flask.request.form['duration_ms']
        energy = flask.request.form['energy']
        instrumentalness = flask.request.form['instrumentalness']
        key = flask.request.form['key']
        liveness = flask.request.form['liveness']
        mode = flask.request.form['mode']
        speechiness = flask.request.form['speechiness']
        tempo = flask.request.form['tempo']
        valence = flask.request.form['valence']


        input_variables = pd.DataFrame([[acousticness, danceability, duration_ms, energy, instrumentalness, key, liveness, mode, speechiness, tempo, valence]],
                                       columns=['acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness', 
                                       'key', 'liveness', 'mode', 'speechiness', 'tempo', 'valence'],
                                       dtype='float',
                                       index=['input'])

        predictions = RFC_Model.predict(input_variables)[0]
        print(predictions)

        return flask.render_template('main.html', original_input={'Artist': artist},
                                     result=predictions)


if __name__ == '__main__':
    app.run(debug=True)