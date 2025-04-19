
import pickle

from flask import Flask
from flask import request
from flask import jsonify

def pickle_load(file_path):
    """
    Load a dictionary vectorizer or model file.
    """
    with open(file_path, 'rb') as f_in:
        loaded_file = pickle.load(f_in)
    return loaded_file


CHURN_LEVEL = 0.5

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    customer = request.get_json()

    # load dict vectorizor and model
    dv = pickle_load('model/dv.bin')
    model = pickle_load('model/model1.bin')

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]

    churn = y_pred >= CHURN_LEVEL

    result = {
        "churn_probability": float(y_pred),
        "churn": bool(churn),
    }

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
