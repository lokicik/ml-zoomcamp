import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model2.bin'

with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

input_dv = "dv.bin"

with open(input_dv, 'rb') as f_in:
    dv = pickle.load(f_in)

app = Flask('poutcome')

@app.route('/predicthomework', methods=['POST'])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0, 1]
    poutcome = y_pred >= 0.5

    result = {
        'poutcome_probability': float(y_pred),
        'poutcome': bool(poutcome)
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)