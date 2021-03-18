from flask import Flask
from flask import request, jsonify, render_template
from utils.utils import text_preprocess
from joblib import load
from models import models

# creation of the Flask application object
app = Flask(__name__)
# start the debugger - if code is not correct than an error will appear when the app is visited
app.config["DEBUG"] = True


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            json_ = request.json
            data = json_

            model_path = "models/stack_overflow_tag_prediction.joblib"
            target_path = "models/target_col.joblib"
            model = load(model_path)
            target_col = load(target_path)

            output = []
            question = data['question']
            processed_text = text_preprocess(question)
            output.append(' '.join(processed_text))

            result = models.prediction(model, 0.35, output, target_col)

            return jsonify({"prediction": result})
            # return json_
        except:
            return jsonify({"trace": traceback.format_exc()})


# routing >> mapping url to function - mapped to path '/'
# methods list is a keyword that lets Flask know what type of request is allowed
# type of request for the web applications:
#  - GET to send data from the application to the user
#  - POST to receive data from the user
@app.route("/", methods=['GET', 'POST'])
def home():
    if request.method == 'POST':

        model_path = "models/stack_overflow_tag_prediction.joblib"
        target_path = "models/target_col.joblib"
        model = load(model_path)
        target_col = load(target_path)

        output = []
        question = request.form.get('question')
        processed_text = text_preprocess(question)
        output.append(' '.join(processed_text))

        result = models.prediction(model, 0.35, output, target_col)
    return render_template('index.html', prediction=result)
    # return render_template('index.html')
#

# # A route to return all of the available entries in our catalog.
# @app.route('/test', methods=['GET'])
# def api_all():
#     return jsonify(meta)

if __name__ == "__main__":

    app.run()  # run the application server
