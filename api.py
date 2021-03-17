from flask import Flask
from flask import request, jsonify
from utils.utils import text_preprocess
from joblib import load

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
            question_processed = text_preprocess(str(data['question']))
            model.predict_proba(question_processed)

            return jsonify({"prediction": question_processed})

        except:
            return jsonify({"trace": traceback.format_exc()})


# routing >> mapping url to function - mapped to path '/'
# methods list is a keyword that lets Flask know what type of request is allowed
# type of request for the web applications:
#  - GET to send data from the application to the user
#  - POST to receive data from the user
# @app.route("/", methods=['GET'])
# def index():
#     return 'test'
#
# # A route to return all of the available entries in our catalog.
# @app.route('/test', methods=['GET'])
# def api_all():
#     return jsonify(meta)

if __name__ == "__main__":
    model_path = "models/stack_overflow_tag_prediction.joblib"
    model = load(model_path)
    app.run()  # run the application server
