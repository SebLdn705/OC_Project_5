from flask import Flask
from flask import request, jsonify
from utils.utils import text_preprocess
from joblib import load
from models import models
import traceback


# creation of the Flask application object
app = Flask(__name__)
# start the debugger - if code is not correct than an error will appear when the app is visited
app.config["DEBUG"] = True

model_path = "models/stack_overflow_tag_prediction.joblib"
target_path = "models/target_col.joblib"
model = load(model_path)
target_col = load(target_path)


@app.route('/predict', methods=['POST'])
def predict():

    try:
        json_ = request.json
        data = json_

        output = []
        question = data['question']
        processed_text = text_preprocess(question)
        output.append(' '.join(processed_text))

        result = models.prediction(model, 0.35, output, target_col)

        return jsonify({"prediction": result})
    except:
        return jsonify({"trace": traceback.format_exc()})


if __name__ == "__main__":
    app.run()  # run the application server
