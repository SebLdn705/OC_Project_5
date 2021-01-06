from flask import Flask
from flask import request, jsonify

# creation of the Flask application object
app = Flask(__name__)
# start the debugger - if code is not correct than an error will appear when the app is visited
app.config["DEBUG"] = True

# creation of test data
meta = [
    {
        'title': 'test question for the API OpenClassrooms project #5'
    }
]


# routing >> mapping url to function - mapped to path '/'
# methods list is a keyword that lets Flask know what type of request is allowed
# type of request for the web applications:
#  - GET to send data from the application to the user
#  - POST to receive data from the user
@app.route("/", methods=['GET'])
def index():
    return "this site is a prototype for the API<p>Project OpenClassrooms ML"

# A route to return all of the available entries in our catalog.
@app.route('/test', methods=['GET'])
def api_all():
    return jsonify(meta)

if __name__ == "__main__":
    app.run()  # run the application server
