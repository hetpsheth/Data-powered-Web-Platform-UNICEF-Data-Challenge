from utils import DataReader
import flask
from flask import request, jsonify

#out of 246 countries, HDI is available for 172 countries.
app = flask.Flask(__name__)
app.config["DEBUG"] = True

@app.route('/get_data', methods=['GET'])
def home():
    obj = DataReader()
    obj.parse_country()
    obj.parse_city()
    obj.parse_features()
    obj.parse_prediction()
    return jsonify(obj.final_data)

if __name__ == '__main__':
    app.run()
