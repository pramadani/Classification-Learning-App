from flask import Flask, jsonify
import random

app = Flask(__name__)

@app.route('/')
def index():
    return 'Server is running.'

@app.route('/api/random', methods=['GET'])
def get_random_data():
    data = {'value': random.randint(1, 100)}
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
