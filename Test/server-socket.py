from flask import Flask
from flask_socketio import SocketIO, emit
import random
import time

app = Flask(__name__)
# app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

def send_random_data():
    while True:
        data = {'value': random.randint(1, 100)}
        socketio.emit('data_response', data)
        time.sleep(1)

@app.route('/')
def index():
    return 'Server is running.'

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.start_background_task(send_random_data)
    socketio.run(app)