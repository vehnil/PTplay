from flask import Flask, render_template, Response, jsonify
from capture import Capture

app = Flask(__name__)
camera = Capture()

def generate_frames(camera):
    while True:
        frame = camera.get_frame()
        if frame is None:
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counter')
def get_counter():
    return jsonify(counter=camera.counter)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
