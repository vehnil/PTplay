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
def landing():
    return render_template('landing.html')

@app.route('/kneadpizza')
def knead():
    return render_template('knead.html')

@app.route('/tosspizza')
def toss():
    return render_template('toss.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/reset_counter')
def reset_counter():
    camera.counter = 0  # Reset the counter
    return jsonify(success=True)

@app.route('/counter')
def get_counter():
    return jsonify(counter=camera.counter)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
