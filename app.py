from flask import Flask, render_template, Response, jsonify
from capture import Capture
from PracticeStudio import PracticeStudio

app = Flask(__name__)
camera = Capture(0)
practice_camera = PracticeStudio(0)

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
    global camera
    camera = Capture(2)
    return render_template('knead.html')

@app.route('/tosspizza')
def toss():
    global camera
    camera = Capture(1)
    return render_template('toss.html')

@app.route('/saucepizza')
def sauce():
    global camera
    camera = Capture(3)
    return render_template('sauce.html')

@app.route('/cheesepizza')
def cheese():
    global camera
    camera = Capture(0)
    return render_template('cheese.html')

@app.route('/toppizza')
def toppings():
    global camera
    camera = Capture(5)
    return render_template('toppings.html')

@app.route('/cookpizza')
def oven():
    global camera
    camera = Capture(6)
    return render_template('oven.html')

@app.route('/cutpizza')
def cut():
    global camera
    camera = Capture(4)
    return render_template('cut.html')

@app.route('/eatpizza')
def eat():
    global camera
    camera = Capture(1)
    return render_template('eat.html')

@app.route('/pizzagame')
def pizzastart():
    return render_template('pizzastart.html')

@app.route('/data')
def data():
    return render_template('data.html')


@app.route('/design')
def design():
    return render_template('design.html')

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

@app.route('/practicestudio')
def practicestudio():
    return render_template('practicestudio.html')

@app.route('/leftcurl')
def leftcurl():
    return render_template('/leftcurl.html')

@app.route('/rightcurl')
def rightcurl():
    global practice_camera
    practice_camera = PracticeStudio(1)
    return render_template('/rightcurl.html')


@app.route('/practice_video_feed')
def practice_video_feed():
    return Response(generate_frames(practice_camera),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
