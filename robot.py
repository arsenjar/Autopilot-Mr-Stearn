import time
import threading
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from PCA9685 import PCA9685
import cv2
app = Flask(__name__)
CORS(app)

HOST_IP  = '0.0.0.0'
PORT     = 8080
SPEED    = 0.6
DURATION = 0.6
last_frame = None
lock = threading.Lock()

def generate_frames(processed=False):
    while True:
        with lock:
            if last_frame is None:
                time.sleep(0.05)
                continue
            frame = last_frame.copy()

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.033)
@app.post('/move_pwm')
def move_pwm():
    data = request.get_json()

    left = float(data.get('left', 0))
    right = float(data.get('right', 0))

    motors(left, right)

    return jsonify({
        'left': left,
        'right': right
    })

pwm = PCA9685(0x40, debug=False)
pwm.setPWMFreq(50)

def motors(left, right):
    pwm.setDutycycle(0, int(abs(left) * 100))

    pwm.setLevel(1, 0 if left >= 0 else 1)
    pwm.setLevel(2, 1 if left >= 0 else 0)
    pwm.setDutycycle(5, int(abs(right) * 100))
    pwm.setLevel(3, 1 if right >= 0 else 0)
    pwm.setLevel(4, 0 if right >= 0 else 1)

def motors_stop():
    pwm.setDutycycle(0, 0)
    pwm.setDutycycle(5, 0)

def drive(left, right):
    motors(left, right)
    time.sleep(DURATION)
    motors_stop()

@app.post('/move/stop')
def move_stop():
    motors_stop()
    return jsonify({'command': 'stop'})
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- Routes ---
@app.post('/move/forward')
def move_forward():
    threading.Thread(target=drive, args=(SPEED, SPEED), daemon=True).start()
    return jsonify({'command': 'forward'})

@app.post('/move/left')
def move_left():
    threading.Thread(target=drive, args=(-SPEED, SPEED), daemon=True).start()
    return jsonify({'command': 'left'})

@app.post('/move/right')
def move_right():
    threading.Thread(target=drive, args=(SPEED, -SPEED), daemon=True).start()
    return jsonify({'command': 'right'})


if __name__ == '__main__':
    app.run(host=HOST_IP, port=PORT, threaded=True)