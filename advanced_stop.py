import cv2
import time
import threading
import requests
import numpy as np
from flask import Flask, Response
from flask_cors import CORS
from LineDetector import LineDetector
app = Flask(__name__)
CORS(app)
POST_DRIVE_TIME = 5.0   #time after detection
HOST_IP    = '0.0.0.0'
PORT       = 8080
STREAM_URL = "http://192.168.240.150:8080/video_feed"
ROBOT_URL  = "http://192.168.240.150:8080"

# ROBOT SETUP
line_detector = LineDetector()
last_frame = None
lock = threading.Lock()

last_seen_error = 0 # last seen line

stop_triggered = False
stop_time = None
POST_TURN_BOOST = 28


def update_camera():
    global last_frame
    cap = cv2.VideoCapture(STREAM_URL)

    while True:
        success, frame = cap.read()

        if not success:
            print("Lost camera connection!!!! :((((")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        with lock:
            last_frame = frame.copy()

threading.Thread(target=update_camera, daemon=True).start()

### Functions ###
def send_command(cmd):
    try:
        requests.post(f"{ROBOT_URL}/move/{cmd}", timeout=1)
        print(f"Sent: {cmd}")
    except Exception:
        print(f"Faild {cmd}")


def send_pwm(left, right):
    try:
        requests.post(
            f"{ROBOT_URL}/move_pwm",
            json={"left": float(left), "right": float(right)},
            timeout=1
        )
    except Exception:
        print("PWM send failed")

# ROBOT VARIBLES
BASE_SPEED = 60
Kp = 0.15

LEFT_TRIM  = 1.0
RIGHT_TRIM = 0.35

DEAD_ZONE = 10
SEARCH_FAST = 34
SEARCH_SLOW = 20

# detection of the horizontal line
def get_lane_center_from_mask(mask):
    h, w = mask.shape
    y = int(h * 0.8)

    row = mask[y]
    xs = np.where(row > 0)[0]

    if len(xs) < 2:
        return None

    center_screen = w // 2

    left_candidates = xs[xs < center_screen]
    right_candidates = xs[xs > center_screen]

    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return None

    left_x = np.max(left_candidates)
    right_x = np.min(right_candidates)

    lane_center = (left_x + right_x) / 2
    return lane_center

def detect_blue_stop_line(frame):
    h, w, _ = frame.shape

    # Define ROI
    y1, y2 = int(h * 0.70), int(h * 0.95)
    x1, x2 = int(w * 0.30), int(w * 0.70)
    roi = frame[y1:y2, x1:x2]
    
    # Color Conversion & Filtering
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Clean up noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of blue objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Minimum size to ignore noise
            x, y, w_obj, h_obj = cv2.boundingRect(cnt)
            aspect_ratio = float(w_obj) / h_obj
            
            # A stop line MUST be wider than it is tall
            # and should take up a decent portion of the ROI width
            if aspect_ratio > 2.0 and w_obj > (roi.shape[1] * 0.5):
                return True

    return False


def draw_blue_roi(frame):
    h, w, _ = frame.shape
    y1 = int(h * 0.70)
    y2 = int(h * 0.95)
    x1 = int(w * 0.30)
    x2 = int(w * 0.70)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    return frame


def control_loop():
    global last_seen_error, stop_triggered, stop_time

    while True:
        time.sleep(0.05)

        with lock:
            if last_frame is None:
                continue
            frame = last_frame.copy()

        # Если уже нашли синюю стоп-линию
        if stop_triggered:
            elapsed = time.time() - stop_time

            if elapsed < POST_DRIVE_TIME:
                try:
                    optimized = line_detector.optimize_frame(frame)
                    transformed = line_detector.transform(optimized)
                    mask = line_detector.threshold_img(transformed)
                    morphed = line_detector.Morphology(mask)

                    lane_center = get_lane_center_from_mask(morphed)

                    if lane_center is not None:
                        center_of_screen = transformed.shape[1] / 2
                        error = lane_center - center_of_screen

                        # For smother
                        if abs(error) < DEAD_ZONE:
                            error = 0

                        if error != 0:
                            last_seen_error = error

                        left_speed = (BASE_SPEED + (Kp * error)) * LEFT_TRIM
                        right_speed = (BASE_SPEED - (Kp * error)) * RIGHT_TRIM

                        left_speed = max(-70, min(70, left_speed))
                        right_speed = max(-100, min(100, right_speed))

                        send_pwm(left_speed, right_speed)

                    else:

                        if last_seen_error > 0:
                            left_speed = BASE_SPEED * LEFT_TRIM
                            right_speed = max(0, (BASE_SPEED - POST_TURN_BOOST) * RIGHT_TRIM)
                            send_pwm(left_speed, right_speed)

                        elif last_seen_error < 0:
                            left_speed = max(0, (BASE_SPEED - POST_TURN_BOOST) * LEFT_TRIM)
                            right_speed = BASE_SPEED * RIGHT_TRIM
                            send_pwm(left_speed, right_speed)
                        else:
                            send_command("stop")
                            print("No lines detected, making stop!!!!")

                except Exception as e:
                    print(e)

                    if last_seen_error > 0:
                        left_speed = BASE_SPEED * LEFT_TRIM
                        right_speed = max(0, (BASE_SPEED - POST_TURN_BOOST) * RIGHT_TRIM)
                        send_pwm(left_speed, right_speed)
                        print("Right")

                    elif last_seen_error < 0:
                        left_speed = max(0, (BASE_SPEED - POST_TURN_BOOST) * LEFT_TRIM)
                        right_speed = BASE_SPEED * RIGHT_TRIM
                        send_pwm(left_speed, right_speed)
                        print("Left")

                    else:
                        send_command("stop")
                        print("Stopped!!!")

            else:
                send_command("stop")
                print("The end!")
            continue
        try:
            if detect_blue_stop_line(frame):
                stop_triggered = True
                stop_time = time.time()
                print(f"Blue lines detected, moving forward for {POST_DRIVE_TIME}")
                continue

            optimized = line_detector.optimize_frame(frame)
            transformed = line_detector.transform(optimized)
            mask = line_detector.threshold_img(transformed)
            morphed = line_detector.Morphology(mask)
            lane_center = get_lane_center_from_mask(morphed)

            if lane_center is None:
                raise ValueError("No lines detected")

            center_of_screen = transformed.shape[1] / 2
            error = lane_center - center_of_screen

            if abs(error) < DEAD_ZONE:
                error = 0

            if error != 0:
                last_seen_error = error

            left_speed = (BASE_SPEED + (Kp * error)) * LEFT_TRIM
            right_speed = (BASE_SPEED - (Kp * error)) * RIGHT_TRIM

            left_speed = max(-70, min(70, left_speed))
            right_speed = max(-100, min(100, right_speed))
            send_pwm(left_speed, right_speed)

        except Exception as e:
            print(e)
            if last_seen_error > 0:
                send_pwm(SEARCH_FAST, SEARCH_SLOW)
            elif last_seen_error < 0:
                send_pwm(SEARCH_SLOW, SEARCH_FAST)
            else:
                send_command("stop")

threading.Thread(target=control_loop, daemon=True).start()


def generate_frames(processed=False):
    while True:
        with lock:
            if last_frame is None:
                time.sleep(0.05)
                continue

            frame = last_frame.copy()

        if processed:
            try:
                frame = line_detector.process_frame(frame)
            except Exception:
                pass

            frame = draw_blue_roi(frame)

            if stop_triggered:
                cv2.putText(frame, "STOPPPPPPP!!!!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        ret, buffer = cv2.imencode('.jpg', frame)

        if not ret:
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' +
            buffer.tobytes() +
            b'\r\n'
        )
        time.sleep(0.03)
### API Requests ###
@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
@app.route('/video_feed/processed')
def video_feed_processed():
    return Response(
        generate_frames(processed=True),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    app.run(host=HOST_IP, port=PORT, threaded=True)
