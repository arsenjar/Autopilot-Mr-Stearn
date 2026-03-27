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

HOST_IP    = '0.0.0.0'
PORT       = 8000
STREAM_URL = 0
STREAM_URL = "http://192.168.240.150:8080/video_feed"
ROBOT_URL  = "http://192.168.240.150:8080"

line_detector = LineDetector()
last_frame = None
lock = threading.Lock()

last_seen_error = 0

def update_camera():
    global last_frame
    cap = cv2.VideoCapture(STREAM_URL)

    while True:
        success, frame = cap.read()

        if not success:
            print("Lost connection to camera. Reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(STREAM_URL)
            continue

        with lock:
            last_frame = frame.copy()

threading.Thread(target=update_camera, daemon=True).start()


def send_command(cmd):
    try:
        requests.post(f"{ROBOT_URL}/move/{cmd}", timeout=1)
        print(f"Sent: {cmd}")
    except:
        print(f"Failed to send: {cmd}")


def send_pwm(left, right):
    try:
        requests.post(
            f"{ROBOT_URL}/move_pwm",
            json={"left": left, "right": right},
            timeout=1
        )
    except:
        print("PWM send failed")


BASE_SPEED = 60
Kp = 0.15

LEFT_TRIM  = 1.0
RIGHT_TRIM = 0.42

DEAD_ZONE = 10
SEARCH_FAST = 34
SEARCH_SLOW = 20


def get_lane_center_from_skeleton(skeleton_clusters, frame_width):
    if len(skeleton_clusters) < 2:
        return None

    cx0 = float(np.mean(skeleton_clusters[0][:, 0]))
    cx1 = float(np.mean(skeleton_clusters[1][:, 0]))

    lane_center = (cx0 + cx1) / 2.0
    return lane_center


def is_stop_line(mask, threshold_ratio=0.4):
    h, w = mask.shape
    

    y_start = int(h * 0.75)
    y_end = int(h * 0.85)

    band = mask[y_start:y_end, :]
    white_pixels = np.sum(band > 0)
    total_pixels = band.shape[0] * band.shape[1]
    

    density = white_pixels / total_pixels
    

    return density > threshold_ratio


def control_loop():
    global last_seen_error

    while True:
        time.sleep(0.05)

        with lock:
            if last_frame is None:
                continue
            frame = last_frame.copy()

        try:
            # optimized = line_detector.optimize_frame(frame)
            # transformed = line_detector.transform(optimized)
            # mask = line_detector.threshold_img(transformed)
            # morphed = line_detector.Morphology(mask)
            optimized = line_detector.optimize_frame(frame)
            mask = line_detector.threshold_img(optimized)
            morphed = line_detector.Morphology(mask)

            print(f"morphed white pixels: {np.sum(morphed > 0)}")  # ← добавь

            skeleton = line_detector.skeletonization_img(morphed)

            print(f"skeleton clusters: {len(skeleton)}")  # ← добавь

            if len(skeleton)== 0:
                print("NO LINES ON SKELETON -> STOP")
                send_pwm(0, 0)
                continue

            # if is_stop_line(morphed, threshold_ratio=0.4):
            #     print("STOP LINE DETECTED")
            #     send_command("stop")
            #     send_pwm(0, 0)
            #     time.sleep(3) 
            #     continue   
            
            lane_center = get_lane_center_from_skeleton(skeleton, optimized.shape[1])
            if lane_center is None:
                raise ValueError("Two lane borders not found")


            center_of_screen = optimized.shape[1] / 2
            error = lane_center - center_of_screen

            if abs(error) < DEAD_ZONE:
                error = 0

            if error != 0:
                last_seen_error = error

            left_speed  = (BASE_SPEED + (Kp * error)) * LEFT_TRIM
            right_speed = (BASE_SPEED - (Kp * error)) * RIGHT_TRIM

            left_speed  = max(-70, min(70, left_speed))
            right_speed = max(-100, min(100, right_speed))

            print(f"lane_center={lane_center:.1f}, error={error:.1f}, last_seen_error={last_seen_error:.1f}")

            send_pwm(left_speed, right_speed)

        except Exception as e:
            print(f"LINE LOST -> SEARCH MODE: {e}")

            if last_seen_error > 0:
                send_pwm(SEARCH_FAST, SEARCH_SLOW)
            elif last_seen_error < 0:
                send_pwm(SEARCH_SLOW, SEARCH_FAST)
            else:
                send_command("stop")
        # except Exception as e:
        #     print(f"LINE LOST -> TURN RIGHT THEN STOP: {e}")
        #     # Повернуть направо
        #     send_pwm(SEARCH_FAST, SEARCH_SLOW)
        #     time.sleep(0.5)
        #     # Остановиться
        #     send_pwm(0, 0)

threading.Thread(target=control_loop, daemon=True).start()


def generate_frames(processed=False):
    while True:
        with lock:
            if last_frame is None:
                time.sleep(0.05)
                continue

            frame = last_frame.copy()

        if processed:
            frame = line_detector.process_frame(frame)

        ret, buffer = cv2.imencode('.jpg', frame)

        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')

        time.sleep(0.033)


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
