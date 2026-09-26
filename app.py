# app.py
import os
import csv
import random
import json
import time
from datetime import datetime
from flask import Flask, render_template, Response, request, redirect, jsonify, send_from_directory, url_for
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import yt_dlp

LIVE_SOURCE = None
WEBCAM_ALERT_RUNNING = False
LAST_WEBCAM_ALERT = 0                # global timestamp
ALERT_COOLDOWN = 60 * 5              # 5 minutes


# -------------------- YOUTUBE RESOLVE --------------------
def resolve_youtube_url(url):
    try:
        ydl_opts = {
            "format": "best[ext=mp4]/best",
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info["url"]
    except Exception as e:
        print("❌ YouTube resolve error:", e)
        return None


# Optional sentinelhub import
try:
    from sentinelhub import SHConfig, BBox, CRS, SentinelHubRequest, MimeType, DataCollection
    SENTINEL_AVAILABLE = True
except Exception:
    SENTINEL_AVAILABLE = False


# -------------------- FLASK SETUP --------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
app.secret_key = os.getenv('FLASK_SECRET', 'change_this_in_production')


# -------------------- EMAIL CONFIG --------------------
EMAIL_ADDRESS = "prajwalgowdask7022@gmail.com"
EMAIL_PASSWORD = "txrk sodo pcfe qkbe"
ALERT_TO = "prajwalgowdask7022@gmail.com"

import smtplib
from email.message import EmailMessage


def send_email_alert(subject, body, attachment_path=None):
    try:
        print("📧 Sending email alert...")
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = ALERT_TO
        msg.set_content(body)

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, "rb") as f:
                ext = os.path.splitext(attachment_path)[1].lower()
                subtype = "jpeg" if ext in [".jpg", ".jpeg"] else "png"
                msg.add_attachment(
                    f.read(),
                    maintype="image",
                    subtype=subtype,
                    filename=os.path.basename(attachment_path)
                )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print("✅ Email alert sent successfully.")
    except Exception as e:
        print("❌ Email alert failed:", e)


# -------------------- YOLO SETUP --------------------
try:
    yolo_model = YOLO("yolov8n.pt")
    print("✅ YOLOv8 model loaded.")
except Exception as e:
    print("❌ YOLOv8 load error:", e)
    yolo_model = None


# -------------------- DETECTION --------------------
def detect_objects(frame):
    if yolo_model is None:
        return frame, False, {}

    results = yolo_model(frame, verbose=False)
    poacher_found = False
    other_flags = {"logging": False, "vehicle": False, "fire": False}
    annotated = frame.copy()

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = yolo_model.names[cls_id].lower()

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        color = (0, 255, 0)

        if label == "person":
            poacher_found = True
            color = (0, 0, 255)

        if label in ["car", "truck", "bus", "motorbike"]:
            other_flags["vehicle"] = True

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{label} {conf:.2f}", (x1, max(20, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated, poacher_found, other_flags


# -------------------- WEBCAM ALERT --------------------
def webcam_email_alert():
    global WEBCAM_ALERT_RUNNING
    print("🔥 Starting webcam alert thread...")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("❌ Webcam not accessible.")
        WEBCAM_ALERT_RUNNING = False
        return

    prev_detected = False
    no_person_frames = 0

    while WEBCAM_ALERT_RUNNING:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Could not read webcam frame")
            time.sleep(0.2)
            continue

        processed, detected, flags = detect_objects(frame)
        print("Webcam running — detected =", detected)

        if detected and not prev_detected:
            print("📌 POACHER APPEARED — sending email!")
            timestamp = int(time.time())
            proof_path = f"uploads/webcam_proof_{timestamp}.jpg"
            cv2.imwrite(proof_path, processed)
            send_email_alert("🚨 Poacher Detected (Webcam)",
                             "Poacher appeared on LIVE webcam.", proof_path)
            prev_detected = True
            no_person_frames = 0

        if not detected:
            no_person_frames += 1
            if no_person_frames > 10:
                prev_detected = False
        else:
            no_person_frames = 0

        time.sleep(0.05)

    cap.release()
    print("🛑 Webcam alert stopped.")


@app.route("/start_webcam_alert")
def start_webcam_alert_route():
    global WEBCAM_ALERT_RUNNING
    WEBCAM_ALERT_RUNNING = True
    import threading
    t = threading.Thread(target=webcam_email_alert)
    t.daemon = True
    t.start()
    return "✔ Webcam alert STARTED."


@app.route("/stop_webcam_alert")
def stop_webcam_alert():
    global WEBCAM_ALERT_RUNNING
    WEBCAM_ALERT_RUNNING = False
    return "🛑 Webcam alert STOPPED."


# -------------------- LOGGING --------------------
def log_detection(source_type, result_text, lat=None, lon=None):
    log_path = "detections.csv"
    file_exists = os.path.isfile(log_path)
    if lat is None:
        lat = round(11.6 + random.uniform(-0.2, 0.2), 4)
    if lon is None:
        lon = round(76.6 + random.uniform(-0.2, 0.2), 4)

    with open(log_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["Timestamp", "Source", "Result", "Latitude", "Longitude"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), source_type,
                         result_text, lat, lon])


# -------------------- SERVE UPLOADS --------------------
@app.route('/uploads/<path:filename>')
def send_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# -------------------- STREAM WEBCAM --------------------
def generate_frames():
    global LAST_WEBCAM_ALERT
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("❌ Cannot access webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))
        processed, poacher_found, flags = detect_objects(frame)

        if poacher_found:
            cv2.putText(processed, "🚨 POACHER DETECTED!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(processed, "✅ No Poacher Detected", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        now = time.time()
        if poacher_found and (now - LAST_WEBCAM_ALERT > ALERT_COOLDOWN):
            LAST_WEBCAM_ALERT = now
            proof_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                      f"webcam_proof_{int(now)}.jpg")
            cv2.imwrite(proof_path, processed)
            log_detection("Webcam", "Poacher Detected")
            send_email_alert("🚨 Poacher Detected (Webcam)",
                             f"A poacher was detected at {datetime.now()}.", proof_path)

        ret2, buffer = cv2.imencode('.jpg', processed)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live_camera')
def live_camera():
    return render_template('live_camera.html')


# -------------------- UPLOAD IMAGE --------------------
@app.route('/upload_image', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        image = cv2.imread(filepath)
        processed_frame, poacher_found, flags = detect_objects(image)
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'result_' + file.filename)
        cv2.imwrite(output_path, processed_frame)

        log_detection("Image", "Poacher Detected" if poacher_found else "No Poacher Detected")

        if poacher_found:
            send_email_alert("🚨 Poacher Detected (Image)",
                             f"Detected at {datetime.now()}.", output_path)
            message, color = "🚨 Poacher Detected in Image!", "red"
        else:
            message, color = "✅ No Poacher Detected in Image.", "green"

        return render_template('result.html', message=message, color=color,
                               proof_image=os.path.basename(output_path))

    return render_template('upload_image.html')


# -------------------- UPLOAD VIDEO --------------------
@app.route('/upload_video', methods=['GET', 'POST'])
def upload_video():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        cap = cv2.VideoCapture(filepath)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{file.filename}")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        poacher_found = False
        proof_frame_saved = False
        proof_image_path = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed, detected, flags = detect_objects(frame)
            out.write(processed)

            if detected and not proof_frame_saved:
                proof_image_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                                "proof_" + os.path.splitext(file.filename)[0] + ".jpg")
                cv2.imwrite(proof_image_path, processed)
                proof_frame_saved = True

            if detected:
                poacher_found = True

        cap.release()
        out.release()

        log_detection("Video", "Poacher Detected" if poacher_found else "No Poacher Detected")

        if poacher_found:
            send_email_alert("🚨 Poacher Detected (Video)",
                             f"Detected at {datetime.now()}.", proof_image_path)
            message, color = "🚨 Poacher Detected in Video!", "red"
        else:
            message, color = "✅ No Poacher Detected in Video.", "green"

        return render_template('result.html', message=message, color=color,
                               proof_image=(os.path.basename(proof_image_path) if proof_image_path else None))

    return render_template('upload_video.html')


# -------------------- DASHBOARD --------------------
@app.route('/dashboard')
def dashboard():
    detections = []
    try:
        with open('detections.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                detections.append(row)
    except FileNotFoundError:
        detections = []

    total = len(detections)
    poachers = len([d for d in detections if d.get("Result", "").strip() == "Poacher Detected"])
    safe = total - poachers if total >= poachers else 0
    last_alert = detections[-1]["Timestamp"] if detections else "None"

    defo = []
    if os.path.exists('deforestation_log.csv'):
        with open('deforestation_log.csv', 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                defo.append(row)

    return render_template('dashboard.html', total=total, poachers=poachers,
                           safe=safe, last_alert=last_alert,
                           detections=detections, deforestation_data=defo)


# -------------------- TREE COUNT (SIMPLE) --------------------
@app.route('/upload_forest', methods=['GET', 'POST'])
def upload_forest():
    def estimate_tree_count(image_path):
        image = cv2.imread(image_path)
        if image is None:
            return 0

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        trees = max(0, num_labels - 1)

        overlay = (mask).astype(np.uint8)
        overlay_colored = cv2.applyColorMap(overlay, cv2.COLORMAP_SUMMER)
        blended = cv2.addWeighted(image, 0.7, overlay_colored, 0.3, 0)

        proof_path = os.path.join(app.config['UPLOAD_FOLDER'], "tree_mask_" + os.path.basename(image_path))
        cv2.imwrite(proof_path, blended)
        return trees, os.path.basename(proof_path)

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return redirect(request.url)

        path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(path)
        count, proof = estimate_tree_count(path)

        with open('tree_counts.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            if os.stat('tree_counts.csv').st_size == 0:
                writer.writerow(["Timestamp", "TreeCount", "Image"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), count, proof])

        return render_template('result_view.html', proof=proof, message=f"🌳 Estimated Trees: {count}")

    return render_template('upload_forest.html')


# -------------------- RESULT VIEW --------------------
@app.route('/result_view')
def result_view():
    proof = request.args.get('proof')
    message = request.args.get('msg', '')
    return render_template('result_view.html', proof=proof, message=message)


# -------------------- LIVE STREAM FEED --------------------
@app.route("/live_stream")
def live_stream():
    return Response(generate_live(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/add_live_source", methods=["GET", "POST"])
def add_live_source():
    global LIVE_SOURCE

    if request.method == "POST":
        input_url = request.form.get("source").strip()

        if "youtube.com" in input_url or "youtu.be" in input_url:
            resolved = resolve_youtube_url(input_url)
            if not resolved:
                return "❌ Could not resolve YouTube live stream."
            LIVE_SOURCE = resolved
            print("✔ YouTube live stream resolved:", LIVE_SOURCE)
            return redirect("/live_stream")

        if input_url.startswith(("rtsp://", "http://", "https://")):
            LIVE_SOURCE = input_url
            print("✔ Live source set to:", LIVE_SOURCE)
            return redirect("/live_stream")

        return "❌ Invalid URL. Only RTSP/HTTP/HTTPS/YouTube supported."

    return render_template("add_live_source.html")


def generate_live():
    global LIVE_SOURCE

    if not LIVE_SOURCE:
        print("❌ No live source set.")
        return

    print(f"🎥 Opening live source: {LIVE_SOURCE}")
    cap = cv2.VideoCapture(LIVE_SOURCE, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("❌ Cannot open live stream.")
        return

    proof_saved = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠ Stream dropped or ended.")
            break

        processed, detected, flags = detect_objects(frame)

        if detected and not proof_saved:
            proof_path = "uploads/live_proof.jpg"
            cv2.imwrite(proof_path, processed)
            proof_saved = True
            send_email_alert("🚨 Poacher Detected (Live Stream)",
                             "A poacher was detected on the live stream.", proof_path)

        ret2, buffer = cv2.imencode(".jpg", processed)
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

    cap.release()


# -------------------- MAIN --------------------
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
