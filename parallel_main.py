
import cv2
import torch
import time
import queue
import threading
import numpy as np

from collections import deque
from datetime import datetime

from ultralytics import YOLO

from config import *
from camera import open_camera
from timesformer_adapter import TimeSformerAdapter
from alert_manager import AlertManager


# =====================================================
# 1. PYTORCH CPU CONFIGURATION
# =====================================================

torch.set_num_threads(TORCH_THREADS)
torch.set_num_interop_threads(TORCH_INTEROP_THREADS)

print("Inference device:", DEVICE)
print("CPU threads:", torch.get_num_threads())


# =====================================================
# 2. SHARED VARIABLES
# =====================================================

stop_event = threading.Event()
results_lock = threading.Lock()

latest_results = {
    "timesformer": None,
    "pose": None,
    "weapon": None
}

model_status = {
    "timesformer": "Starting",
    "pose": "Starting",
    "weapon": "Starting"
}


# =====================================================
# 3. INDEPENDENT MODEL QUEUES
# =====================================================

timesformer_queue = queue.Queue(maxsize=1)
pose_queue = queue.Queue(maxsize=1)
weapon_queue = queue.Queue(maxsize=1)


def put_latest(q, item):

    try:
        q.put_nowait(item)

    except queue.Full:

        try:
            q.get_nowait()
        except queue.Empty:
            pass

        try:
            q.put_nowait(item)
        except queue.Full:
            pass


# =====================================================
# 4. RESULT MANAGEMENT
# =====================================================

def update_result(name, result):

    with results_lock:
        latest_results[name] = result


def update_status(name, status):

    with results_lock:
        model_status[name] = status


def get_results():

    with results_lock:
        return dict(latest_results), dict(model_status)


# =====================================================
# 5. TIMESFORMER WORKER
# =====================================================

def timesformer_worker():

    try:

        model = TimeSformerAdapter()

        update_status("timesformer", "Running")

        print("[TimeSformer] Worker started.")

    except Exception as error:

        update_status("timesformer", f"ERROR: {error}")
        print("[TimeSformer ERROR]", error)
        return

    while not stop_event.is_set():

        try:
            clip, timestamp, submitted_at = (
                timesformer_queue.get(timeout=0.2)
            )

        except queue.Empty:
            continue

        try:

            prediction = model.predict(clip)

            result = {
                "timestamp": timestamp,
                "wall_time": submitted_at,
                "prediction": prediction["class_name"],
                "predicted_class": prediction["predicted_class"],
                "confidence": float(prediction["confidence"]),
                "violence_probability": float(
                    prediction["violence_probability"]
                ),
                "probabilities": prediction["probabilities"]
            }

            update_result("timesformer", result)

            print(
                "[TimeSformer]",
                result["prediction"],
                result["probabilities"]
            )

        except Exception as error:
            print("[TimeSformer ERROR]", error)


# =====================================================
# 6. YOLO POSE WORKER
# =====================================================

def pose_worker():

    try:

        if not POSE_PATH.exists():
            raise FileNotFoundError(POSE_PATH)

        model = YOLO(str(POSE_PATH))

        update_status("pose", "Running")

        print("[Pose] Worker started.")
        print("[Pose] Classes:", model.names)

    except Exception as error:

        update_status("pose", f"ERROR: {error}")
        print("[Pose ERROR]", error)
        return

    while not stop_event.is_set():

        try:
            frame, timestamp, submitted_at = (
                pose_queue.get(timeout=0.2)
            )

        except queue.Empty:
            continue

        try:

            with torch.inference_mode():

                prediction = model.predict(
                    source=frame,
                    conf=POSE_CONFIDENCE,
                    device=DEVICE,
                    verbose=False
                )[0]

            people_count = (
                len(prediction.boxes)
                if prediction.boxes is not None
                else 0
            )

            person_boxes = []

            if prediction.boxes is not None:
                person_boxes = (
                    prediction.boxes.xyxy.cpu().numpy().tolist()
                )

            keypoints = None

            if prediction.keypoints is not None:
                keypoints = (
                    prediction.keypoints.data.cpu().numpy().tolist()
                )

            result = {
                "timestamp": timestamp,
                "wall_time": submitted_at,
                "people_detected": people_count,
                "person_boxes": person_boxes,
                "keypoints": keypoints
            }

            update_result("pose", result)

        except Exception as error:
            print("[Pose ERROR]", error)


# =====================================================
# 7. YOLO WEAPON WORKER
# =====================================================

def weapon_worker():

    try:

        if not WEAPON_PATH.exists():
            raise FileNotFoundError(WEAPON_PATH)

        model = YOLO(str(WEAPON_PATH))

        update_status("weapon", "Running")

        print("[Weapon] Worker started.")
        print("[Weapon] Classes:", model.names)

    except Exception as error:

        update_status("weapon", f"ERROR: {error}")
        print("[Weapon ERROR]", error)
        return

    while not stop_event.is_set():

        try:
            frame, timestamp, submitted_at = (
                weapon_queue.get(timeout=0.2)
            )

        except queue.Empty:
            continue

        try:

            with torch.inference_mode():

                prediction = model.predict(
                    source=frame,
                    conf=WEAPON_CONFIDENCE,
                    device=DEVICE,
                    verbose=False
                )[0]

            detections = []

            if prediction.boxes is not None:

                boxes = prediction.boxes.xyxy.cpu().numpy()

                confidences = (
                    prediction.boxes.conf.cpu().numpy()
                )

                class_ids = (
                    prediction.boxes.cls.cpu().numpy().astype(int)
                )

                for box, confidence, class_id in zip(
                    boxes,
                    confidences,
                    class_ids
                ):

                    detections.append({
                        "class_name": prediction.names[class_id],
                        "confidence": float(confidence),
                        "box": box.tolist()
                    })

            result = {
                "timestamp": timestamp,
                "wall_time": submitted_at,
                "weapon_detected": bool(detections),
                "detections": detections
            }

            update_result("weapon", result)

        except Exception as error:
            print("[Weapon ERROR]", error)


# =====================================================
# 8. START THREE INDEPENDENT WORKERS
# =====================================================

workers = [

    threading.Thread(
        target=timesformer_worker,
        name="TimeSformer",
        daemon=True
    ),

    threading.Thread(
        target=pose_worker,
        name="Pose",
        daemon=True
    ),

    threading.Thread(
        target=weapon_worker,
        name="Weapon",
        daemon=True
    )

]


def start_workers():

    for worker in workers:
        worker.start()


# =====================================================
# 9. TIMESFORMER TEMPORAL CLIP SAMPLING
# =====================================================

def create_timesformer_clip(frame_buffer, now):

    start_time = now - TIMESFORMER_CLIP_SECONDS

    entries = [
        item for item in frame_buffer
        if start_time <= item[0] <= now
    ]

    if len(entries) < 2:
        return None

    frame_times = np.array([
        item[0] for item in entries
    ])

    target_times = np.linspace(
        start_time,
        now,
        TIMESFORMER_NUM_FRAMES
    )

    selected = []

    for target in target_times:

        index = int(
            np.argmin(np.abs(frame_times - target))
        )

        selected.append(entries[index][1])

    return np.stack(selected)


# =====================================================
# 10. DRAW HUMAN POSE
# =====================================================

POSE_CONNECTIONS = [

    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16)

]


def draw_pose(display, pose):

    if not pose:
        return display

    for box in pose.get("person_boxes", []):

        x1, y1, x2, y2 = map(int, box)

        cv2.rectangle(
            display,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            display,
            "Person",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

    all_keypoints = pose.get("keypoints")

    if all_keypoints is None:
        return display

    for person in all_keypoints:

        visible = {}

        for index, point in enumerate(person):

            x = int(point[0])
            y = int(point[1])

            confidence = (
                float(point[2])
                if len(point) > 2
                else 1.0
            )

            if confidence >= 0.25:

                visible[index] = (x, y)

                cv2.circle(
                    display,
                    (x, y),
                    4,
                    (0, 255, 255),
                    -1
                )

        for start, end in POSE_CONNECTIONS:

            if start in visible and end in visible:

                cv2.line(
                    display,
                    visible[start],
                    visible[end],
                    (255, 0, 255),
                    2
                )

    return display


# =====================================================
# 11. DISPLAY ALL THREE MODEL RESULTS
# =====================================================

def draw_results(frame, results, statuses):

    display = frame.copy()

    cv2.putText(
        display,
        "PARALLEL REAL-TIME DETECTION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # TimeSformer

    ts = results["timesformer"]

    if ts:

        age = time.monotonic() - ts["wall_time"]

        text = (
            f"TimeSformer: {ts['prediction']} "
            f"({ts['confidence']:.2f})"
        )

        if age > RESULT_MAX_AGE:
            text += " [STALE]"

        cv2.putText(
            display,
            text,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

        p = ts["probabilities"]

        cv2.putText(
            display,
            (
                f"Normal: {p['Normal']:.2f} "
                f"Fighting: {p['Fighting']:.2f} "
                f"Shooting: {p['Shooting']:.2f}"
            ),
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

    else:

        cv2.putText(
            display,
            "TimeSformer: Waiting",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

    # YOLO Pose

    pose = results["pose"]

    if pose:

        pose_text = (
            f"YOLO Pose: {pose['people_detected']} people"
        )

        if time.monotonic() - pose["wall_time"] > RESULT_MAX_AGE:
            pose_text += " [STALE]"

        draw_pose(display, pose)

    else:
        pose_text = "YOLO Pose: Waiting"

    cv2.putText(
        display,
        pose_text,
        (20, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 0),
        2
    )

    # YOLO Weapon

    weapon = results["weapon"]

    if weapon:

        detections = weapon["detections"]

        weapon_text = (
            f"Weapon detections: {len(detections)}"
        )

        if time.monotonic() - weapon["wall_time"] > RESULT_MAX_AGE:
            weapon_text += " [STALE]"

        cv2.putText(
            display,
            weapon_text,
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2
        )

        for detection in detections:

            x1, y1, x2, y2 = map(
                int,
                detection["box"]
            )

            cv2.rectangle(
                display,
                (x1, y1),
                (x2, y2),
                (0, 165, 255),
                2
            )

            label = (
                f"{detection['class_name']} "
                f"{detection['confidence']:.2f}"
            )

            cv2.putText(
                display,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                2
            )

    else:

        cv2.putText(
            display,
            "Weapon: Waiting",
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2
        )

    # Model status

    y = 220

    for name, status in statuses.items():

        cv2.putText(
            display,
            f"{name}: {status}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        y += 25

    return display


# =====================================================
# 11.5. DRAW ON-SCREEN ALERT BANNER
# =====================================================

def draw_alert_banner(frame, alert):

    if alert is None:
        return frame

    height, width = frame.shape[:2]

    flash = int(time.time() * 3) % 2 == 0

    if alert["severity"] == "CRITICAL":
        color = (0, 0, 255)

    elif alert["severity"] == "HIGH":
        color = (0, 80, 255)

    else:
        color = (0, 165, 255)

    if flash:

        cv2.rectangle(
            frame,
            (0, 0),
            (width, 90),
            color,
            -1
        )

    cv2.putText(
        frame,
        "SECURITY ALERT",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        alert["type"],
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press A to silence alarm",
        (width - 320, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    return frame


# =====================================================
# 11.6. WRAP TEXT FOR INCIDENT SUMMARY PANEL
# =====================================================

def wrap_text(text, max_width, font, scale, thickness):

    words = str(text).split()

    lines = []
    current_line = ""

    for word in words:

        test_line = (
            word if not current_line
            else current_line + " " + word
        )

        text_width = cv2.getTextSize(
            test_line,
            font,
            scale,
            thickness
        )[0][0]

        if text_width <= max_width:
            current_line = test_line

        else:

            if current_line:
                lines.append(current_line)

            current_line = word

    if current_line:
        lines.append(current_line)

    return lines


# =====================================================
# 11.7. DRAW AI INCIDENT SUMMARY PANEL
# =====================================================

def draw_incident_panel(frame, incident):

    if incident is None:
        return frame

    height, width = frame.shape[:2]

    panel_width = min(760, width - 40)

    if panel_width < 250:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX

    title_scale = 0.58
    text_scale = 0.43
    small_scale = 0.38

    padding = 15
    line_gap = 21

    summary = incident.get(
        "summary",
        "Generating AI summary..."
    )

    status = str(
        incident.get("llm_status", "pending")
    ).lower()

    severity = incident.get("severity", "UNKNOWN")

    if status == "pending":
        status_text = "AI SUMMARY: GENERATING..."
    elif status in ["success", "completed"]:
        status_text = "AI SUMMARY: COMPLETED"
    else:
        status_text = "AI SUMMARY: FALLBACK / CHECK"

    # Prepare wrapped summary text

    max_text_width = panel_width - (padding * 2)

    summary_lines = wrap_text(
        summary,
        max_text_width,
        font,
        text_scale,
        1
    )

    # Keep the panel compact
    max_summary_lines = 4

    if len(summary_lines) > max_summary_lines:

        summary_lines = summary_lines[:max_summary_lines]

        last_line = summary_lines[-1]

        while last_line and cv2.getTextSize(
            last_line + "...",
            font,
            text_scale,
            1
        )[0][0] > max_text_width:
            last_line = last_line[:-1]

        summary_lines[-1] = last_line.rstrip() + "..."

    # Detection details

    classification = incident.get(
        "violence_classification",
        "Unavailable"
    )

    confidence = incident.get(
        "violence_confidence"
    )

    if confidence is None:
        confidence_text = "N/A"
    else:
        confidence_text = f"{float(confidence):.2f}"

    people_count = incident.get("people_count", 0)

    weapon_detections = incident.get(
        "weapon_detections",
        []
    )

    weapon_names = []

    for detection in weapon_detections:

        name = detection.get("class_name", "Weapon")

        if name not in weapon_names:
            weapon_names.append(name)

    weapons_text = (
        ", ".join(weapon_names)
        if weapon_names
        else "None detected"
    )

    details = (
        f"Class: {classification} | "
        f"Confidence: {confidence_text} | "
        f"People: {people_count}"
    )

    weapon_line = f"Weapons: {weapons_text}"

    # Calculate panel height

    panel_height = (
        padding
        + 25
        + 22
        + 25
        + len(summary_lines) * line_gap
        + 8
        + 20
        + 22
        + padding
    )

    panel_height = min(panel_height, height - 30)

    x1 = 20
    x2 = x1 + panel_width

    y2 = height - 15
    y1 = max(10, y2 - panel_height)

    # Dark transparent background

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.82,
        frame,
        0.18,
        0,
        frame
    )

    # Severity color

    if severity == "CRITICAL":
        severity_color = (0, 0, 255)

    elif severity == "HIGH":
        severity_color = (0, 100, 255)

    else:
        severity_color = (0, 165, 255)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y1 + 5),
        severity_color,
        -1
    )

    # Panel title

    y = y1 + 28

    cv2.putText(
        frame,
        "LATEST INCIDENT",
        (x1 + padding, y),
        font,
        title_scale,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        severity,
        (x2 - 125, y),
        font,
        0.48,
        severity_color,
        2
    )

    # Incident time and ID

    y += 23

    incident_time = incident.get(
        "timestamp",
        ""
    )

    incident_id = incident.get(
        "incident_id",
        ""
    )

    cv2.putText(
        frame,
        f"{incident_time} | {incident_id}",
        (x1 + padding, y),
        font,
        small_scale,
        (180, 180, 180),
        1
    )

    # AI summary status

    y += 27

    status_color = (
        (0, 255, 255)
        if status == "pending"
        else (0, 255, 0)
        if status in ["success", "completed"]
        else (0, 165, 255)
    )

    cv2.putText(
        frame,
        status_text,
        (x1 + padding, y),
        font,
        0.45,
        status_color,
        2
    )

    # Summary text

    y += 25

    for line in summary_lines:

        if y > y2 - 55:
            break

        cv2.putText(
            frame,
            line,
            (x1 + padding, y),
            font,
            text_scale,
            (255, 255, 255),
            1
        )

        y += line_gap

    # Detection information

    details_y = y2 - 40

    cv2.putText(
        frame,
        details,
        (x1 + padding, details_y),
        font,
        small_scale,
        (180, 220, 255),
        1
    )

    cv2.putText(
        frame,
        weapon_line,
        (x1 + padding, y2 - 18),
        font,
        small_scale,
        (0, 190, 255),
        1
    )

    return frame


# =====================================================
# 12. ALERT + INCIDENT SUMMARIZATION
# =====================================================

def process_alerts(
    alert_manager,
    frame,
    results,
    last_evaluated
):

    ts = results["timesformer"]
    weapon = results["weapon"]
    pose = results["pose"]

    if ts is None or weapon is None:
        return

    now = time.monotonic()

    # Ignore stale results

    if now - ts["wall_time"] > RESULT_MAX_AGE:
        return

    if now - weapon["wall_time"] > RESULT_MAX_AGE:
        return

    # Require both models to have new results

    if ts["wall_time"] == last_evaluated["ts"]:
        return

    if weapon["wall_time"] == last_evaluated["weapon"]:
        return

    last_evaluated["ts"] = ts["wall_time"]
    last_evaluated["weapon"] = weapon["wall_time"]

    # TimeSformer result

    timesformer_result = {

        "class_name": ts["prediction"],

        "confidence": ts["confidence"],

        "violence_probability": ts["violence_probability"],

        "probabilities": ts["probabilities"],

        "wall_time": ts["wall_time"]

    }

    # Weapon detections

    weapon_detections = weapon["detections"]

    # Pose is supporting information only

    people_count = 0

    if (
        pose is not None
        and now - pose["wall_time"] <= RESULT_MAX_AGE
    ):
        people_count = pose["people_detected"]

    # Check for confirmed alert

    alert_manager.check_alert(

        frame=frame,

        timesformer_result=timesformer_result,

        weapon_detections=weapon_detections,

        people_count=people_count

    )


# =====================================================
# 13. MAIN CAMERA LOOP
# =====================================================

def main():

    print("\nStarting three model workers...")

    start_workers()

    print("Connecting to iPhone DroidCam...")

    cap = open_camera()

    frame_buffer = deque()

    last_timesformer_time = 0
    last_detection_time = 0

    alert_manager = AlertManager()

    last_evaluated = {
        "ts": None,
        "weapon": None
    }

    print("\n======================================")
    print("REAL-TIME PARALLEL SYSTEM STARTED")
    print("TimeSformer | YOLO Pose | YOLO Weapon")
    print("Pose skeleton + bounding boxes enabled")
    print("Audio + Visual Alerts Enabled")
    print("AI Incident Summary Panel Enabled")
    print("--------------------------------------")
    print("Press A to silence alert sound")
    print("Press Q to stop")
    print("======================================")

    try:

        while True:

            ret, frame = cap.read()

            if not ret or frame is None:

                print("Camera frame unavailable.")
                break

            now = time.monotonic()

            timestamp = datetime.now().isoformat(
                timespec="milliseconds"
            )

            # ---------------------------------
            # A. TimeSformer frame buffer
            # ---------------------------------

            small_frame = cv2.resize(
                frame,
                (224, 224)
            )

            frame_buffer.append(
                (now, small_frame)
            )

            while (
                frame_buffer
                and now - frame_buffer[0][0]
                > TIMESFORMER_CLIP_SECONDS + 0.5
            ):
                frame_buffer.popleft()

            # ---------------------------------
            # B. TimeSformer clip submission
            # ---------------------------------

            if (
                now - last_timesformer_time
                >= TIMESFORMER_STRIDE_SECONDS

                and frame_buffer

                and now - frame_buffer[0][0]
                >= TIMESFORMER_CLIP_SECONDS
            ):

                clip = create_timesformer_clip(
                    frame_buffer,
                    now
                )

                if clip is not None:

                    put_latest(
                        timesformer_queue,
                        (clip, timestamp, now)
                    )

                    last_timesformer_time = now

            # ---------------------------------
            # C. Pose and weapon submission
            # ---------------------------------

            if now - last_detection_time >= DETECTION_INTERVAL:

                packet = (
                    frame.copy(),
                    timestamp,
                    now
                )

                put_latest(pose_queue, packet)
                put_latest(weapon_queue, packet)

                last_detection_time = now

            # ---------------------------------
            # D. Collect latest results
            # ---------------------------------

            results, statuses = get_results()

            # ---------------------------------
            # E. Alert processing
            # ---------------------------------

            process_alerts(
                alert_manager,
                frame,
                results,
                last_evaluated
            )

            # ---------------------------------
            # F. Draw model results
            # ---------------------------------

            display = draw_results(
                frame,
                results,
                statuses
            )

            # ---------------------------------
            # G. Draw active alert banner
            # ---------------------------------

            active_alert = alert_manager.get_active_alert()

            display = draw_alert_banner(
                display,
                active_alert
            )

            # ---------------------------------
            # H. Draw latest AI incident summary
            # ---------------------------------

            latest_incident = (
                alert_manager.get_latest_incident()
            )

            display = draw_incident_panel(
                display,
                latest_incident
            )

            # ---------------------------------
            # I. Display camera
            # ---------------------------------

            cv2.imshow(
                "Three Model Parallel Detection",
                display
            )

            # ---------------------------------
            # J. Keyboard controls
            # ---------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("a"):

                alert_manager.acknowledge()

            elif key == ord("q"):

                break

    finally:

        stop_event.set()

        cap.release()

        cv2.destroyAllWindows()

        for worker in workers:
            worker.join(timeout=3)

        print("\nReal-time system stopped.")


# =====================================================
# 14. EXECUTION
# =====================================================

if __name__ == "__main__":
    main()
