
import time
import cv2

from config import (
    SCREENSHOT_DIR,
    VIOLENCE_THRESHOLD,
    WEAPON_CONFIDENCE,
    ALERT_COOLDOWN,
    ALERT_CONFIRMATION_COUNT,
    SAVE_INCIDENT_SCREENSHOT
)

from incident_summarizer import save_incident_report

def normalize_weapon_results(yolo_results, confidence_threshold=0.50):
    """
    Convert Ultralytics YOLO/RT-DETR results into
    the format expected by AlertManager.
    """

    detections = []

    if not yolo_results:
        return detections

    result = yolo_results[0]
    boxes = result.boxes

    if boxes is None:
        return detections

    for box in boxes:
        confidence = float(box.conf[0].item())

        if confidence < confidence_threshold:
            continue

        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]

        detections.append({
            "class_name": str(class_name),
            "confidence": confidence
        })

    return detections


def normalize_pose_people_count(pose_results):
    """
    Estimate the number of people from YOLO Pose detections.
    """

    if not pose_results:
        return 0

    result = pose_results[0]

    if result.boxes is None:
        return 0

    return len(result.boxes)


class AlertManager:

    def __init__(self):
        self.last_alert_time = 0.0

        self.violence_count = 0
        self.weapon_count = 0

    def check_alert(
        self,
        frame,
        timesformer_result,
        weapon_detections,
        people_count=0
    ):
        """
        timesformer_result must contain:
        predicted_class, class_name, confidence,
        violence_probability, probabilities

        weapon_detections must be a list of dictionaries.
        """

        current_time = time.time()

        # ======================================
        # 1. READ TIMESFORMER RESULT
        # ======================================

        violence_class = timesformer_result.get(
            "class_name", "Normal"
        )

        violence_confidence = float(
            timesformer_result.get("violence_probability", 0.0)
        )

        violence_detected = (
            violence_class in ("Fighting", "Shooting")
            and violence_confidence >= VIOLENCE_THRESHOLD
        )

        # ======================================
        # 2. CHECK WEAPON DETECTIONS
        # ======================================

        valid_weapons = [
            item for item in weapon_detections
            if float(item.get("confidence", 0.0))
            >= WEAPON_CONFIDENCE
        ]

        weapon_detected = len(valid_weapons) > 0

        # ======================================
        # 3. CONFIRM DETECTIONS
        # ======================================

        if violence_detected:
            self.violence_count += 1
        else:
            self.violence_count = 0

        if weapon_detected:
            self.weapon_count += 1
        else:
            self.weapon_count = 0

        confirmed_violence = (
            self.violence_count >= ALERT_CONFIRMATION_COUNT
        )

        confirmed_weapon = (
            self.weapon_count >= ALERT_CONFIRMATION_COUNT
        )

        if not confirmed_violence and not confirmed_weapon:
            return None

        # ======================================
        # 4. COOLDOWN
        # ======================================

        if current_time - self.last_alert_time < ALERT_COOLDOWN:
            return None

        # ======================================
        # 5. CAPTURE SCREENSHOT
        # ======================================

        screenshot_path = None

        if SAVE_INCIDENT_SCREENSHOT and frame is not None:
            filename = (
                f"incident_{int(current_time * 1000)}.jpg"
            )

            screenshot_path = SCREENSHOT_DIR / filename

            cv2.imwrite(str(screenshot_path), frame)

        # ======================================
        # 6. CREATE INCIDENT REPORT
        # ======================================

        report, report_path = save_incident_report(
            violence_detected=confirmed_violence,
            violence_class=violence_class,
            violence_confidence=violence_confidence,
            weapon_detections=valid_weapons,
            people_count=people_count,
            screenshot_path=screenshot_path
        )

        self.last_alert_time = current_time

        # Reset counters after recording an incident.
        self.violence_count = 0
        self.weapon_count = 0

        # ======================================
        # 7. DISPLAY ALERT
        # ======================================

        print("\n" + "=" * 55)
        print("             SECURITY INCIDENT ALERT")
        print("=" * 55)

        print("Incident ID:", report["incident_id"])
        print("Severity:", report["severity"])
        print("Incident:", report["incident_type"])
        print("Summary:", report["summary"])

        print("Report saved:", report_path)

        if screenshot_path:
            print("Screenshot saved:", screenshot_path)

        print("=" * 55 + "\n")

        return report
