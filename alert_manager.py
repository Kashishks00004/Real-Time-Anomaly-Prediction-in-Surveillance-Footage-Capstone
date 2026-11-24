
import cv2
import json
import time
import threading
import winsound

from pathlib import Path
from datetime import datetime
from llm_summarizer import generate_incident_summary


# ==========================================
# ALERT SETTINGS
# ==========================================

try:
    from config import (
        OUTPUT_DIR,
        INCIDENT_DIR,
        SCREENSHOT_DIR,
        VIOLENCE_THRESHOLD,
        WEAPON_CONFIDENCE,
        WEAPON_CONFIRMATION_COUNT,
        ALERT_COOLDOWN,
        ALERT_CONFIRMATION_COUNT,
        SAVE_INCIDENT_SCREENSHOT
    )

except ImportError:
    OUTPUT_DIR = Path("outputs")
    INCIDENT_DIR = OUTPUT_DIR / "incidents"
    SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"

    VIOLENCE_THRESHOLD = 0.65
    WEAPON_CONFIDENCE = 0.80
    WEAPON_CONFIRMATION_COUNT = 3
    ALERT_COOLDOWN = 5.0
    ALERT_CONFIRMATION_COUNT = 2
    SAVE_INCIDENT_SCREENSHOT = True


INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# ALERT MANAGER
# ==========================================

class AlertManager:

    def __init__(self):

        self.last_alert_time = 0

        self.violence_count = 0
        self.weapon_count = 0

        self.current_alert = None
        self.alert_expiry = 0

        self.latest_incident = None

        self.sound_stop_event = threading.Event()
        self.lock = threading.Lock()


    # ======================================
    # SOUND ALERT
    # ======================================

    def play_sound(self, severity):

        self.sound_stop_event.clear()

        def sound_worker():

            try:

                if severity == "CRITICAL":
                    pattern = [
                        (1500, 300), (1000, 300),
                        (1500, 300), (1000, 300)
                    ]

                elif severity == "HIGH":
                    pattern = [
                        (1200, 350), (1200, 350),
                        (1200, 350)
                    ]

                else:
                    pattern = [(800, 400), (800, 400)]

                for frequency, duration in pattern:

                    if self.sound_stop_event.is_set():
                        break

                    winsound.Beep(frequency, duration)

                    if self.sound_stop_event.wait(0.15):
                        break

            except Exception as error:
                print("Sound alert error:", error)

        threading.Thread(
            target=sound_worker,
            daemon=True
        ).start()


    # ======================================
    # SILENCE SOUND
    # ======================================

    def acknowledge(self):

        self.sound_stop_event.set()

        print("[INFO] Alert sound acknowledged.")


    # ======================================
    # GET ACTIVE ALERT
    # ======================================

    def get_active_alert(self):

        with self.lock:

            if time.time() <= self.alert_expiry:
                return self.current_alert.copy() if self.current_alert else None

            return None


    # ======================================
    # GET LATEST INCIDENT FOR SCREEN
    # ======================================

    def get_latest_incident(self):

        with self.lock:

            if self.latest_incident is None:
                return None

            return self.latest_incident.copy()


    # ======================================
    # SAVE INCIDENT AND GENERATE SUMMARY
    # ======================================

    def save_incident(
        self,
        frame,
        alert_type,
        severity,
        timesformer_result,
        weapon_detections,
        people_count
    ):

        timestamp = datetime.now()

        incident_id = timestamp.strftime(
            "INC_%Y%m%d_%H%M%S_%f"
        )

        screenshot_path = None

        if SAVE_INCIDENT_SCREENSHOT and frame is not None:

            screenshot_path = SCREENSHOT_DIR / f"{incident_id}.jpg"

            cv2.imwrite(str(screenshot_path), frame)

        report_path = INCIDENT_DIR / f"{incident_id}.json"

        incident_report = {
            "incident_id": incident_id,
            "timestamp": timestamp.isoformat(),
            "incident_type": alert_type,
            "severity": severity,
            "timesformer": timesformer_result,
            "weapon_detections": weapon_detections,
            "people_count": people_count,
            "llm_summary": "Generating AI summary...",
            "llm_model": "Ollama - qwen2.5:3b",
            "llm_status": "pending",
            "screenshot": str(screenshot_path) if screenshot_path else None
        }

        # Save initial report

        with open(report_path, "w", encoding="utf-8") as file:

            json.dump(
                incident_report,
                file,
                indent=4,
                ensure_ascii=False,
                default=str
            )

        # Make incident available to screen

        with self.lock:

            self.latest_incident = {
                "incident_id": incident_id,
                "timestamp": timestamp.strftime("%H:%M:%S"),
                "incident_type": alert_type,
                "severity": severity,
                "summary": "Generating AI summary...",
                "llm_status": "pending",
                "llm_model": "Ollama - qwen2.5:3b",

                "violence_classification": timesformer_result.get(
                    "class_name", "Unavailable"
                ),

                "violence_confidence": timesformer_result.get(
                    "confidence", None
                ),

                "violence_probability": timesformer_result.get(
                    "violence_probability", None
                ),

                "weapon_detections": weapon_detections,
                "people_count": people_count,
                "screenshot": str(screenshot_path) if screenshot_path else None
            }


        # ==================================
        # BACKGROUND LLM SUMMARY
        # ==================================

        def generate_summary_in_background():

            try:

                llm_result = generate_incident_summary(
                    timesformer_result,
                    weapon_detections,
                    people_count,
                    alert_type,
                    severity
                )

                summary = llm_result.get(
                    "summary",
                    "Summary could not be generated."
                )

                method = llm_result.get(
                    "method",
                    "Ollama"
                )

                status = llm_result.get(
                    "status",
                    "unknown"
                )

            except Exception as error:

                print("[ERROR] LLM summary generation failed:", error)

                summary = (
                    "AI summary generation failed. "
                    "Please review the incident manually."
                )

                method = "Python fallback"
                status = "error"


            # Update JSON report

            incident_report["llm_summary"] = summary
            incident_report["llm_model"] = method
            incident_report["llm_status"] = status

            try:

                with open(report_path, "w", encoding="utf-8") as file:

                    json.dump(
                        incident_report,
                        file,
                        indent=4,
                        ensure_ascii=False,
                        default=str
                    )

            except Exception as error:

                print("[ERROR] Could not update incident report:", error)


            # Update live screen data

            with self.lock:

                if (
                    self.latest_incident is not None
                    and self.latest_incident["incident_id"] == incident_id
                ):

                    self.latest_incident["summary"] = summary
                    self.latest_incident["llm_model"] = method
                    self.latest_incident["llm_status"] = status


            print("\n[INFO] LLM INCIDENT SUMMARY")
            print(summary)
            print("[INFO] Updated report:", report_path)


        threading.Thread(
            target=generate_summary_in_background,
            daemon=True
        ).start()

        print("[INFO] Incident report created:", report_path)
        print("[INFO] LLM summarization running in background.")


    # ======================================
    # CHECK ALERT CONDITIONS
    # ======================================

    def check_alert(
        self,
        frame,
        timesformer_result,
        weapon_detections,
        people_count=0
    ):

        timesformer_result = timesformer_result or {}
        weapon_detections = weapon_detections or []


        # ==================================
        # TIMESFORMER RESULT
        # ==================================

        class_name = str(
            timesformer_result.get(
                "class_name",
                timesformer_result.get("prediction", "Normal")
            )
        ).strip()

        violence_probability = float(
            timesformer_result.get("violence_probability", 0.0)
        )

        violence_detected = (
            class_name.lower() in ["fighting", "shooting"]
            and violence_probability >= VIOLENCE_THRESHOLD
        )


        # ==================================
        # WEAPON RESULT
        # ==================================

        valid_weapons = []

        for detection in weapon_detections:

            confidence = float(detection.get("confidence", 0))

            class_name = str(
                detection.get("class_name", "")
            ).strip().lower()

            if (
                class_name in ["gun", "knife"]
                and confidence >= WEAPON_CONFIDENCE
            ):

                valid_weapons.append(detection)

        weapon_detected = len(valid_weapons) > 0


        # ==================================
        # CONFIRMATION COUNTERS
        # ==================================

        self.violence_count = (
            self.violence_count + 1 if violence_detected else 0
        )

        self.weapon_count = (
            self.weapon_count + 1 if weapon_detected else 0
        )

        confirmed_violence = (
            self.violence_count >= ALERT_CONFIRMATION_COUNT
        )

        confirmed_weapon = (
            self.weapon_count >= WEAPON_CONFIRMATION_COUNT
        )


        # ==================================
        # DETERMINE INCIDENT
        # ==================================

        if confirmed_violence and confirmed_weapon:

            alert_type = "VIOLENCE AND WEAPON DETECTED"
            severity = "CRITICAL"

        elif confirmed_violence:

            alert_type = "VIOLENCE DETECTED"
            severity = "HIGH"

        elif confirmed_weapon:

            alert_type = "WEAPON DETECTED"
            severity = "WARNING"

        else:
            return None


        now = time.time()


        # ==================================
        # COOLDOWN
        # ==================================

        if now - self.last_alert_time < ALERT_COOLDOWN:
            return None

        self.last_alert_time = now


        alert = {
            "type": alert_type,
            "severity": severity,
            "time": datetime.now().strftime("%H:%M:%S"),
            "violence_probability": violence_probability,
            "weapon_count": len(valid_weapons),
            "people_count": people_count
        }


        with self.lock:

            self.current_alert = alert
            self.alert_expiry = now + 8


        # ==================================
        # PLAY SOUND
        # ==================================

        self.play_sound(severity)


        # ==================================
        # SAVE INCIDENT AND GENERATE SUMMARY
        # ==================================

        self.save_incident(
            frame,
            alert_type,
            severity,
            timesformer_result,
            valid_weapons,
            people_count
        )


        # ==================================
        # RESET COUNTERS
        # ==================================

        self.violence_count = 0
        self.weapon_count = 0

        return alert
