
import json
from datetime import datetime
from pathlib import Path

from config import INCIDENT_DIR


def create_incident_summary(
    violence_detected,
    violence_class,
    violence_confidence,
    weapon_detections,
    people_count
):
    """
    Create a readable summary using the independent model outputs.
    """

    if violence_detected and weapon_detections:
        incident_type = "Violence and Weapon Detected"
        severity = "CRITICAL"

    elif violence_detected:
        incident_type = "Violence Detected"
        severity = "HIGH"

    elif weapon_detections:
        incident_type = "Weapon Detected"
        severity = "HIGH"

    else:
        incident_type = "No Incident"
        severity = "NORMAL"

    if violence_detected:
        violence_text = (
            f"TimeSformer detected {violence_class} "
            f"with confidence {violence_confidence:.2%}."
        )
    else:
        violence_text = "No violence was detected by TimeSformer."

    if weapon_detections:
        weapon_text = "Weapon detections: " + ", ".join(
            f"{item['class_name']} ({item['confidence']:.2%})"
            for item in weapon_detections
        ) + "."
    else:
        weapon_text = "No weapon was detected."

    people_text = f"Pose model detected approximately {people_count} people."

    summary = (
        f"{incident_type}. "
        f"{violence_text} "
        f"{weapon_text} "
        f"{people_text}"
    )

    return {
        "incident_type": incident_type,
        "severity": severity,
        "summary": summary
    }


def save_incident_report(
    violence_detected,
    violence_class,
    violence_confidence,
    weapon_detections,
    people_count,
    screenshot_path=None
):
    """
    Save a complete incident report as JSON.
    """

    timestamp = datetime.now().astimezone()
    timestamp_text = timestamp.isoformat(timespec="seconds")

    report = create_incident_summary(
        violence_detected=violence_detected,
        violence_class=violence_class,
        violence_confidence=violence_confidence,
        weapon_detections=weapon_detections,
        people_count=people_count
    )

    incident_id = timestamp.strftime("INC_%Y%m%d_%H%M%S_%f")

    report.update({
        "incident_id": incident_id,
        "timestamp": timestamp_text,
        "violence_detected": bool(violence_detected),
        "violence_class": violence_class,
        "violence_confidence": float(violence_confidence),
        "weapon_detections": weapon_detections,
        "people_count": int(people_count),
        "screenshot": str(screenshot_path) if screenshot_path else None
    })

    report_path = INCIDENT_DIR / f"{incident_id}.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    return report, report_path
