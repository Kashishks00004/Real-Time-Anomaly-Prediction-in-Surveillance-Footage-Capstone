import json
import urllib.request
import urllib.error

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


def generate_incident_summary(
    timesformer_result,
    weapon_detections,
    people_count,
    alert_type,
    severity
):
    """
    Generate a natural-language incident summary using a local LLM.
    """

    timesformer_result = timesformer_result or {}
    weapon_detections = weapon_detections or []

    incident_data = {
        "incident_type": alert_type,
        "severity": severity,
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
        "people_count": people_count
    }

    prompt = f"""
You are an incident report writing assistant for a video surveillance system.

Write a short, professional incident summary using ONLY the information
provided below.

Rules:
1. Do not invent events, actions, objects, people, or details.
2. Do not claim that a weapon was detected if the detection list is empty.
3. Do not treat pose detection or people count as proof of violence.
4. Clearly distinguish model predictions from confirmed real-world events.
5. Mention the violence classification and confidence if available.
6. Mention detected weapons and their confidence if available.
7. Mention the number of people detected if available.
8. Keep the summary between 3 and 5 sentences.

Detection information:
{json.dumps(incident_data, indent=2)}

Write the incident summary:
"""

    fallback_summary = (
        f"The surveillance system generated a {severity} alert "
        f"for {alert_type}. "
        f"TimeSformer classification: "
        f"{incident_data['violence_classification']}. "
        f"Number of people detected: {people_count}. "
        "The incident requires human review."
    )

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 180
        }
    }

    try:
        request = urllib.request.Request(
            OLLAMA_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))

        summary = result.get("response", "").strip()

        if summary:
            return {
                "summary": summary,
                "method": "Ollama - " + OLLAMA_MODEL,
                "status": "success"
            }

        return {
            "summary": fallback_summary,
            "method": "Python fallback",
            "status": "empty LLM response"
        }

    except Exception as error:
        print("[WARNING] LLM summarization unavailable:", error)

        return {
            "summary": fallback_summary,
            "method": "Python fallback",
            "status": "LLM unavailable"
        }