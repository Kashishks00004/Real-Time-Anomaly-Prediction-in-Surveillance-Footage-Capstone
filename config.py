
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ==========================================
# MODEL PATHS
# ==========================================

MODEL_DIR = BASE_DIR / "models"

TIMESFORMER_PATH = MODEL_DIR / "timesformer" / "timesformerbest.pt"
POSE_PATH = MODEL_DIR / "pose" / "yoloposebest.pt"
WEAPON_PATH = MODEL_DIR / "weapon" / "yoloweaponbest.pt"

# ==========================================
# OUTPUT DIRECTORIES
# ==========================================

OUTPUT_DIR = BASE_DIR / "outputs"
INCIDENT_DIR = OUTPUT_DIR / "incidents"
SCREENSHOT_DIR = OUTPUT_DIR / "screenshots"

INCIDENT_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# HARDWARE
# ==========================================

DEVICE = "cpu"
TORCH_THREADS = 6
TORCH_INTEROP_THREADS = 2

# ==========================================
# CAMERA
# ==========================================

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ==========================================
# TIMESFORMER
# ==========================================

TIMESFORMER_NUM_FRAMES = 8
TIMESFORMER_IMAGE_SIZE = 224

TIMESFORMER_CLIP_SECONDS = 5.0
TIMESFORMER_STRIDE_SECONDS = 2.5

TIMESFORMER_LABELS = {
    0: "Normal",
    1: "Fighting",
    2: "Shooting"
}

# ==========================================
# DETECTION CONFIDENCE
# ==========================================

POSE_CONFIDENCE = 0.25
WEAPON_CONFIDENCE = 0.50

# Initial threshold; validate it using your test videos.
VIOLENCE_THRESHOLD = 0.65

# ==========================================
# INFERENCE SCHEDULING
# ==========================================

DETECTION_INTERVAL = 0.5
RESULT_MAX_AGE = 5.0

# ==========================================
# ALERT SETTINGS
# ==========================================

ALERT_COOLDOWN = 5.0
ALERT_CONFIRMATION_COUNT = 2

# ==========================================
# INCIDENT SETTINGS
# ==========================================

SAVE_INCIDENT_SCREENSHOT = True
SAVE_INCIDENT_JSON = True
