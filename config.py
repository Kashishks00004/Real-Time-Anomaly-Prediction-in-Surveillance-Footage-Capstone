
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

# Model checkpoints
TIMESFORMER_PATH = MODEL_DIR / "timesformer" / "timesformerbest.pt"
POSE_PATH = MODEL_DIR / "pose" / "yoloposebest.pt"
WEAPON_PATH = MODEL_DIR / "weapon" / "yoloweaponbest.pt"

# Output folders
OUTPUT_DIR = BASE_DIR / "outputs"
INCIDENT_DIR = OUTPUT_DIR / "incidents"

INCIDENT_DIR.mkdir(parents=True, exist_ok=True)

# Hardware
DEVICE = "cpu"
TORCH_THREADS = 6
TORCH_INTEROP_THREADS = 2

# Camera
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# TimeSformer - matching training configuration
TIMESFORMER_NUM_FRAMES = 8
TIMESFORMER_IMAGE_SIZE = 224
TIMESFORMER_CLIP_SECONDS = 5.0
TIMESFORMER_STRIDE_SECONDS = 2.5

# Detection confidence
POSE_CONFIDENCE = 0.25
WEAPON_CONFIDENCE = 0.50

# Inference scheduling
DETECTION_INTERVAL = 0.5
RESULT_MAX_AGE = 5.0
ALERT_COOLDOWN = 5.0

# TimeSformer class labels
TIMESFORMER_LABELS = {
    0: "Normal",
    1: "Fighting",
    2: "Shooting"
}
