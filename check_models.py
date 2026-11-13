
import torch
from ultralytics import YOLO

from config import (
    TIMESFORMER_PATH,
    POSE_PATH,
    WEAPON_PATH,
    DEVICE,
    TORCH_THREADS,
    TORCH_INTEROP_THREADS
)

torch.set_num_threads(TORCH_THREADS)
torch.set_num_interop_threads(TORCH_INTEROP_THREADS)


def check_file(path, name):

    print(f"\nChecking {name}:")
    print(path)

    if not path.exists():
        print("ERROR: File does not exist.")
        return False

    if path.stat().st_size == 0:
        print("ERROR: File is empty.")
        return False

    print(
        "File size:",
        round(path.stat().st_size / (1024 * 1024), 2),
        "MB"
    )

    return True


def main():

    for name, path in [
        ("TimeSformer", TIMESFORMER_PATH),
        ("Pose", POSE_PATH),
        ("Weapon", WEAPON_PATH)
    ]:

        if not check_file(path, name):
            return

    print("\nLoading TimeSformer checkpoint...")

    try:

        checkpoint = torch.load(
            TIMESFORMER_PATH,
            map_location="cpu",
            weights_only=False
        )

        print(
            "TimeSformer checkpoint keys:",
            list(checkpoint.keys())
        )

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "model_state_dict is missing."
            )

        print("TimeSformer checkpoint readable.")

    except Exception as error:

        print("TimeSformer checkpoint ERROR:", error)
        return

    print("\nLoading Pose model...")

    try:

        pose = YOLO(str(POSE_PATH))

        print("Pose model loaded.")
        print("Pose classes:", pose.names)

    except Exception as error:

        print("Pose model ERROR:", error)
        return

    print("\nLoading Weapon model...")

    try:

        weapon = YOLO(str(WEAPON_PATH))

        print("Weapon model loaded.")
        print("Weapon classes:", weapon.names)

    except Exception as error:

        print("Weapon model ERROR:", error)
        return

    print("\nALL THREE CHECKPOINTS PASSED LOADING TEST.")


if __name__ == "__main__":
    main()
