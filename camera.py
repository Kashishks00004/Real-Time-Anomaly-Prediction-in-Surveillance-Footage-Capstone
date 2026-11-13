
import cv2
import time

CAMERA_INDEX = 0


def open_camera():

    print("Connecting to DroidCam Video...")

    cap = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_MSMF
    )

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not open DroidCam Video."
        )

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Camera opened. Waiting for valid frames...")

    for attempt in range(30):

        ret, frame = cap.read()

        if (
            ret
            and frame is not None
            and frame.size > 0
            and frame.ndim == 3
            and frame.shape[2] == 3
        ):

            print("SUCCESS: DroidCam frames received!")
            print("Frame dimensions:", frame.shape)

            return cap

        time.sleep(0.2)

    cap.release()

    raise RuntimeError(
        "DroidCam opened but valid frames were not received."
    )


def test_camera():

    cap = open_camera()

    try:

        while True:

            ret, frame = cap.read()

            if not ret or frame is None:
                print("Frame reading failed.")
                break

            cv2.imshow(
                "IPHONE - DROIDCAM",
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    test_camera()
