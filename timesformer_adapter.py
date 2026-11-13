
import torch
import torch.nn as nn
import numpy as np
import cv2

from transformers import (
    TimesformerConfig,
    TimesformerForVideoClassification
)

from config import (
    DEVICE,
    TIMESFORMER_PATH,
    TIMESFORMER_NUM_FRAMES,
    TIMESFORMER_IMAGE_SIZE,
    TIMESFORMER_LABELS
)


class TimeSformerAdapter:

    def __init__(self):

        self.device = torch.device(DEVICE)

        if not TIMESFORMER_PATH.exists():

            raise FileNotFoundError(
                f"TimeSformer checkpoint missing:\n"
                f"{TIMESFORMER_PATH}"
            )

        print("[TimeSformer] Loading checkpoint...")

        # Load your existing trained checkpoint
        checkpoint = torch.load(
            str(TIMESFORMER_PATH),
            map_location="cpu",
            weights_only=False
        )

        if "model_state_dict" not in checkpoint:

            raise KeyError(
                "Expected model_state_dict in TimeSformer checkpoint."
            )

        state_dict = checkpoint["model_state_dict"]

        # Exact configuration from your Kaggle
        # three-model testing cell
        config = TimesformerConfig(
            image_size=224,
            patch_size=16,
            num_frames=8,
            num_channels=3,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            attention_type="divided_space_time",
            num_labels=3
        )

        self.model = TimesformerForVideoClassification(
            config
        )

        self.model.classifier = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(768, 3)
        )

        # Strict loading detects architecture mismatch
        self.model.load_state_dict(
            state_dict,
            strict=True
        )

        self.model.to(self.device)
        self.model.eval()

        # Exact normalization from your notebook
        self.mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        self.std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        print("[TimeSformer] Loaded successfully.")
        print("[TimeSformer] Device:", self.device)

    def preprocess_clip(self, clip):

        if len(clip) != TIMESFORMER_NUM_FRAMES:

            raise ValueError(
                f"Expected {TIMESFORMER_NUM_FRAMES} frames, "
                f"received {len(clip)}"
            )

        processed = []

        for frame in clip:

            # BGR to RGB
            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame = cv2.resize(
                frame,
                (
                    TIMESFORMER_IMAGE_SIZE,
                    TIMESFORMER_IMAGE_SIZE
                )
            )

            frame = frame.astype(
                np.float32
            ) / 255.0

            frame = (
                frame - self.mean
            ) / self.std

            processed.append(frame)

        clip_array = np.array(processed)

        # [8, 224, 224, 3]
        # becomes [8, 3, 224, 224]
        clip_array = np.transpose(
            clip_array,
            (0, 3, 1, 2)
        )

        tensor = torch.tensor(
            clip_array,
            dtype=torch.float32
        ).unsqueeze(0)

        return tensor.to(self.device)

    def predict(self, clip):

        inputs = self.preprocess_clip(clip)

        with torch.inference_mode():

            outputs = self.model(
                pixel_values=inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )[0]

        probabilities = probabilities.cpu().numpy()

        predicted_class = int(
            np.argmax(probabilities)
        )

        violence_probability = float(
            probabilities[1] + probabilities[2]
        )

        return {

            "predicted_class": predicted_class,

            "class_name": TIMESFORMER_LABELS[
                predicted_class
            ],

            "confidence": float(
                probabilities[predicted_class]
            ),

            "violence_probability": violence_probability,

            "probabilities": {
                TIMESFORMER_LABELS[i]: float(
                    probabilities[i]
                )
                for i in range(3)
            }
        }
