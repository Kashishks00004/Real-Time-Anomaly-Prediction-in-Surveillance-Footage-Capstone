# Real-Time Violence and Weapon Detection System Using Deep Learning

## 1. Project Overview

This project presents a real-time video surveillance system designed to detect potentially violent activities and weapons from live camera footage using deep learning techniques.

The system uses three independent deep learning models to analyse video frames and provide visual alerts when suspicious activities are detected.

The project consists of:

- **TimeSformer:** Detects violence-related activities and classifies them into Normal, Fighting, and Shooting.
- **YOLO Pose Estimation:** Detects people and identifies their body keypoints to visualize human posture and movement.
- **YOLO Weapon Detection:** Detects weapons such as guns and other weapon categories included in the trained dataset.

All three models operate independently on the incoming video. Their individual results are displayed together in the application. The system does not use feature fusion or an additional classification model to combine their predictions.

An alert management module monitors the model predictions and generates alerts when the configured detection conditions are satisfied. An AI-based incident summarization module uses a locally running Large Language Model (LLM) through Ollama to generate a textual summary of detected incidents.

## 2. Objectives

- Develop a real-time surveillance system using deep learning.
- Detect violence-related activities in video footage.
- Identify weapons using a trained YOLO object detection model.
- Detect people and visualize their body poses using YOLO Pose.
- Display the outputs of all three models in a single application.
- Generate visual and audio alerts for detected incidents.
- Save incident screenshots and corresponding JSON reports.
- Generate AI-based incident summaries using a locally hosted LLM.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| PyTorch | Deep learning framework |
| TimeSformer | Video-based violence classification |
| YOLO | Weapon detection |
| YOLO Pose | Human pose estimation |
| OpenCV | Video capture and image processing |
| NumPy | Numerical operations |
| Ollama | Local LLM execution |
| Qwen 2.5 3B | AI-based incident summarization |
| Threading and Queues | Parallel model execution |

## 4. System Architecture

The system follows a parallel processing approach.

1. The live camera captures video frames.
2. The incoming frames are passed to three independent processing pipelines.
3. TimeSformer analyses video clips and predicts the activity category.
4. YOLO Pose detects people and their body keypoints.
5. YOLO Weapon identifies weapons present in the frames.
6. The individual model outputs are displayed on the live video.
7. The Alert Manager checks the violence and weapon detection results.
8. When the configured alert conditions are satisfied, an alert is generated.
9. The system saves incident information, including screenshots and JSON reports.
10. The LLM Summarizer generates a textual incident summary in the background.

## 5. Project Folder Structure

The project directory is organized as follows:

```text
RealtimeCapstone/
│
├── models/
│   ├── timesformer/
│   │   └── timesformerbest.pt
│   │
│   ├── pose/
│   │   └── yoloposebest.pt
│   │
│   └── weapon/
│       └── yoloweaponbest.pt
│
├── outputs/
│   ├── incidents/
│   └── screenshots/
│
├── config.py
├── camera.py
├── hardware_check.py
├── model_check.py
├── timesformer_adapter.py
├── alert_manager.py
├── llm_summarizer.py
├── parallel_main.py
│
└── README.md
```

**Note:** Keep all Python files and model files in their respective directories. The model filenames and folder names must match the paths specified in `config.py`.

## 6. System Requirements

### Hardware Requirements

- Windows-based computer.
- Intel or AMD processor.
- Webcam or external camera.
- Minimum 8 GB RAM recommended.
- Sufficient storage for trained models and incident records.

### Software Requirements

- Python 3.10 or a compatible Python version.
- Visual Studio Code or another Python IDE.
- PyTorch.
- OpenCV.
- Ultralytics YOLO.
- Ollama (for AI-based incident summarization).

The system can execute using CPU inference. A CUDA-compatible GPU is not mandatory.

## 7. Installation and Setup

### Step 1: Clone or Download the Project

Download the project and open the project folder in Visual Studio Code.

Open the terminal inside the project directory:

```powershell
cd C:\Users\Kashi\Desktop\CAPSTONE\RealtimeCapstone
```

### Step 2: Create a Virtual Environment

Create a Python virtual environment:

```powershell
python -m venv venv
```

Activate the environment:

```powershell
.\venv\Scripts\activate
```

### Step 3: Install Required Dependencies

Install the required Python packages:

```powershell
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install numpy
pip install timm
pip install requests
```

If a `requirements.txt` file is available, dependencies can alternatively be installed using:

```powershell
pip install -r requirements.txt
```

### Step 4: Add the Trained Models

Place the trained model files in their corresponding folders.

**TimeSformer:**

```text
models/timesformer/timesformerbest.pt
```

**YOLO Pose:**

```text
models/pose/yoloposebest.pt
```

**YOLO Weapon:**

```text
models/weapon/yoloweaponbest.pt
```

Ensure that all three model files are present before executing the application.

### Step 5: Configure the System

Open `config.py` and verify the model paths, camera settings, detection thresholds, and processing configurations.

Important configurations include:

- Camera index: `0`
- Frame resolution: `1280 × 720`
- TimeSformer input frames: `8`
- TimeSformer image size: `224 × 224`
- Weapon confidence threshold: `0.70`
- Violence probability threshold: `0.65`
- Device: `cpu`

These values can be modified according to the system and detection requirements.

## 8. Camera Setup

The system supports webcam input.

If using an iPhone as a camera, install and configure DroidCam on both the phone and computer.

1. Connect the phone and computer to the same Wi-Fi network.
2. Open DroidCam on the phone.
3. Open the DroidCam client on the computer.
4. Establish the camera connection.
5. Verify that the camera is accessible through OpenCV.

The camera is initialized through `camera.py`.

Ensure the camera is connected before starting the real-time detection application.

## 9. Ollama Setup for AI Incident Summarization

Ollama is used to execute the local language model responsible for generating incident summaries.

### Step 1: Install Ollama

Download and install Ollama from:

https://ollama.com/download

### Step 2: Download the Qwen Model

Open PowerShell and execute:

```powershell
ollama pull qwen2.5:3b
```

Wait until the model download is completed.

### Step 3: Verify the Model

Execute:

```powershell
ollama list
```

Ensure that `qwen2.5:3b` appears in the installed models list.

### Step 4: Verify the Ollama Server

Execute:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

If the command returns the available model information, the Ollama server is responding.

If the server is not running, start it using:

```powershell
ollama serve
```

Keep the Ollama server running while using the application.

**Note:** Ollama is required for LLM-generated incident summaries. The detection models can operate independently of the LLM. If Ollama is unavailable, the summarizer can use its fallback summary.

## 10. How to Execute the Project

Follow the execution sequence below whenever you want to run the real-time surveillance system.

### Step 1: Open the Project

Open the `RealtimeCapstone` folder in Visual Studio Code.

Activate the virtual environment:

```powershell
.\venv\Scripts\activate
```

### Step 2: Check Hardware Compatibility

Execute:

```powershell
python hardware_check.py
```

This checks the available hardware and verifies the execution environment.

Ensure that the hardware check completes successfully.

### Step 3: Check the Deep Learning Models

Execute:

```powershell
python model_check.py
```

This verifies that the required trained models can be loaded.

Ensure that TimeSformer, YOLO Pose, and YOLO Weapon load successfully before proceeding.

### Step 4: Start Ollama

Ensure that Ollama is installed, the `qwen2.5:3b` model is downloaded, and its server is running.

If Ollama is already running in the background, no additional command is required.

### Step 5: Connect the Camera

Connect the webcam or configure DroidCam before starting the application.

Ensure that the camera is accessible.

### Step 6: Execute the Real-Time Application

Run:

```powershell
python parallel_main.py
```

The application will:

1. Initialize the camera.
2. Load the three trained deep learning models.
3. Start independent processing threads.
4. Capture live video frames.
5. Perform violence classification, pose estimation, and weapon detection.
6. Display the detection results on the live video.
7. Monitor model outputs for potential incidents.
8. Generate alerts when the configured conditions are satisfied.
9. Save incident information and generate AI-based summaries.

## 11. Real-Time Application Output

The application displays the results of all three models on the live camera feed.

### TimeSformer Output

Displays the predicted activity category:

- Normal
- Fighting
- Shooting

The classification confidence and violence probability are also used by the alert management system.

### YOLO Pose Output

- Detects people in the video.
- Identifies body keypoints.
- Displays the detected human pose and skeleton.

### YOLO Weapon Output

- Identifies weapons present in the video.
- Displays bounding boxes around detected weapons.
- Shows the corresponding weapon class and detection confidence.

### Alert System

The Alert Manager evaluates the TimeSformer and weapon detection results.

The configured alert conditions are:

- **Violence Alert:** Triggered when the predicted activity is Fighting or Shooting and the violence probability satisfies the configured threshold.
- **Weapon Alert:** Triggered when a weapon is detected with sufficient confidence.
- **Critical Alert:** Triggered when both violence and weapon detection conditions are satisfied.

The application provides visual and audio notifications when an incident is detected.

### AI Incident Summary

After an alert is generated:

1. Incident information is collected.
2. A screenshot and JSON report are saved.
3. The incident information is sent to the local Qwen 2.5 3B model through Ollama.
4. The LLM generates a textual summary.
5. The summary is displayed in the application's incident summary panel.

The summary is generated asynchronously, allowing the live detection process to continue.

## 12. Incident Storage

Detected incidents are stored inside the `outputs` directory.

```text
outputs/
│
├── incidents/
│   └── incident_report.json
│
└── screenshots/
    └── incident_screenshot.jpg
```

The JSON report contains information such as:

- Incident ID.
- Timestamp.
- Alert type and severity.
- TimeSformer prediction.
- Classification confidence.
- Violence probability.
- Weapon detection details.
- Number of detected people.
- AI-generated incident summary.

The screenshot provides a visual record of the incident at the time of detection.

## 13. Application Controls

| Key | Function |
|---|---|
| A | Acknowledge or silence the active alert sound |
| Q | Quit the real-time application |

## 14. Troubleshooting

### Camera Not Opening

- Ensure that the webcam or DroidCam is connected.
- Verify the camera index in `config.py`.
- Close other applications currently using the camera.
- Check the camera initialization in `camera.py`.

### Model Loading Error

- Verify that all three `.pt` files are present.
- Check the model paths in `config.py`.
- Ensure that the required Python dependencies are installed.

### Ollama Connection Error

- Check whether Ollama is installed.
- Verify that `qwen2.5:3b` is downloaded.
- Check whether the Ollama server is running.
- Test the server using:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

### AI Summary Not Appearing

- Ensure that Ollama is running.
- Verify that the Qwen model is installed.
- Check the terminal for any summarizer errors.
- Allow some time for the background summary generation to complete.

### Low Processing Speed

- Close unnecessary background applications.
- Verify the CPU thread settings in `config.py`.
- Reduce the camera resolution if necessary.
- Keep in mind that running three deep learning models on a CPU can increase processing time.

## 15. Important Notes

- All three detection models operate independently.
- TimeSformer is responsible for violence-related activity classification.
- YOLO Pose is responsible for human pose estimation.
- YOLO Weapon is responsible for weapon detection.
- Pose estimation is displayed as supporting visual information and does not determine the violence classification.
- The alert system uses configured confidence thresholds and confirmation conditions.
- The LLM generates a textual summary based on the available incident information; it does not independently verify whether a crime has occurred.
- Detection performance depends on camera quality, lighting, model accuracy, and hardware capabilities.

## 16. Conclusion

The proposed system demonstrates the application of deep learning in intelligent video surveillance through real-time violence classification, human pose estimation, and weapon detection.

By integrating independent detection models with an alert management system and local LLM-based incident summarization, the project aims to provide a practical approach to identifying and reporting potentially suspicious activities in surveillance footage.

The system combines real-time video analysis, automated alerts, incident storage, and AI-generated summaries to support more efficient surveillance monitoring.
