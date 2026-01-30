# 'Amusement Park Face Cut-Out' Generator with YOLOv5 and Local Diffusion

This project combines YOLOv5 for real-time face detection with a local text-to-image diffusion pipeline to create custom amusement park-style face cut-out boards. It captures faces, removes the background, and uses text prompts to guide image generation, mimicking the effect of placing your face into a cut-out board at an amusement park.

## Local mode (Phase 1)
- Fully local Stable Diffusion pipeline (SDXL / SDXL-Turbo / SD 1.5) chosen automatically based on GPU VRAM; override with `--model` and `--low-vram`.
- Text-to-image by default; IP-Adapter FaceID can consume the latest detected face crop to improve likeness (see Phase 2 notes below).
- Webcam loop stays responsive (15–30 FPS); generations run asynchronously in a worker thread.
- Expected latency (per prompt, warm): SDXL on ≥12–16 GB GPU ~4–8 s; SDXL-Turbo on 8–12 GB ~2–4 s; SD 1.5 CPU/offload (e.g., P600) 25–40 s at 512px.
- Offline after one-time model download via `python scripts/download_model.py --model <preset>`.
 - IP-Adapter FaceID auto-disables on CPU or when adapter weights cannot load; generation falls back to text-only with a warning.

## Phase 2: FaceID (IP-Adapter)

| Device tier                | Suggested preset              | FaceID status | Notes                                   |
|----------------------------|-------------------------------|---------------|-----------------------------------------|
| CUDA/ROCm ≥12–16 GB        | SDXL + FaceID Plus v2         | On            | Best likeness; higher VRAM/latency      |
| CUDA/ROCm 8–12 GB          | SD 1.5 + FaceID Plus v2       | On            | Lower res; lighter VRAM hit             |
| CUDA/ROCm <8 GB or CPU     | SD 1.5 (text-only fallback)   | Off           | FaceID auto-disables; slow on CPU       |

FaceID uses the most recent detected face crop; if adapters fail to load, the app logs a warning and continues with text-only generation.

## Features

- **Real-Time Face Detection**: Uses YOLOv5 to detect faces from various video sources, including webcams, RTSP streams, and video files.
- **Facial Landmark Mapping**: Identifies and marks key features on each detected face.
- **Real-Time Face Extraction**: Isolates detected faces by making non-face areas transparent, focusing solely on the face.
- **Image Masking**: Prepares faces for further processing by masking them, enabling integration with the generator.
- **Local Diffusion Integration**: Incorporates the isolated faces into custom visuals by passing prompts to a locally hosted diffusion model (SDXL / SDXL-Turbo / SD 1.5), eliminating external API calls.
- **Text Prompt Integration**: Allows users to guide the scene creation by providing prompts, ensuring the detected faces are incorporated into the desired context.

## Example
![ScreenShot1](screenshot1.jpg)
![ScreenShot2](screenshot2.jpg)

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/famesjranko/realtime-dall-e-face-capture.git
   cd realtime-dall-e-face-capture
   ```

2. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download a model once for offline use (optional but recommended)**:
   ```bash
   python scripts/download_model.py --model sdxl  # or sdxl-turbo / sd15 / auto
   ```

## Usage

1. **Running the Script**:
   ```bash
   python detect_face.py --weights <path_to_yolov5_weights> --source <video_source> [--model sdxl|sdxl-turbo|sd15] [--low-vram]
   ```
   - `--weights`: Path to the YOLOv5 weights file.
   - `--source`: Source for video input. Use `0` for the default webcam, or provide a path to a video file or RTSP stream.
   - `--model`: Force a specific preset; default auto-selects based on VRAM/backends.
   - `--low-vram`: Enable extra memory-saving settings and lower resolution.
   - `--face-adapter`: `none`, `faceid-sdxl`, or `faceid-sd15` (auto-selects to match the chosen model); IP-Adapter uses the most recent detected face crop.
   - Prompts are entered interactively; generation runs asynchronously so the webcam stays responsive.

## Changes Made to `utils/datasets.py`

### Problem
The original `LoadStreams` class in `utils/datasets.py` used a basic `cv2.VideoCapture` setup, which had compatibility issues and performance limitations on Windows.

### Solution
To address this, we modified the `LoadStreams` class to try multiple OpenCV backends (`cv2.CAP_DSHOW`, `cv2.CAP_MSMF`, `cv2.CAP_VFW`) when initializing video capture. This approach ensures the best possible performance and compatibility on Windows systems.

### Modified Code
Here's the modified line in the `LoadStreams` class:
```python
# Attempt to open the video source using different backends for better Windows compatibility
cap = None
for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]:
    cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s, backend)
    if cap.isOpened():
        break  # Stop if a backend successfully opens the source
```

## Additional Notes

- Ensure that you have the required OpenCV backends installed on your system. If the webcam or stream does not open, try adjusting the backend order in the script.
- This project assumes a working knowledge of Python, OpenCV, and PyTorch, as well as access to the YOLOv5 model weights.

## Contributing

Feel free to contribute to this project

## Acknowledgements

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics for the face detection model.

### Running the Script:

To start detecting faces and generating cut-out boards, run:

```bash
python detect_face.py --weights <path_to_yolov5_weights> --source <video_source>
```

- `--weights`: Path to the YOLOv5 weights file.
- `--source`: Video input source (use `0` for default webcam, or provide a path to a video file or RTSP stream).

### Interacting with DALL-E:

The script allows you to input prompts for DALL-E to generate images based on the detected faces. The prompts should describe the desired scene (e.g., "a pirate ship with a cartoon character body").

## Modifications to `utils/datasets.py`

### Problem:
The original `LoadStreams` class in `utils/datasets.py` had compatibility and performance issues on Windows when using `cv2.VideoCapture`.

### Solution:
We modified the `LoadStreams` class to try multiple OpenCV backends (`cv2.CAP_DSHOW`, `cv2.CAP_MSMF`, `cv2.CAP_VFW`) for improved performance and compatibility on Windows systems.

### Modified Code:
Here's the updated line in the `LoadStreams` class:

```python
# Attempt to open the video source using different backends for better Windows compatibility
cap = None
for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW]:
    cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s, backend)
    if cap.isOpened():
        break  # Stop if a backend successfully opens the source
```

### Additional Notes:
- Ensure you have the required OpenCV backends installed on your system.
- Adjust the backend order if the webcam or stream does not open correctly.

## Prerequisites

- Working knowledge of Python, OpenCV, and PyTorch.
- Access to the YOLOv5 model weights.
- A valid OpenAI API key and available usage quota.

## Contributing

Contributions are welcome! Please submit pull requests or open issues for any improvements or bugs.

## Acknowledgements

- **YOLOv5** by Ultralytics for face detection.
- **OpenAI** for DALL-E integration.

---

This updated README reflects the amusement park face cut-out concept, detailing how the system works from face detection to generating themed images with DALL-E. Let me know if you’d like any further adjustments!
