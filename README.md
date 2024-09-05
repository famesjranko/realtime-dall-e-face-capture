# YOLOv5 Face Detection and Masking with DALL-E Integration

This project demonstrates a custom script for face detection using YOLOv5 with additional functionality to mask detected regions and generate custom images using OpenAI's DALL-E. The script is designed to work efficiently with webcam inputs, especially on Windows systems, by integrating multiple OpenCV backend options for better compatibility.

## Features

- **Face Detection**: Uses YOLOv5 to detect faces in real-time from video sources like webcams, RTSP streams, or video files.
- **Landmark Detection**: Draws facial landmarks for each detected face.
- **Masking**: Masks detected faces and processes images for further editing or generating new content using DALL-E.
- **DALL-E Integration**: Sends masked images to OpenAI's DALL-E API to generate custom images based on user-defined prompts.

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

3. **Set up OpenAI API key**:
   - Obtain an API key from [OpenAI](https://openai.com/).
   - Set your API key in the script (`openai.api_key = 'your_openai_api_key'`).

## Usage

1. **Running the Script**:
   ```bash
   python detect_face.py
   ```
   - `--weights`: Path to the YOLOv5 weights file.
   - `--source`: Source for video input. Use `0` for the default webcam, or provide a path to a video file or RTSP stream.

2. **Interacting with DALL-E**:
   - The script allows you to input prompts for DALL-E to generate images based on the detected and masked areas of the video feed.

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
- The DALL-E integration requires a valid OpenAI API key and available usage quota.

## Contributing

Feel free to contribute to this project

## Acknowledgements

- [YOLOv5](https://github.com/ultralytics/yolov5) by Ultralytics for the face detection model.
- [OpenAI](https://openai.com/) for the DALL-E API integration.
