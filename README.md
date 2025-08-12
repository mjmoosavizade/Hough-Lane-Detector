# Hough Lane Detector

This C++ project detects road lane lines in images using the Hough Transform with OpenCV.

Pipeline highlights:
- Grayscale + Gaussian blur
- Canny edge detection
- Region-of-interest (trapezoid) mask to focus on the road
- Probabilistic Hough Transform
- Slope-based filtering and left/right lane averaging

The app auto-detects headless environments (no X display) and will skip GUI windows if needed.

## Prerequisites
- CMake >= 3.10
- OpenCV (C++ library)

## Build Instructions
1. Install OpenCV (e.g., `sudo apt install libopencv-dev` on Ubuntu).
2. Build:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Usage
Place your road images in the `data` folder. Run the program from the `build` directory:
```bash
./HoughLaneDetector            # uses ../data by default
# or specify a custom folder
./HoughLaneDetector /absolute/or/relative/path/to/images
```

Outputs are saved alongside the inputs with the `_lanes.png` suffix (e.g., `data/000010_lanes.png`).
