# Stereo Camera Calibration Guide

A comprehensive guide for students to learn camera calibration and stereo system rectification using OpenCV and Python.

## Overview

This repository provides step-by-step instructions and example code to:
1. **Calibrate a single camera** - Determine intrinsic parameters (focal length, principal point, lens distortion)
2. **Rectify a stereo system** - Align stereo image pairs for accurate depth estimation

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Camera Calibration](#camera-calibration)
- [Stereo Rectification](#stereo-rectification)
- [Usage Examples](#usage-examples)
- [Resources](#resources)

## Prerequisites

- Python 3.7 or higher
- A camera (webcam or USB camera)
- A printed checkerboard calibration pattern (recommended: 9x6 or 8x6 inner corners)
- For stereo: Two cameras or a stereo camera rig

## Installation

1. Clone this repository:
```bash
git clone https://github.com/doc-ceb/Stereo_Calibration_Guide.git
cd Stereo_Calibration_Guide
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Camera Calibration

Camera calibration is the process of estimating the intrinsic and extrinsic parameters of a camera. See the detailed guide: [Camera Calibration Guide](docs/camera_calibration.md)

**Quick Start:**
```bash
python scripts/calibrate_camera.py --pattern 9x6 --images calibration_images/
```

### What You'll Learn:
- Camera intrinsic parameters (focal length, principal point, distortion coefficients)
- How to capture calibration images
- How to compute calibration matrices
- How to undistort images

## Stereo Rectification

Stereo rectification transforms stereo image pairs so that corresponding points lie on the same horizontal line, simplifying stereo matching. See the detailed guide: [Stereo Rectification Guide](docs/stereo_rectification.md)

**Quick Start:**
```bash
python scripts/stereo_rectify.py --left left_images/ --right right_images/
```

### What You'll Learn:
- Stereo camera calibration
- Computing rectification transforms
- Epipolar geometry
- Creating disparity maps for depth estimation

## Usage Examples

### Example 1: Calibrate a Single Camera
```python
python scripts/calibrate_camera.py \
    --pattern 9x6 \
    --square_size 0.025 \
    --images calibration_images/camera1/
```

### Example 2: Calibrate Stereo Cameras
```python
python scripts/calibrate_stereo.py \
    --pattern 9x6 \
    --square_size 0.025 \
    --left calibration_images/left/ \
    --right calibration_images/right/
```

### Example 3: Rectify and Display Stereo Pair
```python
python scripts/stereo_rectify.py \
    --calibration stereo_calibration.npz \
    --left test_images/left.jpg \
    --right test_images/right.jpg \
    --output rectified/
```

## Project Structure

```
Stereo_Calibration_Guide/
├── docs/                          # Detailed documentation
│   ├── camera_calibration.md      # Camera calibration guide
│   └── stereo_rectification.md    # Stereo rectification guide
├── scripts/                       # Python scripts
│   ├── calibrate_camera.py        # Single camera calibration
│   ├── calibrate_stereo.py        # Stereo system calibration
│   └── stereo_rectify.py          # Stereo rectification
├── sample_data/                   # Sample calibration images
│   └── README.md                  # Instructions for sample data
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Resources

### Calibration Patterns
- [Download Checkerboard Pattern (9x6)](https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png)
- Print the pattern on A4 paper and mount it on a flat, rigid surface

### Learning Resources
- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV Stereo Images Tutorial](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- Zhang, Z. (2000). "A flexible new technique for camera calibration"

## Tips for Success

1. **Use good lighting** - Ensure the calibration pattern is well-lit and clearly visible
2. **Capture diverse poses** - Move the pattern to different positions, angles, and depths
3. **Keep it rigid** - The calibration pattern must be perfectly flat
4. **Minimum images** - Capture at least 15-20 images for reliable calibration
5. **Check reprojection error** - Values below 1.0 pixel indicate good calibration

## Troubleshooting

**Q: Pattern not detected?**
- Ensure the pattern is well-lit and in focus
- Check that the pattern size matches your command-line arguments
- Try adjusting camera exposure

**Q: High reprojection error?**
- Capture more images with better coverage
- Ensure the pattern is flat and not warped
- Check for motion blur in images

**Q: Stereo images don't align after rectification?**
- Verify both cameras were calibrated with the same pattern
- Ensure synchronization between cameras during calibration
- Check that the stereo baseline is correct

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

See [LICENSE](LICENSE) file for details.