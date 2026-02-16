# Stereo Camera Calibration Guide

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![OpenCV 4.13](https://img.shields.io/badge/OpenCV-4.13-green.svg)](https://opencv.org/)

A comprehensive Jupyter notebook for **stereo camera calibration** using OpenCV and Python. Features a guided capture system with pre-defined target poses for optimal calibration quality.

![Stereo Rectification Demo](./figures/stereo_rectification_demo.gif)
*Real-time stereo rectification with epipolar line validation*

---

## 🎯 Features

- ✅ **Guided Capture System**: Pre-defined target poses ensure optimal spatial coverage and angle diversity
- ✅ **Physics-Based Calibration**: Constraints for realistic camera parameters (square pixels, zero tangental distortion)
- ✅ **Monocular Calibration**: Individual camera intrinsics and distortion coefficients
- ✅ **Stereo Calibration**: Relative pose (R, T) and epipolar geometry (E, F matrices)
- ✅ **Stereo Rectification**: Horizontally align epipolar lines for efficient stereo matching
- ✅ **Live Demo**: Real-time rectified view with corner detection and quality metrics

---

## 📋 Hardware Requirements

### Cameras
- **2× USB webcams** (tested with Logitech C270, C920, C930e)
- Recommended: Same model for both cameras
- Recommended resolution (for block matching): 640×480
- Fixed focus

### Rigid Mounting
- Left-Right camera mounting: [example](https://a.co/d/0dJ8drms "Amazon")

### Calibration Pattern
- **Chessboard**: 9×6 inner corners (10×7 total squares)
- **Square size**: 24-25mm (measure after printing!)
- **Material**: High-contrast printed pattern on rigid surface
  - Recommended: Foam board, acrylic, or rigid cardboard
  - Avoid: Paper alone (warps easily)

### Printing Requirements
- **Paper**: Letter size or A4
- **DPI**: 300 minimum
- **Scaling**: 100% (actual size)
- **Verification**: Measure printed squares with vernier caliper or ruler

---

## 🔧 Software Requirements

### Python Environment
```bash
# Python 3.11+
python --version  # Should be 3.13
```

### Dependencies

**Tested versions:**
```bash
pip install -r requirements.txt 
```
or manually

```bash
pip install opencv-python==4.12.0.88 opencv_contrib_python==4.13.0.92 numpy==2.4.2 matplotlib==3.10.8 jupyter
```
---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone git@github.com:doc-ceb/Stereo_Calibration_Guide.git
cd Stereo_Calibration_Guide
```

### 2. Print Calibration Pattern
```bash
jupyter notebook stereo_calibration_guide.ipynb
```

Run **Section 0** to generate `calibration_chessboard_Letter.png`, then:
1. Print at **100% scale** (no fit-to-page!)
2. Mount on rigid surface (foam board recommended)
3. **Measure actual square size** and update `SQUARE_SIZE_MM` if different

### 3. Detect Cameras
Run **Section 1** to list available cameras:
```python
# Update these indices based on your setup
cam_left_idx = 2   # Change to your left camera index
cam_right_idx = 0  # Change to your right camera index
```

### 4. Calibrate
Run cells sequentially through **Section 5**:
- Section 2: Generate guided target poses
- Section 3: Capture calibration images (follow on-screen overlay)
- Section 4: Monocular calibration (left & right cameras)
- Section 5: Stereo calibration & rectification

### 5. Validate
Run **Section 6** for live rectified demo:
- Verify horizontal epipolar lines
- Check vertical alignment (< 1px offset)
- Save results with 's' key

---

## 📊 Calibration Quality Metrics

### Monocular Calibration
| Quality | RMS Error | Per-Image Max | Assessment |
|---------|-----------|---------------|------------|
| **Excellent** | < 0.3 px | < 0.5 px | Production ready |
| **Good** | < 0.5 px | < 0.8 px | Acceptable for most applications |
| **Acceptable** | < 1.0 px | < 1.5 px | May need refinement |
| **Poor** | > 1.0 px | > 1.5 px | Recalibrate with better data |

### Stereo Calibration
| Quality | RMS Error | Vertical Offset | Rotation | Assessment |
|---------|-----------|-----------------|----------|------------|
| **Excellent** | < 0.3 px | < 0.5 px | < 2° | Excellent rectification |
| **Good** | < 0.5 px | < 1.0 px | < 5° | Good for stereo matching |
| **Acceptable** | < 1.0 px | < 2.0 px | < 10° | May have alignment issues |
| **Poor** | > 1.0 px | > 2.0 px | > 10° | Recalibrate |

---

## 📁 Output Files

### Automatically Generated

| File | Description | Size |
|------|-------------|------|
| `calibration_chessboard_Letter.png` | Printable calibration pattern | ~100 KB |
| `left_camera_params.npz` | Left camera intrinsics (K, D) | ~2 KB |
| `right_camera_params.npz` | Right camera intrinsics (K, D) | ~2 KB |
| `stereo_params.npz` | Complete stereo calibration | ~500 KB |
| `stereo_params_dataset/` | Captured images (optional) | ~10-50 MB |

### Data Structure (stereo_params.npz)

```python
params = np.load('stereo_params.npz', allow_pickle=True)

# Intrinsics
K_left, D_left    # Left camera matrix (3×3) & distortion (5,)
K_right, D_right  # Right camera matrix (3×3) & distortion (5,)

# Stereo extrinsics
R, T              # Rotation (3×3) & translation (3,1) from R to L
E, F              # Essential (3×3) & fundamental (3×3) matrices

# Rectification
R1, R2            # Rectification rotations (3×3) for each camera
P1, P2            # Projection matrices (3×4) after rectification
Q                 # Disparity-to-depth matrix (4×4)
map1_L, map2_L    # Remap tables for fast undistortion+rectification
map1_R, map2_R

# Metadata
baseline_mm       # Distance between cameras (millimeters)
rotation_deg      # Rotation angles on each axis (degrees)
rms_error         # Reprojection error (pixels)
num_images        # Number of image pairs used
```

---

## 🔍 Workflow Details

### Section 0: Generate Calibration Pattern
- Creates high-resolution chessboard pattern
- Configurable square size and paper format
- Validates pattern fits on page
- Provides printing instructions

### Section 1: Camera Detection
- Auto-detects available cameras
- Tests supported resolutions
- Displays first frame from each camera
- Configure indices and resolution

### Section 2: Target Pose Generation
- Creates 24 diverse calibration poses
- Covers center, corners, edges
- Includes tilted and rotated views
- Visualizes all target positions

### Section 3: Guided Calibration Interface
- Real-time overlay of target quadrilateral
- Alignment score calculation
- Automatic capture at threshold
- Progress tracking

### Section 4: Monocular Calibration
- Individual calibration for each camera
- Physics-based constraints (optional)
- Per-image error analysis
- Undistortion visualization

### Section 5: Stereo Calibration
- Initial stereo calibration
- Optional refinement with subset selection
- Rotation split analysis (R1/R2)
- Rectification quality plots

### Section 6: Live Rectified Demo
- Real-time rectified view
- Corner detection validation
- Epipolar line overlay
- Color-coded alignment quality

---

## 🛠️ Troubleshooting

### Poor Calibration Quality

**Symptoms**: RMS error > 1.0px, high vertical offset

**Solutions**:
1. ✅ Verify pattern is **perfectly flat** (use rigid mount)
2. ✅ Check actual square size matches `SQUARE_SIZE_MM` (measure with ruler)
3. ✅ Improve lighting (diffuse, avoid glare/shadows)
4. ✅ Ensure cameras remain **fixed** during entire capture session
5. ✅ Capture more diverse poses (don't skip tilted angles)
6. ✅ Remove blurry or poorly detected images

### Cameras Not Detected

**Symptoms**: `list_available_cameras()` returns empty list

**Solutions**:
1. ✅ Check USB connections (try different ports)
2. ✅ Update camera indices (may change between sessions)
3. ✅ Verify camera permissions (Windows: Settings > Privacy > Camera)
4. ✅ Try different backend: `backend = cv2.CAP_MSMF` (Windows)
5. ✅ Test cameras in other applications first

### Pattern Not Detected

**Symptoms**: "Chessboard not detected" during capture

**Solutions**:
1. ✅ Ensure good lighting (diffuse, not too bright/dark)
2. ✅ Check pattern orientation (9 wide × 6 tall inner corners)
3. ✅ Move pattern slower (avoid motion blur)
4. ✅ Adjust camera focus (manual focus preferred)
5. ✅ Verify pattern contrast (clean printing, white background)

### Rectification Shows Large Vertical Offset

**Symptoms**: Corresponding points have different Y-coordinates (> 2px)

**Solutions**:
1. ✅ Redo stereo calibration with more image pairs
2. ✅ Ensure cameras didn't move between monocular and stereo capture
3. ✅ Check monocular calibration quality first (< 0.5px RMS)
4. ✅ Verify pattern measurements are accurate
5. ✅ Use refinement step (select best image pairs)

---

## 📚 Theory & Background

### Calibration Overview

**Monocular calibration** estimates:
- **Intrinsics** (K): Focal lengths, principal point
- **Distortion** (D): Radial (k1, k2, k3) and tangential (p1, p2)

**Stereo calibration** estimates:
- **Rotation** (R): 3×3 matrix from right to left camera frame
- **Translation** (T): 3×1 vector (baseline direction and magnitude)
- **Epipolar geometry**: Essential (E) and Fundamental (F) matrices

**Rectification** computes:
- **R1, R2**: Rotations to align epipolar lines horizontally
- **P1, P2**: New projection matrices (virtual cameras)
- **Q**: Disparity-to-depth reprojection matrix

### Key Concepts

**Epipolar Geometry**: For any point in the left image, the corresponding point in the right image lies on a line (epipolar line). Rectification makes these lines horizontal.

**Disparity**: Horizontal pixel difference between corresponding points. Inversely proportional to depth:
```
depth = (focal_length × baseline) / disparity
```

**Calibration Quality**: Measured by reprojection error - Euclidean distance between detected corners and their predicted positions using calibrated parameters.

---

## 🎓 Educational Use

This notebook is designed for:
- **Computer vision courses**: Hands-on calibration lab
- **Robotics workshops**: Stereo vision setup
- **Research labs**: Standardized calibration protocol
- **Self-study**: Complete tutorial with theory

### Learning Objectives
1. Understand pinhole camera model and lens distortion
2. Perform monocular camera calibration
3. Estimate stereo geometry and relative pose
4. Compute stereo rectification for epipolar alignment
5. Validate calibration quality with real-time demo

---

## 📖 References

### Academic Papers
- **Zhang, Z. (2000).** "A flexible new technique for camera calibration." *IEEE Transactions on Pattern Analysis and Machine Intelligence.*
- **Hartley, R., & Zisserman, A. (2004).** *Multiple View Geometry in Computer Vision* (2nd ed.). Cambridge University Press.
- **Bouguet, J.-Y. (2001).** "Camera Calibration Toolbox for Matlab." Caltech Technical Report.

### OpenCV Documentation
- [Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [Stereo Calibration](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga246253dcc6de2e0376c599e7d692303a)
- [Stereo Rectification](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-improvement`)
3. Commit changes (`git commit -m 'Add amazing improvement'`)
4. Push to branch (`git push origin feature/amazing-improvement`)
5. Open a Pull Request

### Areas for Contribution
- Additional camera models/configurations
- Automated quality assessment
- Multi-camera calibration (> 2 cameras)
- Alternative calibration patterns (ChArUco, circles)
- Performance optimizations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Authors

- **Angel** - *Initial work* - [doc-ceb](https://github.com/doc-ceb)

---

## 🙏 Acknowledgments

- OpenCV community for excellent computer vision tools
- Zhang's calibration method for robust camera parameter estimation
- Bouguet's rectification algorithm for optimal image alignment
- Students and researchers who provided feedback on this guide

---

## 📧 Contact

For questions, issues, or suggestions:
- **GitHub Issues**: [Report a bug](https://github.com/doc-ceb/Stereo_Calibration_Guide/issues)
- **Email**: [angel.ceballos.esp@gmail.com]

---

**Star ⭐ this repository if you found it helpful!**