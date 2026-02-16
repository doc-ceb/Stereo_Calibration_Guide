# Stereo Rectification Guide

## Introduction

Stereo rectification is the process of transforming two images of a stereo pair so that corresponding points lie on the same horizontal scanline. This simplifies stereo matching and depth estimation.

## Why Rectify?

**Before Rectification:**
- Corresponding points can be anywhere in 2D space
- Stereo matching requires 2D search (computationally expensive)
- Epipolar lines are not horizontal

**After Rectification:**
- Corresponding points are on the same horizontal line (y-coordinate)
- Stereo matching only requires 1D search (along x-axis)
- Epipolar lines are horizontal and aligned

## Prerequisites

Before stereo rectification, you must:
1. Calibrate both cameras individually (see [Camera Calibration Guide](camera_calibration.md))
2. Calibrate the stereo system (compute relative position between cameras)

## Stereo System Calibration

### Step 1: Capture Stereo Calibration Images

Simultaneously capture images of the checkerboard pattern with both cameras:

**Important**:
- **Synchronization**: Capture left and right images at the same time
- **Visibility**: Pattern must be visible in both cameras
- **Coverage**: Capture 15-30 stereo pairs with varied poses
- **Static scene**: Pattern must not move between left/right capture

### Step 2: Detect Corners in Both Views

```python
import cv2
import numpy as np

pattern_size = (9, 6)

# Process left images
left_objpoints = []
left_imgpoints = []
for left_img in left_images:
    gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        left_objpoints.append(objp)
        left_imgpoints.append(corners)

# Process right images (same process)
right_objpoints = []
right_imgpoints = []
# ... similar code for right images
```

### Step 3: Stereo Calibration

Compute the relative position and orientation between cameras:

```python
# Stereo calibration
ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
    objpoints,           # 3D points in real world
    left_imgpoints,      # 2D points in left images
    right_imgpoints,     # 2D points in right images
    mtx1, dist1,         # Left camera calibration
    mtx2, dist2,         # Right camera calibration
    image_size,          # Image dimensions
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
    flags=cv2.CALIB_FIX_INTRINSIC  # Use existing calibrations
)
```

**Returns**:
- **R**: Rotation matrix between cameras
- **T**: Translation vector between cameras (baseline)
- **E**: Essential matrix
- **F**: Fundamental matrix

### Step 4: Compute Rectification Transforms

```python
# Compute rectification transforms
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
    mtx1, dist1,     # Left camera parameters
    mtx2, dist2,     # Right camera parameters
    image_size,      # Image dimensions
    R, T,            # Relative rotation and translation
    alpha=1          # 0=crop, 1=keep all pixels
)
```

**Returns**:
- **R1, R2**: Rectification rotation matrices for left and right cameras
- **P1, P2**: Projection matrices in rectified coordinate system
- **Q**: Disparity-to-depth mapping matrix (for 3D reconstruction)
- **roi1, roi2**: Valid regions of interest in rectified images

### Step 5: Create Remap Matrices

For efficient processing, compute undistortion and rectification maps:

```python
# Left camera maps
map1_left, map2_left = cv2.initUndistortRectifyMap(
    mtx1, dist1, R1, P1, image_size, cv2.CV_32FC1
)

# Right camera maps
map1_right, map2_right = cv2.initUndistortRectifyMap(
    mtx2, dist2, R2, P2, image_size, cv2.CV_32FC1
)
```

### Step 6: Rectify Images

Apply the remap to stereo image pairs:

```python
# Rectify left and right images
left_rectified = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
right_rectified = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)
```

### Step 7: Save Rectification Data

```python
# Save all stereo calibration and rectification data
np.savez('stereo_calibration.npz',
         mtx1=mtx1, dist1=dist1, mtx2=mtx2, dist2=dist2,
         R=R, T=T, E=E, F=F,
         R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         roi1=roi1, roi2=roi2,
         map1_left=map1_left, map2_left=map2_left,
         map1_right=map1_right, map2_right=map2_right)
```

## Verifying Rectification

### Visual Inspection

Draw horizontal lines across rectified images to verify alignment:

```python
import matplotlib.pyplot as plt

# Combine rectified images side by side
combined = np.hstack((left_rectified, right_rectified))

# Draw horizontal lines
for y in range(0, combined.shape[0], 50):
    cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

plt.figure(figsize=(16, 8))
plt.imshow(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
plt.title('Rectified Stereo Pair with Epipolar Lines')
plt.axis('off')
plt.show()
```

**Good rectification**: Corresponding features align horizontally across the images.

### Quantitative Verification

Check that epipolar lines are truly horizontal:

```python
# Select point in left image
pt_left = np.array([[x, y]], dtype=np.float32)

# Compute corresponding epipolar line in right image
# Using fundamental matrix
line = cv2.computeCorrespondEpilines(pt_left.reshape(-1, 1, 2), 1, F)

# For perfect rectification, line should be horizontal (a ≈ 0, b ≈ 1)
a, b, c = line[0][0]
print(f"Epipolar line equation: {a}x + {b}y + {c} = 0")
print(f"Line angle from horizontal: {np.arctan2(a, b) * 180 / np.pi} degrees")
```

## Computing Disparity Maps

Once images are rectified, compute disparity (depth) maps:

```python
# Create stereo matcher
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)

# Or use StereoSGBM for better quality
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*5,      # Must be divisible by 16
    blockSize=5,
    P1=8 * 3 * 5**2,          # Smoothness term
    P2=32 * 3 * 5**2,         # Smoothness term
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# Compute disparity
gray_left = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
gray_right = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)
disparity = stereo.compute(gray_left, gray_right)

# Normalize for visualization
disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
```

## 3D Reconstruction

Convert disparity to 3D points using the Q matrix:

```python
# Reproject to 3D
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Filter valid points (disparity > 0)
mask = disparity > disparity.min()
points = points_3D[mask]
colors = left_rectified[mask]

# Save as point cloud (e.g., PLY format)
# Or visualize with Open3D or other tools
```

## Understanding Key Matrices

### Fundamental Matrix (F)

Relates corresponding points in stereo images:

```
x_right^T * F * x_left = 0
```

Where x_left and x_right are corresponding points in homogeneous coordinates.

### Essential Matrix (E)

Similar to F but for calibrated cameras:

```
E = K_right^T * F * K_left
```

### Q Matrix (Disparity-to-Depth)

Maps disparity to 3D coordinates:

```
[X, Y, Z, W]^T = Q * [u, v, disparity, 1]^T
```

Then: x = X/W, y = Y/W, z = Z/W (3D point in meters)

## Stereo Baseline and Depth Resolution

The **baseline** (distance between cameras) affects depth resolution:

- **Larger baseline**: Better depth resolution at distance, but harder to find correspondences
- **Smaller baseline**: Easier correspondence, but poor depth resolution

**Depth formula**:
```
Z = (f * B) / disparity
```

Where:
- Z: Depth (distance from cameras)
- f: Focal length (pixels)
- B: Baseline (meters)
- disparity: Difference in x-coordinates (pixels)

**Depth uncertainty**:
```
ΔZ = (Z²) / (f * B)
```

Uncertainty grows quadratically with depth!

## Common Issues and Solutions

### Issue: Poor alignment after rectification

**Solutions**:
- Recalibrate stereo system with more image pairs
- Ensure good coverage and synchronization during calibration
- Check that individual camera calibrations are accurate
- Verify that R and T matrices are reasonable

### Issue: Low-quality disparity maps

**Solutions**:
- Adjust StereoBM/StereoSGBM parameters
- Improve lighting (uniform, no shadows)
- Use textured scenes (avoid uniform surfaces)
- Try different algorithms (BM vs SGBM vs deep learning)

### Issue: Black regions after rectification

**Solutions**:
- Use alpha=1 in stereoRectify to keep all pixels
- Accept that invalid regions are unavoidable
- Crop to roi1 and roi2 for valid regions only

### Issue: Vertical misalignment

**Solutions**:
- Recalibrate the entire system
- Check for time delays between left/right capture
- Ensure cameras are rigidly mounted
- Verify synchronization

## Optimization Tips

### For Real-Time Processing

1. **Compute maps once**: Use initUndistortRectifyMap and reuse maps
2. **GPU acceleration**: Use CUDA versions of stereo matchers
3. **Reduce resolution**: Process at lower resolution, upscale results
4. **Optimize parameters**: Balance quality vs speed

### For Better Quality

1. **Use StereoSGBM**: Better than BM, but slower
2. **Post-filtering**: Apply WLS filter or bilateral filter
3. **Use deep learning**: Modern neural network methods (e.g., RAFT-Stereo)
4. **Calibrate carefully**: High-quality calibration improves all results

## Best Practices

1. **Synchronization**: Capture left/right images simultaneously
2. **Rigid mounting**: Cameras must not move relative to each other
3. **Similar cameras**: Use identical camera models with same settings
4. **Horizontal baseline**: Mount cameras horizontally for standard rectification
5. **Sufficient baseline**: 5-20% of working distance
6. **Overlapping FOV**: Ensure significant field-of-view overlap
7. **Regular recalibration**: If cameras are moved or settings changed

## Applications

### Depth Estimation
- Robotics navigation and obstacle avoidance
- 3D scanning and reconstruction
- Autonomous vehicles

### 3D Reconstruction
- Cultural heritage preservation
- Medical imaging
- Virtual reality content creation

### Scene Understanding
- Object recognition and localization
- Augmented reality
- Gesture recognition

## Advanced Topics

### Non-Horizontal Stereo Rigs

For vertical or arbitrary stereo configurations, rectification still works but:
- Epipolar lines won't be horizontal
- May need custom implementations
- OpenCV's stereoRectify assumes horizontal baseline

### Multi-Camera Systems

For more than two cameras:
- Calibrate all pairs
- Use consistent coordinate system
- Consider bundle adjustment for global optimization

### Calibration Refinement

For production systems:
- Use target-less calibration methods
- Implement online calibration
- Monitor and detect calibration drift

## Further Reading

- OpenCV Documentation: [Depth Map from Stereo Images](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)
- Hartley & Zisserman: "Multiple View Geometry in Computer Vision"
- Scharstein & Szeliski: "A Taxonomy and Evaluation of Dense Two-Frame Stereo Correspondence Algorithms"

## Summary

Stereo rectification transforms stereo image pairs to simplify correspondence search:

1. **Calibrate** individual cameras
2. **Calibrate** stereo system (find R and T)
3. **Compute** rectification transforms (R1, R2, P1, P2, Q)
4. **Create** remap matrices
5. **Rectify** image pairs
6. **Verify** horizontal alignment
7. **Compute** disparity maps
8. **Reconstruct** 3D points

With properly rectified images, stereo matching becomes a 1D search problem, enabling efficient and accurate depth estimation.
