# Camera Calibration Guide

## Introduction

Camera calibration is the process of estimating the parameters of a camera. There are two types of parameters:

1. **Intrinsic Parameters**: Internal camera characteristics
   - Focal length (fx, fy)
   - Principal point (cx, cy)
   - Distortion coefficients (k1, k2, k3, p1, p2)

2. **Extrinsic Parameters**: Camera position and orientation in 3D space
   - Rotation matrix (R)
   - Translation vector (T)

## Why Calibrate?

- **Remove lens distortion**: Real lenses cause image distortion (barrel, pincushion)
- **3D reconstruction**: Essential for accurate depth estimation
- **Object measurement**: Convert pixel measurements to real-world units
- **Augmented reality**: Overlay virtual objects correctly

## The Calibration Process

### Step 1: Prepare Calibration Pattern

Use a checkerboard pattern with known dimensions:
- **Recommended**: 9x6 or 8x6 inner corners (black-white intersections)
- **Square size**: Typically 20-30mm (measure accurately!)
- **Print quality**: High-quality print on flat, rigid surface

**Important**: The pattern must be perfectly flat. Mount on cardboard or foam board.

### Step 2: Capture Calibration Images

Capture 15-30 images of the calibration pattern in different positions:

1. **Vary position**: Move pattern to different locations in the frame
2. **Vary orientation**: Tilt and rotate the pattern
3. **Vary distance**: Capture at different depths
4. **Cover the frame**: Include corners and edges of the camera view

**Tips**:
- Use good, uniform lighting
- Avoid motion blur (use tripod if needed)
- Keep pattern in focus
- Ensure pattern is fully visible in each image

### Step 3: Detect Checkerboard Corners

The calibration algorithm automatically detects corner points in each image:

```python
import cv2
import numpy as np

# Define the checkerboard dimensions
pattern_size = (9, 6)  # inner corners (width, height)

# Find corners
ret, corners = cv2.findChessboardCorners(gray_image, pattern_size, None)

# Refine corner positions (sub-pixel accuracy)
if ret:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
```

### Step 4: Calibrate Camera

Use detected corners to compute camera parameters:

```python
# Prepare object points (3D points in real world)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
objp *= square_size  # Scale by actual square size in meters

# Collect object points and image points from all images
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

for each image:
    objpoints.append(objp)
    imgpoints.append(corners)

# Calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)
```

### Step 5: Evaluate Calibration Quality

Check the **reprojection error** - the average distance between detected corners and reprojected corners:

```python
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    mean_error += error

mean_error /= len(objpoints)
print(f"Mean reprojection error: {mean_error}")
```

**Interpretation**:
- **< 0.5 pixels**: Excellent calibration
- **0.5 - 1.0 pixels**: Good calibration
- **> 1.0 pixels**: Poor calibration - capture more/better images

### Step 6: Save Calibration Results

```python
# Save calibration data
np.savez('calibration.npz', 
         mtx=mtx,           # Camera matrix
         dist=dist,         # Distortion coefficients
         rvecs=rvecs,       # Rotation vectors
         tvecs=tvecs)       # Translation vectors
```

## Understanding the Camera Matrix

The camera matrix (intrinsic matrix) is a 3x3 matrix:

```
    [fx  0  cx]
K = [0  fy  cy]
    [0   0   1]
```

Where:
- **fx, fy**: Focal length in pixels (x and y directions)
- **cx, cy**: Principal point (optical center) in pixels
- Typically cx ≈ image_width/2 and cy ≈ image_height/2

## Understanding Distortion Coefficients

Distortion coefficients correct for lens distortion:

```
dist = [k1, k2, p1, p2, k3]
```

Where:
- **k1, k2, k3**: Radial distortion coefficients
- **p1, p2**: Tangential distortion coefficients

**Radial distortion**: Points are displaced radially from the center
- Barrel distortion (k1 < 0): Image bulges outward
- Pincushion distortion (k1 > 0): Image pinches inward

**Tangential distortion**: Lens is not parallel to image plane

## Using Calibration Results

### Undistort Images

Remove lens distortion from images:

```python
# Load calibration
data = np.load('calibration.npz')
mtx = data['mtx']
dist = data['dist']

# Read image
img = cv2.imread('image.jpg')

# Undistort
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)

# Crop image (optional)
x, y, w, h = roi
undistorted = undistorted[y:y+h, x:x+w]
```

### Remap for Faster Processing

For video or multiple images, create remap matrices:

```python
# Compute remap matrices once
map1, map2 = cv2.initUndistortRectifyMap(
    mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1
)

# Apply to many images (fast)
undistorted = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
```

## Common Issues and Solutions

### Issue: Checkerboard not detected

**Solutions**:
- Ensure good lighting and contrast
- Verify pattern size parameters match actual pattern
- Check image focus and clarity
- Try different `cv2.findChessboardCorners` flags

### Issue: High reprojection error

**Solutions**:
- Capture more images (20-30 images recommended)
- Improve coverage (more varied positions/angles)
- Ensure pattern is perfectly flat
- Check for motion blur
- Verify square size measurement

### Issue: Distorted images after "undistortion"

**Solutions**:
- Recalibrate with more images
- Check that correct calibration file is loaded
- Verify image dimensions match calibration

## Best Practices

1. **Pattern quality**: Use high-quality print on rigid, flat surface
2. **Image quantity**: Capture 20-30 images minimum
3. **Image diversity**: Vary position, orientation, and distance
4. **Frame coverage**: Include all areas of the camera view
5. **Lighting**: Use consistent, good lighting
6. **Focus**: Ensure pattern is sharp in all images
7. **Validation**: Always check reprojection error

## Mathematical Background (Optional)

### Pinhole Camera Model

The basic projection equation:

```
s * [u, v, 1]^T = K * [R|t] * [X, Y, Z, 1]^T
```

Where:
- (u, v): 2D image coordinates (pixels)
- (X, Y, Z): 3D world coordinates
- K: Camera intrinsic matrix
- [R|t]: Extrinsic matrix (rotation and translation)
- s: Scale factor

### Distortion Model

Corrected points (x', y') from distorted points (x, y):

```
x' = x * (1 + k1*r² + k2*r⁴ + k3*r⁶) + 2*p1*x*y + p2*(r² + 2*x²)
y' = y * (1 + k1*r² + k2*r⁴ + k3*r⁶) + p1*(r² + 2*y²) + 2*p2*x*y
```

Where r² = x² + y²

## Further Reading

- OpenCV Documentation: [Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- Zhang, Z. (2000). "A flexible new technique for camera calibration". IEEE TPAMI.
- Hartley, R., & Zisserman, A. (2004). "Multiple View Geometry in Computer Vision"

## Next Steps

Once you have calibrated your camera, proceed to [Stereo Rectification Guide](stereo_rectification.md) to calibrate and rectify a stereo system.
