# Sample Data Directory

This directory is for storing your calibration images and test images.

## Recommended Directory Structure

```
sample_data/
├── calibration/
│   ├── camera1/           # Single camera calibration images
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   ├── left/              # Left camera calibration images
│   │   ├── img_001.jpg
│   │   ├── img_002.jpg
│   │   └── ...
│   └── right/             # Right camera calibration images
│       ├── img_001.jpg
│       ├── img_002.jpg
│       └── ...
└── test/
    ├── left.jpg           # Test left image for rectification
    └── right.jpg          # Test right image for rectification
```

## Capturing Calibration Images

### For Single Camera Calibration

1. Print a checkerboard pattern (9x6 or 8x6 inner corners recommended)
2. Mount it on a flat, rigid surface
3. Capture 15-30 images with:
   - Different positions (move pattern around the frame)
   - Different orientations (rotate and tilt pattern)
   - Different distances (closer and farther from camera)
   - Coverage of all parts of the frame (corners and edges)

### For Stereo Calibration

1. Set up both cameras with fixed positions (do not move during calibration)
2. Use the same checkerboard pattern
3. Capture 15-30 synchronized stereo pairs where:
   - Pattern is visible in both cameras simultaneously
   - Pattern is at various positions, orientations, and distances
   - Both images are captured at exactly the same time

## Tips for Good Calibration Images

- ✓ Good, uniform lighting
- ✓ Pattern in focus (no motion blur)
- ✓ Flat pattern (no warping or bending)
- ✓ High contrast (clear black and white squares)
- ✓ Diverse poses (don't just move in one direction)
- ✗ Avoid motion blur
- ✗ Avoid shadows on pattern
- ✗ Avoid reflections
- ✗ Don't capture too many similar images

## Checkerboard Patterns

You can download standard patterns from:
- [OpenCV Pattern (9x6)](https://raw.githubusercontent.com/opencv/opencv/master/doc/pattern.png)

Or create your own using:
- OpenCV's pattern generator
- Online pattern generators
- Drawing software (ensure accurate dimensions)

Remember to measure the actual square size after printing!

## Example Usage

After placing your images in the appropriate directories:

### Single Camera Calibration
```bash
python scripts/calibrate_camera.py \
    --pattern 9x6 \
    --square_size 0.025 \
    --images sample_data/calibration/camera1/
```

### Stereo Calibration
```bash
python scripts/calibrate_stereo.py \
    --pattern 9x6 \
    --square_size 0.025 \
    --left sample_data/calibration/left/ \
    --right sample_data/calibration/right/
```

### Stereo Rectification
```bash
python scripts/stereo_rectify.py \
    --calibration stereo_calibration.npz \
    --left sample_data/test/left.jpg \
    --right sample_data/test/right.jpg \
    --output output/ \
    --disparity
```
