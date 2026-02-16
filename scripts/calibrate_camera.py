#!/usr/bin/env python3
"""
Camera Calibration Script

This script calibrates a single camera using a checkerboard pattern.
It processes a set of calibration images, detects the checkerboard corners,
computes camera intrinsic parameters and distortion coefficients, and saves
the calibration results.

Usage:
    python calibrate_camera.py --pattern 9x6 --square_size 0.025 --images calibration_images/

Author: Camera Calibration Guide
License: MIT
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calibrate a camera using checkerboard pattern images'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        required=True,
        help='Checkerboard pattern size as WIDTHxHEIGHT (e.g., 9x6 for 9 inner corners width, 6 height)'
    )
    parser.add_argument(
        '--square_size',
        type=float,
        default=1.0,
        help='Size of checkerboard square in meters (default: 1.0)'
    )
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Path to directory containing calibration images or glob pattern'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='calibration.npz',
        help='Output file for calibration data (default: calibration.npz)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show detected corners in images'
    )
    parser.add_argument(
        '--extension',
        type=str,
        default='jpg',
        help='Image file extension (default: jpg)'
    )
    
    return parser.parse_args()


def find_images(path, extension):
    """Find all images in the specified path."""
    if os.path.isdir(path):
        pattern = os.path.join(path, f'*.{extension}')
        images = glob.glob(pattern)
    else:
        images = glob.glob(path)
    
    if not images:
        print(f"Error: No images found matching pattern: {path}")
        sys.exit(1)
    
    print(f"Found {len(images)} images")
    return sorted(images)


def detect_corners(image_path, pattern_size, show=False):
    """
    Detect checkerboard corners in an image.
    
    Args:
        image_path: Path to the image file
        pattern_size: Tuple of (width, height) inner corners
        show: Whether to display the detected corners
        
    Returns:
        Tuple of (success, corners, image) or (False, None, None) if detection failed
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Could not read image: {image_path}")
        return False, None, None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # Refine corners to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        if show:
            # Draw and display corners
            img_with_corners = cv2.drawChessboardCorners(img, pattern_size, corners, ret)
            cv2.imshow('Detected Corners', img_with_corners)
            cv2.waitKey(500)
        
        return True, corners, img
    else:
        print(f"Warning: Could not detect pattern in: {image_path}")
        return False, None, None


def calibrate_camera(objpoints, imgpoints, image_size):
    """
    Calibrate camera using object points and image points.
    
    Args:
        objpoints: List of 3D object points
        imgpoints: List of 2D image points
        image_size: Tuple of (width, height) of images
        
    Returns:
        Tuple of (ret, mtx, dist, rvecs, tvecs)
    """
    print("\nCalibrating camera...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    
    return ret, mtx, dist, rvecs, tvecs


def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    """Compute mean reprojection error."""
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    return mean_error / len(objpoints)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Parse pattern size
    try:
        width, height = map(int, args.pattern.split('x'))
        pattern_size = (width, height)
    except ValueError:
        print("Error: Pattern must be in format WIDTHxHEIGHT (e.g., 9x6)")
        sys.exit(1)
    
    print(f"Calibration Configuration:")
    print(f"  Pattern size: {pattern_size[0]}x{pattern_size[1]} inner corners")
    print(f"  Square size: {args.square_size} meters")
    print(f"  Images path: {args.images}")
    print(f"  Output file: {args.output}")
    
    # Prepare object points (3D points in real world space)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= args.square_size
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Find all calibration images
    images = find_images(args.images, args.extension)
    
    # Process each image
    print("\nProcessing images...")
    successful_images = []
    image_size = None
    
    for image_path in images:
        ret, corners, img = detect_corners(image_path, pattern_size, args.show)
        
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            successful_images.append(image_path)
            
            # Get image size from first successful image
            if image_size is None:
                h, w = img.shape[:2]
                image_size = (w, h)
            
            print(f"✓ {os.path.basename(image_path)}")
        else:
            print(f"✗ {os.path.basename(image_path)}")
    
    if args.show:
        cv2.destroyAllWindows()
    
    # Check if we have enough images
    if len(successful_images) < 10:
        print(f"\nError: Only {len(successful_images)} images with detected patterns.")
        print("At least 10 images are required for calibration.")
        sys.exit(1)
    
    print(f"\nSuccessfully processed {len(successful_images)} / {len(images)} images")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(objpoints, imgpoints, image_size)
    
    # Compute reprojection error
    mean_error = compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)
    
    # Print results
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)
    print(f"\nCamera Matrix (K):")
    print(mtx)
    print(f"\nDistortion Coefficients:")
    print(dist)
    print(f"\nReprojection Error: {mean_error:.4f} pixels")
    
    # Interpretation
    print("\nCalibration Quality:")
    if mean_error < 0.5:
        print("  ✓ EXCELLENT (< 0.5 pixels)")
    elif mean_error < 1.0:
        print("  ✓ GOOD (< 1.0 pixels)")
    else:
        print("  ⚠ POOR (> 1.0 pixels) - Consider capturing more/better images")
    
    # Focal length in pixels
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    print(f"\nCamera Parameters:")
    print(f"  Focal length: fx={fx:.2f} pixels, fy={fy:.2f} pixels")
    print(f"  Principal point: cx={cx:.2f} pixels, cy={cy:.2f} pixels")
    print(f"  Image size: {image_size[0]}x{image_size[1]} pixels")
    
    # Save calibration results
    print(f"\nSaving calibration to: {args.output}")
    np.savez(
        args.output,
        mtx=mtx,
        dist=dist,
        rvecs=np.array(rvecs),
        tvecs=np.array(tvecs),
        image_size=image_size,
        reprojection_error=mean_error
    )
    
    print("\n" + "=" * 60)
    print("Calibration complete!")
    print("=" * 60)
    print(f"\nUse this calibration file with undistort or stereo calibration scripts.")


if __name__ == '__main__':
    main()
