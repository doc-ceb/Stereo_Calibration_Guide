#!/usr/bin/env python3
"""
Stereo Camera Calibration Script

This script calibrates a stereo camera system by processing pairs of calibration
images from left and right cameras. It computes the relative position and
orientation between the two cameras.

Usage:
    python calibrate_stereo.py --pattern 9x6 --square_size 0.025 --left left_images/ --right right_images/

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
        description='Calibrate a stereo camera system'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        required=True,
        help='Checkerboard pattern size as WIDTHxHEIGHT (e.g., 9x6)'
    )
    parser.add_argument(
        '--square_size',
        type=float,
        default=1.0,
        help='Size of checkerboard square in meters (default: 1.0)'
    )
    parser.add_argument(
        '--left',
        type=str,
        required=True,
        help='Path to directory containing left camera calibration images'
    )
    parser.add_argument(
        '--right',
        type=str,
        required=True,
        help='Path to directory containing right camera calibration images'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='stereo_calibration.npz',
        help='Output file for stereo calibration data (default: stereo_calibration.npz)'
    )
    parser.add_argument(
        '--extension',
        type=str,
        default='jpg',
        help='Image file extension (default: jpg)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show detected corners in images'
    )

    return parser.parse_args()


def find_image_pairs(left_path, right_path, extension):
    """Find matching pairs of left and right images."""
    left_pattern = os.path.join(left_path, f'*.{extension}')
    right_pattern = os.path.join(right_path, f'*.{extension}')
    
    left_images = sorted(glob.glob(left_pattern))
    right_images = sorted(glob.glob(right_pattern))
    
    if not left_images:
        print(f"Error: No left images found in {left_path}")
        sys.exit(1)
    
    if not right_images:
        print(f"Error: No right images found in {right_path}")
        sys.exit(1)
    
    if len(left_images) != len(right_images):
        print(f"Warning: Number of left images ({len(left_images)}) != right images ({len(right_images)})")
        print("Using minimum of the two")
    
    num_pairs = min(len(left_images), len(right_images))
    print(f"Found {num_pairs} stereo image pairs")
    
    return left_images[:num_pairs], right_images[:num_pairs]


def detect_corners_stereo(left_path, right_path, pattern_size, show=False):
    """
    Detect checkerboard corners in a stereo image pair.
    
    Returns:
        Tuple of (success, left_corners, right_corners, left_img, right_img)
    """
    # Read images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        return False, None, None, None, None
    
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    
    # Find corners in both images
    ret_left, corners_left = cv2.findChessboardCorners(left_gray, pattern_size, None)
    ret_right, corners_right = cv2.findChessboardCorners(right_gray, pattern_size, None)
    
    if ret_left and ret_right:
        # Refine corners
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
        
        if show:
            # Draw corners
            left_display = cv2.drawChessboardCorners(left_img.copy(), pattern_size, corners_left, True)
            right_display = cv2.drawChessboardCorners(right_img.copy(), pattern_size, corners_right, True)
            combined = np.hstack((left_display, right_display))
            cv2.imshow('Stereo Corners', combined)
            cv2.waitKey(500)
        
        return True, corners_left, corners_right, left_img, right_img
    
    return False, None, None, None, None


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
    
    print("Stereo Calibration Configuration:")
    print(f"  Pattern size: {pattern_size[0]}x{pattern_size[1]} inner corners")
    print(f"  Square size: {args.square_size} meters")
    print(f"  Left images: {args.left}")
    print(f"  Right images: {args.right}")
    print(f"  Output file: {args.output}")
    
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= args.square_size
    
    # Arrays to store points
    objpoints = []       # 3D points in real world space
    left_imgpoints = []  # 2D points in left image plane
    right_imgpoints = [] # 2D points in right image plane
    
    # Find image pairs
    left_images, right_images = find_image_pairs(args.left, args.right, args.extension)
    
    # Process each pair
    print("\nProcessing stereo image pairs...")
    successful_pairs = 0
    image_size = None
    
    for left_path, right_path in zip(left_images, right_images):
        ret, left_corners, right_corners, left_img, right_img = detect_corners_stereo(
            left_path, right_path, pattern_size, args.show
        )
        
        if ret:
            objpoints.append(objp)
            left_imgpoints.append(left_corners)
            right_imgpoints.append(right_corners)
            successful_pairs += 1
            
            # Get image size
            if image_size is None:
                h, w = left_img.shape[:2]
                image_size = (w, h)
            
            left_name = os.path.basename(left_path)
            right_name = os.path.basename(right_path)
            print(f"✓ Pair {successful_pairs}: {left_name} + {right_name}")
        else:
            print(f"✗ Failed: {os.path.basename(left_path)} + {os.path.basename(right_path)}")
    
    if args.show:
        cv2.destroyAllWindows()
    
    # Check minimum number of pairs
    if successful_pairs < 10:
        print(f"\nError: Only {successful_pairs} valid stereo pairs.")
        print("At least 10 pairs are required for stereo calibration.")
        sys.exit(1)
    
    print(f"\nSuccessfully processed {successful_pairs} / {len(left_images)} stereo pairs")
    
    # Individual camera calibrations (initial estimates)
    print("\nCalibrating left camera...")
    ret_left, mtx1, dist1, rvecs1, tvecs1 = cv2.calibrateCamera(
        objpoints, left_imgpoints, image_size, None, None
    )
    
    print("Calibrating right camera...")
    ret_right, mtx2, dist2, rvecs2, tvecs2 = cv2.calibrateCamera(
        objpoints, right_imgpoints, image_size, None, None
    )
    
    # Stereo calibration
    print("\nPerforming stereo calibration...")
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        left_imgpoints,
        right_imgpoints,
        mtx1, dist1,
        mtx2, dist2,
        image_size,
        criteria=criteria,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    
    # Compute stereo rectification
    print("Computing rectification transforms...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx1, dist1,
        mtx2, dist2,
        image_size,
        R, T,
        alpha=1  # 0=crop to valid pixels, 1=keep all pixels
    )
    
    # Compute rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        mtx1, dist1, R1, P1, image_size, cv2.CV_32FC1
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        mtx2, dist2, R2, P2, image_size, cv2.CV_32FC1
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("STEREO CALIBRATION RESULTS")
    print("=" * 60)
    
    print("\nLeft Camera Matrix:")
    print(mtx1)
    print("\nRight Camera Matrix:")
    print(mtx2)
    
    print("\nRotation Matrix (R):")
    print(R)
    print("\nTranslation Vector (T):")
    print(T)
    
    # Compute baseline
    baseline = np.linalg.norm(T)
    print(f"\nBaseline: {baseline:.4f} meters ({baseline*1000:.2f} mm)")
    
    # Rotation angle
    rotation_angle = np.arccos((np.trace(R) - 1) / 2) * 180 / np.pi
    print(f"Relative rotation angle: {rotation_angle:.2f} degrees")
    
    # Save results
    print(f"\nSaving stereo calibration to: {args.output}")
    np.savez(
        args.output,
        # Individual camera parameters
        mtx1=mtx1, dist1=dist1,
        mtx2=mtx2, dist2=dist2,
        # Stereo parameters
        R=R, T=T, E=E, F=F,
        # Rectification parameters
        R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
        roi1=roi1, roi2=roi2,
        # Rectification maps
        map1_left=map1_left, map2_left=map2_left,
        map1_right=map1_right, map2_right=map2_right,
        # Metadata
        image_size=image_size,
        baseline=baseline
    )
    
    print("\n" + "=" * 60)
    print("Stereo calibration complete!")
    print("=" * 60)
    print("\nYou can now use this calibration file with stereo_rectify.py")
    print("to rectify stereo image pairs and compute disparity maps.")


if __name__ == '__main__':
    main()
