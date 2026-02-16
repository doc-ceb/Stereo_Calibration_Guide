#!/usr/bin/env python3
"""
Stereo Rectification and Disparity Computation Script

This script rectifies stereo image pairs using pre-computed calibration data
and optionally computes disparity maps for depth estimation.

Usage:
    python stereo_rectify.py --calibration stereo_calibration.npz --left left.jpg --right right.jpg

Author: Camera Calibration Guide
License: MIT
"""

import argparse
import os
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Rectify stereo images and compute disparity'
    )
    parser.add_argument(
        '--calibration',
        type=str,
        required=True,
        help='Path to stereo calibration file (.npz)'
    )
    parser.add_argument(
        '--left',
        type=str,
        required=True,
        help='Path to left image'
    )
    parser.add_argument(
        '--right',
        type=str,
        required=True,
        help='Path to right image'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for rectified images (default: output)'
    )
    parser.add_argument(
        '--disparity',
        action='store_true',
        help='Compute and save disparity map'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display results'
    )
    parser.add_argument(
        '--num_disparities',
        type=int,
        default=80,
        help='Maximum disparity (must be divisible by 16, default: 80)'
    )
    parser.add_argument(
        '--block_size',
        type=int,
        default=5,
        help='Block size for stereo matching (default: 5)'
    )

    return parser.parse_args()


def load_calibration(filepath):
    """Load stereo calibration data from file."""
    if not os.path.exists(filepath):
        print(f"Error: Calibration file not found: {filepath}")
        sys.exit(1)
    
    print(f"Loading calibration from: {filepath}")
    data = np.load(filepath)
    
    # Check required fields
    required = ['map1_left', 'map2_left', 'map1_right', 'map2_right']
    for field in required:
        if field not in data:
            print(f"Error: Calibration file missing required field: {field}")
            sys.exit(1)
    
    return data


def rectify_images(left_img, right_img, calib_data):
    """Rectify stereo image pair."""
    print("Rectifying images...")
    
    # Apply rectification maps
    left_rectified = cv2.remap(
        left_img,
        calib_data['map1_left'],
        calib_data['map2_left'],
        cv2.INTER_LINEAR
    )
    
    right_rectified = cv2.remap(
        right_img,
        calib_data['map1_right'],
        calib_data['map2_right'],
        cv2.INTER_LINEAR
    )
    
    return left_rectified, right_rectified


def compute_disparity(left_gray, right_gray, num_disparities=80, block_size=5):
    """Compute disparity map using StereoSGBM."""
    print(f"Computing disparity map...")
    print(f"  num_disparities: {num_disparities}")
    print(f"  block_size: {block_size}")
    
    # Ensure num_disparities is divisible by 16
    num_disparities = ((num_disparities + 15) // 16) * 16
    
    # Create StereoSGBM object
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    # Compute disparity
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    return disparity


def visualize_disparity(disparity):
    """Create colorized disparity map for visualization."""
    # Normalize disparity for visualization
    disp_min = disparity[disparity > 0].min() if np.any(disparity > 0) else 0
    disp_max = disparity.max()
    
    disparity_normalized = np.zeros_like(disparity)
    mask = disparity > 0
    disparity_normalized[mask] = (disparity[mask] - disp_min) / (disp_max - disp_min) * 255
    disparity_normalized = disparity_normalized.astype(np.uint8)
    
    # Apply colormap
    disparity_color = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)
    
    # Make invalid pixels black
    disparity_color[~mask] = 0
    
    return disparity_color


def draw_epipolar_lines(left_img, right_img, num_lines=20):
    """Draw horizontal epipolar lines on rectified images."""
    combined = np.hstack((left_img, right_img))
    h, w = left_img.shape[:2]
    
    # Draw lines every (h // num_lines) pixels
    step = max(1, h // num_lines)
    for y in range(0, h, step):
        color = (0, 255, 0)  # Green
        cv2.line(combined, (0, y), (2*w, y), color, 1)
    
    return combined


def main():
    """Main function."""
    args = parse_arguments()
    
    # Load calibration
    calib_data = load_calibration(args.calibration)
    
    # Read input images
    print(f"Reading left image: {args.left}")
    left_img = cv2.imread(args.left)
    if left_img is None:
        print(f"Error: Could not read left image: {args.left}")
        sys.exit(1)
    
    print(f"Reading right image: {args.right}")
    right_img = cv2.imread(args.right)
    if right_img is None:
        print(f"Error: Could not read right image: {args.right}")
        sys.exit(1)
    
    # Check image sizes match
    if left_img.shape != right_img.shape:
        print("Error: Left and right images have different sizes")
        print(f"  Left: {left_img.shape}")
        print(f"  Right: {right_img.shape}")
        sys.exit(1)
    
    # Rectify images
    left_rect, right_rect = rectify_images(left_img, right_img, calib_data)
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save rectified images
    left_output = os.path.join(args.output, 'left_rectified.jpg')
    right_output = os.path.join(args.output, 'right_rectified.jpg')
    
    cv2.imwrite(left_output, left_rect)
    cv2.imwrite(right_output, right_rect)
    print(f"\nSaved rectified images:")
    print(f"  {left_output}")
    print(f"  {right_output}")
    
    # Draw epipolar lines
    epipolar = draw_epipolar_lines(left_rect, right_rect)
    epipolar_output = os.path.join(args.output, 'epipolar_lines.jpg')
    cv2.imwrite(epipolar_output, epipolar)
    print(f"  {epipolar_output}")
    
    # Compute disparity if requested
    disparity = None
    if args.disparity:
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        
        disparity = compute_disparity(
            left_gray, right_gray,
            args.num_disparities,
            args.block_size
        )
        
        # Save disparity map
        disparity_output = os.path.join(args.output, 'disparity.npy')
        np.save(disparity_output, disparity)
        print(f"  {disparity_output}")
        
        # Save visualized disparity
        disparity_vis = visualize_disparity(disparity)
        disparity_vis_output = os.path.join(args.output, 'disparity_color.jpg')
        cv2.imwrite(disparity_vis_output, disparity_vis)
        print(f"  {disparity_vis_output}")
        
        # Print statistics
        valid_disparity = disparity[disparity > 0]
        if len(valid_disparity) > 0:
            print(f"\nDisparity Statistics:")
            print(f"  Min: {valid_disparity.min():.2f} pixels")
            print(f"  Max: {valid_disparity.max():.2f} pixels")
            print(f"  Mean: {valid_disparity.mean():.2f} pixels")
            print(f"  Valid pixels: {len(valid_disparity)} / {disparity.size} ({100*len(valid_disparity)/disparity.size:.1f}%)")
            
            # Depth estimation (if baseline available)
            if 'baseline' in calib_data and 'P1' in calib_data:
                baseline = calib_data['baseline']
                fx = calib_data['P1'][0, 0]  # Focal length in pixels
                
                # Compute depth: Z = (f * B) / disparity
                depth = (fx * baseline) / (valid_disparity + 1e-6)
                
                print(f"\nDepth Estimation (baseline={baseline*1000:.1f}mm):")
                print(f"  Min depth: {depth.min():.3f} meters")
                print(f"  Max depth: {depth.max():.3f} meters")
                print(f"  Mean depth: {depth.mean():.3f} meters")
    
    # Display results if requested
    if args.show:
        print("\nDisplaying results... (press any key to continue)")
        
        # Show original images
        original = np.hstack((left_img, right_img))
        cv2.imshow('Original Stereo Pair', original)
        cv2.waitKey(0)
        
        # Show rectified images with epipolar lines
        cv2.imshow('Rectified Stereo Pair with Epipolar Lines', epipolar)
        cv2.waitKey(0)
        
        # Show disparity if computed
        if disparity is not None:
            cv2.imshow('Disparity Map', disparity_vis)
            cv2.waitKey(0)
        
        cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("Rectification complete!")
    print("=" * 60)
    print(f"\nResults saved to: {args.output}/")


if __name__ == '__main__':
    main()
