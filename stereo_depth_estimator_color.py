import cv2
import numpy as np
import time
import os

class StereoDepthEstimator:
    """
    Unified stereo depth estimation class with interactive parameter tuning.
    Supports color-based region detection for targeted disparity computation.
    """
    
    def __init__(self, calibration_data, camera_config=None):
        """
        Initialize the stereo depth estimator.
        
        Parameters:
        -----------
        calibration_data : dict
            Dictionary containing rectification maps ('map1_L', 'map2_L', 'map1_R', 'map2_R')
            and disparity-to-depth matrix ('Q')
        camera_config : dict, optional
            Camera configuration with keys: 'left_idx', 'right_idx', 'width', 'height', 'fps'
        """
        self.rect_maps = {
            'map1_L': calibration_data['map1_L'],
            'map2_L': calibration_data['map2_L'],
            'map1_R': calibration_data['map1_R'],
            'map2_R': calibration_data['map2_R']
        }
        self.Q = calibration_data['Q']
        self.camera_config = camera_config
        
        # Extract baseline and focal length from Q matrix for direct depth calculation
        # Q[3,2] = -1/Tx (where Tx is baseline in same units as focal length)
        # Q[2,3] = focal_length
        self.baseline = 1.0 / abs(self.Q[3, 2])  # in mm
        self.focal_length = abs(self.Q[2, 3])    # in pixels
        
        # Default stereo matching parameters
        self.method = 'SGBM'  # 'BM' or 'SGBM'
        self.min_disp = 0
        self.num_disp = 96  # Must be divisible by 16
        self.block_size = 11
        self.uniqueness_ratio = 10
        self.speckle_window_size = 100
        self.speckle_range = 32
        self.disp12_max_diff = 1
        
        # SGBM-specific parameters
        self.p1 = 8 * 3 * self.block_size**2
        self.p2 = 32 * 3 * self.block_size**2
        
        # Visualization parameters
        self.max_depth_vis = 5000  # mm
        
        # Color-based region detection (HSV color space is better than RGB)
        self.color_detect_enabled = False
        self.h_min = 0     # Hue: 0-179 (OpenCV uses 0-179 for Hue)
        self.h_max = 179
        self.s_min = 0     # Saturation: 0-255
        self.s_max = 255
        self.v_min = 0     # Value (brightness): 0-255
        self.v_max = 255
        self.min_area = 100  # Minimum area for detected region
        
        # FPS tracking
        self.fps = 0.0
        self.fps_update_time = time.time()
        self.fps_frame_count = 0
        
        # For mouse callback functionality
        self.hsv_left_image = None
        self.control_window_name = None
        self.hsv_window_name = 'HSV Color Picker'
        self.hsv_window_created = False
        
        # Interactive measurement box for normal mode
        self.measure_box_enabled = True
        self.measure_box = [100, 100, 100, 100]  # [x, y, width, height]
        self.dragging_box = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.box_offset_x = 0
        self.box_offset_y = 0
        
        # Store default parameters for reset functionality
        self._defaults = {
            'method': self.method,
            'min_disp': self.min_disp,
            'num_disp': self.num_disp,
            'block_size': self.block_size,
            'uniqueness_ratio': self.uniqueness_ratio,
            'speckle_window_size': self.speckle_window_size,
            'speckle_range': self.speckle_range,
            'max_depth_vis': self.max_depth_vis,
            'color_detect_enabled': self.color_detect_enabled,
            'h_min': self.h_min,
            'h_max': self.h_max,
            's_min': self.s_min,
            's_max': self.s_max,
            'v_min': self.v_min,
            'v_max': self.v_max,
            'min_area': self.min_area
        }
        
    def _create_stereo_matcher(self):
        """Create stereo matcher based on current parameters."""
        if self.method == 'SGBM':
            stereo = cv2.StereoSGBM_create(
                minDisparity=self.min_disp,
                numDisparities=self.num_disp,
                blockSize=self.block_size,
                P1=self.p1,
                P2=self.p2,
                disp12MaxDiff=self.disp12_max_diff,
                uniquenessRatio=self.uniqueness_ratio,
                speckleWindowSize=self.speckle_window_size,
                speckleRange=self.speckle_range,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        else:  # BM
            stereo = cv2.StereoBM_create(
                numDisparities=self.num_disp,
                blockSize=self.block_size
            )
            stereo.setMinDisparity(self.min_disp)
            stereo.setSpeckleWindowSize(self.speckle_window_size)
            stereo.setSpeckleRange(self.speckle_range)
            stereo.setUniquenessRatio(self.uniqueness_ratio)
        
        return stereo
    
    def compute_disparity_and_depth(self, img_left, img_right, rectify=True):
        """
        Compute disparity and depth maps from stereo image pair.
        
        Parameters:
        -----------
        img_left, img_right : numpy.ndarray
            Input stereo image pair
        rectify : bool
            Whether to apply rectification (default: True)
            
        Returns:
        --------
        disparity : numpy.ndarray
            Disparity map (float32)
        depth : numpy.ndarray  
            Depth map in millimeters
        disparity_visual : numpy.ndarray
            Normalized disparity for visualization (uint8, 0-255)
        """
        # Rectify images if requested
        if rectify:
            img_left = cv2.remap(img_left, self.rect_maps['map1_L'], 
                                self.rect_maps['map2_L'], cv2.INTER_LINEAR)
            img_right = cv2.remap(img_right, self.rect_maps['map1_R'], 
                                 self.rect_maps['map2_R'], cv2.INTER_LINEAR)
        
        # Convert to grayscale if needed
        if len(img_left.shape) == 3:
            gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        else:
            gray_left = img_left
            gray_right = img_right
        
        # Create stereo matcher
        stereo = self._create_stereo_matcher()
        
        # Pad both images to eliminate black border on left side
        padded_left = cv2.copyMakeBorder(
            gray_left, 
            top=0, bottom=0, left=self.num_disp, right=0, 
            borderType=cv2.BORDER_CONSTANT, value=0
        )
        padded_right = cv2.copyMakeBorder(
            gray_right, 
            top=0, bottom=0, left=self.num_disp, right=0, 
            borderType=cv2.BORDER_CONSTANT, value=0
        )
        
        # Compute disparity with padded images
        disparity = stereo.compute(padded_left, padded_right).astype(np.float32)
        
        # Crop the padding from result
        disparity = disparity[:, self.num_disp:]
        
        # Scale disparity (SGBM/BM returns disparity scaled by 16)
        disparity = disparity / 16.0
        
        # Compute depth using Q matrix
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)
        depth = points_3d[:, :, 2]
        
        # Filter invalid depths
        depth[depth <= 0] = 0
        depth[depth > 10000] = 0  # Max 10 meters
        
        # Create visualization of disparity
        disparity_visual = disparity.copy()
        disparity_visual[disparity_visual < 0] = 0
        
        # Normalize to 0-255
        if disparity_visual.max() > 0:
            disparity_visual = (disparity_visual - disparity_visual.min()) / \
                              (disparity_visual.max() - disparity_visual.min()) * 255
        disparity_visual = disparity_visual.astype(np.uint8)
        
        return disparity, depth, disparity_visual
    
    def _create_trackbars(self, window_name):
        """Create trackbars for interactive parameter tuning."""
        def nothing(x):
            pass
        
        # Method selection (0=BM, 1=SGBM)
        cv2.createTrackbar('Method', window_name, 1 if self.method == 'SGBM' else 0, 1, nothing)
        
        # Disparity parameters
        cv2.createTrackbar('NumDisp16', window_name, self.num_disp // 16, 20, nothing)
        cv2.createTrackbar('BlockSize', window_name, self.block_size, 25, nothing)
        cv2.createTrackbar('Uniqueness', window_name, self.uniqueness_ratio, 100, nothing)
        cv2.createTrackbar('SpeckleWin', window_name, self.speckle_window_size, 200, nothing)
        cv2.createTrackbar('SpeckleRng', window_name, self.speckle_range, 100, nothing)
        cv2.createTrackbar('MaxDepth(m)', window_name, self.max_depth_vis // 1000, 20, nothing)
        
        # Color detection parameters (HSV color space)
        cv2.createTrackbar('ColorDetect', window_name, 1 if self.color_detect_enabled else 0, 1, nothing)
        cv2.createTrackbar('H_min', window_name, self.h_min, 179, nothing)
        cv2.createTrackbar('H_max', window_name, self.h_max, 179, nothing)
        cv2.createTrackbar('S_min', window_name, self.s_min, 255, nothing)
        cv2.createTrackbar('S_max', window_name, self.s_max, 255, nothing)
        cv2.createTrackbar('V_min', window_name, self.v_min, 255, nothing)
        cv2.createTrackbar('V_max', window_name, self.v_max, 255, nothing)
        cv2.createTrackbar('MinArea/100', window_name, self.min_area // 100, 100, nothing)
        
    def _read_trackbars(self, window_name):
        """Read current trackbar values and update parameters."""
        method_val = cv2.getTrackbarPos('Method', window_name)
        self.method = 'SGBM' if method_val == 1 else 'BM'
        
        # Ensure numDisparities is divisible by 16 and at least 16
        num_disp_val = cv2.getTrackbarPos('NumDisp16', window_name)
        self.num_disp = max(16, num_disp_val * 16)
        
        # Ensure blockSize is odd and at least 5
        block_val = cv2.getTrackbarPos('BlockSize', window_name)
        self.block_size = max(5, block_val if block_val % 2 == 1 else block_val + 1)
        
        self.uniqueness_ratio = cv2.getTrackbarPos('Uniqueness', window_name)
        self.speckle_window_size = cv2.getTrackbarPos('SpeckleWin', window_name)
        self.speckle_range = cv2.getTrackbarPos('SpeckleRng', window_name)
        self.max_depth_vis = cv2.getTrackbarPos('MaxDepth(m)', window_name) * 1000
        
        # Color detection parameters
        self.color_detect_enabled = cv2.getTrackbarPos('ColorDetect', window_name) == 1
        self.h_min = cv2.getTrackbarPos('H_min', window_name)
        self.h_max = cv2.getTrackbarPos('H_max', window_name)
        self.s_min = cv2.getTrackbarPos('S_min', window_name)
        self.s_max = cv2.getTrackbarPos('S_max', window_name)
        self.v_min = cv2.getTrackbarPos('V_min', window_name)
        self.v_max = cv2.getTrackbarPos('V_max', window_name)
        self.min_area = max(100, cv2.getTrackbarPos('MinArea/100', window_name) * 100)
        
        # Update P1/P2 for SGBM
        self.p1 = 8 * 3 * self.block_size**2
        self.p2 = 32 * 3 * self.block_size**2
    
    def _update_trackbars(self, window_name):
        """Update trackbar positions to match current parameters."""
        cv2.setTrackbarPos('Method', window_name, 1 if self.method == 'SGBM' else 0)
        cv2.setTrackbarPos('NumDisp16', window_name, self.num_disp // 16)
        cv2.setTrackbarPos('BlockSize', window_name, self.block_size)
        cv2.setTrackbarPos('Uniqueness', window_name, self.uniqueness_ratio)
        cv2.setTrackbarPos('SpeckleWin', window_name, self.speckle_window_size)
        cv2.setTrackbarPos('SpeckleRng', window_name, self.speckle_range)
        cv2.setTrackbarPos('MaxDepth(m)', window_name, self.max_depth_vis // 1000)
        cv2.setTrackbarPos('ColorDetect', window_name, 1 if self.color_detect_enabled else 0)
        cv2.setTrackbarPos('H_min', window_name, self.h_min)
        cv2.setTrackbarPos('H_max', window_name, self.h_max)
        cv2.setTrackbarPos('S_min', window_name, self.s_min)
        cv2.setTrackbarPos('S_max', window_name, self.s_max)
        cv2.setTrackbarPos('V_min', window_name, self.v_min)
        cv2.setTrackbarPos('V_max', window_name, self.v_max)
        cv2.setTrackbarPos('MinArea/100', window_name, self.min_area // 100)
    
    def reset_parameters(self):
        """Reset all parameters to default values."""
        self.method = self._defaults['method']
        self.min_disp = self._defaults['min_disp']
        self.num_disp = self._defaults['num_disp']
        self.block_size = self._defaults['block_size']
        self.uniqueness_ratio = self._defaults['uniqueness_ratio']
        self.speckle_window_size = self._defaults['speckle_window_size']
        self.speckle_range = self._defaults['speckle_range']
        self.max_depth_vis = self._defaults['max_depth_vis']
        self.color_detect_enabled = self._defaults['color_detect_enabled']
        self.h_min = self._defaults['h_min']
        self.h_max = self._defaults['h_max']
        self.s_min = self._defaults['s_min']
        self.s_max = self._defaults['s_max']
        self.v_min = self._defaults['v_min']
        self.v_max = self._defaults['v_max']
        self.min_area = self._defaults['min_area']
        
        # Update P1/P2
        self.p1 = 8 * 3 * self.block_size**2
        self.p2 = 32 * 3 * self.block_size**2
    
    def _mouse_callback_hsv(self, event, x, y, flags, param):
        """
        Mouse callback for clicking on the HSV window to set HSV parameters.
        Click anywhere on the HSV image to auto-set HSV range based on that pixel.
        """
        try:
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"\n=== MOUSE CLICK DETECTED ===")
                print(f"Pixel coords: ({x}, {y})")
                
                if self.hsv_left_image is None:
                    print("✗ HSV image not yet available")
                    return
                
                img_height, img_width = self.hsv_left_image.shape[:2]
                
                # Validate coordinates
                if x < 0 or x >= img_width or y < 0 or y >= img_height:
                    print(f"✗ Click out of bounds (image: {img_width}x{img_height})")
                    return
                
                # Get the HSV values at the clicked pixel (convert to Python int)
                h, s, v = self.hsv_left_image[y, x]
                h, s, v = int(h), int(s), int(v)
                
                print(f"✓ Pixel HSV: H={h}, S={s}, V={v}")
                
                # Set a range around the clicked color
                h_range = 10  # +/- 10 for Hue
                s_range = 40  # +/- 40 for Saturation
                v_range = 40  # +/- 40 for Value
                
                self.h_min = int(max(0, h - h_range))
                self.h_max = int(min(179, h + h_range))
                self.s_min = int(max(0, s - s_range))
                self.s_max = int(min(255, s + s_range))
                self.v_min = int(max(0, v - v_range))
                self.v_max = int(min(255, v + v_range))
                
                print(f"✓ New range: H=[{self.h_min}-{self.h_max}] S=[{self.s_min}-{self.s_max}] V=[{self.v_min}-{self.v_max}]")
                
                # Update trackbars in the control window
                if self.control_window_name:
                    try:
                        cv2.setTrackbarPos('H_min', self.control_window_name, self.h_min)
                        cv2.setTrackbarPos('H_max', self.control_window_name, self.h_max)
                        cv2.setTrackbarPos('S_min', self.control_window_name, self.s_min)
                        cv2.setTrackbarPos('S_max', self.control_window_name, self.s_max)
                        cv2.setTrackbarPos('V_min', self.control_window_name, self.v_min)
                        cv2.setTrackbarPos('V_max', self.control_window_name, self.v_max)
                        print("✓ Trackbars updated!")
                    except Exception as e:
                        print(f"✗ Error updating trackbars: {e}")
                
                print("=== END ===\n")
        except Exception as e:
            print(f"\n✗ EXCEPTION IN CALLBACK: {e}")
            import traceback
            traceback.print_exc()
    
    def _mouse_callback_display(self, event, x, y, flags, param):
        """
        Mouse callback for the main display window to move the measurement box.
        """
        if not self.color_detect_enabled and self.measure_box_enabled:
            # Get image dimensions (display is 2x2, we want top-left quadrant)
            h_img = param['height']
            w_img = param['width']
            
            # Only handle clicks in the top-left quadrant (left rectified image)
            if x < w_img and y < h_img:
                if event == cv2.EVENT_LBUTTONDOWN:
                    # Check if click is inside the box
                    bx, by, bw, bh = self.measure_box
                    if bx <= x <= bx + bw and by <= y <= by + bh:
                        self.dragging_box = True
                        self.drag_start_x = x
                        self.drag_start_y = y
                        self.box_offset_x = x - bx
                        self.box_offset_y = y - by
                
                elif event == cv2.EVENT_MOUSEMOVE:
                    if self.dragging_box:
                        # Update box position
                        new_x = x - self.box_offset_x
                        new_y = y - self.box_offset_y
                        
                        # Clamp to image boundaries
                        new_x = max(0, min(new_x, w_img - self.measure_box[2]))
                        new_y = max(0, min(new_y, h_img - self.measure_box[3]))
                        
                        self.measure_box[0] = new_x
                        self.measure_box[1] = new_y
                
                elif event == cv2.EVENT_LBUTTONUP:
                    self.dragging_box = False
    
    def _detect_color_region(self, img_left, img_right):
        """
        Detect colored regions in both images and return aligned bounding boxes.
        
        Returns:
        --------
        bbox_left, bbox_right : tuple or None
            Bounding boxes as (x, y, w, h) or None if no region detected
        mask_left, mask_right : numpy.ndarray
            Binary masks of detected regions
        """
        # Convert to HSV color space
        hsv_l = cv2.cvtColor(img_left, cv2.COLOR_BGR2HSV)
        hsv_r = cv2.cvtColor(img_right, cv2.COLOR_BGR2HSV)
        
        # Create color threshold mask
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        
        mask_l = cv2.inRange(hsv_l, lower, upper)
        mask_r = cv2.inRange(hsv_r, lower, upper)
        
        # Apply morphological operations to clean up masks
        kernel = np.ones((5, 5), np.uint8)
        mask_l = cv2.morphologyEx(mask_l, cv2.MORPH_CLOSE, kernel)
        mask_l = cv2.morphologyEx(mask_l, cv2.MORPH_OPEN, kernel)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_CLOSE, kernel)
        mask_r = cv2.morphologyEx(mask_r, cv2.MORPH_OPEN, kernel)
        
        # Find contours in both images
        contours_l, _ = cv2.findContours(mask_l, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_r, _ = cv2.findContours(mask_r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour in each image
        bbox_l = None
        bbox_r = None
        
        if contours_l:
            # Filter by area and get largest
            valid_contours_l = [c for c in contours_l if cv2.contourArea(c) >= self.min_area]
            if valid_contours_l:
                largest_l = max(valid_contours_l, key=cv2.contourArea)
                bbox_l = cv2.boundingRect(largest_l)
        
        if contours_r:
            valid_contours_r = [c for c in contours_r if cv2.contourArea(c) >= self.min_area]
            if valid_contours_r:
                largest_r = max(valid_contours_r, key=cv2.contourArea)
                bbox_r = cv2.boundingRect(largest_r)
        
        # If both regions found, align Y-axis and ensure same dimensions
        if bbox_l is not None and bbox_r is not None:
            x_l, y_l, w_l, h_l = bbox_l
            x_r, y_r, w_r, h_r = bbox_r
            
            # Use shared Y coordinates (take minimum to encompass both)
            y_shared = min(y_l, y_r)
            h_shared = max(y_l + h_l, y_r + h_r) - y_shared
            
            # Ensure height doesn't exceed image bounds
            img_h = img_left.shape[0]
            if y_shared + h_shared > img_h:
                h_shared = img_h - y_shared
            
            # Update bounding boxes with shared Y and height
            bbox_l = (x_l, y_shared, w_l, h_shared)
            bbox_r = (x_r, y_shared, w_r, h_shared)
        
        return bbox_l, bbox_r, mask_l, mask_r
    
    def run_realtime(self):
        """
        Run real-time depth estimation with interactive controls and FPS counter.
        Features color-based region detection for targeted disparity computation.
        Press ESC to exit, R to reset parameters.
        """
        if self.camera_config is None:
            raise ValueError("Camera configuration required for real-time mode")
        
        # Initialize cameras
        cap_left = cv2.VideoCapture(self.camera_config['left_idx'], cv2.CAP_DSHOW)
        cap_right = cv2.VideoCapture(self.camera_config['right_idx'], cv2.CAP_DSHOW)
        
        for cap in [cap_left, cap_right]:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_config['width'])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_config['height'])
            cap.set(cv2.CAP_PROP_FPS, self.camera_config.get('fps', 30))  # Set FPS (default 30)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        # Create separate windows for display and controls
        display_window = 'Stereo Depth Estimation - Press ESC to exit'
        control_window = 'Controls'
        self.control_window_name = control_window  # Store for mouse callback
        
        cv2.namedWindow(display_window, cv2.WINDOW_NORMAL)
        cv2.namedWindow(control_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(control_window, 500, 600)  # Wider window for better label visibility
        
        # Set mouse callback for display window (for measurement box)
        cv2.setMouseCallback(display_window, self._mouse_callback_display, 
                           {'width': self.camera_config['width'], 'height': self.camera_config['height']})
        
        # Create trackbars
        self._create_trackbars(control_window)
        
        print("Real-time depth estimation starting...")
        print("Controls:")
        print("  - ESC: Exit")
        print("  - R: Reset parameters to defaults")
        print("  - M: Toggle measurement box on/off")
        print("  - W/S: Increase/Decrease measurement box height")
        print("  - A/D: Decrease/Increase measurement box width")
        print("  - Drag measurement box to move it (normal mode only)")
        print("  - Adjust trackbars in 'Controls' window to tune parameters")
        print("  - Enable ColorDetect to show HSV window (much faster processing)")
        print("  - Click on HSV window to auto-set HSV range from that pixel")
        print(f"  - Current method: {self.method}")
        
        frame_count = 0
        self.fps_update_time = time.time()
        self.fps_frame_count = 0
        
        while True:
            ret_l, frame_l = cap_left.read()
            ret_r, frame_r = cap_right.read()
            
            if not ret_l or not ret_r:
                break
            
            # Read trackbar values every 5 frames to reduce overhead
            if frame_count % 5 == 0:
                self._read_trackbars(control_window)
            
            # Get image dimensions
            h, w = frame_l.shape[:2]
            
            # Rectify full frames first
            rect_l = cv2.remap(frame_l, self.rect_maps['map1_L'], 
                             self.rect_maps['map2_L'], cv2.INTER_LINEAR)
            rect_r = cv2.remap(frame_r, self.rect_maps['map1_R'], 
                             self.rect_maps['map2_R'], cv2.INTER_LINEAR)
            
            # Apply color detection if enabled
            region_detected = False
            roi_bbox = None
            avg_depth_mm = 0
            
            if self.color_detect_enabled:
                # Detect colored regions in rectified frames
                bbox_l, bbox_r, mask_l, mask_r = self._detect_color_region(rect_l, rect_r)
                
                if bbox_l is not None and bbox_r is not None:
                    region_detected = True
                    # Extract bounding box coordinates
                    x_l, y_l, w_l, h_l = bbox_l
                    x_r, y_r, w_r, h_r = bbox_r
                    roi_bbox = (x_l, y_l, w_l, h_l)
                    
                    # BETTER ROI APPROACH:
                    # Crop HEIGHT to detected region (saves computation)
                    # Keep full WIDTH so stereo matcher has full scanlines to search
                    # Then mask result to show only detected region
                    
                    # Use detected Y-coordinates (height) from left
                    y_crop = y_l
                    h_crop = h_l
                    
                    # Keep full width (all columns) for both images
                    rect_l_roi = rect_l[y_crop:y_crop+h_crop, :]  # All columns
                    rect_r_roi = rect_r[y_crop:y_crop+h_crop, :]  # All columns
                    
                    min_h = min(rect_l_roi.shape[0], rect_r_roi.shape[0])
                    min_w = min(rect_l_roi.shape[1], rect_r_roi.shape[1])
                    rect_l_roi = rect_l_roi[:min_h, :min_w]
                    rect_r_roi = rect_r_roi[:min_h, :min_w]
                    
                    # Debug output
                    if frame_count % 30 == 0:
                        print(f"[CROP] Height: y={y_crop}, h={h_crop} | Width: full ({min_w} cols) | ROI mask: x={x_l}:{x_l+w_l}")
                    
                    # Check if ROI is large enough for SGBM parameters
                    # Need at least: num_disp + block_size + some margin
                    min_width_required = self.num_disp + self.block_size + 20
                    min_height_required = self.block_size + 10
                    roi_too_small = (min_w < min_width_required or min_h < min_height_required)
                    
                    if roi_too_small:
                        # ROI too small - fall back to full frame computation
                        if frame_count % 30 == 0:
                            print(f"\n⚠ ROI too small ({min_w}x{min_h}) for SGBM (need {min_width_required}x{min_height_required})")
                            print(f"  Falling back to full-frame processing...")
                        
                        disparity, depth, disparity_visual = self.compute_disparity_and_depth(
                            rect_l, rect_r, rectify=False
                        )
                        
                        # Extract depth statistics from the ROI region
                        depth_roi = depth[y_l:y_l+h_l, x_l:x_l+w_l]
                        disparity_roi = disparity[y_l:y_l+h_l, x_l:x_l+w_l]
                    else:
                        # ROI large enough - compute disparity on horizontal strip (full width, cropped height)
                        # Matcher has full scanlines to search, then we mask to detected region
                        disparity_strip, _, disparity_visual_strip = self.compute_disparity_and_depth(
                            rect_l_roi, rect_r_roi, rectify=False
                        )
                        
                        # Verify dimensions match expectations
                        if frame_count % 30 == 0:
                            print(f"[DEBUG] Computed disparity on full-width strip: {rect_l_roi.shape[1]}x{rect_l_roi.shape[0]}")
                            print(f"        Masking to detected region: x={x_l}:{x_l+w_l}")
                        
                        # Extract only the detected region from the strip (mask horizontally)
                        # The strip is full width, we want x_l:x_l+w_l columns
                        disparity_roi = disparity_strip[:, x_l:x_l+w_l]
                        disparity_visual_roi = disparity_visual_strip[:, x_l:x_l+w_l]
                        
                        # Use direct depth formula on the masked region
                        depth_roi = np.zeros_like(disparity_roi, dtype=np.float32)
                        valid_disp_mask = disparity_roi > 0
                        
                        # Direct depth calculation (much faster and more robust for ROI)
                        depth_roi[valid_disp_mask] = (self.baseline * self.focal_length) / disparity_roi[valid_disp_mask]
                        
                        # Filter invalid depths in ROI
                        depth_roi[depth_roi <= 0] = 0
                        depth_roi[depth_roi > 10000] = 0
                        
                        # Create full-size arrays for visualization
                        # Place the masked ROI at the correct position
                        disparity = np.zeros((h, w), dtype=np.float32)
                        disparity[y_l:y_l+disparity_roi.shape[0], x_l:x_l+disparity_roi.shape[1]] = disparity_roi
                        
                        depth = np.zeros((h, w), dtype=np.float32)
                        depth[y_l:y_l+depth_roi.shape[0], x_l:x_l+depth_roi.shape[1]] = depth_roi
                        
                        disparity_visual = np.zeros((h, w), dtype=np.uint8)
                        disparity_visual[y_l:y_l+disparity_visual_roi.shape[0], x_l:x_l+disparity_visual_roi.shape[1]] = disparity_visual_roi
                    
                    # Debug: Print disparity and depth statistics (every 30 frames to reduce spam)
                    if frame_count % 30 == 0:
                        valid_disp = disparity_roi[disparity_roi > 0]
                        valid_depth_vals = depth_roi[depth_roi > 0]
                        
                        # Speedup calculation: compare full frame vs horizontal strip
                        strip_pixels = min_w * min_h  # Full-width × cropped-height
                        full_pixels = w * h
                        pixel_reduction = full_pixels / strip_pixels if strip_pixels > 0 else 1
                        processing_mode = "FALLBACK (full)" if roi_too_small else "OPTIMIZED (strip+mask)"
                        
                        print(f"\n=== ROI Processing Stats (frame {frame_count}) ===")
                        print(f"Mode: {processing_mode}")
                        print(f"Detected region: x={x_l}, y={y_l}, w={w_l}, h={h_l}")
                        print(f"Strip size: {min_w}x{min_h} ({strip_pixels:,} px) → Masked to: {w_l}x{h_l} ({w_l*h_l:,} px)")
                        print(f"Full frame: {w}x{h} ({full_pixels:,} px)")
                        print(f"Speedup: {pixel_reduction:.1f}x fewer pixels" if not roi_too_small else f"No speedup (ROI too small)")
                        print(f"Disparity: valid_pixels={len(valid_disp)}, ", end="")
                        if len(valid_disp) > 0:
                            print(f"min={valid_disp.min():.1f}, max={valid_disp.max():.1f}, mean={valid_disp.mean():.1f}")
                        else:
                            print("NO VALID DISPARITY!")
                        
                        print(f"Depth: valid_pixels={len(valid_depth_vals)}, ", end="")
                        if len(valid_depth_vals) > 0:
                            print(f"min={valid_depth_vals.min():.1f}, max={valid_depth_vals.max():.1f}, mean={valid_depth_vals.mean():.1f}mm")
                            avg_depth_mm = valid_depth_vals.mean()  # Calculate here to avoid redundant computation
                        else:
                            print("NO VALID DEPTH!")
                        print("===========================\n")
                    else:
                        # Only calculate avg depth when not printing debug (for display)
                        valid_depth_vals = depth_roi[depth_roi > 0]
                        if len(valid_depth_vals) > 0:
                            avg_depth_mm = valid_depth_vals.mean()
                    
                    # No additional masking needed - already placed at correct position
                    # (disparity, depth, and disparity_visual are zero everywhere except the detected region)
                    
                    # Draw bounding boxes on rectified frames
                    cv2.rectangle(rect_l, (x_l, y_l), (x_l+w_l-1, y_l+h_l-1), (0, 255, 0), 2)
                    cv2.rectangle(rect_r, (x_r, y_r), (x_r+w_r-1, y_r+h_r-1), (0, 255, 0), 2)
                else:
                    # No region detected - show empty/full frame
                    disparity, depth, disparity_visual = self.compute_disparity_and_depth(
                        rect_l, rect_r, rectify=False
                    )
            else:
                # Process full frame (pass already rectified)
                disparity, depth, disparity_visual = self.compute_disparity_and_depth(
                    rect_l, rect_r, rectify=False
                )
            
            # Apply colormaps
            disp_color = cv2.applyColorMap(disparity_visual, cv2.COLORMAP_JET)
            
            # Depth visualization
            depth_vis = depth.copy()
            depth_vis[depth_vis <= 0] = 0
            
            # For ROI mode, normalize based on actual depth values in the region
            if region_detected and roi_bbox is not None and avg_depth_mm > 0:
                x_l, y_l, w_l, h_l = roi_bbox
                
                # Get valid depth values from the ROI region only
                depth_roi_slice = depth[y_l:y_l+h_l, x_l:x_l+w_l]
                valid_depth_vals = depth_roi_slice[depth_roi_slice > 0]
                
                if len(valid_depth_vals) > 0:
                    # Use the max depth from the ROI or the global max_depth_vis setting
                    roi_max_depth = valid_depth_vals.max()
                    if self.max_depth_vis > 0:
                        max_d = min(roi_max_depth, self.max_depth_vis)
                    else:
                        max_d = roi_max_depth
                    
                    # Normalize: values > max_d get clipped to max_d
                    depth_vis_clipped = np.clip(depth_vis, 0, max_d)
                    depth_vis = (depth_vis_clipped / max_d * 255).astype(np.uint8)
                else:
                    depth_vis = depth_vis.astype(np.uint8)
            else:
                # Normal full-frame visualization
                if self.max_depth_vis > 0:
                    depth_vis[depth_vis > self.max_depth_vis] = self.max_depth_vis
                if depth_vis.max() > 0:
                    depth_vis = (depth_vis / self.max_depth_vis * 255).astype(np.uint8)
                else:
                    depth_vis = depth_vis.astype(np.uint8)
            
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
            
            # Draw bounding box on depth and disparity colormaps if region detected
            if region_detected and roi_bbox is not None:
                x_l, y_l, w_l, h_l = roi_bbox
                cv2.rectangle(disp_color, (x_l, y_l), (x_l+w_l-1, y_l+h_l-1), (0, 255, 0), 2)
                cv2.rectangle(depth_color, (x_l, y_l), (x_l+w_l-1, y_l+h_l-1), (0, 255, 0), 2)
            
            # Create 2x2 display (always RGB)
            top_row = np.hstack([rect_l, rect_r])
            bot_row = np.hstack([disp_color, depth_color])
            display = np.vstack([top_row, bot_row])
            
            # Show HSV window when color detection is enabled
            if self.color_detect_enabled:
                hsv_l = cv2.cvtColor(rect_l, cv2.COLOR_BGR2HSV)
                self.hsv_left_image = hsv_l.copy()
                
                # Create HSV window on first use
                if not self.hsv_window_created:
                    cv2.namedWindow(self.hsv_window_name, cv2.WINDOW_NORMAL)
                    cv2.setMouseCallback(self.hsv_window_name, self._mouse_callback_hsv)
                    self.hsv_window_created = True
                    print(f"\n✓ HSV window created: '{self.hsv_window_name}'")
                    print("  Click on the HSV image to auto-set color range\n")
                
                cv2.imshow(self.hsv_window_name, hsv_l)
            else:
                # Close HSV window if color detection is disabled
                if self.hsv_window_created:
                    cv2.destroyWindow(self.hsv_window_name)
                    self.hsv_window_created = False
                    self.hsv_left_image = None
                    print("\n✗ HSV window closed (ColorDetect disabled)\n")
            
            # Calculate FPS (update every 10 frames for stability)
            self.fps_frame_count += 1
            if self.fps_frame_count >= 10:
                current_time = time.time()
                elapsed = current_time - self.fps_update_time
                self.fps = self.fps_frame_count / elapsed
                self.fps_update_time = current_time
                self.fps_frame_count = 0
            
            # Add labels and info
            font = cv2.FONT_HERSHEY_SIMPLEX
            
            # Draw measurement box on normal mode
            if not self.color_detect_enabled and self.measure_box_enabled:
                bx, by, bw, bh = self.measure_box
                
                # Draw box on all views (semi-transparent overlay style)
                cv2.rectangle(display, (bx, by), (bx + bw, by + bh), (0, 255, 255), 2)
                cv2.rectangle(display, (bx + w, by), (bx + bw + w, by + bh), (0, 255, 255), 1)
                cv2.rectangle(display, (bx, by + h), (bx + bw, by + bh + h), (0, 255, 255), 1)
                cv2.rectangle(display, (bx + w, by + h), (bx + bw + w, by + bh + h), (0, 255, 255), 1)
                
                # Calculate average depth in measurement box
                depth_measure_region = depth[by:by+bh, bx:bx+bw]
                valid_depth_measure = depth_measure_region[depth_measure_region > 0]
                
                if len(valid_depth_measure) > 0:
                    avg_measure_depth = np.mean(valid_depth_measure)
                    # Display measurement above the box
                    measure_text = f"{avg_measure_depth:.1f}mm ({avg_measure_depth/10:.1f}cm)"
                    text_y = by - 10 if by > 30 else by + bh + 25
                    cv2.putText(display, measure_text, (bx, text_y), font, 0.6, (0, 255, 255), 2)
            
            # Add labels (always RGB view)
            cv2.putText(display, "Left (Rectified)",  (10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Right (Rectified)", (w + 10, 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Disparity - {self.method}", (10, h + 30), font, 0.7, (255, 255, 255), 2)
            cv2.putText(display, f"Depth (0-{self.max_depth_vis}mm)", (w + 10, h + 30), font, 0.7, (255, 255, 255), 2)
            
            # FPS display (top right corner in green)
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(display, fps_text, (w * 2 - 150, h + 30), font, 0.7, (0, 255, 0), 2)
            
            # Display average depth if region detected
            if region_detected and avg_depth_mm > 0:
                depth_text = f"Avg Depth: {avg_depth_mm:.1f}mm ({avg_depth_mm/10:.1f}cm)"
                cv2.putText(display, depth_text, (w + 10, h + 60), font, 0.6, (0, 255, 255), 2)
            
            # Parameter info with color detection status
            color_status = "ON" if self.color_detect_enabled else "OFF"
            info_text = f"Disp:{self.num_disp} Block:{self.block_size} Uniq:{self.uniqueness_ratio} | Color:{color_status}"
            cv2.putText(display, info_text, (10, h * 2 - 10), font, 0.5, (0, 255, 0), 1)
            
            cv2.imshow(display_window, display)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                break
            elif key == ord('r') or key == ord('R'):  # Reset parameters
                print("Resetting parameters to defaults...")
                self.reset_parameters()
                self._update_trackbars(control_window)
                print(f"Reset complete: {self.method}, numDisp={self.num_disp}, blockSize={self.block_size}")
            elif key == ord('m') or key == ord('M'):  # Toggle measurement box
                self.measure_box_enabled = not self.measure_box_enabled
                print(f"Measurement box: {'ON' if self.measure_box_enabled else 'OFF'}")
            elif key == ord('w') or key == ord('W'):  # Increase height
                self.measure_box[3] = min(self.measure_box[3] + 10, h - self.measure_box[1])
            elif key == ord('s') or key == ord('S'):  # Decrease height
                self.measure_box[3] = max(20, self.measure_box[3] - 10)
            elif key == ord('a') or key == ord('A'):  # Decrease width
                self.measure_box[2] = max(20, self.measure_box[2] - 10)
            elif key == ord('d') or key == ord('D'):  # Increase width
                self.measure_box[2] = min(self.measure_box[2] + 10, w - self.measure_box[0])
            
            frame_count += 1
        
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
        
        print(f"Processed {frame_count} frames")
        print(f"Average FPS: {self.fps:.1f}")
        print(f"Final parameters: {self.method}, numDisp={self.num_disp}, blockSize={self.block_size}")
    
    def get_parameters(self):
        """Return current stereo matching parameters."""
        return {
            'method': self.method,
            'min_disp': self.min_disp,
            'num_disp': self.num_disp,
            'block_size': self.block_size,
            'uniqueness_ratio': self.uniqueness_ratio,
            'speckle_window_size': self.speckle_window_size,
            'speckle_range': self.speckle_range,
            'p1': self.p1,
            'p2': self.p2
        }
    
    def set_parameters(self, **kwargs):
        """Set stereo matching parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # Update P1/P2 if block_size changed
        if 'block_size' in kwargs:
            self.p1 = 8 * 3 * self.block_size**2
            self.p2 = 32 * 3 * self.block_size**2
        
        # Update defaults after setting parameters
        self._defaults = {
            'method': self.method,
            'min_disp': self.min_disp,
            'num_disp': self.num_disp,
            'block_size': self.block_size,
            'uniqueness_ratio': self.uniqueness_ratio,
            'speckle_window_size': self.speckle_window_size,
            'speckle_range': self.speckle_range,
            'max_depth_vis': self.max_depth_vis,
            'color_detect_enabled': self.color_detect_enabled,
            'h_min': self.h_min,
            'h_max': self.h_max,
            's_min': self.s_min,
            's_max': self.s_max,
            'v_min': self.v_min,
            'v_max': self.v_max,
            'min_area': self.min_area
        }

def load_stereo_calibration(filename='stereo_params.npz'):
    """
    Load stereo calibration parameters from file.
    
    Returns a dictionary with all calibration parameters.
    """
    print(f"Loading stereo calibration from {filename}...")
    
    data = np.load(filename)
    
    params = {
        # Monocular intrinsics
        'K_left': data['K_left'],
        'D_left': data['D_left'],
        'K_right': data['K_right'],
        'D_right': data['D_right'],
        
        # Stereo extrinsics
        'R': data['R'],
        'T': data['T'],
        'E': data['E'],
        'F': data['F'],
        
        # Rectification transforms
        'R1': data['R1'],
        'R2': data['R2'],
        'P1': data['P1'],
        'P2': data['P2'],
        'Q': data['Q'],
        
        # Rectification maps
        'map1_L': data['map1_L'],
        'map2_L': data['map2_L'],
        'map1_R': data['map1_R'],
        'map2_R': data['map2_R'],
        
        # Metadata
        'baseline_mm': float(data['baseline_mm']),
        'rotation_deg': data['rotation_deg'],
        'rms_error': float(data['rms_error']),
        'num_images': int(data['num_images']),
        'image_shape': tuple(data['image_shape'])
    }
    
    print(f"Loaded successfully!")
    print(f"\nBaseline: {params['baseline_mm']:.2f} mm")
    print(f"RMS Error: {params['rms_error']:.4f} px")
    print(f"Calibrated with: {params['num_images']} image pairs")
    print(f"Image size: {params['image_shape']}")
    
    return params


if __name__ == "__main__":    # Load calibration parameters

    # config params
    cam_left_idx = 2
    cam_right_idx = 0
    width, height = 640, 480
    PATTERN_SQUARES_X = 9
    PATTERN_SQUARES_Y = 6
    pattern_size = (PATTERN_SQUARES_X, PATTERN_SQUARES_Y)
    loaded_params = load_stereo_calibration('stereo_params.npz')

    # Reconstruct rect_results dictionary
    rect_results = {
        'R1': loaded_params['R1'],
        'R2': loaded_params['R2'],
        'P1': loaded_params['P1'],
        'P2': loaded_params['P2'],
        'Q': loaded_params['Q'],
        'map1_L': loaded_params['map1_L'],
        'map2_L': loaded_params['map2_L'],
        'map1_R': loaded_params['map1_R'],
        'map2_R': loaded_params['map2_R']
    }

    # Initialize the estimator
    camera_cfg = {
        'left_idx': cam_left_idx,
        'right_idx': cam_right_idx,
        'width': width,
        'height': height
    }

    estimator = StereoDepthEstimator(
        calibration_data=rect_results,
        camera_config=camera_cfg
    )

    # Set initial parameters
    estimator.set_parameters(
        method='BM',
        num_disp=96,
        block_size=15,
        uniqueness_ratio=10
    )

    print("="*65)
    print("   REAL-TIME DEPTH ESTIMATION - INTERACTIVE MODE")
    print("="*65)
    print("\nTwo windows:")
    print("  • Main Display (full-size) - 2×2 grid view")
    print("  • Controls (400×280) - All parameter trackbars")
    print("\nTrackbars:")
    print("  Method, NumDisp, BlockSize, Uniqueness,")
    print("  SpeckleWin, SpeckleRng, MaxDepth")
    print("\nPress ESC to exit")
    print("="*65)

    # Run
    estimator.run_realtime()

    # Summary
    print("\n" + "="*65)
    print("   FINAL PARAMETERS")
    print("="*65)
    for key, value in estimator.get_parameters().items():
        print(f"  {key:25s} = {value}")
    print("="*65)