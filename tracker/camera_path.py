import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from typing import List, Dict, Tuple, Optional

class CameraPath:
    def __init__(self, frame_count: int):
        self.frame_count = frame_count
        self.raw_points: Dict[int, Dict] = {} # frame_idx -> {x, y, bbox_w, bbox_h}
        self.keyframes: List[Dict] = [] # List of {frame, x, y}

    def add_raw_point(self, frame_idx: int, data: Dict):
        self.raw_points[frame_idx] = data

    def add_keyframe(self, frame_idx: int, x: float, y: float):
        # Remove if already exists for this frame
        self.keyframes = [kf for kf in self.keyframes if kf['frame'] != frame_idx]
        self.keyframes.append({"frame": frame_idx, "x": x, "y": y})
        self.keyframes.sort(key=lambda k: k['frame'])

    def interpolate_keyframes(self, method: str = "cubic") -> List[Tuple[float, float]]:
        """Interpolates between keyframes to create a full path."""
        if len(self.keyframes) < 2:
            # Not enough keyframes to interpolate
            if len(self.keyframes) == 1:
                kf = self.keyframes[0]
                return [(kf['x'], kf['y'])] * self.frame_count
            return []

        frames = [kf['frame'] for kf in self.keyframes]
        x_coords = [kf['x'] for kf in self.keyframes]
        y_coords = [kf['y'] for kf in self.keyframes]

        # Ensure first and last frames are covered if not already
        if frames[0] > 0:
            frames.insert(0, 0)
            x_coords.insert(0, x_coords[0])
            y_coords.insert(0, y_coords[0])
        if frames[-1] < self.frame_count - 1:
            frames.append(self.frame_count - 1)
            x_coords.append(x_coords[-1])
            y_coords.append(y_coords[-1])

        target_frames = np.arange(self.frame_count)
        
        if method == "cubic" and len(frames) >= 4:
            cs_x = CubicSpline(frames, x_coords)
            cs_y = CubicSpline(frames, y_coords)
            interp_x = cs_x(target_frames)
            interp_y = cs_y(target_frames)
        elif method == "bezier":
            # Simple bezier approximation using interp1d with quadratic/cubic
            kind = 'quadratic' if len(frames) == 3 else 'cubic' if len(frames) > 3 else 'linear'
            f_x = interp1d(frames, x_coords, kind=kind)
            f_y = interp1d(frames, y_coords, kind=kind)
            interp_x = f_x(target_frames)
            interp_y = f_y(target_frames)
        else: # Linear
            f_x = interp1d(frames, x_coords, kind='linear')
            f_y = interp1d(frames, y_coords, kind='linear')
            interp_x = f_x(target_frames)
            interp_y = f_y(target_frames)

        return list(zip(interp_x, interp_y))

    def get_raw_path(self) -> List[Tuple[float, float]]:
        """Returns the raw tracking path, filling gaps with linear interpolation."""
        if not self.raw_points:
            return [(0.0, 0.0)] * self.frame_count
            
        indices = sorted(self.raw_points.keys())
        x_coords = [self.raw_points[i]['x'] for i in indices]
        y_coords = [self.raw_points[i]['y'] for i in indices]
        
        if len(indices) == 1:
            return [(x_coords[0], y_coords[0])] * self.frame_count
            
        # Fill gaps
        target_frames = np.arange(self.frame_count)
        f_x = interp1d(indices, x_coords, kind='linear', fill_value="extrapolate")
        f_y = interp1d(indices, y_coords, kind='linear', fill_value="extrapolate")
        
        interp_x = f_x(target_frames)
        interp_y = f_y(target_frames)
        
        return list(zip(interp_x, interp_y))

    def get_bbox_path(self) -> List[Tuple[float, float]]:
        """Returns the bounding box size path (w, h)."""
        if not self.raw_points:
            return [(0.0, 0.0)] * self.frame_count
            
        indices = sorted(self.raw_points.keys())
        w_coords = [self.raw_points[i].get('bbox_w', 0) for i in indices]
        h_coords = [self.raw_points[i].get('bbox_h', 0) for i in indices]
        
        if len(indices) == 1:
            return [(w_coords[0], h_coords[0])] * self.frame_count
            
        target_frames = np.arange(self.frame_count)
        f_w = interp1d(indices, w_coords, kind='linear', fill_value="extrapolate")
        f_h = interp1d(indices, h_coords, kind='linear', fill_value="extrapolate")
        
        interp_w = f_w(target_frames)
        interp_h = f_h(target_frames)
        
        return list(zip(interp_w, interp_h))
