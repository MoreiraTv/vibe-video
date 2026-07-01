import argparse
import json
import os
import subprocess
import tempfile
from types import SimpleNamespace
from pathlib import Path

try:
    import cv2
    import torch
    from ultralytics import YOLO
except ImportError:
    cv2 = SimpleNamespace(
        VideoCapture=None,
        CAP_PROP_FPS=5,
        CAP_PROP_FRAME_COUNT=7,
        CAP_PROP_FRAME_WIDTH=3,
        CAP_PROP_FRAME_HEIGHT=4,
    )
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    YOLO = None


def ensure_tracking_deps() -> None:
    if YOLO is None or getattr(cv2, "VideoCapture", None) is None:
        raise ImportError(
            "ultralytics, torch or opencv-python not installed. Please install them or run `uv sync`"
        )

def apply_ema(values, alpha=0.05):
    """Applies Exponential Moving Average (EMA) for cinematic smoothing."""
    if not values:
        return []
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed

def track_stage(video_path: Path, output_json: Path, alpha: float = 0.05, stride: int = 1):
    """Tracks the most prominent person on stage and applies cinematic smoothing."""
    ensure_tracking_deps()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLOv11 model on device '{device}' for stage tracking on {video_path}...")
    
    model = YOLO("yolo11n.pt")  # Auto-downloads nano model on first run
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    print(f"Tracking {total_frames} frames (Mode: Stage, Stride: {stride})...")
    
    proxy_path = None
    if stride > 1:
        temp_dir = Path(tempfile.gettempdir())
        proxy_path = temp_dir / f"proxy_{video_path.stem}.mp4"
        print(f"Creating optimized proxy video with FFMPEG (stride={stride}) to bypass OpenCV decoding bottleneck...")
        
        if device == "cuda":
            cmd = [
                "ffmpeg", "-y", 
                "-hwaccel", "cuda",
                "-i", str(video_path),
                "-vf", f"framestep={stride}",
                "-an", 
                "-c:v", "h264_nvenc", 
                "-preset", "fast", 
                str(proxy_path)
            ]
        else:
            cmd = [
                "ffmpeg", "-y", 
                "-i", str(video_path),
                "-vf", f"framestep={stride}",
                "-an", 
                "-c:v", "libx264", 
                "-preset", "ultrafast", 
                "-crf", "28",
                str(proxy_path)
            ]
            
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            tracking_source = str(proxy_path)
            yolo_stride = 1 # Video is already strided
        else:
            print(f"FFMPEG Error, falling back to OpenCV: {res.stderr[:200]}")
            tracking_source = str(video_path)
            yolo_stride = stride
    else:
        tracking_source = str(video_path)
        yolo_stride = 1
    
    # We use YOLO built-in tracking (ByteTrack). 
    # imgsz=480 (multiple of 32) speeds up inference without degrading large-body detection.
    results = model.track(
        source=tracking_source, 
        tracker="bytetrack.yaml", 
        stream=True, 
        verbose=False, 
        classes=[0], # class 0 is 'person'
        device=device,
        imgsz=480,
        vid_stride=yolo_stride
    )

    raw_centers_x = []
    raw_centers_y = []
    frame_indices = []
    
    target_id = None
    
    for r in results:
        boxes = r.boxes
        
        # Determine cx, cy for this inference step
        cx, cy = None, None
        
        if boxes is not None and boxes.id is not None and len(boxes.id) > 0:
            if target_id is None:
                max_area = 0
                for box in boxes:
                    if box.id is not None:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            target_id = int(box.id[0])
            
            found_target = False
            for box in boxes:
                if box.id is not None and int(box.id[0]) == target_id:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    found_target = True
                    break
                    
            if not found_target and raw_centers_x:
                cx = raw_centers_x[-1]
                cy = raw_centers_y[-1]
        elif raw_centers_x:
            cx = raw_centers_x[-1]
            cy = raw_centers_y[-1]
            
        # If we still don't have a cx, we haven't found anyone yet. Skip padding.
        if cx is not None and cy is not None:
            # Duplicate the reading for 'stride' amount of frames to fill gaps
            for _ in range(stride):
                raw_centers_x.append(cx)
                raw_centers_y.append(cy)
                frame_indices.append(len(frame_indices) + 1)
        else:
            # We haven't found a target yet, just pad with zeros or skip
            for _ in range(stride):
                raw_centers_x.append(0)
                raw_centers_y.append(0)
                frame_indices.append(len(frame_indices) + 1)

    # Truncate any overshoot caused by the stride loop
    raw_centers_x = raw_centers_x[:total_frames]
    raw_centers_y = raw_centers_y[:total_frames]
    frame_indices = frame_indices[:total_frames]
    
    print("Applying cinematic smoothing (EMA)...")
    # EMA naturally smooths out the "stair-steps" created by the stride duplication!
    smooth_x = apply_ema(raw_centers_x, alpha=alpha)
    smooth_y = apply_ema(raw_centers_y, alpha=alpha)
    
    # Package into JSON
    tracking_data = {
        "mode": "stage",
        "fps": fps,
        "total_frames": total_frames,
        "target_id": target_id,
        "frames": []
    }
    
    for i in range(len(frame_indices)):
        timestamp = frame_indices[i] / fps
        tracking_data["frames"].append({
            "frame": frame_indices[i],
            "time": timestamp,
            "cx": round(smooth_x[i], 2),
            "cy": round(smooth_y[i], 2),
            "raw_cx": round(raw_centers_x[i], 2),
            "raw_cy": round(raw_centers_y[i], 2)
        })
        
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(tracking_data, f, indent=2)
        
    if proxy_path and proxy_path.exists():
        try:
            os.remove(proxy_path)
        except Exception:
            pass
            
    print(f"Tracking complete. Saved cinematic coordinates to {output_json}")

def track_podcast(video_path: Path, output_json: Path, alpha: float = 0.2, stride: int = 1):
    """Tracks active speakers in podcast format by dynamically focusing on the most prominent person in each frame."""
    ensure_tracking_deps()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLOv11 model on device '{device}' for podcast tracking on {video_path}...")
    
    model = YOLO("yolo11n.pt")  # Auto-downloads nano model on first run
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Default center
    default_cx = video_w / 2
    default_cy = video_h / 2
    
    print(f"Tracking {total_frames} frames (Mode: Podcast, Stride: {stride})...")
    
    proxy_path = None
    if stride > 1:
        temp_dir = Path(tempfile.gettempdir())
        proxy_path = temp_dir / f"proxy_{video_path.stem}.mp4"
        print(f"Creating optimized proxy video with FFMPEG (stride={stride}) to bypass OpenCV decoding bottleneck...")
        
        if device == "cuda":
            cmd = [
                "ffmpeg", "-y", 
                "-hwaccel", "cuda",
                "-i", str(video_path),
                "-vf", f"framestep={stride}",
                "-an", 
                "-c:v", "h264_nvenc", 
                "-preset", "fast", 
                str(proxy_path)
            ]
        else:
            cmd = [
                "ffmpeg", "-y", 
                "-i", str(video_path),
                "-vf", f"framestep={stride}",
                "-an", 
                "-c:v", "libx264", 
                "-preset", "ultrafast", 
                "-crf", "28",
                str(proxy_path)
            ]
            
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            tracking_source = str(proxy_path)
            yolo_stride = 1 # Video is already strided
        else:
            print(f"FFMPEG Error, falling back to OpenCV: {res.stderr[:200]}")
            tracking_source = str(video_path)
            yolo_stride = stride
    else:
        tracking_source = str(video_path)
        yolo_stride = 1
    
    results = model.track(
        source=tracking_source, 
        tracker="bytetrack.yaml", 
        stream=True, 
        verbose=False, 
        classes=[0], # class 0 is 'person'
        device=device,
        imgsz=480,
        vid_stride=yolo_stride
    )

    raw_centers_x = []
    raw_centers_y = []
    frame_indices = []
    
    for r in results:
        boxes = r.boxes
        cx, cy = None, None
        
        if boxes is not None and len(boxes) > 0:
            # Find the box with the maximum area (most prominent person)
            max_area = 0
            best_box = None
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    best_box = box
            
            if best_box is not None:
                x1, y1, x2, y2 = best_box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                
        if cx is None or cy is None:
            # Fallback to last known position, or center if none
            if raw_centers_x:
                cx = raw_centers_x[-1]
                cy = raw_centers_y[-1]
            else:
                cx = default_cx
                cy = default_cy
                
        for _ in range(stride):
            raw_centers_x.append(cx)
            raw_centers_y.append(cy)
            frame_indices.append(len(frame_indices) + 1)

    # Truncate any overshoot caused by the stride loop
    raw_centers_x = raw_centers_x[:total_frames]
    raw_centers_y = raw_centers_y[:total_frames]
    frame_indices = frame_indices[:total_frames]
    
    print("Applying cinematic smoothing (EMA)...")
    # For podcast mode, we want a faster response to switch focus quickly
    smooth_x = apply_ema(raw_centers_x, alpha=alpha)
    smooth_y = apply_ema(raw_centers_y, alpha=alpha)
    
    # Package into JSON
    tracking_data = {
        "mode": "podcast",
        "fps": fps,
        "total_frames": total_frames,
        "frames": []
    }
    
    for i in range(len(frame_indices)):
        timestamp = frame_indices[i] / fps
        tracking_data["frames"].append({
            "frame": frame_indices[i],
            "time": timestamp,
            "cx": round(smooth_x[i], 2),
            "cy": round(smooth_y[i], 2),
            "raw_cx": round(raw_centers_x[i], 2),
            "raw_cy": round(raw_centers_y[i], 2)
        })
        
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(tracking_data, f, indent=2)
        
    if proxy_path and proxy_path.exists():
        try:
            os.remove(proxy_path)
        except Exception:
            pass
            
    print(f"Tracking complete. Saved cinematic coordinates to {output_json}")

def track_vlog(video_path: Path, output_json: Path, alpha: float = 0.02, stride: int = 5):
    """Tracks groups of people in vlog format using Center of Mass and a Deadzone."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLOv11 model on device '{device}' for vlog tracking on {video_path}...")
    
    model = YOLO("yolo11n.pt")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    default_cx = video_w / 2
    default_cy = video_h / 2
    
    # Deadzone is 20% of width and 10% of height
    deadzone_w = video_w * 0.20
    deadzone_h = video_h * 0.10
    
    print(f"Tracking {total_frames} frames (Mode: Vlog, Stride: {stride})...")
    
    proxy_path = None
    if stride > 1:
        temp_dir = Path(tempfile.gettempdir())
        proxy_path = temp_dir / f"proxy_{video_path.stem}.mp4"
        print(f"Creating optimized proxy video with FFMPEG (stride={stride}) to bypass OpenCV decoding bottleneck...")
        
        if device == "cuda":
            cmd = [
                "ffmpeg", "-y", 
                "-hwaccel", "cuda",
                "-i", str(video_path),
                "-vf", f"framestep={stride}",
                "-an", 
                "-c:v", "h264_nvenc", 
                "-preset", "fast", 
                str(proxy_path)
            ]
        else:
            cmd = [
                "ffmpeg", "-y", 
                "-i", str(video_path),
                "-vf", f"framestep={stride}",
                "-an", 
                "-c:v", "libx264", 
                "-preset", "ultrafast", 
                "-crf", "28",
                str(proxy_path)
            ]
            
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            tracking_source = str(proxy_path)
            yolo_stride = 1
        else:
            print(f"FFMPEG Error, falling back to OpenCV: {res.stderr[:200]}")
            tracking_source = str(video_path)
            yolo_stride = stride
    else:
        tracking_source = str(video_path)
        yolo_stride = 1
    
    results = model.track(
        source=tracking_source, 
        tracker="bytetrack.yaml", 
        stream=True, 
        verbose=False, 
        classes=[0], # class 0 is 'person'
        device=device,
        imgsz=480,
        vid_stride=yolo_stride
    )

    raw_centers_x = []
    raw_centers_y = []
    frame_indices = []
    
    target_cx = default_cx
    target_cy = default_cy
    
    for r in results:
        boxes = r.boxes
        cx, cy = None, None
        
        if boxes is not None and len(boxes) > 0:
            max_area = 0
            valid_boxes = []
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1)
                valid_boxes.append((area, (x1 + x2) / 2, (y1 + y2) / 2))
                if area > max_area:
                    max_area = area
            
            # Filter boxes that are at least 30% the size of the largest person
            prominent_boxes = [b for b in valid_boxes if b[0] >= max_area * 0.3]
            
            if prominent_boxes:
                # Center of mass
                mass_cx = sum(b[1] for b in prominent_boxes) / len(prominent_boxes)
                mass_cy = sum(b[2] for b in prominent_boxes) / len(prominent_boxes)
                
                # Apply Deadzone logic
                if mass_cx > target_cx + deadzone_w / 2:
                    target_cx = mass_cx - deadzone_w / 2
                elif mass_cx < target_cx - deadzone_w / 2:
                    target_cx = mass_cx + deadzone_w / 2
                    
                if mass_cy > target_cy + deadzone_h / 2:
                    target_cy = mass_cy - deadzone_h / 2
                elif mass_cy < target_cy - deadzone_h / 2:
                    target_cy = mass_cy + deadzone_h / 2
                    
                cx, cy = target_cx, target_cy
                
        if cx is None or cy is None:
            cx = target_cx
            cy = target_cy
                
        for _ in range(stride):
            raw_centers_x.append(cx)
            raw_centers_y.append(cy)
            frame_indices.append(len(frame_indices) + 1)

    raw_centers_x = raw_centers_x[:total_frames]
    raw_centers_y = raw_centers_y[:total_frames]
    frame_indices = frame_indices[:total_frames]
    
    print("Applying vlog cinematic smoothing (High EMA)...")
    smooth_x = apply_ema(raw_centers_x, alpha=alpha)
    smooth_y = apply_ema(raw_centers_y, alpha=alpha)
    
    tracking_data = {
        "mode": "vlog",
        "fps": fps,
        "total_frames": total_frames,
        "frames": []
    }
    
    for i in range(len(frame_indices)):
        timestamp = frame_indices[i] / fps
        tracking_data["frames"].append({
            "frame": frame_indices[i],
            "time": timestamp,
            "cx": round(smooth_x[i], 2),
            "cy": round(smooth_y[i], 2),
            "raw_cx": round(raw_centers_x[i], 2),
            "raw_cy": round(raw_centers_y[i], 2)
        })
        
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(tracking_data, f, indent=2)
        
    if proxy_path and proxy_path.exists():
        try:
            os.remove(proxy_path)
        except Exception:
            pass
            
    print(f"Tracking complete. Saved cinematic coordinates to {output_json}")

def main():
    parser = argparse.ArgumentParser(description="Active Speaker and Stage Tracking (PTZ Virtual Camera)")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--edit-dir", type=Path, default=Path("edit"), help="Output directory")
    parser.add_argument("--mode", choices=["stage", "podcast", "vlog"], default="stage", 
                        help="Tracking mode. 'stage' tracks one body with cinematic smoothing. 'podcast' tracks multiple speakers. 'vlog' tracks groups with deadzones and high smoothing.")
    parser.add_argument("--smoothing", type=float, default=0.05, 
                        help="EMA alpha for smoothing (lower = smoother/slower, higher = faster/jittery)")
    parser.add_argument("--stride", type=int, default=5, 
                        help="Frame skip (e.g. 5 = process 1 frame every 5). Default is 5 for a great speed/accuracy balance.")
    
    args = parser.parse_args()
    
    if not args.video.exists():
        print(f"Error: Video file {args.video} not found.")
        exit(1)
        
    output_json = args.edit_dir / f"{args.video.stem}_tracking.json"
    
    # Adjust default smoothing for podcast/vlog if not explicitly set to custom value
    smoothing = args.smoothing
    if args.mode == "podcast" and smoothing == 0.05:
        smoothing = 0.2  # Faster response to focus cuts in podcast mode
    elif args.mode == "vlog" and smoothing == 0.05:
        smoothing = 0.02 # High smoothing for shaky cameras
        
    if args.mode == "stage":
        track_stage(args.video, output_json, alpha=smoothing, stride=args.stride)
    elif args.mode == "podcast":
        track_podcast(args.video, output_json, alpha=smoothing, stride=args.stride)
    elif args.mode == "vlog":
        track_vlog(args.video, output_json, alpha=smoothing, stride=args.stride)

if __name__ == "__main__":
    main()
