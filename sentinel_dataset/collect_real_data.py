#!/usr/bin/env python3
"""
SENTINEL — Real data collection script
Run on Raspberry Pi 4 with Camera Module v2 connected via CSI.

Usage:
  python3 collect_real_data.py --output /home/pi/sentinel_real --frames 2500

Dependencies (on RPi):
  sudo apt install -y python3-picamera2
  pip3 install opencv-python-headless numpy

Camera: Sony IMX219 | 640x480 @ 30fps | FOV 62.2° H x 48.8° V
"""

import argparse
import time
import os
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output",  default="/home/pi/sentinel_real",
                   help="Output directory")
    p.add_argument("--frames",  type=int, default=2500,
                   help="Number of frames to collect")
    p.add_argument("--fps",     type=int, default=5,
                   help="Collection framerate (extract every N frames from 30fps stream)")
    p.add_argument("--preview", action="store_true",
                   help="Show preview window (requires display)")
    return p.parse_args()

def main():
    args = parse_args()
    out  = Path(args.output)
    (out / "images").mkdir(parents=True, exist_ok=True)

    # ── Initialise camera ─────────────────────────────────────
    try:
        from picamera2 import Picamera2
        cam = Picamera2()
        config = cam.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            controls={"FrameRate": 30}
        )
        cam.configure(config)
        cam.start()
        time.sleep(2)   # warm-up
        use_picamera2 = True
        print("✓ picamera2 initialised (640x480 @ 30fps)")
    except ImportError:
        # Fallback: OpenCV VideoCapture (older systems)
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_FPS, 30)
        use_picamera2 = False
        print("✓ OpenCV VideoCapture initialised")

    # ── Collection loop ───────────────────────────────────────
    interval    = 30 // args.fps    # capture every N frames
    frame_count = 0
    saved_count = 0
    metadata    = []
    session_id  = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\nStarting collection: {args.frames} frames at ~{args.fps}fps")
    print("Drive the car along the track. Press Ctrl+C to stop.")
    print(f"Output: {out}/images/")

    try:
        while saved_count < args.frames:
            # Capture frame
            if use_picamera2:
                frame_rgb = cam.capture_array()
                frame     = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            else:
                ret, frame = cam.read()
                if not ret:
                    continue

            frame_count += 1
            if frame_count % interval != 0:
                continue

            # ── Preprocessing (match synthetic pipeline) ──────
            # Resize to training resolution
            frame_small = cv2.resize(frame, (320, 240))

            # Convert to HSV for quick QA check
            hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)

            # Save frame
            fname    = f"real_{session_id}_{saved_count:06d}.png"
            img_path = out / "images" / fname
            cv2.imwrite(str(img_path), frame_small)

            # Basic metadata (no labels yet — labeling done in next step)
            metadata.append({
                "image":     f"images/{fname}",
                "source":    "real",
                "has_lane":  None,   # to be labeled
                "timestamp": time.time(),
                "frame_num": frame_count,
            })

            saved_count += 1

            if saved_count % 100 == 0:
                print(f"  Saved {saved_count}/{args.frames} frames")

            # Optional preview
            if args.preview:
                cv2.imshow("Collection preview", frame_small)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print(f"\nStopped by user at {saved_count} frames")

    finally:
        if use_picamera2:
            cam.stop()
        else:
            cam.release()
        if args.preview:
            cv2.destroyAllWindows()

    # Save metadata
    meta_path = out / f"metadata_{session_id}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Collection done")
    print(f"  Frames saved: {saved_count}")
    print(f"  Metadata:     {meta_path}")
    print(f"\nNext step: transfer {out} to your laptop and run")
    print(f"  label_real_data.ipynb  (or add labels in this notebook)")

if __name__ == "__main__":
    main()
