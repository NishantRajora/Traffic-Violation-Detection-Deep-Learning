"""
Traffic Violation Detection - Live Video Object Detection
Detects 'helmet', 'no_helmet', and 'overloading' with bounding boxes in real-time video playback.
Default video: v1.mp4 (Does NOT save files to disk; displays live playback window directly).
"""

import os
# Prevent OpenMP runtime conflict crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_MODEL_PATHS = [
    BASE_DIR / "saved_models" / "helmet_detection_model.pt",
    BASE_DIR / "runs" / "detect" / "Traffic_Detection_Runs" / "helmet_detection-5" / "weights" / "best.pt",
    BASE_DIR / "runs" / "detect" / "Traffic_Detection_Runs" / "helmet_detection" / "weights" / "best.pt",
]

# Color codes in BGR for OpenCV overlay
CLASS_COLORS = {
    "helmet": (34, 197, 94),      # Green (BGR)
    "no_helmet": (68, 68, 239),   # Red (BGR)
    "overloading": (38, 38, 220), # Dark Red (BGR)
}


def locate_model() -> Path:
    """Finds the trained YOLO detection weights."""
    for path in CANDIDATE_MODEL_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find detection model weights! "
        "Ensure saved_models/helmet_detection_model.pt exists."
    )


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    total_frames: int,
    counts: dict,
    fps: float,
    device_name: str
) -> np.ndarray:
    """Draws a professional Heads-Up Display (HUD) banner at the top of the frame."""
    h, w, _ = frame.shape
    banner_height = 68

    # Create semi-transparent overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (12, 16, 24), -1)

    violation_present = counts.get("no_helmet", 0) > 0 or counts.get("overloading", 0) > 0
    compliant_present = counts.get("helmet", 0) > 0

    if violation_present:
        status_text = f"TRAFFIC VIOLATION DETECTED: {counts.get('no_helmet', 0)} No Helmet, {counts.get('overloading', 0)} Overloaded"
        status_color = (68, 68, 239)   # Red
        border_color = (68, 68, 239)
    elif compliant_present:
        status_text = f"COMPLIANT / SAFE: {counts.get('helmet', 0)} Helmet(s) Detected"
        status_color = (34, 197, 94)   # Green
        border_color = (34, 197, 94)
    else:
        status_text = "SCANNING TRAFFIC: No Targets Above Threshold"
        status_color = (148, 163, 184) # Muted gray
        border_color = (40, 50, 70)

    # Blend overlay
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    cv2.line(frame, (0, banner_height), (w, banner_height), border_color, 2)

    # Status title text
    cv2.putText(
        frame,
        status_text,
        (16, 26),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        status_color,
        2,
        cv2.LINE_AA
    )

    # Subtext: Detailed counters, FPS and progress
    progress_str = f"Frame: {frame_idx}/{total_frames}" if total_frames > 0 else f"Frame: {frame_idx}"
    telemetry_text = (
        f"Helmets: {counts.get('helmet', 0)} | "
        f"No-Helmets: {counts.get('no_helmet', 0)} | "
        f"FPS: {fps:.1f} | {device_name} | {progress_str} | [Q: Quit, Space: Pause]"
    )

    cv2.putText(
        frame,
        telemetry_text,
        (16, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 225, 235),
        1,
        cv2.LINE_AA
    )

    return frame


def process_video(
    source_path: str = "v1.mp4",
    conf_threshold: float = 0.15,
    save_output: bool = False,
    output_path: str = None,
    show_window: bool = True
):
    """
    Processes video frame-by-frame with YOLO detection and displays a live playback window.
    Does NOT save output files to disk unless explicitly requested via save_output=True.
    """
    
    # 1. Device and Model Initialization
    if torch.cuda.is_available():
        device = 0
        device_name = f"GPU ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
        device_name = "CPU"

    model_path = locate_model()
    print("=" * 70)
    print("  TRAFFIC VIOLATION DETECTION - LIVE PLAYBACK")
    print("=" * 70)
    print(f"  Model Weights : {model_path.name}")
    print(f"  Hardware      : {device_name}")
    print(f"  Source Video  : {source_path}")
    print(f"  Confidence    : {int(conf_threshold * 100)}% ({conf_threshold})")
    print(f"  Live Window   : {'ACTIVE' if show_window else 'DISABLED'}")
    print(f"  File Saving   : {'ENABLED (' + str(output_path) + ')' if save_output else 'DISABLED (No files saved)'}")
    print("=" * 70)

    model = YOLO(str(model_path))

    # 2. Open Video Capture
    is_webcam = str(source_path).isdigit()
    cap_source = int(source_path) if is_webcam else str(source_path)
    cap = cv2.VideoCapture(cap_source)

    if not cap.isOpened():
        print(f"ERROR: Could not open video source '{source_path}'")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if fps_in <= 0 or np.isnan(fps_in):
        fps_in = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1
    target_delay_ms = int(1000 / fps_in)

    # 3. Setup Video Writer ONLY if explicitly requested
    writer = None
    if save_output:
        if not output_path:
            src_name = Path(source_path).stem if not is_webcam else "webcam"
            output_path = str(BASE_DIR / f"{src_name}_detected.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps_in, (width, height))
        print(f"  Writing to    : {output_path}")

    # 4. Processing & Live Playback Loop
    frame_idx = 0
    total_violation_frames = 0
    total_compliant_frames = 0
    class_detections_count = {"helmet": 0, "no_helmet": 0, "overloading": 0}
    start_total_time = time.perf_counter()
    fps_smooth = fps_in

    window_name = f"Traffic Violation Detection - {Path(source_path).name}"

    print("\nLive playback started!")
    print("  - Press 'Q' or 'ESC' to exit")
    print("  - Press 'SPACE' or 'P' to pause/resume\n")

    try:
        while True:
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Run YOLO Bounding Box Detection
            results = model.predict(
                source=frame,
                device=device,
                conf=conf_threshold,
                verbose=False
            )
            result = results[0]

            # Render Bounding Boxes on image
            annotated_frame = result.plot(line_width=2, font_size=1)

            # Count classes for this frame
            frame_counts = {"helmet": 0, "no_helmet": 0, "overloading": 0}
            if result.boxes is not None and len(result.boxes) > 0:
                for b in result.boxes:
                    cls_id = int(b.cls[0])
                    cls_name = model.names.get(cls_id, "")
                    if cls_name in frame_counts:
                        frame_counts[cls_name] += 1
                        class_detections_count[cls_name] += 1

            if frame_counts["no_helmet"] > 0 or frame_counts["overloading"] > 0:
                total_violation_frames += 1
            elif frame_counts["helmet"] > 0:
                total_compliant_frames += 1

            # Calculate FPS
            t_diff = time.perf_counter() - t0
            current_fps = 1.0 / t_diff if t_diff > 0 else 30.0
            fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps

            # Draw top HUD
            display_frame = draw_hud(
                annotated_frame,
                frame_idx,
                total_frames,
                frame_counts,
                fps_smooth,
                device_name
            )

            # Write frame only if saving is enabled
            if writer:
                writer.write(display_frame)

            # Show in Live Window
            if show_window:
                cv2.imshow(window_name, display_frame)

                # Sync playback timing with video's native frame rate
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                wait_ms = max(1, target_delay_ms - elapsed_ms)
                key = cv2.waitKey(wait_ms) & 0xFF

                # Quit on 'q' or ESC (27)
                if key in (ord("q"), ord("Q"), 27):
                    print("\nLive playback closed by user.")
                    break

                # Pause on 'p' or SPACE
                elif key in (ord("p"), ord("P"), 32):
                    print("  [PAUSED] Press Space or Enter to resume, Q to quit...")
                    while True:
                        k = cv2.waitKey(50) & 0xFF
                        if k in (ord("p"), ord("P"), 32, 13):
                            print("  [RESUMED]")
                            break
                        elif k in (ord("q"), ord("Q"), 27):
                            return

            # Periodic console progress
            if frame_idx % 25 == 0 or frame_idx == total_frames:
                prog = f"[{frame_idx}/{total_frames}]" if total_frames > 0 else f"[{frame_idx}]"
                print(
                    f"  {prog} FPS: {fps_smooth:.1f} | "
                    f"Violations: {frame_counts['no_helmet']} | "
                    f"Compliant: {frame_counts['helmet']}"
                )

    finally:
        cap.release()
        if writer:
            writer.release()
        if show_window:
            cv2.destroyAllWindows()

    total_time = time.perf_counter() - start_total_time
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    # 5. Summary Report
    print("\n" + "=" * 70)
    print("  PLAYBACK COMPLETE - SUMMARY REPORT")
    print("=" * 70)
    print(f"  Total Frames Viewed      : {frame_idx}")
    print(f"  Playback Time            : {total_time:.2f} seconds")
    print(f"  Average Processing Speed : {avg_fps:.1f} FPS")
    print(f"  Violation Frames         : {total_violation_frames} ({(total_violation_frames/max(1,frame_idx))*100:.1f}%)")
    print(f"  Compliant Frames         : {total_compliant_frames} ({(total_compliant_frames/max(1,frame_idx))*100:.1f}%)")
    print("  --------------------------------------------------")
    print(f"  Total 'Helmet' Detections     : {class_detections_count['helmet']}")
    print(f"  Total 'No Helmet' Detections  : {class_detections_count['no_helmet']}  🚨 (Violation)")
    print(f"  Total 'Overloading' Detections: {class_detections_count['overloading']}  🚨 (Violation)")
    if save_output and output_path:
        print(f"  Saved File                    : {output_path}")
    else:
        print("  Disk Storage                  : No video files were written to disk.")
    print("=" * 70)


def interactive_menu():
    """Interactive prompt when run without arguments."""
    print("=" * 70)
    print("     🚦 TRAFFIC VIOLATION DETECTION - LIVE VIDEO PLAYBACK 🚦")
    print("=" * 70)
    print("  Select video source:")
    print("    [1] Default sample video: v1.mp4")
    print("    [2] Custom video file path")
    print("    [3] Live Webcam (Device 0)")
    print("=" * 70)

    choice = input("Enter choice (1/2/3) [Default: 1]: ").strip()

    if choice == "2":
        source = input("Enter video file path: ").strip().strip('"').strip("'")
        if not Path(source).exists():
            print(f"File not found: {source}. Falling back to 'v1.mp4'")
            source = "v1.mp4"
    elif choice == "3":
        source = "0"
    else:
        source = "v1.mp4"

    # Confidence prompt
    conf_input = input("Enter confidence threshold (5-80%) [Default: 15]: ").strip()
    try:
        conf_val = float(conf_input) / 100.0 if float(conf_input) > 1.0 else float(conf_input)
    except ValueError:
        conf_val = 0.15

    # Process and display live window (save_output=False ensures NO file is created)
    process_video(
        source_path=source,
        conf_threshold=conf_val,
        save_output=False,
        show_window=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Violation Object Detection on Video (Live Playback Window, No File Saving by Default)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to input video file (e.g. v1.mp4) or '0' for webcam."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.15,
        help="Confidence threshold between 0.05 and 0.90 (default: 0.15)"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Optional: save annotated video to disk (default: False - live playback only)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated video if --save is passed."
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable live playback window (headless mode)"
    )

    args = parser.parse_args()

    # If CLI arguments are provided
    if args.source is not None:
        process_video(
            source_path=args.source,
            conf_threshold=args.conf,
            save_output=args.save,
            output_path=args.output,
            show_window=not args.no_show
        )
    else:
        # Run friendly interactive prompt
        interactive_menu()
