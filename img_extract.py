import cv2
import os
import shutil

# =====================================
# CONFIG
# =====================================
video_path = "Data/Raw_data/Outdoorfail3.mp4"
output_dir = "Data/Processed_data/Fail_data"
interval_ms = 200

# Target output resolution
target_width = 1280
target_height = 720

CLEAR_TARGET_FOLDER = False
# =====================================


def format_ratio(r):
    return f"{r:.4f}"


# =====================================
# OUTPUT DIRECTORY HANDLING
# =====================================
if CLEAR_TARGET_FOLDER and os.path.exists(output_dir):
    print(f"Clearing folder: {output_dir}")
    shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=True)


# =====================================
# VIDEO SETUP
# =====================================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise Exception("Error: Could not open video.")

video_filename = os.path.basename(video_path)
video_name = os.path.splitext(video_filename)[0]

# =========================
# VIDEO PROPERTIES
# =========================
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration_sec = frame_count_total / fps
duration_ms = duration_sec * 1000

orig_ar = orig_width / orig_height

# =========================
# PRINT VIDEO INFO
# =========================
print("====== VIDEO INFO ======")
print(f"File Name         : {video_filename}")
print(f"Resolution        : {orig_width} x {orig_height}")
print(f"Aspect Ratio      : {format_ratio(orig_ar)}")
print(f"FPS               : {fps:.2f}")
print(f"ms/Frame          : {1000/fps:.2f}")
print(f"Total Frames      : {frame_count_total}")
print(f"Duration          : {duration_sec:.2f} sec ({int(duration_ms)} ms)")
print("========================\n")

# =========================
# HANDLE ORIENTATION
# =========================
needs_rotation = orig_width < target_width or orig_height < target_height

if needs_rotation:
    print("Portrait video detected. Frames will be rotated for landscape crop.\n")
    work_width, work_height = orig_height, orig_width
else:
    work_width, work_height = orig_width, orig_height

# =========================
# VALIDATION
# =========================
if work_width < target_width or work_height < target_height:
    raise ValueError(
        f"Requested resolution ({target_width}x{target_height}) "
        f"is larger than working resolution ({work_width}x{work_height}). "
        f"Clean crop is impossible."
    )

# Precompute centered crop
x_start = (work_width - target_width) // 2
y_start = (work_height - target_height) // 2

if needs_rotation:
    print("Cropping mode: ROTATE 90° + HARD CENTER CROP")
else:
    print("Cropping mode: HARD CENTER CROP")
print(f"Crop region: x={x_start}:{x_start+target_width}, "
      f"y={y_start}:{y_start+target_height}\n")

# =========================
# FRAME EXTRACTION
# =========================
frame_interval = int((interval_ms / 1000) * fps)
if frame_interval <= 0:
    frame_interval = 1  # Safety for very small intervals

frame_count = 0
saved_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        # Rotate if needed
        if needs_rotation:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        cropped = frame[
            y_start:y_start + target_height,
            x_start:x_start + target_width
        ]

        saved_count += 1
        filename = f"{video_name}_{saved_count:05d}.jpg"
        filepath = os.path.join(output_dir, filename)

        cv2.imwrite(filepath, cropped)

    frame_count += 1

cap.release()

print(f"Done. Saved {saved_count} images to '{output_dir}'")