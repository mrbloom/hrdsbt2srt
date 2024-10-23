import cv2
import pytesseract
import numpy as np
from datetime import timedelta
from glob import glob

# Function to save the subtitle in SRT format
def save_subtitle_to_srt(subtitle_list, srt_file):
    with open(srt_file, 'w', encoding='utf-8') as f:
        for i, (start_time, end_time, text) in enumerate(subtitle_list, 1):
            f.write(f"{i}\n")
            f.write(f"{format_time(start_time)} --> {format_time(end_time)}\n")
            f.write(f"{text.strip()}\n\n")

# Convert time to SRT time format (hh:mm:ss,ms)
def format_time(seconds):
    ms = int((seconds % 1) * 1000)
    time_str = str(timedelta(seconds=int(seconds)))
    return f"{time_str},{ms:03d}"

# Apply a mask for yellow subtitles with black borders
def apply_yellow_subtitle_mask(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for yellow color (tuned for subtitle yellow)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a mask for the yellow color
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)

    # Find the contours of the masked yellow regions
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for black border around yellow text
    black_mask = np.zeros_like(yellow_mask)

    for cnt in contours:
        # Create bounding box for each contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Define a slightly larger bounding box to capture the black border
        border_x, border_y, border_w, border_h = x - 2, y - 2, w + 4, h + 4
        cv2.rectangle(black_mask, (border_x, border_y), (border_x + border_w, border_y + border_h), 255, thickness=-1)

    # Combine the yellow mask and black border mask
    combined_mask = cv2.bitwise_and(black_mask, yellow_mask)

    # Extract the relevant region
    masked_frame = cv2.bitwise_and(frame, frame, mask=combined_mask)

    return masked_frame

# Crop to bottom 30% of the frame
def crop_bottom_30_percent(frame):
    height = frame.shape[0]
    crop_start = int(height * 0.7)  # Crop the top 70% of the frame
    cropped_frame = frame[crop_start:, :]
    return cropped_frame

# Extract subtitles using OCR
def extract_subtitles_from_video(video_path, srt_file):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps)  # Capture one frame per second

    subtitle_list = []
    previous_text = ""
    start_time = 0

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Crop the bottom 30% of the frame
            cropped_frame = crop_bottom_30_percent(frame)

            # Apply yellow subtitle mask to the cropped frame
            masked_frame = apply_yellow_subtitle_mask(cropped_frame)

            # Convert to grayscale for better OCR accuracy
            gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

            # Use Tesseract to extract text from the masked region
            text = pytesseract.image_to_string(gray_frame, lang='eng')  # Adjust the language as needed

            if text.strip() and text != previous_text:
                end_time = frame_count / fps
                if previous_text:
                    subtitle_list.append((start_time, end_time, previous_text))
                    print(previous_text)

                start_time = end_time
                previous_text = text
            print(f"secs = {frame_count / frame_interval}")

        frame_count += 1

    if previous_text:
        end_time = frame_count / fps
        subtitle_list.append((start_time, end_time, previous_text))

    cap.release()

    save_subtitle_to_srt(subtitle_list, srt_file)
    print(f"Subtitles extracted to {srt_file}")

# Example usage
# video_path = "Kvodo S02E01.mkv"
# srt_file = "Kvodo S02E01.srt"

for p in glob("*.mkv"):    
    video_path = p
    srt_file = p[:-3]+"srt"
    print(f"For {video_path} make {srt_file}")
    extract_subtitles_from_video(video_path, srt_file)
