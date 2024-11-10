import os
import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from vision_agent.agent import VisionAgent
from vision_agent.tools import extract_frames_and_timestamps, owl_v2_image, overlay_bounding_boxes, save_video
from vision_agent.lmm import AnthropicLMM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize VisionAgent and LMM
agent = VisionAgent()
lmm = AnthropicLMM()

# Safety items to detect
SAFETY_ITEMS = "helmet, harness, footwear, gloves, parachute"
ACROBATIC_MANEUVERS = "spiral dive, wing over, full stall, asymmetric collapse"

def analyze_image(image):
    """Analyze a single image using VisionAgent."""
    detections = owl_v2_image(SAFETY_ITEMS, image)
    annotated_image = overlay_bounding_boxes(image, detections)
    
    # Extract detected items
    detected_items = [d['label'] for d in detections]
    return annotated_image, detected_items

def display_image_analysis(image):
    """Display the analyzed image with bounding boxes."""
    st.image(image, caption="Annotated Image", use_container_width=True)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def display_safety_gear_checklist(detected_items):
    """Display checkmarks or crosses for detected safety gear items."""
    st.subheader("Safety Gear Checklist")
    for item in SAFETY_ITEMS.split(", "):
        if item.lower() in [d.lower() for d in detected_items]:
            st.success(f"✅ {item.capitalize()} detected")
        else:
            st.error(f"❌ {item.capitalize()} not detected")

def detect_safety_gear_and_acrobatics(video_path):
    """Track safety gear and acrobatics in a video frame by frame."""
    # st.write("Extracting frames from video...")
    frames_and_ts = extract_frames_and_timestamps(video_path, fps=1)
    frames = [f["frame"] for f in frames_and_ts]

    # Initialize progress bar
    # st.write("Analyzing frames for safety gear and acrobatics...")
    total_frames = len(frames)
    progress_bar = st.progress(0)

    visualized_frames = []
    for idx, frame in enumerate(frames):
        try:
            # Analyze each frame individually
            detections = owl_v2_image(ACROBATIC_MANEUVERS, frame)
            visualized_frame = overlay_bounding_boxes(frame, detections)
            visualized_frames.append(visualized_frame)
        except Exception as e:
            st.error(f"Error analyzing frame {idx + 1}: {e}")
            visualized_frames.append(frame)

        # Update the progress bar
        progress_bar.progress((idx + 1) / total_frames)

    # Save the annotated frames as a video
    output_path = "annotated_paragliding.mp4"
    save_video(visualized_frames, output_path)
    return output_path

def analyze_video_with_lmm(video_path):
    """Use LMM to analyze the video and get a description."""
    try:
        response = lmm([
            {
                "role": "user",
                "content": "Describe the video in detail as a story of paragliding.",
                "media": [video_path]
            }
        ])
        return response
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        return None

def main():
    st.title("Welcome to Happy Landing Paragliding Service")

    # Image Analysis Section
    st.header("Step 1: Come in front of the Camera and have your Safety Gear Verified")
    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file:
        # Read the uploaded image
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Analyze the image
        st.write("Analyzing image for safety gear...")
        analyzed_image, detected_items = analyze_image(image)
        display_image_analysis(analyzed_image)
        display_safety_gear_checklist(detected_items)

        # Save the annotated image
        cv2.imwrite("annotated_image.png", analyzed_image)
        st.success("Verified image saved as 'verified_image.png'")

    # Video Analysis Section
    st.header("Step 2: Video feed for Acrobatics Analysis and Story-telling")
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if video_file:
        # Save the uploaded video temporarily
        video_input_path = "uploaded_video.mp4"
        with open(video_input_path, "wb") as f:
            f.write(video_file.read())

        st.video(video_input_path)

        # Detect safety gear and acrobatics
        st.write("Analyzing video for acrobatics...")
        output_video = detect_safety_gear_and_acrobatics(video_input_path)
        if output_video and os.path.exists(output_video):
            st.subheader("Analyzed Video")
            st.video(output_video)

        # LMM Analysis for the video
        st.write("Analyzing video using LMM for telling your story...")
        lmm_response = analyze_video_with_lmm(video_input_path)
        if lmm_response:
            st.subheader("Your Paragliding Story:")
            st.write(lmm_response)

        # Clean up uploaded video
        os.remove(video_input_path)

if __name__ == "__main__":
    main()
