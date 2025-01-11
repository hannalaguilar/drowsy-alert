import streamlit as st
import cv2
import mediapipe as mp
import pandas as pd
from collections import deque
import numpy as np
import time

# Thresholds
EYE_AR_THRESH = 0.2
MOUTH_AR_THRESH = 0.35
CHIN_THRESH = 0.12
SCORE_THRESH = 5

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)
# Streamlit Layout
st.set_page_config(layout="wide")

st.title("Drowsy Alert System")

# Create columns for the layout
col1, col2, col3 = st.columns([4, 2, 2])  # Left column (for webcam) is wider

# Webcam Feed on Left (col1)
with col1:
    st.markdown(
    """
    ### Real-Time Monitoring
    This application detects drowsiness using face landmarks and provides live graphs of relevant metrics such as:
    - **Mouth Openness Ratio** (indicating yawning or drowsiness)
    - **Eye Aspect Ratio (EAR)** (indicating eye closure levels)
    """
    )
    # Start Webcam and Detection
    #run = st.checkbox("Start Detection")
    run=st.toggle("Activate Detection")
    FRAME_WINDOW = st.image([])

# Graphs on Right (col2)
with col2:
    st.subheader("Live Metrics")
    st.markdown("Graphs updating in real-time:")
    st.markdown("1. **Mouth Openness Ratio**")
    mouth_placeholder = st.empty()
    st.markdown("3. **Right Eye Aspect Ratio (EAR)**")
    right_eye_placeholder = st.empty()


with col3:
    st.subheader(" ")
    st.markdown(" ")
    st.markdown("2. **Fainted Chin**")
    fainted_chin_placeholder = st.empty()
    st.markdown("4. **Left Eye Aspect Ratio (EAR)**")
    left_eye_placeholder = st.empty()


# Deques for graph data
time_data = deque(maxlen=50)
mouth_data = deque(maxlen=50)
left_eye_data = deque(maxlen=50)
right_eye_data = deque(maxlen=50)
fainted_chin_data = deque(maxlen=50)

# Function to calculate distances between two landmarks
def calculate_distance(landmark1, landmark2, frame_shape):
    height, width, _ = frame_shape
    x1, y1 = int(landmark1.x * width), int(landmark1.y * height)
    x2, y2 = int(landmark2.x * width), int(landmark2.y * height)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Function to calculate EAR (Eye Aspect Ratio)
def calculate_ear(eye_landmarks, frame_shape):
    vertical1 = calculate_distance(eye_landmarks[1], eye_landmarks[5], frame_shape)
    vertical2 = calculate_distance(eye_landmarks[2], eye_landmarks[4], frame_shape)
    horizontal = calculate_distance(eye_landmarks[0], eye_landmarks[3], frame_shape)
    return (vertical1 + vertical2) / (2.0 * horizontal)

# Function to calculate mouth aspect ratio (MAR)
def calculate_mouth_openness(landmarks, frame_shape):
    vertical = calculate_distance(landmarks[13], landmarks[14], frame_shape)
    horizontal = calculate_distance(landmarks[78], landmarks[308], frame_shape)
    return vertical / horizontal

def calculate_fainted_chin(landmarks, frame_shape):
    chin = landmarks[152]
    nose = landmarks[1]
    height, width, _ = frame_shape
    chin_y = chin.y * height
    nose_y = nose.y * height
    drop_ratio = (chin_y - nose_y) / height
    return drop_ratio



cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Unable to access webcam. Please check your camera settings.")
    st.stop()
count = score = 0

# Variables to track eye closure duration
eye_closed_time = 0
eye_threshold = 0.2  # Threshold for determining if the eyes are closed
warning_duration = 2  # Time in seconds for both eyes to be closed
warning_shown = False

# ear_flag = False
# mouth_flag = False
# fain_flag = False
while run:
    ret, frame = cap.read()
    if not ret:
        st.warning("Failed to capture frame. Please check your webcam.")
        break

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    count += 1
    # process every nth frame
    n = 5
    if count % n == 0:
        results = face_mesh.process(rgb_frame)

        # Process landmarks and calculate metrics
        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # Draw landmarks on the frame
                #mp_drawing.draw_landmarks(
                    #frame,
                    #landmarks,
                   # mp_face_mesh.FACEMESH_TESSELATION,
                   # mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                   # mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),
               # )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_iris_connections_style()
                )

                # Left and right eyes
                left_eye_indices = [362, 385, 387, 263, 373, 380]
                right_eye_indices = [33, 160, 158, 133, 153, 144]
                left_eye_landmarks = [landmarks.landmark[i] for i in left_eye_indices]
                right_eye_landmarks = [landmarks.landmark[i] for i in right_eye_indices]

                # Calculate metrics
                mouth_openness = calculate_mouth_openness(landmarks.landmark, frame.shape)
                left_ear = calculate_ear(left_eye_landmarks, frame.shape)
                right_ear = calculate_ear(right_eye_landmarks, frame.shape)
                ear_mean = (left_ear + right_ear) / 2.0
                fainted_chin = calculate_fainted_chin(landmarks.landmark, frame.shape)

                # Threshold
                ear_flag = ear_mean < EYE_AR_THRESH
                mouth_flag = mouth_openness > MOUTH_AR_THRESH
                fain_flag = fainted_chin < CHIN_THRESH

                # Debugging
                print(f"EAR: {ear_mean}, Mouth: {mouth_openness}, Chin: {fainted_chin}")
                print(f"Flags - EAR: {ear_flag}, Mouth: {mouth_flag}, Chin: {fain_flag}")

                if ear_mean < EYE_AR_THRESH:
                    ear_flag = True
                if mouth_openness > MOUTH_AR_THRESH:
                    mouth_flag = True
                if fainted_chin < CHIN_THRESH:
                    fain_flag = True

                if ear_flag or mouth_flag or fain_flag:
                    score += 1  # Increment if any flag is True
                else:
                    score -= 1 # Decrement if all flags are False
                    if score < 0:
                        score = 0

                score_text = f'Score: {score}'
                text_x = 10
                text_y = frame.shape[0] - 10
                text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame, (text_x, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
                cv2.putText(frame, score_text, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if score >= SCORE_THRESH:
                    cv2.putText(frame, 'Drowsy', (rgb_frame.shape[1] - 130, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)


                # Update data for graphs
                time_data.append(len(time_data))
                mouth_data.append(mouth_openness)
                left_eye_data.append(left_ear)
                right_eye_data.append(right_ear)
                fainted_chin_data.append(fainted_chin)

                # Draw a bounding box
                x_min = int(min([lm.x for lm in landmarks.landmark]) * frame.shape[1])
                y_min = int(min([lm.y for lm in landmarks.landmark]) * frame.shape[0])
                x_max = int(max([lm.x for lm in landmarks.landmark]) * frame.shape[1])
                y_max = int(max([lm.y for lm in landmarks.landmark]) * frame.shape[0])
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 0), 2)

                # Check if both eyes are closed
                if left_ear < eye_threshold and right_ear < eye_threshold:
                    if eye_closed_time == 0:
                        eye_closed_time = time.time()
                    elif time.time() - eye_closed_time > warning_duration and not warning_shown:
                        st.warning("Both eyes are closed for more than 2 seconds! Please wake up!")
                        warning_shown = True
                else:
                    eye_closed_time = 0  # Reset timer if eyes are not closed
                    warning_shown = False

        # Update graphs
        mouth_chart = pd.DataFrame({"Time": list(time_data), "Mouth Openness": list(mouth_data)})
        left_eye_chart = pd.DataFrame({"Time": list(time_data), "Left EAR": list(left_eye_data)})
        right_eye_chart = pd.DataFrame({"Time": list(time_data), "Right EAR": list(right_eye_data)})
        fainted_chin_chart = pd.DataFrame({"Time": list(time_data), "Fainted Chin": list(fainted_chin_data)})

        mouth_placeholder.line_chart(mouth_chart["Mouth Openness"], x_label= "Time", y_label="Mouth Openness", width= 300, height=300, use_container_width=False)
        left_eye_placeholder.line_chart(left_eye_chart["Left EAR"], x_label= "Time", y_label="Left Eye EAR",  width= 300 ,height=300,use_container_width=False)
        right_eye_placeholder.line_chart(right_eye_chart["Right EAR"], x_label= "Time", y_label="Right Eye EAR", width= 300 ,height=300,use_container_width=False)
        fainted_chin_placeholder.line_chart(fainted_chin_chart["Fainted Chin"], x_label="Time", y_label="Fainted Chin Ratio", width=300,
                                         height=300, use_container_width=False)

        with col1:
            FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()

