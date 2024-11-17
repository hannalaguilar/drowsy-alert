from typing import Union
import shutil
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

pathType = Union[str, Path]


def extract_frames_by_class(base_path: pathType,
                            output_path: pathType,
                            frames_per_second: int=2):

    base_path = Path(base_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    for video_file in base_path.rglob("*.avi"):
        # Extract the class from the video file name
        class_name = video_file.stem.split("-")[-1]
        class_folder = output_path / class_name
        class_folder.mkdir(exist_ok=True)

        # Open the video
        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / frames_per_second)  # Calculate how many frames to skip

        frame_count = 0
        saved_frame_count = 0
        success, frame = cap.read()

        while success:
            # Save frames at the desired interval
            if frame_count % frame_interval == 0:
                frame_filename = class_folder / f"{video_file.stem}_frame_{saved_frame_count:05d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                saved_frame_count += 1

            frame_count += 1
            success, frame = cap.read()

        cap.release()
        print(f"Extracted frames from {video_file} into {class_folder}")


def move_images(source_folder: pathType, target_folder:pathType):
    source_folder = Path(source_folder)
    target_folder = Path(target_folder)
    target_folder.mkdir(parents=True, exist_ok=True)

    # Move all image files
    for file in source_folder.iterdir():
        if file.is_file() and file.suffix.lower() in {".jpg", ".png", ".jpeg"}:
            shutil.move(str(file), target_folder / file.name)

    print(f"All images moved from {source_folder} to {target_folder}")


def draw_landmarks_on_image(img_path, detection_result):
    image = mp.Image.create_from_file(str(img_path))
    image = image.numpy_view()
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(image)

    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles
            .get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
              landmark_drawing_spec=None,
              connection_drawing_spec=mp.solutions.drawing_styles
              .get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Extract the face blendshapes category names and scores.
    face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
    face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
    # The blendshapes are ordered in decreasing score value.
    face_blendshapes_ranks = range(len(face_blendshapes_names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
    ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
    ax.invert_yaxis()

    # Label each bar with values
    for score, patch in zip(face_blendshapes_scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")
    ax.set_xlabel('Score')
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


def tsne_plot(X, y, perplexity: int=30):
    t_sne = TSNE(n_components=2, learning_rate='auto',
                     init='random', perplexity=perplexity, random_state=0)
    X_embedded = t_sne.fit_transform(X)
    fig, ax = plt.subplots(1, 1)
    sns.scatterplot(x=X_embedded[:, 0],
                    y=X_embedded[:, 1],
                    hue=y,
                    palette='tab10', ax=ax)
    ax.set_xlabel('tsne 1')
    ax.set_ylabel('tsne 2')
    plt.show()


def plot_feature_importance(sorted_features, sorted_importances, n: int=10):
    plt.figure(figsize=(10, 6))
    plt.barh(sorted_features[:10][::-1], sorted_importances[:10][::-1], align='center')
    plt.xlabel('Feature Importance')
    plt.title(f'Top {n} Feature Importances')
    plt.show()


def get_rect_points(landmarks, height, width):
    x_coords = [landmark.x for landmark in landmarks]
    y_coords = [landmark.y for landmark in landmarks]
    x_min = min(x_coords)
    x_max = max(x_coords)
    y_min = min(y_coords)
    y_max = max(y_coords)
    x_min_px = int(x_min * width)
    x_max_px = int(x_max * width)
    y_min_px = int(y_min * height)
    y_max_px = int(y_max * height)
    return x_min_px, x_max_px, y_min_px, y_max_px
