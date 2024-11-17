import joblib
import argparse
from pathlib import Path
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.train import SELECTED_COLUMNS
from src.utils import get_rect_points



def parse_args():
    parser = argparse.ArgumentParser(description="Drowsy-alert model")
    parser.add_argument("--video_name",type=int,default=0,required=False)
    parser.add_argument("--root_path", type=str, default=str(Path(__file__).parent), required=False)
    args = parser.parse_args()
    return args


def load_clf_model(root_path: str):
    return joblib.load(root_path + "/models/random_forest_model_1.pkl")


def main():
    args = parse_args()
    video_name = args.video_name
    root_path = args.root_path

    # load clf_model
    clf = load_clf_model(root_path)

    # load video
    cap = cv2.VideoCapture(video_name)

    # load mediapipe face landmark model
    base_options = python.BaseOptions(
        model_asset_path=root_path + '/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)

    # run the video
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            success, frame = cap.read()

            # mediapipe model inference
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            detection_result = landmarker.detect(mp_frame)
            if len(detection_result.face_blendshapes) > 0:
                # face_blendshapes
                face_blendshapes = detection_result.face_blendshapes[0]
                scores_dict = {blendshape.category_name: blendshape.score for blendshape in face_blendshapes}

                # classification model
                ordered_scores = [scores_dict.get(column, 0.0) for column in SELECTED_COLUMNS]
                y_pred = clf.predict([ordered_scores])

                # bounding box
                height, width, _ = frame.shape
                x_min_px, x_max_px, y_min_px, y_max_px = get_rect_points(detection_result.face_landmarks[0], height, width)

                print(f"Predicted class: {y_pred[0]}")

                if y_pred=='normal':
                    font_color=(0, 255, 0)
                elif y_pred=='yawning':
                    font_color=(0, 0, 255)

                cv2.putText(frame, y_pred[0], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, font_color, 2, cv2.LINE_AA)
                cv2.rectangle(frame, (x_min_px, y_min_px), (x_max_px, y_max_px), font_color, 2)

            if not success:
                break

            cv2.imshow('frame', frame)
            if cv2.waitKey(100) == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()