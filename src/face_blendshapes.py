from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.utils import draw_landmarks_on_image, plot_face_blendshapes_bar_graph


def get_face_blendshapes(img_path, detector):
    image = mp.Image.create_from_file(str(img_path))
    detection_result = detector.detect(image)
    return detection_result


def main(folder_path) -> pd.DataFrame:
    data = []
    category_names = None
    folder_path = Path(folder_path)
    # dd = load_model()

    base_options = python.BaseOptions(
        model_asset_path='/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    # detector = vision.FaceLandmarker.create_from_options(options)

    with vision.FaceLandmarker.create_from_options(options) as dd:
        for i, img_path in enumerate(tqdm(sorted(folder_path.iterdir()), desc="Processing images", unit="images")):
            if img_path.is_file():
                # 13-FemaleNoGlasses-Normal_frame_00034.jpg
                try:
                    detection_result = get_face_blendshapes(img_path, dd)
                    if len(detection_result.face_blendshapes) > 0:
                        face_blendshapes = detection_result.face_blendshapes[0]

                        if face_blendshapes:
                            # Extract category names (once, from the first result)
                            if category_names is None:
                                category_names = [category.category_name for category in face_blendshapes]
                            scores = [category.score for category in face_blendshapes]
                            data.append([img_path.name] + scores)

                        if i % 500 == 0:
                            annotated_image = draw_landmarks_on_image(img_path, detection_result)
                            plt.figure()
                            plt.axis('off')
                            plt.imshow(annotated_image)
                            plt.show()
                            # plot_face_blendshapes_bar_graph(face_blendshapes)
                    else:
                        print(f'{img_path.name} not enough landmarks found')
                except Exception as e:
                    print(f' {img_path.name}: something went wrong')



    # Create a DataFrame
    if category_names and data:
        df = pd.DataFrame(data, columns=['image_name'] + category_names)

    return df


if __name__ == '__main__':
    df = main("../data/YawDD-dataset-frames/Yawning")
    df.to_csv('Yawning.csv', index=False)



