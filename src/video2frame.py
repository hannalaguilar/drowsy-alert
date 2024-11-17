from pathlib import Path

from src.utils import extract_frames_by_class, move_images


DATA_PATH = Path("data/YawDD-dataset/Mirror/Male_mirror/")


if __name__ == '__main__':
    extract_frames_by_class(DATA_PATH, 'Extracted_frames_male', 2)
    # move_images("/home/hanna/MAI/2024-I/ISP/project/drowsy-alert/Extracted_frames_male/Yawning", 'Yawning')
