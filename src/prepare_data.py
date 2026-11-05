"""
Data preparation for the WCEBleedGen dataset (Data Preparation.ipynb as a script).

Takes the raw dataset downloaded from Zenodo (doi:10.5281/zenodo.7548320) :

    WCEBleedGen/
        bleeding/Images/*.png
        bleeding/Bounding boxes/TXT/*.txt
        non-bleeding/images/*.png

and redistributes it into two folders.

1) Classification (format accepted by torchvision's ImageFolder) :

    classification/
        train/{bleeding, non-bleeding}/
        test/{bleeding, non-bleeding}/

2) Detection (bleeding frames only, boxes aligned with images) :

    detection/
        train/{Images, Bounding Boxes}/
        test/{Images, Bounding Boxes}/

Usage :
    python src/prepare_data.py --raw_dir /path/to/WCEBleedGen --out_dir /path/to/WCEBleedGen_prepared
"""

import argparse
import re
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split, GroupShuffleSplit

numbers = re.compile(r'(\d+)')


def numericalSort(value):
    """Sort key so that 'img- (2).png' comes before 'img- (10).png'."""
    parts = numbers.split(str(value))
    parts[1::2] = map(int, parts[1::2])
    return parts


def split_pairs(first : list, second : list, test_size : float, seed : int, block_size : int = 0):
  """
  Splits two aligned lists the same way. (train_test_split when block_size = 0)

  block_size > 0 : consecutive frames img- (1) ... img- (block_size) form one block and a block goes
  completely to train or completely to test, so neighbouring (near identical) frames are never
  shared between the two splits. Needs the lists to be sorted with numericalSort
  """

  if block_size == 0:
    return train_test_split(first, second, test_size=test_size, random_state=seed)

  groups = [int(numbers.findall(path.stem)[-1]) // block_size for path in first]
  train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(first, groups=groups))

  return [first[i] for i in train_idx], [first[i] for i in test_idx], [second[i] for i in train_idx], [second[i] for i in test_idx]


def walk_through_dir(dir_path):
    import os
    for dirpath, dirname, filename in os.walk(dir_path):
        print(f"There are {len(dirname)} directories and {len(filename)} files in {dirpath}")


def copy_files(label_split: dict, target_dir: Path, folder_of) -> None:
    """Copies every file of every split into target_dir/<split>/<folder_of(file)>/"""
    for split, paths in label_split.items():
        for path in paths:
            dest = target_dir / split / folder_of(path) / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
        print(f"[INFO] Copied {len(paths)} files to {target_dir / split}")


def prepare_classification_data(data_dir: Path, target_dir: Path, test_size: float = 0.2, seed: int = 42, block_size: int = 0) -> None:

    bleeding_data_set_dir = data_dir / "bleeding/Images"
    non_bleeding_data_set_dir = data_dir / "non-bleeding/images"

    # NOTE : sorted so that the split is identical on every machine (glob order depends on the filesystem)
    bleeding_images = sorted(bleeding_data_set_dir.glob("*.png"), key=numericalSort)
    non_bleeding_images = sorted(non_bleeding_data_set_dir.glob("*.png"), key=numericalSort)
    print(f"Bleeding images : {len(bleeding_images)} | Non bleeding images : {len(non_bleeding_images)}")

    # Splitting the 2 classes separately so that both stay balanced in train and test
    bleeding_label = [bleeding_data_set_dir.parent.stem] * len(bleeding_images)
    bleeding_train, bleeding_test, _, _ = split_pairs(bleeding_images, bleeding_label, test_size, seed, block_size)

    non_bleeding_label = [non_bleeding_data_set_dir.parent.stem] * len(non_bleeding_images)
    non_bleeding_train, non_bleeding_test, _, _ = split_pairs(non_bleeding_images, non_bleeding_label, test_size, seed, block_size)

    label_split = {}
    label_split['train'] = bleeding_train + non_bleeding_train
    label_split['test'] = bleeding_test + non_bleeding_test

    # class folder = name of the folder one level above the image folder ("bleeding" / "non-bleeding")
    copy_files(label_split, target_dir, folder_of=lambda p: p.parent.parent.stem)
    walk_through_dir(target_dir)


def prepare_detection_data(data_dir: Path, target_dir: Path, test_size: float = 0.2, seed: int = 42, block_size: int = 0) -> None:

    bleeding_data_set_dir = data_dir / "bleeding/Images"
    bleeding_data_bboxes = data_dir / "bleeding/Bounding boxes/TXT"

    bleeding_images = list(bleeding_data_set_dir.glob("*.png"))
    bleeding_bboxes = list(bleeding_data_bboxes.glob("*.txt"))
    assert len(bleeding_images) == len(bleeding_bboxes), "number of images and bounding box files differ"

    # The files are retrieved in arbitrary order (images get mapped to different bounding boxes),
    # to tackle this we sort both lists on the image number contained in the file name
    sorted_bleeding_images = sorted(bleeding_images, key=numericalSort)
    sorted_bleeding_bboxes = sorted(bleeding_bboxes, key=numericalSort)

    bleeding_images_train, bleeding_images_test, bleeding_bboxes_train, bleeding_bboxes_test = split_pairs(
        sorted_bleeding_images, sorted_bleeding_bboxes, test_size, seed, block_size)

    # sanity check : every image is paired with the box file of the same number
    for img, box in zip(bleeding_images_train + bleeding_images_test, bleeding_bboxes_train + bleeding_bboxes_test):
        assert img.stem == box.stem, f"{img.name} paired with {box.name}"
    print(f"Detection split : {len(bleeding_images_train)} train | {len(bleeding_images_test)} test")

    label_split_images = {"train": bleeding_images_train, "test": bleeding_images_test}
    label_split_bboxes = {"train": bleeding_bboxes_train, "test": bleeding_bboxes_test}

    copy_files(label_split_images, target_dir, folder_of=lambda p: "Images")
    copy_files(label_split_bboxes, target_dir, folder_of=lambda p: "Bounding Boxes")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", type=Path, required=True, help="Path of the downloaded WCEBleedGen folder")
    parser.add_argument("--out_dir", type=Path, required=True, help="Where the prepared data is written")
    parser.add_argument("--block_size", type=int, default=0,
                        help="0 : random frame level split (original). N > 0 : blocks of N consecutive frames go together to train or test")
    parser.add_argument("--make_zip", action="store_true", help="Also create a zip of the classification data")
    args = parser.parse_args()

    classification_dir = args.out_dir / "classification"
    detection_dir = args.out_dir / "detection"

    print(f"Creating directory: '{args.out_dir}'")
    prepare_classification_data(args.raw_dir, classification_dir, block_size=args.block_size)
    prepare_detection_data(args.raw_dir, detection_dir, block_size=args.block_size)

    if args.make_zip:
        print(shutil.make_archive(str(classification_dir), format="zip", root_dir=classification_dir.parent))
