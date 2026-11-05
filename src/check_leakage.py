"""
Is the random frame level train / test split of WCEBleedGen leaking near identical frames ?

Two checks on the bleeding frames (downsampled to 32x32 and flattened) :

    1) Are neighbouring frame numbers visually closer than two random frames ?
       (if not, the numbering is not temporal and a block split makes no sense)
    2) For every test frame, how far is the closest training frame ?
       random split (as used in the notebook) vs block split.
       A much smaller distance for the random split means train / test share near duplicates.

Usage :
    python src/check_leakage.py --raw_dir /path/to/WCEBleedGen --block_size 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from prepare_data import numericalSort, split_pairs


def load_flat(paths : list) -> torch.Tensor:
  """Loads the images, resizes them to 32x32 and flattens them -> (number of images, 3072)"""

  images = [np.asarray(Image.open(path).convert("RGB").resize((32,32)), dtype=np.float32).flatten() / 255 for path in paths]
  return torch.tensor(np.array(images))


def nearest_train_distance(train_data : torch.Tensor, test_data : torch.Tensor) -> np.ndarray:
  """Distance of every test frame to its closest training frame"""

  return torch.cdist(test_data, train_data).min(dim=1).values.numpy()


if __name__ == "__main__":

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--raw_dir", type=Path, required=True, help="Path of the downloaded WCEBleedGen folder")
  parser.add_argument("--block_size", type=int, default=50)
  parser.add_argument("--out_file", type=Path, default=Path("runs/leakage_check.json"))
  args = parser.parse_args()

  images = sorted((args.raw_dir / "bleeding/Images").glob("*.png"), key=numericalSort)
  data = load_flat(images)
  print(f"{len(images)} bleeding frames loaded : {tuple(data.shape)}")

  # 1) neighbouring frames vs random pairs
  neighbour_distance = (data[1:] - data[:-1]).norm(dim=1)
  random_pairs = torch.randint(0, len(data), (len(data) - 1, 2))
  random_distance = (data[random_pairs[:, 0]] - data[random_pairs[:, 1]]).norm(dim=1)

  results = {"median distance frame n and n+1" : neighbour_distance.median().item(),
             "median distance random pair" : random_distance.median().item()}

  # 2) nearest training frame of every test frame
  for name, block_size in [("random split", 0), (f"block split ({args.block_size} frames)", args.block_size)]:

    train_images, test_images, _, _ = split_pairs(images, images, 0.2, 42, block_size)
    index = {str(path) : i for i, path in enumerate(images)}

    distances = nearest_train_distance(data[[index[str(p)] for p in train_images]], data[[index[str(p)] for p in test_images]])
    results[name] = {"train" : len(train_images), "test" : len(test_images),
                     "median distance to closest train frame" : float(np.median(distances)),
                     "10th percentile" : float(np.percentile(distances, 10))}

  for key, value in results.items():
    print(key, ":", value)

  args.out_file.parent.mkdir(parents=True, exist_ok=True)
  args.out_file.write_text(json.dumps(results, indent=1))
