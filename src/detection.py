"""
Bleeding region detection (Object Detection part of
'WCE Bleeding Region Classification and Detection.ipynb' as a script).

Fine tunes a COCO pre-trained Faster R-CNN (ResNet50-FPN) : backbone and ROI heads frozen,
Region Proposal Network trained, box predictor replaced by a 2 class (background / bleeding) one.

Expects the folder created by prepare_data.py :
    <data_dir>/train/{Images, Bounding Boxes}
    <data_dir>/test/{Images, Bounding Boxes}

Usage :
    python src/detection.py --data_dir /path/to/prepared/detection --out_dir runs/detection
"""

import argparse
import json
import os
import pathlib
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from matplotlib import patches
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, ops
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm.auto import tqdm

try:
    from torchinfo import summary
except ImportError:
    summary = None

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------------------------
# Custom Dataset Class for the Detection Dataset
# ----------------------------------------------------------------------------------------------
def numericalSort(value):

  numbers = re.compile(r'(\d+)')
  parts = numbers.split(str(value))
  parts[1::2] = map(int, parts[1::2])
  return parts

transform = transforms.Compose([transforms.ToTensor()])

class BleedingRegionDataset(Dataset):

  """
  Custom Dataset Creator for Bleeding Region detection dataset
  """

  def __init__(self, root_dir : str, bbox_dir : str, transforms = None):

    """
    Arguments :
      root_dir : Path where all the images are stored
      bbox_dir : Path where all the bounding box .txt files are stored
      transforms (optional) : transform to be applied on a sample.

    Returns :
      None
    """

    self.sorted_bleeding_images = sorted(list(Path(root_dir).glob("*.png")), key=numericalSort)
    self.sorted_bleeding_bboxes = sorted(list(Path(bbox_dir).glob("*.txt")), key=numericalSort)
    self.transforms = transforms

  def __len__(self):

    return len(self.sorted_bleeding_images)

  def rearrange_bbox(self, path : pathlib.PosixPath):

    with open(path) as f:
      output = np.array([int(number) for number in f.readlines()[0].split()]).reshape((-1,4))

    return output

  def __getitem__(self, index):

    image_path = self.sorted_bleeding_images[index]
    img_tensor = transform(Image.open(image_path).convert("RGB"))

    label = {}
    label["boxes"] = torch.FloatTensor(self.rearrange_bbox(self.sorted_bleeding_bboxes[index]))
    label["labels"] = torch.ones(size=[label["boxes"].shape[0]], dtype = torch.int64)

    return img_tensor, label


def collate_fn(batch_data):

  image = []
  label = []

  for b in batch_data:
    image.append(b[0])
    label.append(b[1])

  return image, label


def labels_to_device(labels):
  for label in labels:
    label["boxes"] = label["boxes"].to(device)
    label["labels"] = label["labels"].to(device)


# ----------------------------------------------------------------------------------------------
# Train function and IoU metric
# ----------------------------------------------------------------------------------------------
def train(model, train_dataloader : DataLoader, device, optimizer: torch.optim.Optimizer):

  model.train()

  total_loss_classifier = 0
  total_loss_box_reg = 0
  total_loss_objectness = 0
  total_loss_rpn_box_reg = 0

  for images, labels in train_dataloader:

    images = [image.to(device) for image in images]
    labels_to_device(labels)

    loss_dict = model(images, labels)
    total_loss_classifier += loss_dict['loss_classifier'].item()
    total_loss_box_reg += loss_dict['loss_box_reg'].item()
    total_loss_objectness += loss_dict['loss_objectness'].item()
    total_loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()

    losses = sum(loss for loss in loss_dict.values())

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

  return total_loss_classifier, total_loss_box_reg, total_loss_objectness, total_loss_rpn_box_reg


# IoU evaluation metric

def calculate_IoU(bboxA : torch.Tensor, bboxB : torch.Tensor) -> float:

  x0_A, y0_A, x1_A, y1_A = bboxA
  x0_B, y0_B, x1_B, y1_B = bboxB

  x0_I = torch.max(x0_A, x0_B)
  y0_I = torch.max(y0_A, y0_B)
  x1_I = torch.min(x1_A, x1_B)
  y1_I = torch.min(y1_A, y1_B)

  width_I = x1_I - x0_I
  height_I = y1_I - y0_I

  width_I = torch.clamp(width_I, min=0)
  height_I = torch.clamp(height_I, min=0)

  intersection_area = width_I*height_I

  width_A = x1_A - x0_A
  height_A = y1_A - y0_A

  width_B = x1_B - x0_B
  height_B = y1_B - y0_B

  union_area = (width_A*height_A) + (width_B*height_B) - intersection_area

  IoU = intersection_area / union_area

  return IoU


# ----------------------------------------------------------------------------------------------
# Evaluation (IoU based)
# ----------------------------------------------------------------------------------------------
def get_predictions(model, dataloader : DataLoader, device):
  """Runs the model on a whole dataloader, returns the list of predictions and the list of ground truths."""

  model.eval()
  all_predictions = []
  all_labels = []

  with torch.inference_mode():

    for images, labels in dataloader:

      images = [image.to(device) for image in images]
      labels_to_device(labels)

      all_predictions.extend(model(images))
      all_labels.extend(labels)

  return all_predictions, all_labels


def evaluate(predictions, labels, score_threshold : float, iou_threshold_nms : float, iou_match : float = 0.5):
  """
  For every image : keep the predictions with score >= score_threshold, apply NMS, then

    1) every actual bleeding region is matched with the predicted box having the highest IoU
       (0 if there is no predicted box) -> mean IoU and recall
    2) every predicted box is matched with the actual region having the highest IoU -> precision

  A match is a hit if IoU >= iou_match
  """

  best_iou_per_region = []
  best_iou_per_prediction = []

  for prediction, label in zip(predictions, labels):

    keeps = prediction['scores'] >= score_threshold
    boxes = prediction['boxes'][keeps]
    scores = prediction['scores'][keeps]

    keeps = ops.nms(boxes=boxes, scores=scores, iou_threshold=iou_threshold_nms)
    boxes = boxes[keeps]

    for actual_box in label['boxes']:

      ious = [calculate_IoU(actual_box, predicted_box) for predicted_box in boxes]
      best_iou_per_region.append(max(ious).item() if len(ious) > 0 else 0.0)

    for predicted_box in boxes:

      ious = [calculate_IoU(predicted_box, actual_box) for actual_box in label['boxes']]
      best_iou_per_prediction.append(max(ious).item())

  best_iou_per_region = np.array(best_iou_per_region)
  best_iou_per_prediction = np.array(best_iou_per_prediction)

  return {"score_threshold" : score_threshold,
          "mean_IoU_per_region" : best_iou_per_region.mean(),
          f"recall@{iou_match}" : (best_iou_per_region >= iou_match).mean(),
          f"precision@{iou_match}" : (best_iou_per_prediction >= iou_match).mean() if len(best_iou_per_prediction) > 0 else 0.0,
          "predicted_boxes_per_image" : len(best_iou_per_prediction) / len(labels),
          "actual_boxes_per_image" : len(best_iou_per_region) / len(labels)}


# ----------------------------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------------------------
def plot_loss_curves(results : dict, path : Path) -> None:

  fig, ax = plt.subplots(ncols=2, nrows=2, figsize = (8,8))

  epochs = len(results['loss_classifier'])

  ax[0][0].plot(range(epochs),results['loss_box_reg'])
  ax[0][0].set_title("loss_box_reg")

  ax[0][1].plot(range(epochs),results['loss_classifier'])
  ax[0][1].set_title("loss_classifier")

  ax[1][0].plot(range(epochs),results['loss_objectness'])
  ax[1][0].set_title("loss_objectness")

  ax[1][1].plot(range(epochs),results['loss_rpn_box_reg'])
  ax[1][1].set_title("loss_rpn_box_reg")

  fig.savefig(path, bbox_inches="tight")
  plt.close(fig)


def prediction_plot(model ,test_dataloader : DataLoader, device, batch_size : int, iou_threshould : float, path : Path, score_threshold : float = 0.0) -> None:
  """Green -> actual bleeding region, Red -> model prediction (after NMS), first batch of the test set."""

  model.eval()
  with torch.inference_mode():

    images, labels = next(iter(test_dataloader))

    images = [image.to(device) for image in images]
    labels_to_device(labels)

    prediction = model(images)

  fig, ax = plt.subplots(nrows=batch_size, ncols=1, figsize = (20,5*batch_size))
  index = 0

  for image, label, prediction_label in zip(images,labels,prediction):

    ax[index].imshow(image.permute(1,2,0).cpu())

    for box in label['boxes']:

      x1,y1,x2,y2 = box.cpu()

      w = x2 - x1
      h = y2 - y1

      ax[index].add_patch(patches.Rectangle((x1,y1),w,h, fill=False, edgecolor='green', lw=2))

    keeps = ops.nms(boxes=prediction_label['boxes'], scores=prediction_label['scores'], iou_threshold=iou_threshould)
    keeps = [box_idx for box_idx in keeps if prediction_label['scores'][box_idx] >= score_threshold]

    for box_idx in keeps:

      x1,y1,x2,y2 = prediction_label['boxes'][box_idx].cpu()

      w = x2 - x1
      h = y2 - y1

      ax[index].add_patch(patches.Rectangle((x1,y1),w,h, fill=False, edgecolor='red', lw=2))

    index+=1

  fig.savefig(path, bbox_inches="tight", dpi=50)
  plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--data_dir", type=Path, required=True, help="folder containing train/ and test/ (Images, Bounding Boxes)")
  parser.add_argument("--out_dir", type=Path, default=Path("runs/detection"))
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
  parser.add_argument("--no_shuffle", action="store_true", help="original notebook behaviour : train batches follow the frame order")
  parser.add_argument("--score_threshold", type=float, default=0.5, help="predictions below this score are dropped in the plots")
  parser.add_argument("--seed", type=int, default=None, help="optional, the original notebook did not fix a seed")
  args = parser.parse_args()

  args.out_dir.mkdir(parents=True, exist_ok=True)
  if args.seed is not None:
    torch.manual_seed(args.seed)
  print(f"torch {torch.__version__} | torchvision {torchvision.__version__} | device : {device}")

  # Datasets
  train_object_detection_data = BleedingRegionDataset(root_dir = args.data_dir / "train/Images",
                                                      bbox_dir = args.data_dir / "train/Bounding Boxes")

  test_object_detection_data = BleedingRegionDataset(root_dir = args.data_dir / "test/Images",
                                                     bbox_dir = args.data_dir / "test/Bounding Boxes")

  print(f"train : {len(train_object_detection_data)} | test : {len(test_object_detection_data)}")
  assert len(train_object_detection_data.sorted_bleeding_images) == len(train_object_detection_data.sorted_bleeding_bboxes)
  assert len(test_object_detection_data.sorted_bleeding_images) == len(test_object_detection_data.sorted_bleeding_bboxes)

  # DataLoaders
  train_object_detection_dataloader = DataLoader(dataset = train_object_detection_data,
                                                 batch_size = args.batch_size,
                                                 num_workers = args.num_workers,
                                                 shuffle = not args.no_shuffle,
                                                 collate_fn=collate_fn)

  test_object_detection_dataloader = DataLoader(dataset = test_object_detection_data,
                                                batch_size = args.batch_size,
                                                num_workers = args.num_workers,
                                                shuffle = False,
                                                collate_fn=collate_fn)

  # Loading a Pretrained model (upper limit of 10 box detections per image)
  object_detection_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT", box_detections_per_img = 10).to(device)

  # Freeze Base Layers
  for param in object_detection_model.backbone.parameters():
    param.requires_grad = False

  for param in object_detection_model.roi_heads.parameters():
    param.requires_grad = False

  # Changing Layer for our use case (91 COCO classes -> background + bleeding)
  num_classes = 2
  in_feature = object_detection_model.roi_heads.box_predictor.cls_score.in_features

  object_detection_model.roi_heads.box_predictor = FastRCNNPredictor(in_feature, num_classes).to(device)

  trainable = sum(p.numel() for p in object_detection_model.parameters() if p.requires_grad)
  total = sum(p.numel() for p in object_detection_model.parameters())
  print(f"Trainable parameters : {trainable:,} / {total:,}")

  # Forward Pass Check
  images, labels = next(iter(train_object_detection_dataloader))
  images = [image.to(device) for image in images]
  labels_to_device(labels)
  output = object_detection_model(images, labels)  # Returns losses and detections
  print(output)

  # Optimiser and LR scheduler
  optimiser = torch.optim.SGD(params=object_detection_model.parameters(), lr=0.01, momentum=0.9)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser,step_size=2,gamma=0.9)

  # Training Function
  EPOCHS = args.epochs
  results = defaultdict(list)

  for epoch in tqdm(range(EPOCHS)):

    total_loss_classifier, total_loss_box_reg, total_loss_objectness, total_loss_rpn_box_reg = train(model = object_detection_model,
                                                                                                     train_dataloader=train_object_detection_dataloader,
                                                                                                     device = device,
                                                                                                     optimizer = optimiser)

    results["loss_classifier"].append(total_loss_classifier)
    results["loss_box_reg"].append(total_loss_box_reg)
    results["loss_objectness"].append(total_loss_objectness)
    results["loss_rpn_box_reg"].append(total_loss_rpn_box_reg)

    if epoch%4==0:
      print(f"Epoch {epoch+1} | loss classifier : {total_loss_classifier} | loss box reg : {total_loss_box_reg} | loss objectness : {total_loss_objectness} | loss_rpn_box_reg : {total_loss_rpn_box_reg}")

    scheduler.step()

  torch.save(object_detection_model.state_dict(), args.out_dir / "Object Detection Model Parameters")
  (args.out_dir / "history.json").write_text(json.dumps(results, indent=1))

  # Plot Loss Curves and Predictions
  plot_loss_curves(results, args.out_dir / "loss_curves.png")
  prediction_plot(model = object_detection_model,
                  test_dataloader = test_object_detection_dataloader,
                  device = device,
                  batch_size = args.batch_size,
                  iou_threshould = 0.2,
                  path = args.out_dir / "test_predictions.png",
                  score_threshold = args.score_threshold)

  # IoU based evaluation on the whole test set, for a few score thresholds
  predictions, labels = get_predictions(object_detection_model, test_object_detection_dataloader, device)

  metrics = []
  for score_threshold in [0.3, 0.5, 0.7]:
    metrics.append(evaluate(predictions, labels, score_threshold = score_threshold, iou_threshold_nms = 0.2))
    print(metrics[-1])

  (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
