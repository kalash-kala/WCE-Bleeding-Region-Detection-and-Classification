"""
Bleeding / Non bleeding frame classification (Classification part of
'WCE Bleeding Region Classification and Detection.ipynb' as a script).

Expects the folder created by prepare_data.py :
    <data_dir>/train/{bleeding, non-bleeding}
    <data_dir>/test/{bleeding, non-bleeding}

Usage :
    python src/classification.py --data_dir /path/to/prepared/classification --out_dir runs/classification
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from timeit import default_timer as timer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm.auto import tqdm

try:
    from torchinfo import summary
except ImportError:
    summary = None

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------------
class WCEBleedGen(nn.Module):

  def __init__(self) -> None:
    super().__init__()

    self.layer_1 = nn.Sequential(
        nn.Conv2d(in_channels=3,out_channels=8,kernel_size=3,padding='same'),
        nn.BatchNorm2d(num_features=8),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.layer_2 = nn.Sequential(
        nn.Conv2d(in_channels=8,out_channels=16,kernel_size=3,padding='same'),
        nn.BatchNorm2d(num_features=16),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.layer_3 = nn.Sequential(
        nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding='same'),
        nn.BatchNorm2d(num_features=32),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )

    self.layer_4 = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=32*12*12,out_features=32),
        nn.Linear(in_features=32, out_features=2)
    )

  def forward(self, x : torch.Tensor) -> torch.Tensor:

    return self.layer_4(self.layer_3(self.layer_2(self.layer_1(x))))


# ----------------------------------------------------------------------------------------------
# Train and test functions
# ----------------------------------------------------------------------------------------------
def train_step(train_dataloader : torch.utils.data.DataLoader,
          model : nn.Module,
          loss_fn : nn.Module,
          optimiser : torch.optim.Optimizer,
          ):

  # Put the model in training mode
  model.train()

  # metrics to calculate
  training_loss = 0
  training_accuracy = 0

  for batch, (X,y) in enumerate(train_dataloader):

    # put data on correct device
    X = X.to(device)
    y = y.to(device)

    # make a prediction
    y_pred = model(X) #Logits
    y_pred_label = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

    # Calculate loss and accuracy
    loss = loss_fn(y_pred, y)
    training_loss += loss
    training_accuracy += (y==y_pred_label).sum().item()/len(y)

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

  # Calculate training loss and accuracy per batch
  training_loss/=len(train_dataloader)
  training_accuracy/=len(train_dataloader)

  return training_loss.item(), training_accuracy


def test_step(test_dataloader : torch.utils.data.DataLoader,
          model : nn.Module,
          loss_fn : nn.Module,
          ):

  #Put the model in evaluation mode
  model.eval()

  test_loss = 0
  test_accuracy = 0

  with torch.inference_mode():

    for batch, (X,y) in enumerate(test_dataloader):

      # Put the data on right device
      X = X.to(device)
      y = y.to(device)

      test_pred = model(X) # Logits
      test_pred_label = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)

      test_loss += loss_fn(test_pred, y)
      test_accuracy += (test_pred_label==y).sum().item()/len(y)

    test_loss /= len(test_dataloader)
    test_accuracy /= len(test_dataloader)

  return test_loss.item(), test_accuracy


def train(model : nn.Module,
          train_dataloader : torch.utils.data.DataLoader,
          test_dataloader : torch.utils.data.DataLoader,
          loss_fn : nn.Module,
          optimiser : torch.optim.Optimizer,
          scheduler,
          epochs : int = 5
          ):

  result = defaultdict(list)

  for epoch in tqdm(range(epochs)):

    # train and test step
    train_loss, train_acc = train_step(train_dataloader=train_dataloader,
                                       model = model,
                                       loss_fn = loss_fn,
                                       optimiser = optimiser)
    test_loss, test_acc = test_step(test_dataloader=test_dataloader,
                                    model = model,
                                    loss_fn = loss_fn)

    # Store the results
    result['train_loss'].append(train_loss)
    result['test_loss'].append(test_loss)
    result['train_accuracy'].append(train_acc)
    result['test_accuracy'].append(test_acc)

    scheduler.step()

    # print what's happening
    print(f"Epoch : {epoch} | training loss = {train_loss} | test loss = {test_loss} | training accuracy = {train_acc} | test accuracy = {test_acc}")

  return result


# ----------------------------------------------------------------------------------------------
# Evaluation helpers
# ----------------------------------------------------------------------------------------------
def classification_metrics(model : nn.Module, test_dataset) -> str:

  y_pred = []
  y_true = []

  model.eval()
  with torch.inference_mode():

    for images, labels in test_dataset:

      y_pred_logits = model(images.unsqueeze(dim=0).to(device))
      y_pred_labels = torch.argmax(torch.softmax(y_pred_logits, dim=1), dim=1)

      y_pred.append(y_pred_labels.item())
      y_true.append(labels)

  return classification_report(y_pred = y_pred, y_true = y_true, target_names = test_dataset.classes), \
         classification_report(y_pred = y_pred, y_true = y_true, output_dict = True)


def predict_folder(model : nn.Module, folder : Path, idx_to_class : dict) -> pd.DataFrame:
  """Predicts the class of every .png of a folder (used for the Auto-WCEBleedGen challenge test sets)."""

  validation_transform = transforms.Compose([transforms.Resize(size=(100,100)),
                                             transforms.ToTensor()])

  results = defaultdict(list)

  model.eval()
  with torch.inference_mode():
    for image in sorted(folder.glob("*.png")):

      results['Image name'].append(image.stem)

      img = Image.open(image).convert("RGB")
      transformed_img = validation_transform(img).unsqueeze(dim=0).to(device)

      pred_label = torch.argmax(torch.softmax(model(transformed_img), dim=1), dim=1).item()
      results['Predicted Class label'].append(pred_label)

  results_df = pd.DataFrame.from_dict(results)
  results_df['Predicted Class label'] = results_df['Predicted Class label'].map(idx_to_class)
  return results_df


def plot_results(result : dict, epochs : int, path : Path) -> None:

  fig = plt.figure(figsize=(10,10))
  # Plot loss
  plt.subplot(1, 2, 1)
  plt.plot(range(epochs), result['train_loss'], label='train loss')
  plt.plot(range(epochs), result['test_loss'], label='test loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.subplot(1, 2, 2)
  plt.plot(range(epochs), result['train_accuracy'], label='train_accuracy')
  plt.plot(range(epochs), result['test_accuracy'], label='test_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();

  fig.savefig(path, bbox_inches="tight")
  plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------
if __name__ == "__main__":

  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--data_dir", type=Path, required=True, help="folder containing train/ and test/")
  parser.add_argument("--out_dir", type=Path, default=Path("runs/classification"))
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--batch_size", type=int, default=128)
  parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count()))
  parser.add_argument("--seed", type=int, default=None, help="optional, the original notebook did not fix a seed")
  parser.add_argument("--predict_dir", type=Path, default=None, help="optional folder of .png frames to label after training")
  args = parser.parse_args()

  args.out_dir.mkdir(parents=True, exist_ok=True)
  if args.seed is not None:
    torch.manual_seed(args.seed)
  print(f"torch {torch.__version__} | device : {device}")

  # Creating train and test transforms
  train_transform = transforms.Compose([
      transforms.Resize(size=(100,100)),
      transforms.TrivialAugmentWide(num_magnitude_bins=31),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0,0,0],std=[1,1,1],inplace=True)
  ])

  test_transforms = transforms.Compose([
      transforms.Resize(size=(100,100)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0,0,0],std=[1,1,1],inplace=True)
  ])

  # Create Dataset from the data
  train_dataset = datasets.ImageFolder(root = args.data_dir / "train", transform=train_transform)
  test_dataset = datasets.ImageFolder(root = args.data_dir / "test", transform=test_transforms)
  print(train_dataset, test_dataset, train_dataset.class_to_idx, sep="\n")

  # Create DataLoader from the dataset
  train_dataloader = DataLoader(dataset=train_dataset,
                                shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

  test_dataloader = DataLoader(dataset=test_dataset,
                               shuffle=False,
                               batch_size=args.batch_size,
                               num_workers=args.num_workers)

  # Model, forward pass check
  model = WCEBleedGen().to(device)

  batch_img, batch_label = next(iter(train_dataloader))
  model.eval()
  with torch.inference_mode():
    print(f"Forward pass check : {model(batch_img.to(device)).shape}")
  if summary is not None:
    summary(model, input_size=[1, 3, 100, 100])

  # Loss Function and Optimiser
  loss_fn = nn.CrossEntropyLoss()
  optimiser = torch.optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9)
  scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimiser,step_size=1,gamma=0.9)

  # Actual Training
  start_time = timer()
  result = train(model = model,
                 train_dataloader = train_dataloader,
                 test_dataloader = test_dataloader,
                 loss_fn = loss_fn,
                 optimiser = optimiser,
                 scheduler = scheduler,
                 epochs = args.epochs
                 )
  end_time = timer()
  print(f"Total time taken for training : {end_time - start_time} seconds")

  plot_results(result, args.epochs, args.out_dir / "loss_accuracy_curves.png")
  torch.save(model.state_dict(), args.out_dir / "Classification Model Parameters")

  # Classification Metrics (test set)
  report_text, report_dict = classification_metrics(model, test_dataset)
  print(report_text)
  (args.out_dir / "classification_report.txt").write_text(report_text)
  (args.out_dir / "history.json").write_text(json.dumps({"history": result, "report": report_dict, "train_seconds": end_time - start_time}, indent=1))

  # Optional : labels for a folder of unseen frames
  if args.predict_dir is not None:
    idx_to_class = {val: key for key, val in train_dataset.class_to_idx.items()}
    predict_folder(model, args.predict_dir, idx_to_class).to_csv(args.out_dir / "predictions.csv", index=False)
