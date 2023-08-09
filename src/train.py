# Trains the model in model.py with the datasets in `(train|test)set.csv`
# The trainset is the first 36000 rows of the big dataset
# The testset is the last 4000 rows.

from torch import nn
import torch
import model
from make_token import tokenize
import csv
import numpy as np

data_path = "./emotion_weights.pth"

RETRAIN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = model.Network()
net.to(device)
net = net.double()

dataset: list[tuple[str,str]] = []

# Load the dataset
with open("trainset.csv", newline="") as f:
  creader = csv.reader(f, delimiter=",")
  for row in creader:
    dataset.append(row)


CLASSES = {'empty': 1, 'sadness': 2, 'boredom': 3, 'anger': 4, 'relief': 5, 'fun': 6, 'surprise': 7, 'happiness': 8, 'hate': 9, 'love': 10, 'worry': 11, 'enthusiasm': 12, 'neutral': 13}

if RETRAIN:
  import torch.optim as optim

  criterion = nn.L1Loss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  print("Start training")
  try:
    for epoch in range(500):
      print("epoch #", epoch)
      running_loss = 0.0
      for i,data in enumerate(dataset):
        # collect the data
        label, inputs = data
        tlabel = torch.Tensor([CLASSES[label]], device=device)
        in_array = tokenize(inputs)
        np_array = np.array(in_array).astype(np.double)
        tin = torch.from_numpy(np_array)

        optimizer.zero_grad()
        outputs = net(tin)

        loss = criterion(outputs, tlabel)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"[{epoch + 1}, {i+1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0
  except KeyboardInterrupt:
    pass
  print("Finished training")

  torch.save(net.state_dict(),data_path)
else:
  net.load_state_dict(torch.load(data_path))

print("Accuracy checking")

dataset: list[tuple[str,str]] = []

# Load the dataset
with open("testset.csv", newline="") as f:
  creader = csv.reader(f, delimiter=",")
  for row in creader:
    dataset.append(row)

total = 0
correct = 0

with torch.no_grad():
  for data in dataset:
    label, text = data
    tlabel = torch.Tensor([CLASSES[label]], device=device)
    in_array = tokenize(text)
    np_array = np.array(in_array).astype(np.double)
    tin = torch.from_numpy(np_array)
    outputs = net(tin)
    prediction = torch.round(outputs)
    total += 1
    correct += (1 if prediction == tlabel else 0)
    if total % 500 == 0:
      print(f"step {total}: {100 * correct / total:.2f}%")