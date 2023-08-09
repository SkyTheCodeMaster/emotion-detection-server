# Trains the model in model.py with the datasets in `(train|test)set.csv`
# The trainset is the first 36000 rows of the big dataset
# The testset is the last 4000 rows.

from torch import nn
import torch
import model
from make_token import tokenize
import csv
import numpy as np
from random import shuffle

data_path = "./emotion_weights.pth"

RETRAIN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = model.Network()
net.to(device)
net = net.double()

trainset: list[tuple[str,str]] = []

# Load the dataset
with open("trainset.csv", newline="") as f:
  creader = csv.reader(f, delimiter=",")
  for row in creader:
    trainset.append(row)

testset: list[tuple[str,str]] = []

# Load the dataset
with open("testset.csv", newline="") as f:
  creader = csv.reader(f, delimiter=",")
  for row in creader:
    testset.append(row)


CLASSES = {'empty': 1, 'sadness': 2, 'boredom': 3, 'anger': 4, 'relief': 5, 'fun': 6, 'surprise': 7, 'happiness': 8, 'hate': 9, 'love': 10, 'worry': 11, 'enthusiasm': 12, 'neutral': 13}

if RETRAIN:
  import torch.optim as optim

  criterion = nn.L1Loss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  print("Start training")
  try:
    print("Shuffling trainset")
    shuffle(trainset)
    epoch = 0
    while True:
      epoch += 1
      print("epoch #", epoch)
      running_loss = 0.0
      for i,data in enumerate(trainset):
        # collect the data
        label, inputs = data
        tlabel = torch.tensor([CLASSES[label]], device=device)
        in_array = tokenize(inputs)
        np_array = np.array(in_array).astype(np.double)
        tin = torch.from_numpy(np_array).to(device)

        optimizer.zero_grad()
        outputs = net(tin)

        loss = criterion(outputs, tlabel)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 2000 == 1999:
            print(f"[{epoch}, {i+1:5d}] loss: {running_loss / 2000:.3f}")
            running_loss = 0.0
      # save the dataset incase power goes out or something
      torch.save(net.state_dict(),data_path)
      # also do an accuracy test. if accuracy is >90%, finish.
      total = 0
      correct = 0
      
      for data in testset:
        label, text = data
        tlabel = torch.tensor([CLASSES[label]], device=device)
        in_array = tokenize(text)
        np_array = np.array(in_array).astype(np.double)
        tin = torch.from_numpy(np_array).to(device)
        outputs = net(tin)
        prediction = torch.round(outputs)
        total += 1
        correct += (1 if prediction == tlabel else 0)

      print(f"epoch #{epoch} accuracy: {(correct / total)*100:.2f}%")
      if (correct / total) > 0.95:
        print(">95% accuracy achieved, exiting")
        torch.save(net.state_dict(),data_path)
        break
      
  except KeyboardInterrupt:
    pass
  print("Finished training")

  torch.save(net.state_dict(),data_path)
else:
  net.load_state_dict(torch.load(data_path))

print("Accuracy checking")

total = 0
correct = 0

for data in testset:
  label, text = data
  tlabel = torch.tensor([CLASSES[label]], device=device)
  in_array = tokenize(text)
  np_array = np.array(in_array).astype(np.double)
  tin = torch.from_numpy(np_array).to(device)
  outputs = net(tin)
  prediction = torch.round(outputs)
  total += 1
  correct += (1 if prediction == tlabel else 0)

print(f"Network Accuracy: {(correct / total)*100:.2f}%")