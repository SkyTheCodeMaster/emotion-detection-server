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
import pygad.torchga

data_path = "./emotion_weights.pth"

RETRAIN = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = model.Network()
net.to(device)
net = net.double()

torchga = pygad.torchga.TorchGA(model=net,num_solutions=10)

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

label = []
inputs = []
for row in trainset:
  label.append(row[0])
  inputs.append(row[1])

in_data_1 = []
for data in inputs:
  in_data_1.append(tokenize(data))
np_array = np.array(in_data_1).astype(np.double)
tin = torch.from_numpy(np_array).to(device)
labels = []
for data in label:
  labels.append([CLASSES[data]])
np_array = np.array(labels).astype(np.double)
tlabels = torch.from_numpy(np_array).to(device)

criterion = nn.CrossEntropyLoss()

def fitness_func(ga, solution, sol_idx):
  #print(dir(torchga))
  #model_weights_dict = torchga.model_weights_as_dict(model=model, weights_vector=solution )
  #model.load_state_dict(model_weights_dict)
  predictions = pygad.torchga.predict(net, solution, tin)
  print(len(predictions))
  solution_fitness = 1.0 / (criterion(predictions.unsqueeze(0), tlabels).detach().numpy() + 0.00000001)
  return solution_fitness
        
def callback_generation(ga_instance):
  print(f"Generation {ga_instance.generations_completed}")
  print(f"Fitness    {ga_instance.best_solution()[1]}")

num_generations = 250
num_parents_mating = 5
initial_population = torchga.population_weights
parent_selection_type = "sss" # Type of parent selection.
crossover_type = "single_point" # Type of the crossover operator.
mutation_type = "random" # Type of the mutation operator.
mutation_percent_genes = 10 # Percentage of genes to mutate. This parameter has no action if the parameter mutation_num_genes exists.
keep_parents = -1 # Number of parents to keep in the next population. -1 means keep all parents and 0 means keep nothing.


ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       initial_population=initial_population,
                       fitness_func=fitness_func,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       keep_parents=keep_parents,
                       on_generation=callback_generation)

ga_instance.run()

ga_instance.plot_result(title="iter v fitness",linewidth=4)

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
  #print(outputs)
  total += 1
  correct += (outputs.argmax(0) == tlabel).sum().item()

print(f"Network Accuracy: {(correct / total)*100:.2f}%")