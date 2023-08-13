import torch
#import torchga
import pygad.torchga
import csv
from make_token import tokenize
import numpy as np
from model import Network

RETRAIN = True
data_path = "./emotion_weights.pth"

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

def accuracy():
  total = 0
  correct = 0

  for data in testset:
    label, text = data
    tlabel = torch.tensor([CLASSES[label]], device=device)
    in_array = tokenize(text)
    np_array = np.array(in_array).astype(np.float32)
    tin = torch.from_numpy(np_array).to(device)
    outputs = model(tin)
    print(outputs.argmax(0), tlabel)
    total += 1
    correct += (outputs.argmax(0) == tlabel).sum().item()
  return correct,total

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Network()
if RETRAIN:
  label = []
  inputs = []
  for row in trainset:
    label.append(row[0])
    inputs.append(row[1])
  
  in_data_1 = []
  for data in inputs:
    in_data_1.append(tokenize(data))
  np_array = np.array(in_data_1).astype(np.float32)
  data_inputs = torch.from_numpy(np_array).to(device).to(torch.float32)
  labels = []
  for data in label:
    labels.append(CLASSES[data])
  np_array = np.array(labels).astype(np.float32)
  data_outputs = torch.from_numpy(np_array).to(device).to(torch.float32)
  
  label = []
  inputs = []
  for row in testset:
    label.append(row[0])
    inputs.append(row[1])
  
  in_data_1 = []
  for data in inputs:
    in_data_1.append(tokenize(data))
  np_array = np.array(in_data_1).astype(np.float32)
  test_inputs = torch.from_numpy(np_array).to(device).to(torch.float32)
  labels = []
  for data in label:
    labels.append(CLASSES[data])
  np_array = np.array(labels).astype(np.float32)
  test_outputs = torch.from_numpy(np_array).to(device).to(torch.float32)
  
  def shuffle():
    global data_inputs, data_outputs
    c = torch.randperm(len(data_inputs))
    data_inputs = data_inputs[c]
    data_outputs = data_outputs[c]
  
  def fitness_func(ga_instance, solution, sol_idx):
      global data_inputs, data_outputs, torch_ga, model, loss_function
  
      predictions = pygad.torchga.predict(model=model,
                                          solution=solution,
                                          data=data_inputs)
      #print(predictions.argmax(1)[0], data_outputs[0])
      solution_fitness = 1.0 / (loss_function(predictions.argmax(1).float(), data_outputs.float()).detach().numpy() + 0.00000001)
  
      return solution_fitness
  
  def on_generation(ga_instance):
      print("Generation = {generation}".format(generation=ga_instance.generations_completed))
      print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))
      shuffle()
  
  # Create the PyTorch model.
  # print(model)
  
  # Create an instance of the pygad.torchga.TorchGA class to build the initial population.
  torch_ga = pygad.torchga.TorchGA(model=model,
                             num_solutions=100)
  
  loss_function = torch.nn.CrossEntropyLoss()
  
  # Prepare the PyGAD parameters. Check the documentation for more information: https://pygad.readthedocs.io/en/latest/pygad.html#pygad-ga-class
  num_generations = 25000 # Number of generations.
  num_parents_mating = 5 # Number of solutions to be selected as parents in the mating pool.
  initial_population = torch_ga.population_weights # Initial population of network weights.
  print(initial_population)
  fitness_function = fitness_func
  
  sol_per_pop = 8
  num_genes = 40
  
  init_range_low = -2
  init_range_high = 5
  
  parent_selection_type = "sss"
  keep_parents = 1
  
  crossover_type = "uniform"
  
  mutation_type = "random"
  mutation_percent_genes = 10
  # Create an instance of the pygad.GA class
  #ga_instance = pygad.GA(num_generations=num_generations,
  #                       num_parents_mating=num_parents_mating,
  #                       initial_population=initial_population,
  #                       mutation_percent_genes=50,
  #                       mutation_type="random",
  #                       fitness_func=fitness_func,
  #                       on_generation=on_generation)
  ga_instance = pygad.GA(num_generations=num_generations,
                         num_parents_mating=num_parents_mating,
                         fitness_func=fitness_function,
                         initial_population=initial_population,
                         #sol_per_pop=sol_per_pop,
                         #num_genes=num_genes,
                         init_range_low=init_range_low,
                         init_range_high=init_range_high,
                         parent_selection_type=parent_selection_type,
                         keep_parents=keep_parents,
                         crossover_type=crossover_type,
                         mutation_type=mutation_type,
                         mutation_percent_genes=mutation_percent_genes,
                         on_generation=on_generation)
  
  # Start the genetic algorithm evolution.
  try:
    ga_instance.run()
  except KeyboardInterrupt:
    print("cancel training")
  
  
  # Returning the details of the best solution.
  solution, solution_fitness, solution_idx = ga_instance.best_solution()
  print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))
  print("Index of the best solution : {solution_idx}".format(solution_idx=solution_idx))
  
  print("Saving best solution")
  weights = pygad.torchga.model_weights_as_dict(model, solution)
  print(weights)
  model.load_state_dict(weights)
  torch.save(model.state_dict(), data_path)
  
  # After the generations complete, some plots are showed that summarize how the outputs/fitness values evolve over generations.
  ga_instance.plot_fitness(title="PyGAD & PyTorch - Iteration vs. Fitness", linewidth=4)
  
  
  
  # Make predictions based on the best solution.
  test_predictions = pygad.torchga.predict(model=model,
                                      solution=solution,
                                      data=test_inputs)
  print("Predictions : \n", test_predictions.detach().numpy())
  
  # Calculate the binary crossentropy for the trained model.
  print("Binary Crossentropy : ", loss_function(test_predictions.argmax(1).float(), test_outputs.float()).detach().numpy())

else:
  model.load_state_dict(torch.load(data_path))

correct,total = accuracy()
print(f"Network Accuracy: {correct / total:.2f}%")