# Provides the NN.Module class, and a function to run text through the model (in inference mode)

from torch import nn, tensor
import torch
from make_token import tokenize

CLASSES = ['empty', 'sadness', 'boredom', 'anger', 'relief', 'fun', 'surprise', 'happiness', 'hate', 'love', 'worry', 'enthusiasm', 'neutral']

class Network(nn.Module):
  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.linear_relu_stack = nn.Sequential(                                                                                                               nn.Linear(40,30), # 40 input tokens, whittle it down to 13 neurons                                                                                  nn.ReLU(),                                                                                                                                          nn.Linear(30,25),
      nn.ReLU(),
      nn.Linear(25,20),
      nn.ReLU(),
      nn.Linear(20,1) # 1 output neurons, 1 for each class.
    )

  def forward(self, x):
    logits = self.linear_relu_stack(x)
    return logits
  
def infer(text: str, net: Network, *, device = "cpu") -> str: # Convert text into an emotion
  # Move the network to the correct device
  net.to(device)

  # Tokenize the text
  tokens = tokenize(text)
  # pad tokens until 40
  diff = 40-len(tokens)
  if diff > 0:
    for _ in range(diff):
      tokens.append(0)
  # because this could still end up with >40 tokens, trim it down to 40 max.
  tokens = tokens[:40]

  # Turn the tokens into a tensor, and move it to the correct device.
  ttokens = tensor(tokens,dtype=torch.LongTensor).to(device)

  # run it through the model.
  outputs = net(ttokens)
  print(outputs)
  
  _, prediction = torch.max(outputs, 1)

  print(prediction)