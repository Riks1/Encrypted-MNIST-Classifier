import torch
import torch.nn as nn
class PolyActivation(nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.125 * x * x + 0.5 * x

class MNISTNet(nn.Module):
  def __init__(self,input_dim: int=784, hidden_dim: int= 128,num_classes: int = 10):
    super().__init__()
    self.net = nn.Sequential(
       nn.Linear(input_dim,hidden_dim),
       PolyActivation(),
       nn.Linear(hidden_dim,num_classes))
    self._init_weights()
  def _init_weights(self):
     for m in self.modules():
        if isinstance(m,nn.Linear):
           nn.init.kaiming_normal(m.weight,nonlinearity='relu')
           nn.init.zeros_(m.bias)
  def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.view(x.size(0), -1))
     

# Backward compatibility
MNIST = MNISTNet