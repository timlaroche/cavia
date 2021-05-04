import torch
import torch.nn as nn
import torch.nn.functional as F

class NMLinear(nn.Module):
	def __init__(self, input_channels, output_channels):
		super(NMLinear, self).__init__()
		self.ws = nn.Linear(input_channels, output_channels) # Hardcode 8 for now, TODO: Make this a parameter in constructor this
		self.wm = nn.Linear(input_channels, 8)
		self.wn = nn.Linear(8, output_channels)

	def forward(self, input):
		y = F.relu(self.wm(input))
		y_prime = self.wn(y).tanh() # F.tanh is deprecated, source code for F.tanh(input) just calls input.tanh
		y_prime_updated_sign = torch.sign(y_prime)
		y_prime_updated_sign[y_prime_updated_sign == 0.] = 1. # 0 values have sign 1 rather than 0

		z = self.ws(input)

		output = F.relu(z * y_prime_updated_sign)
		return output
