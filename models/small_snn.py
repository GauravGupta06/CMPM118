import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

class SmallSNN(nn.Module):
    def __init__(self, num_inputs=128*128, num_hidden=256, num_outputs=11, beta=0.9):
        super().__init__()
        spike_grad = surrogate.atan()
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        spk2_rec = []
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step].view(x.size(1), -1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec)
