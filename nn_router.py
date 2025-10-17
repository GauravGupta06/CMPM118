import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.datasets as datasets
from torchvision.transforms import v2

"""
Possible Approaches:

1. --Architecture--
- ANN: Simple, fast, easy to implement. But if fed some form of images/frames, it may ignore correlations that a CNN would pick up.
       Will need some input to work.
- CNN: Can detect visual patterns in frames. But may be more computationally expensive.

2. --ANN Inputs--
- LZC: Alone is too simple. It'd likely be better to use a threshold. So, this must be combined with something else
- Entropy: Maybe just use LZC instead? https://direct.mit.edu/neco/article/16/4/717/6821/Estimating-the-Entropy-Rate-of-Spike-Trains-via
- Standard Deviation
- The actual spike inputs (converting events into frames)
- Spike count
- (Average) Firing rate
- Perhaps, the most appealing idea is to combine most or all of these possible inputs, and feed all of it into the ANN. This would be complex,
  but it would make our ANN as accurate as possible

3. --ANN Outputs--
- Two choices: either "use small SNN" or "use large SNN". Simple enough
- Perhaps include confidence/probability

4. --Training Methods--
- Supervised learning: Label each input with which SNN would achieve better accuracy. This seems like it might take a while though, but we can automate it
                       by running the entire dataset through both SNNs, noting which SNN gets it right (if the small SNN gets it right, label "use the small
                       SNN", else "use large SNN"). Maybe we save it onto a CSV I dunno.
- Reinforced learning: nah let's just do supervised learning
- Make sure we reward the model for saving energy. If our SNNs are built for classification, then I think this isn't as much of an issue, since all we need
  to do is make sure the ANN picks the small SNN whenever the small SNN could get an input right. In our loss function, we do need to add loss if the ANN
  picks the large SNN when the small SNN would've worked.

5. --Things to Look Out For--
- Class imbalance: As Jason mentioned. Essentially, if most problems can be correctly handled using only one of the SNNs, then our dataset is imbalanced (it
                   probably will be), and our SNN may just always pick that one SNN for everything. We can mitigate this issue by: assigning higher weight
                   to the minority class samples in the loss function, oversampling minority class samples (perhaps with repeated/augmented examples) and
                   undersampling majority class samples. Also, use precision, recall, & F1-score to evaluate model results, not general accuracy.
"""

class ANN_Router(nn.Module):
    def __init__(self):
        super().__init__()
        #self.layer1 = nn.Linear()
        #self.activation = nn.ReLU()