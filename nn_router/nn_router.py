import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.datasets as datasets
from torchvision.transforms import v2
import pandas as pd
import matplotlib.pyplot as plt
import snntorch as snn
from sklearn.preprocessing import StandardScaler

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

class ANN_Router_Dataset(Dataset):
    def __init__(self, feature_csv, label_csv):
        self.features = pd.read_csv(feature_csv)
        self.labels = pd.read_csv(label_csv)
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        input = self.features.iloc[index]
        output = self.labels.iloc[index]
        x = input.iloc[[1,2,3,4]].values.astype(np.float32)
        y = output.iloc[4].astype(np.float32)

        return x, y

class ANN_Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.activation = nn.ReLU()

    def forward(self, input):
        partial = self.activation(self.layer1(input))
        partial = self.activation(self.layer2(partial))
        output = self.layer3(partial)
        return output
    
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

full_dataset = ANN_Router_Dataset("data_train.csv", "data_train_snn.csv")
train_len = int(0.8 * full_dataset.length)
validation_len = full_dataset.length - train_len

# TODO: ADD SCALAR
"""
scaler_x = StandardScaler()
features_scaled = scaler_x.fit_transform(self.features.values)
"""

train_dataset, validation_dataset = random_split(full_dataset, [train_len, validation_len])

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=16)

model = ANN_Router().to(device)

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

NUM_EPOCHS = 1000

for i in range(NUM_EPOCHS):
    model.train()

    # For printing average
    total_loss = 0
    total_pred = 0
    correct_pred = 0

    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        
        # PREDICT
        pred = model(x)

        # SCORE
        loss = loss_function(pred, y.float().unsqueeze(1))
        total_loss += loss.item()
        total_pred += y.size(0)
        
        prob = torch.sigmoid(pred)
        decision = (prob > 0.5).long()
        correct_pred += (decision == y).sum().item()

        # LEARN
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



    # VALIDATION LOOP
    model.eval()
    val_loss = 0.0
    val_correct_pred = 0
    val_total_pred = 0

    with torch.no_grad():
        for x, y in validation_dataloader:
                
            x = x.to(device)
            y = y.to(device)

            # PREDICT
            pred = model(x)

            # SCORE
            loss = loss_function(pred, y.float().unsqueeze(1))
            val_loss += loss.item()
            val_total_pred += y.size(0)
                
            prob = torch.sigmoid(pred)
            decision = (prob > 0.5).long()
            val_correct_pred += (decision == y).sum().item()

    avg_loss = total_loss / total_pred
    accuracy = correct_pred / total_pred
    print(f"TRAINING Epoch: {i+1} - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Average loss and accuracy calculation after one epoch on validation
    avg_val_loss = val_loss / val_total_pred
    accuracy_val = val_correct_pred / val_total_pred
    print(f"VALIDATION Epoch {i+1} - Average Loss: {avg_val_loss:.4f}, Accuracy: {accuracy_val:.4f}\n")


model.eval()

# TESTING LOOP
"""
with torch.no_grad():
    
    # For printing average
    total_loss = 0
    num_batches = 0

    for x, y in test_dataloader:
        # PREDICT
        pred = model(x)

        # SCORE
        loss = loss_function(pred, y)

        total_loss += loss.item()
        num_batches += 1

    print(f"TEST LOOP / Loss Average: {total_loss / num_batches}")
"""

# PLOT
"""
fig, axs = plt.subplots(2, 1)
linear = np.linspace(0, NUM_EPOCHS, 680)
avg_linear = np.linspace(0, NUM_EPOCHS, NUM_EPOCHS)
axs[0].plot(linear, all_losses)
axs[0].set_title('All Losses')
axs[1].plot(avg_linear, average)
axs[1].set_title('Average')
plt.show()
"""