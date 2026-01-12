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

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, random_split, WeightedRandomSampler
import torchvision.datasets as datasets
from torchvision.transforms import v2
import pandas as pd
import matplotlib.pyplot as plt
import snntorch as snn
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ANN_Router_Dataset(Dataset):
    def __init__(self, feature_csv, label_csv):
        features_df = pd.read_csv(feature_csv)
        scaler_x = StandardScaler()
        self.features = scaler_x.fit_transform(features_df.iloc[:, [1,2,3,4]]).astype(np.float32)

        self.labels = pd.read_csv(label_csv)
        self.labels = self.labels["use_dense"].map({True: 1.0, False: 0.0}).values.astype(np.float32)
        self.length = len(self.features)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]

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

train_dataset, validation_dataset = random_split(full_dataset, [train_len, validation_len])

target_pos_frac = 0.35

train_indices = train_dataset.indices  
train_labels = full_dataset.labels[train_indices].astype(int)
n_train = len(train_labels)
n_pos = train_labels.sum()
n_neg = n_train - n_pos

desired_pos = int(target_pos_frac * n_train)
desired_neg = n_train - desired_pos

class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]

pos_scale = (desired_pos / n_pos) if n_pos > 0 else 1.0
neg_scale = (desired_neg / n_neg) if n_neg > 0 else 1.0
sample_weights *= np.where(train_labels == 1, pos_scale, neg_scale)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)
train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=sampler)


#train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=16)

model = ANN_Router().to(device)

"""
# In data_train_snn.csv, use_dense has 829 False, 248 True samples
weights = torch.tensor([1.0, 829/248], dtype=torch.float32).to(device)
loss_function = nn.BCEWithLogitsLoss(pos_weight=weights[1])
"""
loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001) 

NUM_EPOCHS = 1000
current_threshold = 0.8

for i in range(NUM_EPOCHS):
    model.train()

    """
    # For printing average
    total_loss = 0
    total_pred = 0
    correct_pred = 0
    """

    all_labels = []
    all_decisions = []

    for x, y in train_dataloader:
        x = x.to(device)
        y = y.to(device)
        
        # PREDICT
        pred = model(x)

        # SCORE
        loss = loss_function(pred, y.float().unsqueeze(1))

        #total_loss += loss.item() * y.size(0)
        #total_pred += y.size(0)
        
        prob = torch.sigmoid(pred)
        decision = (prob > 0.6).long()
        #correct_pred += (decision == y).sum().item()

        all_labels.append(y.cpu())
        all_decisions.append(decision.cpu())

        # LEARN
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    
    # VALIDATION LOOP
    model.eval()

    """
    val_loss = 0.0
    val_correct_pred = 0
    val_total_pred = 0
    """

    val_labels = []
    val_labels_2 = []
    val_probs = []
    val_decisions = []

    with torch.no_grad():
        for x, y in validation_dataloader:
                
            x = x.to(device)
            y = y.to(device)

            # PREDICT
            pred = model(x)

            # SCORE
            loss = loss_function(pred, y.float().unsqueeze(1))

            #val_loss += loss.item() * y.size(0)
            #val_total_pred += y.size(0)
                
            prob = torch.sigmoid(pred)
            decision = (prob > 0.6).long()
            #val_correct_pred += (decision == y).sum().item()

            val_labels.append(y.cpu().numpy().ravel())
            val_labels_2.append(y.cpu())
            val_probs.append(prob.cpu().numpy().ravel())
            val_decisions.append(decision.cpu())
    
    """
    avg_loss = total_loss / total_pred
    accuracy = correct_pred / total_pred
    print(f"TRAINING Epoch: {i+1} - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    """

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_decisions).numpy()

    precision = precision_score(y_true, y_pred)  # More precision -> less false positives
    recall = recall_score(y_true, y_pred)        # More recall    -> less false negatives
    f1 = f1_score(y_true, y_pred)                # More f1        -> more balance between precision & recall

    print(f"TRAINING Epoch: {i+1} Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    """
    # Average loss and accuracy calculation after one epoch on validation
    avg_val_loss = val_loss / val_total_pred
    accuracy_val = val_correct_pred / val_total_pred
    print(f"VALIDATION Epoch {i+1} - Average Loss: {avg_val_loss:.4f}, Accuracy: {accuracy_val:.4f}\n")
    """

    val_true = torch.cat(val_labels_2).numpy()
    val_pred = torch.cat(val_decisions).numpy()

    val_precision = precision_score(val_true, val_pred)  # More precision -> less false positives
    val_recall = recall_score(val_true, val_pred)        # More recall    -> less false negatives
    val_f1 = f1_score(val_true, val_pred)                # More f1        -> more balance between precision & recall

    print(f"VALIDATION Epoch: {i+1} Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

    # Searching for the best threshold
    val_probs = np.concatenate(val_probs)
    val_labels = np.concatenate(val_labels)

    thresholds = np.linspace(0.05, 0.95, 19)
    best_threshold = 0.5
    best_f1 = 0.0
    for t in thresholds:
        preds = (val_probs > t).astype(int)
        curr_f1 = f1_score(val_labels, preds)
        if curr_f1 > best_f1:
            best_f1 = curr_f1
            best_threshold = t
            current_threshold = t

    print(f"Best value for F1: {best_f1:.4f} at threshold {best_threshold:.2f}\n")

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