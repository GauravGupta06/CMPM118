"""

This file exists as a result of me following ChatGPT and letting it describe what could be the problem with the ANN router.
Not a lot of strong changes have been added. It'll exist just to be in the GitHub history, and then I'll delete the file once I've
gotten everything I wanted out of it and have transferred any new code back to `nn_router.py`.

The conclusion is that the features aren't descriptive enough, which makes sense. They don't capture any temporal aspects of
the data. I will see if I can keep the ANN as an ANN and not a CNN, so I'll try to create new features.

Gathered statistics (after validation loop and completed training):

VAL AUC: 0.5663

Confusion Matrix:
 [[ 52 114]     [[ TP FN]
 [  9  41]]     [  FP TN]]

Correlation with label:
spikes_per_lzc   -0.186185
log_spikes       -0.207274
std              -0.223759
entropy          -0.233131
entropy_std      -0.249172
spike_count      -0.253289
lzc              -0.267106
dtype: float64

^^^ This apparently can be interpreted as 7% of the label variability being explained by the features, and 93% as noise
    or something not captured. Therefore, features have to change. We might need to move this to the next quarter if
    no progress is made
"""



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

features_df = pd.read_csv("data_train.csv")
labels_df = pd.read_csv("data_train_snn.csv")

lzc          = features_df.iloc[:,1].values.astype(np.float32)
entropy      = features_df.iloc[:,2].values.astype(np.float32)
std          = features_df.iloc[:,3].values.astype(np.float32)
spike_count  = features_df.iloc[:,4].values.astype(np.float32)

# Bonus features
log_spikes = np.log1p(spike_count)
spikes_per_lzc = spike_count / (lzc + 1e-6)
entropy_std = entropy * std

x = np.column_stack([lzc, entropy, std, spike_count, log_spikes, spikes_per_lzc, entropy_std]).astype(np.float32)
y = labels_df["use_dense"].map({True: 1.0, False: 0.0}).values.astype(np.float32)

train_idx, val_idx = train_test_split(np.arange(len(x)), test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler().fit(x[train_idx])
x_scaled = scaler.transform(x)

class ArrayDataset(Dataset):
    def __init__(self, x_arr, y_arr, indices):
        self.x = x_arr
        self.y = y_arr
        self.idx = np.array(indices)

    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, i):
        ind = self.idx[i]
        return self.x[ind], self.y[ind]

train_dataset = ArrayDataset(x_scaled, y, train_idx)
validation_dataset = ArrayDataset(x_scaled, y, val_idx)

class ANN_Router(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(7, 16)
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

train_labels = y[train_idx].astype(int)
n_train = len(train_labels)
n_pos = train_labels.sum()
n_neg = n_train - n_pos

target_pos_frac = 0.40   # Tune 0.35-0.5
desired_pos = int(target_pos_frac * n_train)
desired_neg = n_train - desired_pos

class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
sample_weights = class_weights[train_labels]

pos_scale = (desired_pos / n_pos) if n_pos > 0 else 1.0
neg_scale = (desired_neg / n_neg) if n_neg > 0 else 1.0
sample_weights = sample_weights * np.where(train_labels == 1, pos_scale, neg_scale)

sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

model = ANN_Router().to(device)

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




from sklearn.metrics import roc_auc_score, confusion_matrix
auc = roc_auc_score(val_labels, val_probs)
print("VAL AUC:", round(auc,4))

# histogram (inspect visually in notebook or save plot)
import matplotlib.pyplot as plt
plt.hist(val_probs[val_labels==0], bins=25, alpha=0.6, label='neg')
plt.hist(val_probs[val_labels==1], bins=25, alpha=0.6, label='pos')
plt.legend(); plt.title("Validation prob distributions"); plt.show()

# confusion with your best threshold
th = current_threshold
preds = (val_probs > th).astype(int)
print("Confusion:\n", confusion_matrix(val_labels, preds))

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

feature_names = [
    "lzc",
    "entropy",
    "std",
    "spike_count",
    "log_spikes",
    "spikes_per_lzc",
    "entropy_std"
]

x_df = pd.DataFrame(x, columns=feature_names)
y_series = pd.Series(y, name="label")

for feat in feature_names:
    plt.figure(figsize=(6,4))
    plt.title(f"{feat} by label")
    plt.boxplot(
        [x_df[feat][y_series == 0], x_df[feat][y_series == 1]],
        labels=["neg (0)", "pos (1)"]
    )
    plt.ylabel(feat)
    plt.show()

corrs = x_df.corrwith(y_series)
print("\nCorrelation with label:")
print(corrs.sort_values(ascending=False))

import itertools

pairs = list(itertools.combinations(range(len(feature_names)), 2))

for i, j in pairs:
    plt.figure(figsize=(5,5))
    plt.scatter(
        x[:, i], x[:, j],
        c=y, alpha=0.4, s=10, cmap="coolwarm"
    )
    plt.xlabel(feature_names[i])
    plt.ylabel(feature_names[j])
    plt.title(f"{feature_names[i]} vs {feature_names[j]}")
    plt.show()