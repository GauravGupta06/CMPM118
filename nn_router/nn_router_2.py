"""
11/28

This file exists as a result of me following ChatGPT and letting it describe what could be the problem with the ANN router.
Not a lot of strong changes have been added. It'll exist just to be in the GitHub history, and then I'll delete the file once I've
gotten everything I wanted out of it and have transferred any new code back to `nn_router.py`.

The conclusion is that the features aren't descriptive enough, which makes sense. They don't capture any temporal aspects of
the data. I will see if I can keep the ANN as an ANN and not a CNN, so I'll try to create new features.

Gathered sample (after validation loop and completed training):

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

---------------------------------------------------------------------------------------------------------------------------------------------
11/29

Continuing to let ChatGPT take the wheel, I've added new features, which can be found in `create_dataset.py`.

It's still messy and I'm still trying to review what it all means. I'll be spending the rest of the night trying to understand what the
features mean & playing with the model.

It works, but it seems like the features could still be better. This might indicate that an ANN might not be the best idea for a router,
and that a CNN or something else more energy-costly would be better. Or, I just need to find the right features to put in. I'll see what
I can do tomorrow

Gathered sample (after validation loop and completed training):

VAL AUC: 0.5219

Confusion:
 [[139  27]
 [ 38  12]]

feat_16    0.090929
feat_24    0.088710
feat_1     0.085855
feat_0     0.085855
feat_2     0.084535
feat_18    0.083834
feat_22    0.083699
feat_14    0.081359
feat_20    0.080341
feat_12    0.078725
feat_9     0.078322
feat_10    0.076270
feat_3     0.059119
feat_4     0.054655
feat_41    0.051706
feat_29    0.043629
feat_8     0.039591
feat_36    0.018746
feat_33    0.017931
feat_40    0.011487
idx        0.008910
feat_28    0.006866
feat_34    0.002571
feat_26   -0.000923
feat_7    -0.004777
feat_32   -0.009698
feat_39   -0.010147
feat_30   -0.013166
feat_35   -0.017950
feat_37   -0.023533
feat_38   -0.035839
feat_27   -0.052472
feat_31   -0.056064
feat_5    -0.078400
feat_6    -0.080755
feat_11         NaN
feat_13         NaN
feat_15         NaN
feat_17         NaN
feat_19         NaN
feat_21         NaN
feat_23         NaN
feat_25         NaN
dtype: float64

Feature 0: MI=0.0370
Feature 1: MI=0.0405
Feature 2: MI=0.0000
Feature 3: MI=0.0000
Feature 4: MI=0.0000
Feature 5: MI=0.0145
Feature 6: MI=0.0151
Feature 7: MI=0.0079
Feature 8: MI=0.0014
Feature 9: MI=0.0471
Feature 10: MI=0.0000
Feature 11: MI=0.0000
Feature 12: MI=0.0480
Feature 13: MI=0.0000
Feature 14: MI=0.0000
Feature 15: MI=0.0000
Feature 16: MI=0.0000
Feature 17: MI=0.0071
Feature 18: MI=0.0000
Feature 19: MI=0.0000
Feature 20: MI=0.0347
Feature 21: MI=0.0000
Feature 22: MI=0.0215
Feature 23: MI=0.0213
Feature 24: MI=0.0291
Feature 25: MI=0.0000
Feature 26: MI=0.0000
Feature 27: MI=0.0836
Feature 28: MI=0.0199
Feature 29: MI=0.0000
Feature 30: MI=0.0000
Feature 31: MI=0.0000
Feature 32: MI=0.0000
Feature 33: MI=0.0000
Feature 34: MI=0.0809
Feature 35: MI=0.0000
Feature 36: MI=0.0000
Feature 37: MI=0.0400
Feature 38: MI=0.0000
Feature 39: MI=0.0000
Feature 40: MI=0.0462
Feature 41: MI=0.0114

Basically this is pretty bad
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

features_df = pd.read_csv("data_train_v2.csv")
x = features_df.drop(columns=["idx"]).values.astype(np.float32)

labels_df = pd.read_csv("data_train_snn.csv")
y = labels_df["use_dense"].map({True: 1.0, False: 0.0}).values.astype(np.float32)

train_idx, val_idx = train_test_split(np.arange(len(x)), test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler().fit(x[train_idx])
x_scaled = scaler.transform(x).astype(np.float32)

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
        self.layer1 = nn.Linear(42, 8)
        self.layer2 = nn.Linear(8, 4)
        self.layer3 = nn.Linear(4, 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, input):
        partial = self.dropout(self.activation(self.layer1(input)))
        partial = self.dropout(self.activation(self.layer2(partial)))
        output = self.layer3(partial)
        return output
    
device = "cuda" if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

train_labels = y[train_idx].astype(int)
n_train = len(train_labels)
n_pos = train_labels.sum()
n_neg = n_train - n_pos

target_pos_frac = 0.45  # Tune 0.35-0.5
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

#train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

model = ANN_Router().to(device)

# In data_train_snn.csv, use_dense has 829 False, 248 True samples
#weights = torch.tensor([1.0, 829/248], dtype=torch.float32).to(device)
#loss_function = nn.BCEWithLogitsLoss(pos_weight=weights[1])
loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3) 

NUM_EPOCHS = 1000
current_threshold = 0.8
temporary_threshold = 0.8

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
        decision = (prob > temporary_threshold).long()
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
            decision = (prob > temporary_threshold).long()
            #val_correct_pred += (decision == y).sum().item()

            val_labels.append(y.cpu().numpy().ravel())
            val_labels_2.append(y.cpu())
            val_probs.append(prob.cpu().numpy().ravel())
            val_decisions.append(decision.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_decisions).numpy()

    precision = precision_score(y_true, y_pred)  # More precision -> less false positives
    recall = recall_score(y_true, y_pred)        # More recall    -> less false negatives
    f1 = f1_score(y_true, y_pred)                # More f1        -> more balance between precision & recall

    print(f"TRAINING Epoch: {i+1} Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")



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

th = temporary_threshold
preds = (val_probs > th).astype(int)
print("Confusion:\n", confusion_matrix(val_labels, preds))

"""
corr = features_df.corrwith(labels_df['use_dense'].astype(int))
print(corr.sort_values(ascending=False))

from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(x, y)
for i, score in enumerate(mi):
    print(f"Feature {i}: MI={score:.4f}")
"""