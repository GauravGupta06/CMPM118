# All imports go here
import tonic
import torch
from SNN_model import *
from LoadDataset import load_dataset


# Set device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)



# Load the dataset
cached_train, cached_test, num_classes = load_dataset(
    dataset_name="DVSGesture",
    dataset_path='./data',
    w=32,
    h=32,
    n_frames=32,
)


# Define the dense model
dense_fc = DVSGestureSNN_FC(
    w=32, h=32, n_frames=32,
    beta=0.8,
    spike_lam=0,  # No spike penalty for dense
    model_type="dense",
    device=device
)


# Define the sparse model
sparse_fc = DVSGestureSNN_FC(
    w=32, h=32, n_frames=32,
    beta=0.4,
    spike_lam=0,  # Spike penalty for sparse
    model_type="sparse",
    device=device
)



# Define data loaders
train_loader = torch.utils.data.DataLoader(cached_train, batch_size=1028, shuffle=False, num_workers = 17, drop_last=True, 
                                           collate_fn=tonic.collation.PadTensors(batch_first=False))
test_loader = torch.utils.data.DataLoader(cached_test, batch_size=64, shuffle=False, num_workers = 17, drop_last=True, 
                                          collate_fn=tonic.collation.PadTensors(batch_first=False))


# # Train the model
# print("starting training dense model")
# dense_fc.train_model(train_loader, test_loader, num_epochs = 150)
# dense_fc.save_model()

print("starting training sparse model")
sparse_fc.train_model(train_loader, test_loader, num_epochs = 250)
sparse_fc.save_model()


