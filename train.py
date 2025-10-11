import torch
from models.small_snn import SmallSNN
from models.large_snn import LargeSNN
from utils.datasets import load_dvsgesture
from utils.training import train_snn
from utils.evaluation import evaluate_snn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Choose model type: "small" or "large"
model_type = "small"

if model_type == "small":
    model = SmallSNN()
else:
    model = LargeSNN()

# OPTIONAL: Add your Google Drive file ID (replace below)
google_drive_file_id = "YOUR_FILE_ID_HERE"

trainloader, testloader = load_dvsgesture(
    batch_size=32,
    drive_file_id=google_drive_file_id
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_snn(model, trainloader, optimizer, num_epochs=5, device=device)
evaluate_snn(model, testloader, device=device)
