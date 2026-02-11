"""Simple UCI HAR model accuracy test - hardcoded paths."""
import torch
from torch.utils.data import DataLoader
from models.uci_har_model import UCIHARSNN
from datasets.uci_har import UCIHARDataset

# ========== HARDCODED PATHS ==========
MODEL_PATH = "./workspace/uci_har/sparse/models/Take1_T128_Epochs20.pth"
DATA_PATH = "./data"
# =====================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"Model path: {MODEL_PATH}")

# Load hyperparameters from checkpoint
hp = UCIHARSNN.load_hyperparams(MODEL_PATH, device=device)
print(f"Hyperparameters: tau_mem={hp['tau_mem']}, tau_syn={hp['tau_syn']}, spike_lam={hp['spike_lam']}")

# Create model
model = UCIHARSNN(
    input_size=hp.get('input_size', 9),
    n_frames=hp.get('n_frames', 128),
    tau_mem=hp['tau_mem'],
    tau_syn=hp['tau_syn'],
    spike_lam=hp['spike_lam'],
    model_type=hp.get('model_type', 'dense'),
    device=device,
    num_classes=hp.get('num_classes', 6),
    dt=hp.get('dt', 0.02),
    threshold=hp.get('threshold', 1.0),
    has_bias=hp.get('has_bias', True)
)
model.load_model(MODEL_PATH)

# Load test dataset
dataset_loader = UCIHARDataset(DATA_PATH, n_frames=hp.get('n_frames', 128))
_, test_dataset = dataset_loader.load_uci_har()
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Test samples: {len(test_dataset)}")

# Evaluate
model.net.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, targets in test_loader:
        data = data.to(device).float()
        targets = targets.to(device)
        
        output, _, _ = model.net(data)
        logits = output.mean(dim=1)
        predictions = logits.argmax(dim=1)
        
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

accuracy = 100 * correct / total
print(f"\n=== RESULTS ===")
print(f"Correct: {correct}/{total}")
print(f"Accuracy: {accuracy:.2f}%")
