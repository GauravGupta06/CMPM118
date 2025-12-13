import torch
import tonic
import torch_directml as tdm

from LoadDataset import load_dataset
from SNN_model import *

# Model hyperparameters
w_large = 32
h_large = 32
n_frames_large = 32

w_small = 32
h_small = 32
n_frames_small = 32


def evaluate_model_on_test(model, test_loader, device):
	"""Return (accuracy, avg_spikes_per_sample)."""
	# Use the model's validate method for accuracy (returns ratio)
	accuracy = model.validate_model(test_loader)

	# Compute average spikes per sample
	total_spikes = 0.0
	total_samples = 0
	with torch.no_grad():
		for data, targets in test_loader:
			data = data.to(device)
			spk_rec, spike_count = model.forward_pass(data)
			# spike_count is total spikes across the batch
			batch_size = data.shape[1] if data.dim() >= 2 else 1
			total_spikes += float(spike_count)
			total_samples += int(batch_size)

	avg_spikes = total_spikes / total_samples if total_samples > 0 else 0.0
	return float(accuracy), float(avg_spikes)


def main():
	# Setup device
	device = tdm.device() if tdm.is_available() else torch.device("cpu")
	print("Using device:", device)
	print(device)

	# Load in the preprocessed dataset
	cached_train, cached_test, num_classes = load_dataset(
		dataset_name="DVSGesture",
		dataset_path="./data",
		w=32,
		h=32,
		n_frames=32
	)

	active_cores = 7
	test_loader = torch.utils.data.DataLoader(
		cached_test,
		batch_size=1,
		shuffle=False,
		num_workers=active_cores,
		drop_last=False,
		collate_fn=tonic.collation.PadTensors(batch_first=False)
	)

	# Create and load model
	model = DVSGestureSNN_FC(
		w=w_small,
		h=h_small,
		n_frames=n_frames_small,
		beta=0.4,
		spike_lam=1e-7,
		slope=25,
		model_type="dense",
		device=device
	)
	model.load_model("results/large/models/Non_Sparse_Take93_32x32_T32_FC_Epochs150.pth")

	# Evaluate
	accuracy, avg_spikes = evaluate_model_on_test(model, test_loader, device)

	print(f"\nResults on test set:")
	print(f"- Accuracy: {accuracy * 100:.2f}%")
	print(f"- Average spikes per sample: {avg_spikes:.2f}")


if __name__ == "__main__":
	import torch.multiprocessing
	torch.multiprocessing.set_start_method('spawn', force=True)
	main()

