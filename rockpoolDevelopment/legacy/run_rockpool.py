import torch
import tonic
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from LoadDataset import load_dataset
from RockpoolSNN_model import DVSGestureSNN_FC


def evaluate_model_on_test(model, test_loader, device):
    """
    Evaluate model on test set.

    Returns:
        accuracy: Test accuracy (0-1)
        avg_spikes_per_sample: Average number of spikes per sample
    """
    # Use the model's validate method for accuracy
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
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device:", device)

    # Model hyperparameters
    w = 32
    h = 32
    n_frames = 32

    # Load the preprocessed dataset
    print("\nLoading DVSGesture dataset...")
    cached_train, cached_test, num_classes = load_dataset(
        dataset_name="DVSGesture",
        dataset_path="./data",
        w=w,
        h=h,
        n_frames=n_frames
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
    print("\nCreating Rockpool model...")
    model = DVSGestureSNN_FC(
        w=w,
        h=h,
        n_frames=n_frames,
        tau_mem=0.02,
        spike_lam=1e-7,
        model_type="dense",
        device=device,
        num_classes=num_classes
    )

    # Load trained weights
    # TODO: Update this path to your trained model
    model_path = "../results/large/models/Rockpool_Non_Sparse_Take1_32x32_T32_FC_Rockpool_Epochs150.pth"

    if os.path.exists(model_path):
        model.load_model(model_path)
    else:
        print(f"\nWarning: Model file not found at {model_path}")
        print("Please train a model first using train_rockpool.py")
        print("Running evaluation on untrained model...\n")

    # Evaluate
    print("\n" + "="*60)
    print("Evaluating on test set...")
    print("="*60 + "\n")

    accuracy, avg_spikes = evaluate_model_on_test(model, test_loader, device)

    print(f"\nResults on test set:")
    print(f"- Accuracy: {accuracy * 100:.2f}%")
    print(f"- Average spikes per sample: {avg_spikes:.2f}")

    # Convert to Xylo for energy estimation
    print("\n" + "="*60)
    print("Converting to Xylo format...")
    print("="*60 + "\n")

    try:
        xylo_model, metadata = model.to_xylo_compatible()
        print("Successfully converted to XyloSim!")
        print(f"\nXylo Model Metadata:")
        print(f"- Architecture: {metadata['architecture']}")
        print(f"- Total parameters: {metadata['total_params']:,}")
        print(f"- Membrane time constant: {metadata['tau_mem']*1000:.1f}ms")
        print(f"- Quantization scales:")
        print(f"  - Input: {metadata['quantization_scales']['input']:.4f}")
        print(f"  - Recurrent: {metadata['quantization_scales']['recurrent']:.4f}")
        print(f"  - Output: {metadata['quantization_scales']['output']:.4f}")

    except Exception as e:
        print(f"Error converting to Xylo: {e}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
