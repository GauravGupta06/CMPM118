"""
Test script to verify Rockpool setup is working correctly.
"""
import torch
import numpy as np
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

print("="*60)
print("ROCKPOOL SETUP TEST")
print("="*60)

# Test 1: Import Rockpool
print("\n[Test 1] Testing Rockpool imports...")
try:
    from rockpool.nn.modules import LIFTorch, LinearTorch
    from rockpool.nn.combinators import Sequential
    from rockpool.devices.xylo.syns63300 import XyloSim
    print("✓ Rockpool imports successful (using Xylo-Audio 3 syns63300)")
except ImportError as e:
    print(f"✗ Failed to import Rockpool: {e}")
    sys.exit(1)

# Test 2: Import our models
print("\n[Test 2] Testing model imports...")
try:
    from RockpoolSNN_model import DVSGestureSNN_FC, SHDSNN, BaseSNNModel
    print("✓ Model imports successful")
except ImportError as e:
    print(f"✗ Failed to import models: {e}")
    sys.exit(1)

# Test 3: Import dataset loader
print("\n[Test 3] Testing dataset loader import...")
try:
    from LoadDataset import load_dataset
    print("✓ Dataset loader import successful")
except ImportError as e:
    print(f"✗ Failed to import LoadDataset: {e}")
    sys.exit(1)

# Test 4: Create DVSGesture model
print("\n[Test 4] Creating DVSGesture model...")
try:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"  Using device: {device}")

    model = DVSGestureSNN_FC(
        w=32,
        h=32,
        n_frames=32,
        tau_mem=0.02,
        spike_lam=1e-7,
        model_type="dense",
        device=device,
        num_classes=11
    )
    print("✓ Model creation successful")
    print(f"  Model architecture:\n{model.net}")
except Exception as e:
    print(f"✗ Failed to create model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test forward pass with dummy data
print("\n[Test 5] Testing forward pass with dummy data...")
try:
    # Create dummy input: [T, B, C, H, W]
    T, B = 32, 2
    dummy_input = torch.randn(T, B, 2, 32, 32).to(device)

    print(f"  Input shape: {dummy_input.shape}")

    # Forward pass
    with torch.no_grad():
        output, spike_count = model.forward_pass(dummy_input)

    print(f"  Output shape: {output.shape}")
    print(f"  Total spikes: {spike_count.item():.0f}")
    print("✓ Forward pass successful")
except Exception as e:
    print(f"✗ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Test Xylo conversion
print("\n[Test 6] Testing Xylo conversion...")
try:
    xylo_model, metadata = model.to_xylo_compatible()

    print("✓ Xylo conversion successful")
    print(f"  Architecture: {metadata['architecture']}")
    print(f"  Total parameters: {metadata['total_params']:,}")
    print(f"  Tau_mem: {metadata['tau_mem']*1000:.1f}ms")
    print(f"  Quantization scales:")
    print(f"    - Input: {metadata['quantization_scales']['input']:.4f}")
    print(f"    - Recurrent: {metadata['quantization_scales']['recurrent']:.4f}")
    print(f"    - Output: {metadata['quantization_scales']['output']:.4f}")
except Exception as e:
    print(f"✗ Xylo conversion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Test XyloSim inference
print("\n[Test 7] Testing XyloSim inference...")
try:
    # Prepare input for Xylo: [B, T, features]
    B, T = 1, 32
    xylo_input = np.random.randn(B, T, 2048).astype(np.float32)

    print(f"  XyloSim input shape: {xylo_input.shape}")

    # Run inference with recording enabled
    output, state, recordings = xylo_model(xylo_input, record=True)

    print(f"  XyloSim output shape: {output.shape}")
    print(f"  Total spikes: {recordings['spikes'].sum():.0f}")

    # Calculate energy estimate
    total_spikes = recordings['spikes'].sum()
    energy_pJ = total_spikes * 23  # picojoules per spike
    energy_nJ = energy_pJ / 1000

    print(f"  Energy estimate: {energy_pJ:.2f} pJ ({energy_nJ:.2f} nJ)")
    print("✓ XyloSim inference successful")
except Exception as e:
    print(f"✗ XyloSim inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Test training components
print("\n[Test 8] Testing training components...")
try:
    # Create dummy batch
    dummy_data = torch.randn(32, 4, 2, 32, 32).to(device)
    dummy_targets = torch.randint(0, 11, (4,)).to(device)

    print(f"  Dummy batch - Data: {dummy_data.shape}, Targets: {dummy_targets.shape}")

    # Forward pass
    model.net.train()
    spk_rec, spike_count = model.forward_pass(dummy_data)
    spike_counts = spk_rec.sum(0)

    # Loss calculation
    loss = model.loss_fn(spike_counts, dummy_targets) + model.spike_regularizer(spike_count, lam=model.spike_lam)

    # Backward pass
    model.optimizer.zero_grad()
    loss.backward()
    model.optimizer.step()

    print(f"  Loss: {loss.item():.4f}")
    print(f"  Spike regularization term: {(model.spike_regularizer(spike_count, lam=model.spike_lam)).item():.6f}")
    print("✓ Training components working")
except Exception as e:
    print(f"✗ Training test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 9: Check data directory
print("\n[Test 9] Checking data directory...")
data_path = "./data"
if os.path.exists(data_path):
    print(f"✓ Data directory exists: {data_path}")
    # List datasets
    if os.path.exists(f"{data_path}/dvsgesture"):
        print("  - DVSGesture dataset found")
    if os.path.exists(f"{data_path}/shd"):
        print("  - SHD dataset found")
else:
    print(f"⚠ Data directory not found: {data_path}")
    print("  You'll need to download datasets before training")

# Test 10: Check results directory
print("\n[Test 10] Checking results directory...")
results_path = "../results"
if os.path.exists(results_path):
    print(f"✓ Results directory exists: {results_path}")
    if os.path.exists(f"{results_path}/large"):
        print("  - Large models directory found")
    if os.path.exists(f"{results_path}/small"):
        print("  - Small models directory found")
else:
    print(f"⚠ Results directory not found: {results_path}")

# Summary
print("\n" + "="*60)
print("SETUP TEST SUMMARY")
print("="*60)
print("✓ All core tests passed!")
print("\nYou can now:")
print("  1. Train a model: python train_rockpool.py --dataset DVSGesture")
print("  2. Evaluate a model: python run_rockpool.py")
print("  3. Run energy simulations with XyloSim")
print("\nNote: Make sure datasets are downloaded before training.")
print("="*60)
