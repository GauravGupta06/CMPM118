"""
Momentum-Based Router for Rockpool SNN models.
Uses cheap similarity comparisons and momentum tracking to reduce LZC computation overhead.

Core Idea:
- Instead of computing expensive LZC for every sample, compare new samples to a reference
- If similar enough, reuse the previous routing decision
- Track momentum to detect gradual drift in complexity
- Only recalculate LZC when similarity is low or momentum exceeds threshold

Usage:
    python momentum_router.py --sparse_model_path <path> --dense_model_path <path> [options]
"""

import numpy as np
import os
import tonic
import torch
from lempel_ziv_complexity import lempel_ziv_complexity
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from scipy.stats import entropy
import json
from datetime import datetime
import argparse
import time

# Dataset imports
from datasets.shd_dataset import SHDDataset
from datasets.dvsgesture_dataset import DVSGestureDataset
from datasets.uci_har import UCIHARDataset

# Model imports
from models.shd_model import SHDSNN_FC
from models.dvsgesture_model import DVSGestureSNN_FC
from models.uci_har_model import UCIHARSNN_FC

from core.base_model import BaseSNNModel
from torch.utils.data import DataLoader


# ========== COMPLEXITY METRICS ==========

def count_spikes_from_recording(recording_dict):
    """
    Count total spikes from model recording dictionary.
    Works with Rockpool's Sequential recording format for ANY spiking neuron type.
    """
    total_spikes = 0

    for layer_name, layer_data in recording_dict.items():
        if torch.is_tensor(layer_data):
            unique_vals = torch.unique(layer_data)
            is_binary = len(unique_vals) <= 2 and torch.all((unique_vals >= 0) & (unique_vals <= 1))
            if is_binary and layer_data.numel() > 0:
                total_spikes += layer_data.sum().item()
        elif isinstance(layer_data, dict) and 'spikes' in layer_data:
            spikes = layer_data['spikes']
            total_spikes += spikes.sum().item()

    return int(total_spikes)


def compute_lzc_from_events(events):
    """
    Compute Lempel-Ziv Complexity from spike events.
    This is the EXPENSIVE operation we want to minimize.
    """
    if torch.is_tensor(events):
        events = events.cpu().numpy()

    spike_seq = (events).astype(int).flatten()
    spike_seq_string = ''.join(map(str, spike_seq.tolist()))
    lz_score = lempel_ziv_complexity(spike_seq_string)
    return lz_score


# ========== CHEAP SIMILARITY METRICS ==========

def compute_xor_similarity(sample_a, sample_b):
    """
    Compute cheap XOR-based similarity between two samples.
    Returns normalized difference (0 = identical, 1 = completely different).

    This is O(n) and trivially parallelizable - much cheaper than LZC.
    """
    if torch.is_tensor(sample_a):
        sample_a = sample_a.cpu()
    if torch.is_tensor(sample_b):
        sample_b = sample_b.cpu()

    # Binarize the samples (threshold at 0.5 for spike data)
    binary_a = (sample_a > 0.5).float()
    binary_b = (sample_b > 0.5).float()

    # XOR: count differing bits
    xor_result = torch.abs(binary_a - binary_b)
    diff_count = xor_result.sum().item()

    # Normalize by total elements
    total_elements = binary_a.numel()
    normalized_diff = diff_count / total_elements if total_elements > 0 else 0.0

    return normalized_diff


def compute_cosine_similarity(sample_a, sample_b):
    """
    Compute cosine similarity between two samples.
    Returns 1 - cosine_sim so that 0 = identical, 1 = orthogonal.
    """
    if torch.is_tensor(sample_a):
        sample_a = sample_a.cpu().float().flatten()
    if torch.is_tensor(sample_b):
        sample_b = sample_b.cpu().float().flatten()

    dot_product = torch.dot(sample_a, sample_b)
    norm_a = torch.norm(sample_a)
    norm_b = torch.norm(sample_b)

    if norm_a == 0 or norm_b == 0:
        return 1.0  # Completely different if one is zero

    cosine_sim = dot_product / (norm_a * norm_b)
    return 1.0 - cosine_sim.item()


def compute_l1_similarity(sample_a, sample_b):
    """
    Compute normalized L1 (Manhattan) distance between two samples.
    Returns normalized difference (0 = identical).
    """
    if torch.is_tensor(sample_a):
        sample_a = sample_a.cpu().float()
    if torch.is_tensor(sample_b):
        sample_b = sample_b.cpu().float()

    l1_dist = torch.abs(sample_a - sample_b).sum().item()

    # Normalize by number of elements and max possible value
    total_elements = sample_a.numel()
    normalized_diff = l1_dist / total_elements if total_elements > 0 else 0.0

    return normalized_diff


# ========== MOMENTUM ROUTER CLASS ==========

class MomentumRouter:
    """
    Momentum-based router that tracks complexity state and minimizes LZC computations.

    Key idea: consecutive samples in a stream are often similar, so we can reuse
    routing decisions when the new sample is similar to what we've seen before.
    """

    def __init__(
        self,
        similarity_threshold=0.1,
        momentum_alpha=0.9,
        drift_threshold=0.3,
        lzc_routing_threshold=6094.0,
        similarity_method='xor'
    ):
        """
        Initialize the momentum router.

        Args:
            similarity_threshold: Max difference to consider samples "similar" (0-1)
            momentum_alpha: Exponential moving average factor for momentum (0-1)
            drift_threshold: Momentum threshold that triggers LZC recalculation
            lzc_routing_threshold: LZC value threshold for sparse vs dense routing
            similarity_method: Method for comparing samples ('xor', 'cosine', 'l1')
        """
        self.similarity_threshold = similarity_threshold
        self.momentum_alpha = momentum_alpha
        self.drift_threshold = drift_threshold
        self.lzc_routing_threshold = lzc_routing_threshold
        self.similarity_method = similarity_method

        # State variables
        self.reference_vector = None
        self.cached_lzc = None
        self.momentum = 0.0
        self.is_initialized = False

        # Statistics tracking
        self.stats = {
            'lzc_computations': 0,
            'similarity_checks': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'momentum_triggers': 0,
            'total_samples': 0
        }

    def _compute_similarity(self, sample_a, sample_b):
        """Compute similarity using the configured method."""
        if self.similarity_method == 'xor':
            return compute_xor_similarity(sample_a, sample_b)
        elif self.similarity_method == 'cosine':
            return compute_cosine_similarity(sample_a, sample_b)
        elif self.similarity_method == 'l1':
            return compute_l1_similarity(sample_a, sample_b)
        else:
            raise ValueError(f"Unknown similarity method: {self.similarity_method}")

    def _should_recalculate(self, diff):
        """
        Determine if we should recalculate LZC based on similarity and momentum.

        Returns:
            bool: True if LZC should be recalculated
            str: Reason for the decision
        """
        # Check similarity threshold
        if diff >= self.similarity_threshold:
            return True, 'similarity_exceeded'

        # Check momentum (drift detection)
        if abs(self.momentum) >= self.drift_threshold:
            return True, 'momentum_drift'

        return False, 'cache_hit'

    def update(self, new_sample):
        """
        Process a new sample and return the routing decision.

        Args:
            new_sample: Input spike tensor

        Returns:
            dict: {
                'route': 'sparse' or 'dense',
                'lzc_value': float (actual or cached),
                'used_cache': bool,
                'similarity': float (if cache was checked),
                'momentum': float
            }
        """
        self.stats['total_samples'] += 1

        # First sample: must compute LZC
        if not self.is_initialized:
            lzc_value = compute_lzc_from_events(new_sample)
            self.stats['lzc_computations'] += 1

            self.reference_vector = new_sample.clone() if torch.is_tensor(new_sample) else new_sample.copy()
            self.cached_lzc = lzc_value
            self.momentum = 0.0
            self.is_initialized = True

            route = 'sparse' if lzc_value < self.lzc_routing_threshold else 'dense'

            return {
                'route': route,
                'lzc_value': lzc_value,
                'used_cache': False,
                'similarity': 0.0,
                'momentum': self.momentum,
                'reason': 'initialization'
            }

        # Compute cheap similarity
        self.stats['similarity_checks'] += 1
        diff = self._compute_similarity(new_sample, self.reference_vector)

        # Update momentum (exponential moving average of differences)
        self.momentum = self.momentum_alpha * self.momentum + (1 - self.momentum_alpha) * diff

        # Decide whether to recalculate
        should_recalc, reason = self._should_recalculate(diff)

        if should_recalc:
            # Recalculate LZC
            lzc_value = compute_lzc_from_events(new_sample)
            self.stats['lzc_computations'] += 1
            self.stats['cache_misses'] += 1

            if reason == 'momentum_drift':
                self.stats['momentum_triggers'] += 1

            # Update reference and cached value
            self.reference_vector = new_sample.clone() if torch.is_tensor(new_sample) else new_sample.copy()
            self.cached_lzc = lzc_value
            self.momentum = 0.0  # Reset momentum after recalculation

            used_cache = False
        else:
            # Use cached LZC value
            lzc_value = self.cached_lzc
            self.stats['cache_hits'] += 1
            used_cache = True

            # Partially update reference vector toward new sample (optional blending)
            # This helps the reference track gradual changes
            if torch.is_tensor(new_sample):
                blend_factor = 0.1
                self.reference_vector = (
                    (1 - blend_factor) * self.reference_vector +
                    blend_factor * new_sample.clone()
                )

        route = 'sparse' if lzc_value < self.lzc_routing_threshold else 'dense'

        return {
            'route': route,
            'lzc_value': lzc_value,
            'used_cache': used_cache,
            'similarity': diff,
            'momentum': self.momentum,
            'reason': reason
        }

    def reset(self):
        """Reset the router state."""
        self.reference_vector = None
        self.cached_lzc = None
        self.momentum = 0.0
        self.is_initialized = False

    def get_stats(self):
        """Get router statistics."""
        stats = self.stats.copy()
        if stats['total_samples'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_samples']
            stats['lzc_computation_rate'] = stats['lzc_computations'] / stats['total_samples']
        else:
            stats['cache_hit_rate'] = 0.0
            stats['lzc_computation_rate'] = 0.0
        return stats


# ========== EVALUATION FUNCTIONS ==========

def evaluate_models_on_dataset(dataLoader, sparse_model, dense_model):
    """
    Evaluate both sparse and dense models on entire dataset.
    Also computes pairwise similarity between consecutive samples.
    """
    results = []
    prev_events = None

    for batch in dataLoader:
        events, label = batch
        events = events.to(sparse_model.device)
        label = label.item()

        # Compute LZC (for ground truth comparison)
        lz_value = compute_lzc_from_events(events)

        # Compute similarity to previous sample
        if prev_events is not None:
            similarity_xor = compute_xor_similarity(events, prev_events)
            similarity_cosine = compute_cosine_similarity(events, prev_events)
            similarity_l1 = compute_l1_similarity(events, prev_events)
        else:
            similarity_xor = 0.0
            similarity_cosine = 0.0
            similarity_l1 = 0.0

        # Get predictions and spike counts
        with torch.no_grad():
            sparse_output, _, sparse_recording = sparse_model.net(events, record=True)
            dense_output, _, dense_recording = dense_model.net(events, record=True)

        spike_count_sparse = count_spikes_from_recording(sparse_recording)
        spike_count_dense = count_spikes_from_recording(dense_recording)

        sparse_logits = sparse_output.mean(dim=1)
        dense_logits = dense_output.mean(dim=1)

        sparse_pred = sparse_logits.argmax(1).item()
        dense_pred = dense_logits.argmax(1).item()

        # Ground truth: dense needed if dense correct and sparse wrong
        true_complex = 1 if (dense_pred == label and sparse_pred != label) else 0

        results.append({
            'label': label,
            'lz_value': lz_value,
            'sparse_pred': sparse_pred,
            'dense_pred': dense_pred,
            'true_complex': true_complex,
            'dense_spikes': spike_count_dense,
            'sparse_spikes': spike_count_sparse,
            'similarity_xor': similarity_xor,
            'similarity_cosine': similarity_cosine,
            'similarity_l1': similarity_l1
        })

        prev_events = events.clone()

    return results


def analyze_similarity_distribution(results):
    """
    Analyze the distribution of similarities between consecutive samples.
    This helps determine if momentum-based routing is viable.
    """
    similarities_xor = [r['similarity_xor'] for r in results[1:]]  # Skip first (no prev)
    similarities_cosine = [r['similarity_cosine'] for r in results[1:]]
    similarities_l1 = [r['similarity_l1'] for r in results[1:]]

    print("\n" + "="*60)
    print("SIMILARITY DISTRIBUTION ANALYSIS")
    print("="*60)

    for name, sims in [('XOR', similarities_xor), ('Cosine', similarities_cosine), ('L1', similarities_l1)]:
        if len(sims) > 0:
            print(f"\n{name} Similarity (0=identical, 1=different):")
            print(f"  Mean:   {np.mean(sims):.4f}")
            print(f"  Std:    {np.std(sims):.4f}")
            print(f"  Min:    {np.min(sims):.4f}")
            print(f"  Max:    {np.max(sims):.4f}")
            print(f"  Median: {np.median(sims):.4f}")

            # What percentage would be cache hits at various thresholds?
            for thresh in [0.05, 0.1, 0.15, 0.2, 0.3]:
                hit_rate = sum(1 for s in sims if s < thresh) / len(sims)
                print(f"  Cache hit rate at threshold {thresh}: {hit_rate*100:.1f}%")

    return {
        'xor': similarities_xor,
        'cosine': similarities_cosine,
        'l1': similarities_l1
    }


def find_optimal_similarity_threshold(results, similarity_type='xor'):
    """
    Find optimal similarity threshold using ROC analysis.

    The idea: we want to find a threshold that correctly identifies when
    consecutive samples can share a routing decision vs when they need separate LZC.
    """
    # For each consecutive pair, determine if they should have the same routing decision
    # Ground truth: same routing decision if both would route to same model

    y_true = []
    similarities = []

    for i in range(1, len(results)):
        curr = results[i]
        prev = results[i-1]

        # Would both samples route to the same model?
        curr_route = 'dense' if curr['true_complex'] == 1 else 'sparse'
        prev_route = 'dense' if prev['true_complex'] == 1 else 'sparse'

        # Also check based on LZC threshold (we'll need to determine this)
        # For now, use a reasonable default
        lzc_threshold = 6094.0
        curr_lzc_route = 'dense' if curr['lz_value'] >= lzc_threshold else 'sparse'
        prev_lzc_route = 'dense' if prev['lz_value'] >= lzc_threshold else 'sparse'

        # If routes match, we COULD use cache (label = 0 means "can use cache")
        # If routes differ, we SHOULD NOT use cache (label = 1 means "must recalculate")
        should_recalculate = 1 if curr_lzc_route != prev_lzc_route else 0

        y_true.append(should_recalculate)

        if similarity_type == 'xor':
            similarities.append(curr['similarity_xor'])
        elif similarity_type == 'cosine':
            similarities.append(curr['similarity_cosine'])
        else:
            similarities.append(curr['similarity_l1'])

    y_true = np.array(y_true)
    similarities = np.array(similarities)

    # ROC analysis: similarity as predictor of "should recalculate"
    fpr, tpr, thresholds = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold using G-mean
    gmean = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(gmean)
    optimal_threshold = thresholds[idx]

    return optimal_threshold, roc_auc, fpr, tpr, thresholds, gmean


def threshold_sweep_and_roc(results, sparse_model, dense_model, plotting_only=False):
    """
    Perform threshold sweep for LZC-based routing (same as original router).
    """
    y_true = np.array([r['true_complex'] for r in results])
    lz_scores = np.array([r['lz_value'] for r in results])
    fpr, tpr, thresholds = roc_curve(y_true, lz_scores)
    roc_auc = auc(fpr, tpr)
    gmean = np.sqrt(tpr * (1 - fpr))
    idx = np.argmax(gmean)
    optimal_threshold = thresholds[idx]

    average_spike_dense = np.mean([r['dense_spikes'] for r in results])
    average_spike_sparse = np.mean([r['sparse_spikes'] for r in results])

    if plotting_only:
        return (optimal_threshold, roc_auc, average_spike_dense, average_spike_sparse)

    print(f"Average spike dense: {average_spike_dense:.2f}")
    print(f"Average spike sparse: {average_spike_sparse:.2f}")
    print(f"Optimal LZC threshold: {optimal_threshold:.4f} (G-mean={gmean[idx]:.4f}) (AUC={roc_auc:.4f})")

    # Extract parameters from models
    input_size = sparse_model.input_size
    n_frames = sparse_model.n_frames

    # Plot LZC ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.scatter(fpr[idx], tpr[idx], color='red', label=f'Optimal G-mean\n(Threshold={optimal_threshold:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for LZC-based Routing')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    os.makedirs("results/momentum_router/ROC_curves", exist_ok=True)
    graph_save_path = f"results/momentum_router/ROC_curves/LZC_Input{input_size}_T{n_frames}.png"
    plt.savefig(graph_save_path)
    plt.show()

    return optimal_threshold, roc_auc, average_spike_dense, average_spike_sparse


def evaluate_momentum_router(dataLoader, sparse_model, dense_model, router, results):
    """
    Evaluate the momentum-based router on the dataset.

    Args:
        dataLoader: Test data loader
        sparse_model: Sparse SNN model
        dense_model: Dense SNN model
        router: MomentumRouter instance
        results: Pre-computed results with LZC values (for comparison)
    """
    router.reset()

    correct_sparse = 0
    correct_dense = 0
    route_counts = {'sparse': 0, 'dense': 0}
    routing_decisions = []

    for i, batch in enumerate(dataLoader):
        events, label = batch
        events = events.to(sparse_model.device)
        label = label.item()

        # Get routing decision from momentum router
        decision = router.update(events)
        route = decision['route']

        routing_decisions.append({
            'index': i,
            'label': label,
            'route': route,
            'lzc_value': decision['lzc_value'],
            'used_cache': decision['used_cache'],
            'similarity': decision['similarity'],
            'momentum': decision['momentum'],
            'reason': decision['reason'],
            'actual_lzc': results[i]['lz_value']  # Ground truth LZC
        })

        with torch.no_grad():
            if route == 'sparse':
                route_counts['sparse'] += 1
                output, _, _ = sparse_model.net(events, record=False)
                logits = output.mean(dim=1)
                pred = logits.argmax(1).item()
                if pred == label:
                    correct_sparse += 1
            else:
                route_counts['dense'] += 1
                output, _, _ = dense_model.net(events, record=False)
                logits = output.mean(dim=1)
                pred = logits.argmax(1).item()
                if pred == label:
                    correct_dense += 1

    # Calculate accuracies
    accuracy_dense_routed = correct_dense / route_counts['dense'] if route_counts['dense'] > 0 else 0
    accuracy_sparse_routed = correct_sparse / route_counts['sparse'] if route_counts['sparse'] > 0 else 0

    total_correct = correct_dense + correct_sparse
    total_samples = route_counts['sparse'] + route_counts['dense']
    total_accuracy = total_correct / total_samples

    # Get router stats
    router_stats = router.get_stats()

    return {
        'total_accuracy': total_accuracy,
        'accuracy_dense_routed': accuracy_dense_routed,
        'accuracy_sparse_routed': accuracy_sparse_routed,
        'route_counts': route_counts,
        'router_stats': router_stats,
        'routing_decisions': routing_decisions
    }


def compare_with_baseline(results, momentum_results, sparse_model, dense_model):
    """
    Compare momentum router with baseline (always compute LZC) router.
    """
    total_samples = len(results)

    # Baseline: always compute LZC
    sparse_correct_baseline = sum(1 for r in results if r['sparse_pred'] == r['label'])
    dense_correct_baseline = sum(1 for r in results if r['dense_pred'] == r['label'])
    sparse_accuracy_baseline = sparse_correct_baseline / total_samples
    dense_accuracy_baseline = dense_correct_baseline / total_samples

    print("\n" + "="*70)
    print("COMPARISON: MOMENTUM ROUTER vs BASELINE (ALWAYS-LZC)")
    print("="*70)

    print(f"\nBaseline (Sparse only):  {sparse_accuracy_baseline*100:.2f}%")
    print(f"Baseline (Dense only):   {dense_accuracy_baseline*100:.2f}%")
    print(f"Momentum Router:         {momentum_results['total_accuracy']*100:.2f}%")

    stats = momentum_results['router_stats']
    print(f"\nLZC Computation Savings:")
    print(f"  Total samples:         {stats['total_samples']}")
    print(f"  LZC computations:      {stats['lzc_computations']}")
    print(f"  Cache hits:            {stats['cache_hits']}")
    print(f"  Cache hit rate:        {stats['cache_hit_rate']*100:.1f}%")
    print(f"  LZC calls saved:       {stats['cache_hits']} ({stats['cache_hit_rate']*100:.1f}%)")
    print(f"  Momentum triggers:     {stats['momentum_triggers']}")

    # Routing distribution
    route_counts = momentum_results['route_counts']
    print(f"\nRouting Distribution:")
    print(f"  Routed to sparse:      {route_counts['sparse']} ({route_counts['sparse']/total_samples*100:.1f}%)")
    print(f"  Routed to dense:       {route_counts['dense']} ({route_counts['dense']/total_samples*100:.1f}%)")

    print(f"\nAccuracy by Route:")
    print(f"  Sparse-routed accuracy: {momentum_results['accuracy_sparse_routed']*100:.2f}%")
    print(f"  Dense-routed accuracy:  {momentum_results['accuracy_dense_routed']*100:.2f}%")


def plot_similarity_distribution(similarities_dict, sparse_model):
    """
    Plot histograms of similarity distributions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, sims) in zip(axes, similarities_dict.items()):
        if len(sims) > 0:
            ax.hist(sims, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(sims), color='r', linestyle='--', label=f'Mean: {np.mean(sims):.3f}')
            ax.axvline(np.median(sims), color='g', linestyle='--', label=f'Median: {np.median(sims):.3f}')
            ax.set_xlabel(f'{name.upper()} Similarity')
            ax.set_ylabel('Frequency')
            ax.set_title(f'{name.upper()} Similarity Distribution\n(0=identical, 1=different)')
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("results/momentum_router/similarity_analysis", exist_ok=True)
    input_size = sparse_model.input_size
    n_frames = sparse_model.n_frames
    save_path = f"results/momentum_router/similarity_analysis/distribution_Input{input_size}_T{n_frames}.png"
    plt.savefig(save_path)
    plt.show()
    print(f"Saved similarity distribution plot to: {save_path}")


def plot_momentum_routing_analysis(routing_decisions, sparse_model):
    """
    Plot analysis of momentum router decisions over time.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    indices = [d['index'] for d in routing_decisions]
    actual_lzc = [d['actual_lzc'] for d in routing_decisions]
    used_lzc = [d['lzc_value'] for d in routing_decisions]
    similarities = [d['similarity'] for d in routing_decisions]
    momentums = [d['momentum'] for d in routing_decisions]
    used_cache = [d['used_cache'] for d in routing_decisions]

    # Plot 1: LZC values (actual vs used)
    ax1 = axes[0]
    ax1.plot(indices, actual_lzc, 'b-', alpha=0.5, label='Actual LZC')
    ax1.scatter(indices, used_lzc, c=['green' if uc else 'red' for uc in used_cache],
                s=10, label='Used LZC (green=cached, red=computed)')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('LZC Value')
    ax1.set_title('LZC Values: Actual vs Used by Router')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Similarity and cache hits
    ax2 = axes[1]
    ax2.plot(indices, similarities, 'b-', alpha=0.7, label='Similarity to Reference')
    cache_hit_indices = [i for i, uc in enumerate(used_cache) if uc]
    cache_miss_indices = [i for i, uc in enumerate(used_cache) if not uc]
    ax2.scatter(cache_hit_indices, [similarities[i] for i in cache_hit_indices],
                c='green', s=20, label='Cache Hit', zorder=5)
    ax2.scatter(cache_miss_indices, [similarities[i] for i in cache_miss_indices],
                c='red', s=20, label='Cache Miss', zorder=5)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Similarity (0=identical)')
    ax2.set_title('Similarity to Reference Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Momentum over time
    ax3 = axes[2]
    ax3.plot(indices, momentums, 'purple', alpha=0.7, label='Momentum')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Momentum')
    ax3.set_title('Momentum (Drift Detection) Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    os.makedirs("results/momentum_router/routing_analysis", exist_ok=True)
    input_size = sparse_model.input_size
    n_frames = sparse_model.n_frames
    save_path = f"results/momentum_router/routing_analysis/momentum_analysis_Input{input_size}_T{n_frames}.png"
    plt.savefig(save_path)
    plt.show()
    print(f"Saved momentum routing analysis to: {save_path}")


def save_momentum_run_to_json(
    results,
    momentum_results,
    optimal_lzc_threshold,
    roc_auc,
    average_spike_dense,
    average_spike_sparse,
    router,
    sparse_model,
    dense_model
):
    """
    Save momentum router run results to JSON file.
    """
    os.makedirs("results/momentum_router/run_logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_size = sparse_model.input_size
    n_frames = sparse_model.n_frames

    save_path = f"results/momentum_router/run_logs/run_Input{input_size}_T{n_frames}_{timestamp}.json"

    output = {
        "metadata": {
            "timestamp": timestamp,
            "router_type": "momentum",
            "sparse_model": {
                "input_size": input_size,
                "n_frames": n_frames,
                "tau_mem": sparse_model.tau_mem,
                "spike_lam": sparse_model.spike_lam,
            },
            "dense_model": {
                "input_size": dense_model.input_size,
                "n_frames": dense_model.n_frames,
                "tau_mem": dense_model.tau_mem,
                "spike_lam": dense_model.spike_lam,
            },
        },
        "router_config": {
            "similarity_threshold": router.similarity_threshold,
            "momentum_alpha": router.momentum_alpha,
            "drift_threshold": router.drift_threshold,
            "lzc_routing_threshold": router.lzc_routing_threshold,
            "similarity_method": router.similarity_method
        },
        "lzc_roc_results": {
            "optimal_threshold": float(optimal_lzc_threshold),
            "roc_auc": float(roc_auc)
        },
        "routing_metrics": {
            "overall_accuracy": float(momentum_results['total_accuracy']),
            "dense_routed_accuracy": float(momentum_results['accuracy_dense_routed']),
            "sparse_routed_accuracy": float(momentum_results['accuracy_sparse_routed']),
            "route_counts": momentum_results['route_counts'],
            "avg_dense_spikes": float(average_spike_dense),
            "avg_sparse_spikes": float(average_spike_sparse)
        },
        "router_stats": momentum_results['router_stats'],
        "per_sample_results": results
    }

    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nSaved momentum router results to: {save_path}\n")


def print_latex_table(
    total_accuracy,
    accuracy_dense_routed,
    accuracy_sparse_routed,
    avg_dense_spikes,
    avg_sparse_spikes,
    route_counts,
    roc_auc,
    optimal_threshold,
    router_stats
):
    """
    Print LaTeX table with momentum routing performance metrics.
    """
    print("\n\n===================== LATEX TABLE =====================\n")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\begin{tabular}{l c}")
    print(r"\hline")
    print(r"Metric & Value \\")
    print(r"\hline")
    print(fr"Optimal LZC threshold & {optimal_threshold:.2f} \\")
    print(fr"ROC-AUC & {roc_auc:.3f} \\")
    print(fr"Total accuracy & {total_accuracy:.3f} \\")
    print(fr"Dense-route accuracy & {accuracy_dense_routed:.3f} \\")
    print(fr"Sparse-route accuracy & {accuracy_sparse_routed:.3f} \\")
    print(fr"Avg dense spikes & {avg_dense_spikes:.1f} \\")
    print(fr"Avg sparse spikes & {avg_sparse_spikes:.1f} \\")
    print(fr"Samples to dense & {route_counts['dense']} \\")
    print(fr"Samples to sparse & {route_counts['sparse']} \\")
    print(r"\hline")
    print(fr"LZC computations & {router_stats['lzc_computations']} \\")
    print(fr"Cache hit rate & {router_stats['cache_hit_rate']*100:.1f}\% \\")
    print(fr"LZC calls saved & {router_stats['cache_hits']} \\")
    print(r"\hline")
    print(r"\end{tabular}")
    print(r"\caption{Momentum router performance and energy savings.}")
    print(r"\end{table}")
    print("\n=======================================================\n")


# ========== MAIN ==========

def main():
    parser = argparse.ArgumentParser(
        description="Momentum-Based Router for Rockpool SNN models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python momentum_router.py \\
    --sparse_model_path ./results/small/models/Sparse.pth \\
    --dense_model_path ./results/large/models/Dense.pth

  # With custom momentum parameters
  python momentum_router.py \\
    --sparse_model_path ./results/small/models/Sparse.pth \\
    --dense_model_path ./results/large/models/Dense.pth \\
    --similarity_threshold 0.15 \\
    --momentum_alpha 0.85 \\
    --drift_threshold 0.25
        """
    )

    # Required arguments
    parser.add_argument('--sparse_model_path', type=str, required=True,
                       help='Path to pre-trained sparse model (.pth file)')
    parser.add_argument('--dense_model_path', type=str, required=True,
                       help='Path to pre-trained dense model (.pth file)')

    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, default='./data',
                       help='Path to dataset cache directory (default: ./data)')
    parser.add_argument('--input_size', type=int, default=700,
                       help='Number of frequency bins (default: 700)')
    parser.add_argument('--n_frames', type=int, default=100,
                       help='Number of temporal bins (default: 100)')

    # Momentum router arguments
    parser.add_argument('--similarity_threshold', type=float, default=0.1,
                       help='Similarity threshold for cache hit (default: 0.1)')
    parser.add_argument('--momentum_alpha', type=float, default=0.9,
                       help='Momentum EMA factor (default: 0.9)')
    parser.add_argument('--drift_threshold', type=float, default=0.3,
                       help='Momentum threshold for drift detection (default: 0.3)')
    parser.add_argument('--similarity_method', type=str, default='xor',
                       choices=['xor', 'cosine', 'l1'],
                       help='Similarity computation method (default: xor)')

    # Model hyperparameter arguments
    parser.add_argument('--tau_mem_sparse', type=float, default=0.01)
    parser.add_argument('--tau_mem_dense', type=float, default=0.02)
    parser.add_argument('--spike_lam_sparse', type=float, default=1e-6)
    parser.add_argument('--spike_lam_dense', type=float, default=1e-8)

    # Additional arguments
    parser.add_argument('--NUM_CHANNELS', type=int, default=700)
    parser.add_argument('--NUM_POLARITIES', type=int, default=2)
    parser.add_argument('--net_dt', type=float, default=10e-3)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Print configuration
    print("\n" + "="*80)
    print("MOMENTUM ROUTER CONFIGURATION")
    print("="*80)
    print(f"Sparse model path:     {args.sparse_model_path}")
    print(f"Dense model path:      {args.dense_model_path}")
    print(f"Similarity threshold:  {args.similarity_threshold}")
    print(f"Momentum alpha:        {args.momentum_alpha}")
    print(f"Drift threshold:       {args.drift_threshold}")
    print(f"Similarity method:     {args.similarity_method}")
    print("="*80 + "\n")

    # ==================== DATASET LOADING ====================
    # Uncomment ONE of the following dataset sections:

    # ----- UCI-HAR Dataset -----
    print("Loading UCI-HAR dataset...")
    data = UCIHARDataset(
        dataset_path=args.dataset_path,
        n_frames=128,
        time_first=True,
        normalize=True
    )
    _, cached_test = data.load_uci_har()
    test_loader = DataLoader(
        cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True
    )

    # ----- SHD Dataset -----
    # print("Loading SHD dataset...")
    # data = SHDDataset(
    #     dataset_path=args.dataset_path,
    #     NUM_CHANNELS=args.NUM_CHANNELS,
    #     NUM_POLARITIES=args.NUM_POLARITIES,
    #     n_frames=args.n_frames,
    #     net_dt=args.net_dt
    # )
    # _, cached_test = data.load_shd()
    # test_loader = DataLoader(
    #     cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
    #     num_workers=args.num_workers, pin_memory=True,
    #     collate_fn=tonic.collation.PadTensors(batch_first=True)
    # )

    # ----- DVSGesture Dataset -----
    # print("Loading DVSGesture dataset...")
    # data = DVSGestureDataset(
    #     dataset_path=args.dataset_path,
    #     NUM_CHANNELS=args.NUM_CHANNELS,
    #     NUM_POLARITIES=args.NUM_POLARITIES,
    #     n_frames=args.n_frames,
    #     net_dt=args.net_dt
    # )
    # _, cached_test = data.load_dvsgesture()
    # test_loader = DataLoader(
    #     cached_test, batch_size=args.batch_size, shuffle=False, drop_last=True,
    #     num_workers=args.num_workers, pin_memory=True,
    #     collate_fn=tonic.collation.PadTensors(batch_first=True)
    # )

    # ==================== END DATASET LOADING ====================

    # Load hyperparameters from checkpoint files
    print("\nLoading hyperparameters from checkpoints...")
    sparse_hp = BaseSNNModel.load_hyperparams(args.sparse_model_path)
    dense_hp = BaseSNNModel.load_hyperparams(args.dense_model_path)

    print(f"Sparse model: tau_mem={sparse_hp['tau_mem']}, tau_syn={sparse_hp['tau_syn']}")
    print(f"Dense model:  tau_mem={dense_hp['tau_mem']}, tau_syn={dense_hp['tau_syn']}")

    # ==================== MODEL CREATION ====================
    # Uncomment ONE of the following model sections (must match dataset above):

    # ----- UCI-HAR Models -----
    print("\nCreating UCI-HAR sparse model...")
    sparse_model = UCIHARSNN_FC(
        input_size=sparse_hp['input_size'],
        n_frames=sparse_hp['n_frames'],
        tau_mem=sparse_hp['tau_mem'],
        tau_syn=sparse_hp['tau_syn'],
        spike_lam=sparse_hp['spike_lam'],
        model_type=sparse_hp['model_type'],
        device=device,
        num_classes=sparse_hp['num_classes'],
        dt=sparse_hp['dt'],
        threshold=sparse_hp['threshold'],
        has_bias=sparse_hp['has_bias']
    )
    print("Creating UCI-HAR dense model...")
    dense_model = UCIHARSNN_FC(
        input_size=dense_hp['input_size'],
        n_frames=dense_hp['n_frames'],
        tau_mem=dense_hp['tau_mem'],
        tau_syn=dense_hp['tau_syn'],
        spike_lam=dense_hp['spike_lam'],
        model_type=dense_hp['model_type'],
        device=device,
        num_classes=dense_hp['num_classes'],
        dt=dense_hp['dt'],
        threshold=dense_hp['threshold'],
        has_bias=dense_hp['has_bias']
    )

    # ----- SHD Models -----
    # print("\nCreating SHD sparse model...")
    # sparse_model = SHDSNN_FC(
    #     input_size=sparse_hp['input_size'],
    #     n_frames=sparse_hp['n_frames'],
    #     tau_mem=sparse_hp['tau_mem'],
    #     tau_syn=sparse_hp['tau_syn'],
    #     spike_lam=sparse_hp['spike_lam'],
    #     model_type=sparse_hp['model_type'],
    #     device=device,
    #     num_classes=sparse_hp['num_classes'],
    #     dt=sparse_hp['dt'],
    #     threshold=sparse_hp['threshold'],
    #     has_bias=sparse_hp['has_bias']
    # )
    # print("Creating SHD dense model...")
    # dense_model = SHDSNN_FC(
    #     input_size=dense_hp['input_size'],
    #     n_frames=dense_hp['n_frames'],
    #     tau_mem=dense_hp['tau_mem'],
    #     tau_syn=dense_hp['tau_syn'],
    #     spike_lam=dense_hp['spike_lam'],
    #     model_type=dense_hp['model_type'],
    #     device=device,
    #     num_classes=dense_hp['num_classes'],
    #     dt=dense_hp['dt'],
    #     threshold=dense_hp['threshold'],
    #     has_bias=dense_hp['has_bias']
    # )

    # ----- DVSGesture Models -----
    # print("\nCreating DVSGesture sparse model...")
    # sparse_model = DVSGestureSNN_FC(
    #     input_size=sparse_hp['input_size'],
    #     n_frames=sparse_hp['n_frames'],
    #     tau_mem=sparse_hp['tau_mem'],
    #     tau_syn=sparse_hp['tau_syn'],
    #     spike_lam=sparse_hp['spike_lam'],
    #     model_type=sparse_hp['model_type'],
    #     device=device,
    #     num_classes=sparse_hp['num_classes'],
    #     dt=sparse_hp['dt'],
    #     threshold=sparse_hp['threshold'],
    #     has_bias=sparse_hp['has_bias']
    # )
    # print("Creating DVSGesture dense model...")
    # dense_model = DVSGestureSNN_FC(
    #     input_size=dense_hp['input_size'],
    #     n_frames=dense_hp['n_frames'],
    #     tau_mem=dense_hp['tau_mem'],
    #     tau_syn=dense_hp['tau_syn'],
    #     spike_lam=dense_hp['spike_lam'],
    #     model_type=dense_hp['model_type'],
    #     device=device,
    #     num_classes=dense_hp['num_classes'],
    #     dt=dense_hp['dt'],
    #     threshold=dense_hp['threshold'],
    #     has_bias=dense_hp['has_bias']
    # )

    # ==================== END MODEL CREATION ====================

    # Load pre-trained weights
    print(f"\nLoading sparse model from: {args.sparse_model_path}")
    sparse_model.load_model(args.sparse_model_path)

    print(f"Loading dense model from: {args.dense_model_path}")
    dense_model.load_model(args.dense_model_path)

    print("\n" + "="*80)
    print("EVERYTHING LOADED SUCCESSFULLY")
    print("="*80 + "\n")

    # ==================== EVALUATION ====================

    # 1. Evaluate both models and compute similarities
    print("Step 1: Evaluating models and computing similarities...")
    results = evaluate_models_on_dataset(test_loader, sparse_model, dense_model)

    # 2. Analyze similarity distribution (key for viability check)
    print("\nStep 2: Analyzing similarity distribution...")
    similarities_dict = analyze_similarity_distribution(results)
    plot_similarity_distribution(similarities_dict, sparse_model)

    # 3. Find optimal LZC threshold via ROC analysis
    print("\nStep 3: Finding optimal LZC threshold...")
    optimal_lzc_threshold, roc_auc, avg_dense_spikes, avg_sparse_spikes = \
        threshold_sweep_and_roc(results, sparse_model, dense_model)

    # 4. Create and evaluate momentum router
    print("\nStep 4: Evaluating momentum router...")
    router = MomentumRouter(
        similarity_threshold=args.similarity_threshold,
        momentum_alpha=args.momentum_alpha,
        drift_threshold=args.drift_threshold,
        lzc_routing_threshold=optimal_lzc_threshold,
        similarity_method=args.similarity_method
    )

    momentum_results = evaluate_momentum_router(
        test_loader, sparse_model, dense_model, router, results
    )

    # 5. Plot momentum routing analysis
    print("\nStep 5: Plotting momentum routing analysis...")
    plot_momentum_routing_analysis(momentum_results['routing_decisions'], sparse_model)

    # 6. Compare with baseline
    compare_with_baseline(results, momentum_results, sparse_model, dense_model)

    # 7. Save results
    save_momentum_run_to_json(
        results, momentum_results, optimal_lzc_threshold, roc_auc,
        avg_dense_spikes, avg_sparse_spikes, router,
        sparse_model, dense_model
    )

    # 8. Print LaTeX table
    print_latex_table(
        momentum_results['total_accuracy'],
        momentum_results['accuracy_dense_routed'],
        momentum_results['accuracy_sparse_routed'],
        avg_dense_spikes, avg_sparse_spikes,
        momentum_results['route_counts'],
        roc_auc, optimal_lzc_threshold,
        momentum_results['router_stats']
    )


if __name__ == "__main__":
    import torch.multiprocessing
    torch.multiprocessing.set_start_method('spawn', force=True)
    main()
