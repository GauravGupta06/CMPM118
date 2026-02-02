

import unittest
import torch
import numpy as np
from unittest.mock import MagicMock, patch, call
import sys
import os
import argparse

# Add the project root to the path so we can import momentum_router
sys.path.append(os.getcwd())

from momentum_router import (
    MomentumRouter,
    count_spikes_from_recording,
    compute_lzc_from_events,
    compute_xor_similarity,
    compute_cosine_similarity,
    compute_l1_similarity,
    evaluate_momentum_router,
    find_optimal_similarity_threshold,
    main
)

class TestMomentumRouterHelpers(unittest.TestCase):

    def test_count_spikes_from_recording_tensor(self):
        # Case 1: Tensor with binary values
        recording = {'layer1': torch.tensor([1.0, 0.0, 1.0, 1.0])}
        self.assertEqual(count_spikes_from_recording(recording), 3)

        # Case 2: Tensor with non-binary values (should be ignored based on function logic?)
        # The function checks: is_binary = len(unique_vals) <= 2 and torch.all((unique_vals >= 0) & (unique_vals <= 1))
        recording_non_binary = {'layer1': torch.tensor([0.5, 0.2])} 
        self.assertEqual(count_spikes_from_recording(recording_non_binary), 0)

        # Case 3: Empty tensor
        recording_empty = {'layer1': torch.tensor([])}
        self.assertEqual(count_spikes_from_recording(recording_empty), 0)

    def test_count_spikes_from_recording_dict(self):
        # Case: Dictionary with 'spikes' key
        recording = {'layer1': {'spikes': torch.tensor([1.0, 1.0])}}
        self.assertEqual(count_spikes_from_recording(recording), 2)

    @patch('momentum_router.lempel_ziv_complexity')
    def test_compute_lzc_from_events(self, mock_lzc):
        mock_lzc.return_value = 42.0
        events = torch.tensor([[1, 0], [0, 1]])
        
        score = compute_lzc_from_events(events)
        
        # Expected string: "1001" (flattened row-major: 1,0 then 0,1)
        # Note: implementation is: ''.join(map(str, spike_seq.tolist()))
        # [1, 0, 0, 1] -> "1001"
        mock_lzc.assert_called_with("1001")
        self.assertEqual(score, 42.0)

    def test_compute_xor_similarity(self):
        # Identical
        a = torch.tensor([1.0, 0.0, 1.0])
        b = torch.tensor([1.0, 0.0, 1.0])
        self.assertEqual(compute_xor_similarity(a, b), 0.0)

        # Completely different
        a = torch.tensor([0.0, 0.0, 0.0])
        b = torch.tensor([1.0, 1.0, 1.0])
        self.assertEqual(compute_xor_similarity(a, b), 1.0)

        # Half different
        a = torch.tensor([1.0, 1.0])
        b = torch.tensor([1.0, 0.0])
        self.assertEqual(compute_xor_similarity(a, b), 0.5)

        # Threshold logic check (values > 0.5 become 1)
        a = torch.tensor([0.6]) # becomes 1
        b = torch.tensor([0.4]) # becomes 0
        self.assertEqual(compute_xor_similarity(a, b), 1.0)

    def test_compute_cosine_similarity(self):
        # Identical vectors (normalized dist should be 0)
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([1.0, 0.0])
        self.assertAlmostEqual(compute_cosine_similarity(a, b), 0.0)

        # Orthogonal vectors (normalized dist should be 1)
        a = torch.tensor([1.0, 0.0])
        b = torch.tensor([0.0, 1.0])
        self.assertAlmostEqual(compute_cosine_similarity(a, b), 1.0)

        # Zero vector check
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([1.0, 1.0])
        self.assertEqual(compute_cosine_similarity(a, b), 1.0)

    def test_compute_l1_similarity(self):
        # Identical
        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([1.0, 2.0])
        self.assertEqual(compute_l1_similarity(a, b), 0.0)

        # Diff
        a = torch.tensor([0.0, 0.0])
        b = torch.tensor([1.0, 1.0])
        # l1 sum = 2. numel = 2. result = 1.0
        self.assertEqual(compute_l1_similarity(a, b), 1.0)


class TestMomentumRouterLogic(unittest.TestCase):

    def setUp(self):
        self.router = MomentumRouter(
            similarity_threshold=0.1,
            momentum_alpha=0.5, # Simple alpha for easy math
            drift_threshold=0.3,
            lzc_routing_threshold=10.0,
            similarity_method='xor'
        )

    @patch('momentum_router.compute_lzc_from_events')
    def test_initialization(self, mock_lzc):
        mock_lzc.return_value = 5.0 # Below threshold -> sparse
        sample = torch.tensor([1, 0, 1])

        result = self.router.update(sample)

        self.assertTrue(self.router.is_initialized)
        self.assertEqual(result['route'], 'sparse')
        self.assertEqual(result['reason'], 'initialization')
        self.assertEqual(result['lzc_value'], 5.0)
        self.assertFalse(result['used_cache'])
        
        # Verify reference is set
        self.assertTrue(torch.equal(self.router.reference_vector, sample))

    @patch('momentum_router.compute_lzc_from_events')
    def test_cache_hit_similarity(self, mock_lzc):
        # 1. Initialize
        mock_lzc.return_value = 5.0
        sample1 = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # 10 elements
        self.router.update(sample1)

        # 2. Similar sample (1 diff out of 10 = 0.1 diff). 
        # Threshold is 0.1 (inclusive check might be >=, let's verify code)
        # Code: if diff >= self.similarity_threshold: return True (recalc)
        # So 0.1 >= 0.1 -> Recalculate. 
        # Let's make it slightly smaller difference to force hit.
        # Actually, let's just make sample IDENTICAL for a guaranteed hit first.
        sample2 = sample1.clone()
        
        result = self.router.update(sample2)
        
        self.assertEqual(result['reason'], 'cache_hit')
        self.assertTrue(result['used_cache'])
        self.assertEqual(result['similarity'], 0.0)
        self.assertEqual(result['momentum'], 0.0) # 0.5 * 0 + 0.5 * 0

    @patch('momentum_router.compute_lzc_from_events')
    def test_cache_miss_similarity_exceeded(self, mock_lzc):
        # 1. Initialize
        mock_lzc.return_value = 5.0
        sample1 = torch.tensor([1, 0, 0, 0, 0])
        self.router.update(sample1)

        # 2. Different sample (2 diffs out of 5 = 0.4 > 0.1)
        sample2 = torch.tensor([0, 1, 0, 0, 0])
        mock_lzc.return_value = 15.0 # High complexity -> dense

        result = self.router.update(sample2)

        self.assertEqual(result['reason'], 'similarity_exceeded')
        self.assertFalse(result['used_cache'])
        self.assertEqual(result['route'], 'dense')
        # Reference should update
        self.assertTrue(torch.equal(self.router.reference_vector, sample2))
        # Momentum should reset
        self.assertEqual(self.router.momentum, 0.0)

    @patch('momentum_router.compute_lzc_from_events')
    def test_momentum_drift_trigger(self, mock_lzc):
        # Setup specific params
        self.router.similarity_threshold = 0.2
        self.router.momentum_alpha = 0.5 # slower decay, higher memory
        self.router.drift_threshold = 0.15 # strict drift check
        
        # 1. Initialize
        mock_lzc.return_value = 5.0
        sample = torch.zeros(10)
        self.router.update(sample) # Mom: 0

        # 2. Small change (diff=0.1 < 0.2 threshold) -> Cache Hit
        # Update momentum: m = 0.5*0 + 0.5*0.1 = 0.05
        sample[0] = 1 
        res = self.router.update(sample)
        self.assertEqual(res['reason'], 'cache_hit')
        self.assertAlmostEqual(self.router.momentum, 0.05)

        # 3. Another small change relative to REFERENCE (which is blended)
        # Wait, code says: reference_vector updates with blending on cache hit.
        # self.reference_vector = (0.9 * ref + 0.1 * new)
        # This makes exact calculation tricky, but the logic is:
        # If we keep having small errors, momentum builds up.
        
        # Let's force a sequence that builds momentum.
        # We need `diff` to be consistently non-zero.
        
        # Force accumulation
        self.router.momentum = 0.14 # Just below threshold
        
        # Next sample has diff 0.1 (below sim threshold 0.2)
        # New momentum = 0.8 * 0.14 + 0.2 * 0.1 = 0.112 + 0.02 = 0.132 (Decayed?)
        # Wait, formula is: momentum = alpha * momentum + (1 - alpha) * diff
        # If alpha=0.8, (1-alpha)=0.2.
        
        # Let's try to trigger it.
        self.router.momentum = 0.149
        # diff = 0.1.
        # new_mom = 0.8 * 0.149 + 0.2 * 0.1 = 0.1192 + 0.02 = 0.1392 (Still decaying towards diff)
        
        # To build momentum, diff must be > momentum?
        # If momentum tracks average error, and error suddenly jumps but stays under SimThreshold...
        
        # Let's try: alpha = 0.5. drift_thresh = 0.15. sim_thresh = 0.2.
        self.router.momentum_alpha = 0.5
        self.router.drift_threshold = 0.15
        self.router.similarity_threshold = 0.2
        
        # Current momentum 0.
        # Step 1: Diff 0.18 (Almost threshold). 
        # m = 0.5*0 + 0.5*0.18 = 0.09. (No trigger)
        
        # Step 2: Diff 0.18 again.
        # m = 0.5*0.09 + 0.5*0.18 = 0.045 + 0.09 = 0.135 (No trigger)
        
        # Step 3: Diff 0.18 again.
        # m = 0.5*0.135 + 0.5*0.18 = 0.0675 + 0.09 = 0.1575 (> 0.15 Trigger!)
        
        # We need to craft inputs to match this.
        # Init
        mock_lzc.return_value = 5.0
        base = torch.zeros(100)
        self.router.reset()
        self.router.update(base)
        
        # We need diff to be 0.18. So 18 elements different.
        # Note: Reference updates (blends) on cache hit. 
        # To keep diff constant, we need to move further away from the MOVING reference.
        
        # Actually, let's just manually inject the diff logic by mocking _compute_similarity
        # to ensure we isolate momentum logic from vector math.
        with patch.object(self.router, '_compute_similarity') as mock_sim:
            mock_sim.return_value = 0.18
            
            # Call 1
            res = self.router.update(base) # Cache hit, m=0.09
            self.assertEqual(res['reason'], 'cache_hit')
            
            # Call 2
            res = self.router.update(base) # Cache hit, m=0.135
            self.assertEqual(res['reason'], 'cache_hit')
            
            # Call 3
            res = self.router.update(base) # Drift trigger! m=0.1575
            self.assertEqual(res['reason'], 'momentum_drift')
            self.assertFalse(res['used_cache'])
            self.assertEqual(self.router.momentum, 0.0) # Should reset

    def test_routing_decision(self):
        # LZC < threshold -> sparse
        # LZC >= threshold -> dense
        self.router.lzc_routing_threshold = 100.0
        
        res = self.router.update(torch.zeros(10)) # Init
        # Mocking logic via patching is cleaner, but let's assume init works.
        # If we manually set cached_lzc
        self.router.cached_lzc = 50.0
        self.router.is_initialized = True
        
        # Force cache hit
        with patch.object(self.router, '_compute_similarity', return_value=0.0):
            res = self.router.update(torch.zeros(10))
            self.assertEqual(res['route'], 'sparse')
            
            self.router.cached_lzc = 150.0
            res = self.router.update(torch.zeros(10))
            self.assertEqual(res['route'], 'dense')


class TestMomentumRouterIntegration(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device('cpu')
        self.sparse_model = MagicMock()
        self.sparse_model.device = self.device
        self.dense_model = MagicMock()
        self.dense_model.device = self.device
        
        # Setup mock network outputs
        # Format: output, state, recording
        self.sparse_model.net.return_value = (
            torch.randn(1, 10, 2), # Output [Batch, Time, Classes] (Using 2 classes)
            None,
            {'spikes': torch.tensor([1., 1.])} # Recording
        )
        self.dense_model.net.return_value = (
            torch.randn(1, 10, 2),
            None,
            {'spikes': torch.tensor([1., 1., 1., 1.])}
        )
        
        self.router = MomentumRouter(
            similarity_threshold=0.1,
            lzc_routing_threshold=10.0
        )
        
        self.results = [
            {'lz_value': 5.0, 'true_complex': 0},  # sparse OK
            {'lz_value': 15.0, 'true_complex': 1}  # needs dense
        ]

    def test_evaluate_momentum_router(self):
        # Mock DataLoader
        # Batch: events, label
        loader = [
            (torch.zeros(1, 10), torch.tensor([0])),
            (torch.ones(1, 10), torch.tensor([1]))
        ]
        
        with patch('momentum_router.compute_lzc_from_events') as mock_lzc:
            mock_lzc.side_effect = [5.0, 15.0] # Return values for the 2 calls
            
            metrics = evaluate_momentum_router(
                loader,
                self.sparse_model,
                self.dense_model,
                self.router,
                self.results
            )
            
            # Check if metrics contain expected keys
            self.assertIn('total_accuracy', metrics)
            self.assertIn('route_counts', metrics)
            self.assertIn('router_stats', metrics)
            
            # Check route counts (1 sparse, 1 dense based on LZC values and threshold 10.0)
            self.assertEqual(metrics['route_counts']['sparse'], 1)
            self.assertEqual(metrics['route_counts']['dense'], 1)
            
            # Check if models were called correct number of times
            self.assertEqual(self.sparse_model.net.call_count, 1)
            self.assertEqual(self.dense_model.net.call_count, 1)

    def test_find_optimal_similarity_threshold(self):
        # Create dummy results with some crossing 6094.0 to avoid one-class ROC
        results = [
            {'true_complex': 0, 'lz_value': 5000.0, 'similarity_xor': 0.05}, 
            {'true_complex': 0, 'lz_value': 5000.0, 'similarity_xor': 0.05},
            {'true_complex': 1, 'lz_value': 7000.0, 'similarity_xor': 0.8}, 
            {'true_complex': 1, 'lz_value': 7000.0, 'similarity_xor': 0.05} 
        ]
        
        optimal, auc_score, _, _, _, _ = find_optimal_similarity_threshold(results, 'xor')
        
        self.assertIsInstance(optimal, float)
        self.assertIsInstance(auc_score, float)
    
    @patch('momentum_router.UCIHARDataset')
    @patch('momentum_router.UCIHARSNN_FC')
    @patch('momentum_router.BaseSNNModel')
    @patch('momentum_router.DataLoader')
    @patch('momentum_router.evaluate_models_on_dataset')
    @patch('momentum_router.analyze_similarity_distribution')
    @patch('momentum_router.threshold_sweep_and_roc')
    @patch('momentum_router.evaluate_momentum_router')
    @patch('momentum_router.plot_similarity_distribution')
    @patch('momentum_router.plot_momentum_routing_analysis')
    @patch('momentum_router.save_momentum_run_to_json')
    @patch('momentum_router.compare_with_baseline')
    @patch('momentum_router.print_latex_table')
    def test_main_execution(self, mock_print_table, mock_compare, mock_save, mock_plot_analysis, 
                           mock_plot_sim, mock_eval_router, mock_sweep, mock_analyze, mock_eval_models,
                           mock_loader, mock_base_model, mock_model_cls, mock_dataset):
        
        # Setup mocks
        mock_dataset.return_value.load_uci_har.return_value = (None, ['mock_data'])
        mock_base_model.load_hyperparams.return_value = {
            'input_size': 9, 'n_frames': 128, 'tau_mem': 0.01, 'tau_syn': 0.005,
            'spike_lam': 0.0, 'model_type': 'fc', 'num_classes': 6, 'dt': 1e-3,
            'threshold': 1.0, 'has_bias': False
        }
        mock_eval_models.return_value = []
        mock_analyze.return_value = {}
        mock_sweep.return_value = (0.5, 0.8, 10, 5) # threshold, auc, dense_spikes, sparse_spikes
        mock_eval_router.return_value = {
            'total_accuracy': 0.9, 'accuracy_dense_routed': 0.9, 'accuracy_sparse_routed': 0.9,
            'route_counts': {'sparse': 10, 'dense': 10},
            'router_stats': {'lzc_computations': 5, 'cache_hit_rate': 0.5, 'cache_hits': 5, 'momentum_triggers': 1, 'total_samples': 20},
            'routing_decisions': []
        }

        # Mock sys.argv
        test_args = [
            'momentum_router.py',
            '--sparse_model_path', 'mock_sparse.pth',
            '--dense_model_path', 'mock_dense.pth'
        ]
        
        with patch.object(sys, 'argv', test_args):
            main()
            
        # Verify main flow
        mock_dataset.assert_called()
        mock_model_cls.assert_called() # Called twice for sparse/dense
        mock_eval_models.assert_called()
        mock_eval_router.assert_called()
        mock_save.assert_called()

if __name__ == '__main__':
    unittest.main()
