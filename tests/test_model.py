"""
tests/test_model.py
Unit tests for the neuro-symbolic model implementation.
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.neuro_symbolic import NeuroSymbolicKGC, ComplEx


class TestComplEx(unittest.TestCase):
    """Test cases for ComplEx neural model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_entities = 100
        self.n_relations = 20
        self.embedding_dim = 32
        self.batch_size = 16
        
        # Create model
        self.model = ComplEx(
            n_entities=self.n_entities,
            n_relations=self.n_relations,
            embedding_dim=self.embedding_dim
        )
        
        # Create random triples
        self.triples = torch.randint(
            0, 
            self.n_entities, 
            (self.batch_size, 3), 
            dtype=torch.long
        )
        # Ensure relations are within bounds
        self.triples[:, 1] = torch.randint(0, self.n_relations, (self.batch_size,))
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.n_entities, self.n_entities)
        self.assertEqual(self.model.n_relations, self.n_relations)
        self.assertEqual(self.model.embedding_dim, self.embedding_dim)
        
        # Check embedding dimensions
        self.assertEqual(self.model.entity_re.weight.shape, (self.n_entities, self.embedding_dim))
        self.assertEqual(self.model.entity_im.weight.shape, (self.n_entities, self.embedding_dim))
        self.assertEqual(self.model.relation_re.weight.shape, (self.n_relations, self.embedding_dim))
        self.assertEqual(self.model.relation_im.weight.shape, (self.n_relations, self.embedding_dim))
        
    def test_forward_shape(self):
        """Test forward pass output shape."""
        scores = self.model(self.triples)
        self.assertEqual(scores.shape, (self.batch_size,))
        
    def test_forward_differentiable(self):
        """Test that forward pass is differentiable."""
        scores = self.model(self.triples)
        loss = scores.sum()
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(self.model.entity_re.weight.grad)
        
    def test_predict_all_tails_shape(self):
        """Test predict_all_tails output shape."""
        heads = self.triples[:, 0]
        relations = self.triples[:, 1]
        
        scores = self.model.predict_all_tails(heads, relations)
        self.assertEqual(scores.shape, (self.batch_size, self.n_entities))
        
    def test_predict_all_tails_values(self):
        """Test that predict_all_tails produces reasonable values."""
        heads = torch.tensor([0, 1])
        relations = torch.tensor([0, 1])
        
        scores = self.model.predict_all_tails(heads, relations)
        
        # Scores should be finite
        self.assertTrue(torch.isfinite(scores).all())
        
    def test_embedding_device(self):
        """Test that embeddings are on correct device."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ComplEx(self.n_entities, self.n_relations, self.embedding_dim).to(device)
        
        self.assertEqual(model.entity_re.weight.device.type, device.type)


class TestNeuroSymbolicKGC(unittest.TestCase):
    """Test cases for NeuroSymbolicKGC model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_entities = 100
        self.n_relations = 20
        self.embedding_dim = 32
        self.batch_size = 16
        self.lambda_logic = 0.1
        self.temperature = 1.0
        self.margin = 5.0
        
        # Create model
        self.model = NeuroSymbolicKGC(
            n_entities=self.n_entities,
            n_relations=self.n_relations,
            embedding_dim=self.embedding_dim,
            lambda_logic=self.lambda_logic,
            temperature=self.temperature,
            margin=self.margin
        )
        
        # Create random triples
        self.triples = torch.randint(
            0, 
            self.n_entities, 
            (self.batch_size, 3), 
            dtype=torch.long
        )
        self.triples[:, 1] = torch.randint(0, self.n_relations, (self.batch_size,))
        
        # Create random negative scores
        self.n_neg = 10
        self.pos_scores = torch.randn(self.batch_size)
        self.neg_scores = torch.randn(self.batch_size * self.n_neg)
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.n_entities, self.n_entities)
        self.assertEqual(self.model.n_relations, self.n_relations)
        self.assertEqual(self.model.lambda_logic, self.lambda_logic)
        self.assertEqual(self.model.temperature, self.temperature)
        self.assertEqual(self.model.margin, self.margin)
        
        # Check rule confidence initialization
        self.assertEqual(self.model.rule_confidence.shape, (self.n_relations,))
        self.assertTrue(torch.all(self.model.rule_confidence == 0))
        self.assertFalse(self.model.has_rules)
        
    def test_forward_shape(self):
        """Test forward pass output shapes."""
        combined, neural, symbolic = self.model(self.triples)
        
        self.assertEqual(combined.shape, (self.batch_size,))
        self.assertEqual(neural.shape, (self.batch_size,))
        self.assertEqual(symbolic.shape, (self.batch_size,))
        
    def test_forward_no_rules(self):
        """Test forward pass when no rules are set."""
        combined, neural, symbolic = self.model(self.triples)
        
        # Without rules, symbolic scores should be zero
        self.assertTrue(torch.all(symbolic == 0))
        self.assertTrue(torch.all(combined == neural))
        
    def test_set_rules(self):
        """Test setting rules."""
        # Create mock rules
        rules = [
            {
                'type': 'inverse',
                'body': [('?X', 0, '?Y')],
                'head': ('?Y', 1, '?X'),
                'confidence': 0.8,
                'support': 100
            },
            {
                'type': 'symmetric',
                'body': [('?X', 2, '?Y')],
                'head': ('?Y', 2, '?X'),
                'confidence': 0.9,
                'support': 150
            }
        ]
        
        # Create relation2id mapping
        relation2id = {str(i): i for i in range(self.n_relations)}
        
        self.model.set_rules(rules, relation2id)
        
        # Check that rules were set
        self.assertTrue(self.model.has_rules)
        self.assertFalse(torch.all(self.model.rule_confidence == 0))
        
    def test_forward_with_rules(self):
        """Test forward pass with rules set."""
        # Set some rules
        relation2id = {str(i): i for i in range(self.n_relations)}
        rules = [
            {
                'type': 'inverse',
                'body': [('?X', 0, '?Y')],
                'head': ('?Y', 1, '?X'),
                'confidence': 0.8,
                'support': 100
            }
        ]
        self.model.set_rules(rules, relation2id)
        
        # Create triples with relation that has rules
        triples = torch.tensor([[0, 0, 1], [1, 1, 2], [2, 2, 3]])
        combined, neural, symbolic = self.model(triples)
        
        # First triple (relation 0) should have non-zero symbolic score
        self.assertFalse(symbolic[0] == 0)
        # Second triple (relation 1) might have rules from head
        # Third triple (relation 2) should have zero symbolic score
        self.assertTrue(symbolic[2] == 0)
        
    def test_evaluate_ranks_shape(self):
        """Test evaluate_ranks output shape."""
        ranks = self.model.evaluate_ranks(self.triples)
        self.assertEqual(ranks.shape, (self.batch_size,))
        
    def test_evaluate_ranks_values(self):
        """Test that ranks are within valid range."""
        ranks = self.model.evaluate_ranks(self.triples)
        
        # Ranks should be between 1 and n_entities
        self.assertTrue(torch.all(ranks >= 1))
        self.assertTrue(torch.all(ranks <= self.n_entities))
        
    def test_evaluate_ranks_with_filter(self):
        """Test evaluate_ranks with filter mask."""
        # Create filter mask (mask out some entities)
        filter_mask = torch.zeros(self.batch_size, self.n_entities, dtype=torch.bool)
        for i in range(self.batch_size):
            # Mask out first 10 entities for each batch
            filter_mask[i, :10] = True
            # Unmask the true tail
            filter_mask[i, self.triples[i, 2]] = False
            
        ranks = self.model.evaluate_ranks(self.triples, filter_mask)
        
        # Ranks should be valid
        self.assertTrue(torch.all(ranks >= 1))
        self.assertTrue(torch.all(ranks <= self.n_entities))
        
    def test_compute_loss_single_negative(self):
        """Test loss computation with single negative per positive."""
        pos_scores = torch.randn(self.batch_size)
        neg_scores = torch.randn(self.batch_size)
        
        loss = self.model.compute_loss(pos_scores, neg_scores)
        
        # Loss should be a scalar
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        
    def test_compute_loss_multiple_negatives(self):
        """Test loss computation with multiple negatives per positive."""
        n_neg = 10
        pos_scores = torch.randn(self.batch_size)
        neg_scores = torch.randn(self.batch_size * n_neg)
        
        loss = self.model.compute_loss(pos_scores, neg_scores)
        
        # Loss should be a scalar
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        
    def test_compute_loss_reshaped_negatives(self):
        """Test loss computation with already reshaped negatives."""
        n_neg = 10
        pos_scores = torch.randn(self.batch_size)
        neg_scores = torch.randn(self.batch_size, n_neg)
        
        loss = self.model.compute_loss(pos_scores, neg_scores)
        
        # Loss should be a scalar
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        
    def test_compute_loss_differentiable(self):
        """Test that loss is differentiable."""
        pos_scores = torch.randn(self.batch_size, requires_grad=True)
        neg_scores = torch.randn(self.batch_size * 5)
        
        loss = self.model.compute_loss(pos_scores, neg_scores)
        loss.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(pos_scores.grad)
        
    def test_get_regularization(self):
        """Test regularization computation."""
        reg = self.model.get_regularization(self.triples)
        
        # Regularization should be a positive scalar
        self.assertEqual(reg.dim(), 0)
        self.assertTrue(reg > 0)
        self.assertTrue(torch.isfinite(reg))
        
    def test_get_regularization_differentiable(self):
        """Test that regularization is differentiable."""
        reg = self.model.get_regularization(self.triples)
        reg.backward()
        
        # Check that gradients are computed
        self.assertIsNotNone(self.model.neural_model.entity_re.weight.grad)
        
    def test_device_movement(self):
        """Test moving model between devices."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model = self.model.to(device)
            
            # Check that all parameters are on CUDA
            for param in model.parameters():
                self.assertEqual(param.device.type, 'cuda')
                
            # Check buffers
            self.assertEqual(model.rule_confidence.device.type, 'cuda')


class TestNeuroSymbolicKGCIntegration(unittest.TestCase):
    """Integration tests for NeuroSymbolicKGC."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.n_entities = 50
        self.n_relations = 10
        self.embedding_dim = 16
        self.batch_size = 8
        
        self.model = NeuroSymbolicKGC(
            n_entities=self.n_entities,
            n_relations=self.n_relations,
            embedding_dim=self.embedding_dim
        )
        
    def test_training_step(self):
        """Test a single training step."""
        # Create batch
        batch = torch.randint(0, self.n_entities, (self.batch_size, 3))
        batch[:, 1] = torch.randint(0, self.n_relations, (self.batch_size,))
        
        # Create negative batch (simple corruption)
        neg_batch = batch.clone()
        neg_batch[:, 0] = torch.randint(0, self.n_entities, (self.batch_size,))
        
        # Forward pass
        pos_scores, _, _ = self.model(batch)
        neg_scores, _, _ = self.model(neg_batch)
        
        # Compute loss
        loss = self.model.compute_loss(pos_scores, neg_scores)
        
        # Loss should be finite
        self.assertTrue(torch.isfinite(loss))
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"No gradient for {name}")
            
    def test_evaluation_step(self):
        """Test a single evaluation step."""
        batch = torch.randint(0, self.n_entities, (self.batch_size, 3))
        batch[:, 1] = torch.randint(0, self.n_relations, (self.batch_size,))
        
        with torch.no_grad():
            ranks = self.model.evaluate_ranks(batch)
            
        # Compute MRR
        mrr = (1.0 / ranks.float()).mean()
        
        # MRR should be between 0 and 1
        self.assertTrue(0 <= mrr <= 1)


if __name__ == '__main__':
    unittest.main()

