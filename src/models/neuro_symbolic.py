"""
models/neuro_symbolic.py 

This module implements a neuro-symbolic knowledge graph completion model:
- Neural component: ComplEx (complex embeddings) for learning latent patterns
- Symbolic component: Rule confidence integration with learnable weights
- Includes evaluation with filtered metrics and self-adversarial loss

The symbolic component uses mined biological rules to boost scores for 
relations that follow known patterns (inverse, symmetric, chain rules).
Rules are converted to confidence scores per relation type, which are then
used to modulate the neural predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional

# =============================================================================
# Mixed Precision Training Support
# =============================================================================
# Mixed precision (FP16) training speeds up computation and reduces memory usage
# We provide fallbacks for environments without AMP support
try:
    from torch.cuda.amp import autocast
    MIXED_PRECISION = True
except ImportError:
    MIXED_PRECISION = False
    # Dummy autocast for environments without AMP
    class autocast:
        def __init__(self): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass


# =============================================================================
# Neural Component: ComplEx (Complex Embeddings)
# =============================================================================
class ComplEx(nn.Module):
    """
    ComplEx: Knowledge graph embeddings in complex space.
    
    Entities and relations are represented as complex vectors (real + imaginary parts).
    The score for a triple (h, r, t) is the real part of the complex inner product:
    score = Re( <h, r, conj(t)> )
    
    This captures symmetric and antisymmetric relations effectively.
    """
    
    def __init__(self, n_entities, n_relations, embedding_dim = 200):
        """
        Args:
            n_entities: Number of unique entities in the graph
            n_relations: Number of unique relation types
            embedding_dim: Dimension of embeddings (real and imaginary parts each)
        """
        super().__init__()
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embedding_dim = embedding_dim
        
        # Separate embeddings for real and imaginary parts
        # This gives us 2 * embedding_dim total dimensions per entity/relation
        self.entity_re = nn.Embedding(n_entities, embedding_dim)  # Real part
        self.entity_im = nn.Embedding(n_entities, embedding_dim)  # Imaginary part
        self.relation_re = nn.Embedding(n_relations, embedding_dim)
        self.relation_im = nn.Embedding(n_relations, embedding_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize embeddings with small normal distribution for stability."""
        nn.init.normal_(self.entity_re.weight, mean=0, std=0.1)
        nn.init.normal_(self.entity_im.weight, mean=0, std=0.1)
        nn.init.normal_(self.relation_re.weight, mean=0, std=0.1)
        nn.init.normal_(self.relation_im.weight, mean=0, std=0.1)
        
    def forward(self, triples):
        """
        Compute scores for a batch of triples.
        
        Args:
            triples: Tensor of shape (batch_size, 3) containing (head, relation, tail) indices
            
        Returns:
            scores: Tensor of shape (batch_size,) with scores for each triple
        """
        # Extract indices
        h_idx, r_idx, t_idx = triples[:, 0], triples[:, 1], triples[:, 2]
        
        # Get embeddings for each component
        h_re = self.entity_re(h_idx)    # Head real part
        h_im = self.entity_im(h_idx)    # Head imaginary part
        r_re = self.relation_re(r_idx)  # Relation real part
        r_im = self.relation_im(r_idx)  # Relation imaginary part
        t_re = self.entity_re(t_idx)    # Tail real part
        t_im = self.entity_im(t_idx)    # Tail imaginary part (NOTE: not conjugated here)
        
        # Complex multiplication: (h_re + i*h_im) * (r_re + i*r_im)
        # = (h_re*r_re - h_im*r_im) + i*(h_re*r_im + h_im*r_re)
        re_part = h_re * r_re - h_im * r_im
        im_part = h_re * r_im + h_im * r_re
        
        # Score = Re( <h * r, conj(t)> ) = Re( (h_re*r_re - h_im*r_im) * t_re + 
        #                                      (h_re*r_im + h_im*r_re) * t_im )
        # Note: We don't conjugate t_im here because we keep t_im as is
        scores = torch.sum(re_part * t_re + im_part * t_im, dim=1)
        return scores
    
    def predict_all_tails(self, heads, relations):
        """
        Compute scores for all possible tails given heads and relations.
        Used during evaluation for ranking all entities.
        
        Args:
            heads: Tensor of shape (batch_size,) with head indices
            relations: Tensor of shape (batch_size,) with relation indices
            
        Returns:
            scores: Tensor of shape (batch_size, n_entities) with scores for all tails
        """
        # Get embeddings for heads and relations
        h_re = self.entity_re(heads)
        h_im = self.entity_im(heads)
        r_re = self.relation_re(relations)
        r_im = self.relation_im(relations)
        
        # Get embeddings for all possible tails
        all_t_re = self.entity_re.weight  # Shape: (n_entities, embedding_dim)
        all_t_im = self.entity_im.weight  # Shape: (n_entities, embedding_dim)
        
        # Compute the complex product for each head-relation pair
        re_part = h_re * r_re - h_im * r_im  # Shape: (batch_size, embedding_dim)
        im_part = h_re * r_im + h_im * r_re  # Shape: (batch_size, embedding_dim)
        
        # Matrix multiplication gives scores for all tails at once
        # (batch_size, embedding_dim) @ (embedding_dim, n_entities) -> (batch_size, n_entities)
        scores = torch.mm(re_part, all_t_re.t()) + torch.mm(im_part, all_t_im.t())
        return scores


# =============================================================================
# Neuro-Symbolic Model (Neural + Symbolic components)
# =============================================================================
class NeuroSymbolicKGC(nn.Module):
    """
    Neuro-symbolic model for knowledge graph completion that integrates:
    
    1. Neural Component: ComplEx embeddings that learn latent patterns from graph structure
    2. Symbolic Component: Rule confidence scores derived from mined biological rules
       (inverse, symmetric, chain, and composition rules)
    
    The symbolic component provides a boost to scores for relations that have
    high-confidence rules. The weight of this boost (lambda_logic) is learnable,
    allowing the model to adapt the influence of symbolic knowledge during training.
    
    Rules are mined separately using BiologicalRuleMiner and then converted to
    relation-specific confidence scores via set_rules(). These confidences are
    stored in a buffer and used during forward passes to modulate predictions.
    
    The final score is: score = neural_score + lambda_logic * tanh(rule_confidence / temperature)
    """
    
    def __init__(self, n_entities, n_relations, embedding_dim=200, lambda_logic=0.1, temperature=1.0, **kwargs):
        """
        Args:
            n_entities: Number of unique entities
            n_relations: Number of unique relations
            embedding_dim: Dimension of embeddings
            lambda_logic: Weight for symbolic component (0 = no symbolic influence)
            temperature: Temperature for softening symbolic scores (higher = softer)
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__()
        self.neural_model = ComplEx(n_entities, n_relations, embedding_dim)
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.lambda_logic = lambda_logic
        self.temperature = temperature
        
        # Rule confidence matrix (stores confidence score per relation type)
        # Values are in [0,1] after normalization, indicating rule strength
        self.register_buffer('rule_confidence', torch.zeros(n_relations))
        self.has_rules = False
        
    def set_rules(self, rules, relation2id):
        """
        Set rules by populating rule confidence matrix.
        
        This function processes mined rules (from BiologicalRuleMiner) and
        converts them into confidence scores per relation. Multiple rules
        affecting the same relation are combined by taking the maximum confidence.
        
        Args:
            rules: List of mined rule dictionaries with 'type', 'body', 'head', 'confidence'
            relation2id: Mapping from relation names to integer IDs
        """
        if not rules or not relation2id:
            return
            
        # Reset confidence
        self.rule_confidence.zero_()
        rule_count = 0
        
        for rule in rules:
            try:
                if not isinstance(rule, dict):
                    continue
                    
                confidence = rule.get('confidence', 0.5)
                
                # Extract relations from the rule body
                if 'body' in rule and isinstance(rule['body'], list):
                    for elem in rule['body']:
                        if isinstance(elem, (list, tuple)) and len(elem) >= 2:
                            rel = elem[1]
                            if isinstance(rel, int) and rel < self.n_relations:
                                self.rule_confidence[rel] = max(
                                    self.rule_confidence[rel].item(), confidence
                                )
                                rule_count += 1
                
                # Extract relations from the rule head
                if 'head' in rule and isinstance(rule['head'], (list, tuple)) and len(rule['head']) >= 2:
                    rel = rule['head'][1]
                    if isinstance(rel, int) and rel < self.n_relations:
                        self.rule_confidence[rel] = max(
                            self.rule_confidence[rel].item(), confidence
                        )
                        rule_count += 1
                        
            except Exception:
                continue
        
        self.has_rules = rule_count > 0
        if self.has_rules:
            # Normalize confidences to [0,1] range
            self.rule_confidence = self.rule_confidence / self.rule_confidence.max()
            print(f"Loaded rules for {rule_count} relation occurrences")
        
    @autocast()
    def forward(self, triples):
        """
        Forward pass through the model.
        
        Computes both neural and symbolic scores, then combines them.
        
        Args:
            triples: Tensor of shape (batch_size, 3) with (head, relation, tail) indices
            
        Returns:
            Tuple of (combined_scores, neural_scores, symbolic_scores)
            - combined_scores: Final scores used for training/evaluation
            - neural_scores: Pure neural component scores (for analysis)
            - symbolic_scores: Pure symbolic component scores (for analysis)
        """
        # Neural scores from ComplEx
        neural_scores = self.neural_model(triples)
        
        # Symbolic scores (boost for relations with rules)
        symbolic_scores = torch.zeros_like(neural_scores)
        
        if self.has_rules:
            r = triples[:, 1]  # relation indices
            rule_conf = self.rule_confidence[r]
            # Apply temperature and tanh for smoothing
            # Tanh ensures symbolic scores are bounded in [-1, 1]
            symbolic_scores = torch.tanh(rule_conf / self.temperature)
        
        # Combined scores with lambda weighting
        combined_scores = neural_scores + self.lambda_logic * symbolic_scores
        
        return combined_scores, neural_scores, symbolic_scores
    
    @torch.no_grad()
    def evaluate_ranks(self, triples, filter_mask=None):
        """
        Evaluate ranks for triples with optional filtering.
        This is used during validation/testing to compute MRR and Hits@K.
        
        Implements the standard filtered evaluation protocol for KGC:
        - Scores all possible entities as tails
        - Masks out other known true triples (to avoid false negatives)
        - Computes rank of the true tail
        
        Args:
            triples: Tensor of shape (batch_size, 3) with (head, relation, tail) indices
            filter_mask: Optional boolean mask of shape (batch_size, n_entities)
                        True indicates entities that should be excluded from ranking
                        (other known true triples)
            
        Returns:
            ranks: Tensor of shape (batch_size,) with the rank of each true tail
                  (1 = best, higher = worse)
        """
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        
        # Get scores for all possible tails
        all_scores = self.neural_model.predict_all_tails(h, r)  # Shape: (batch_size, n_entities)
        
        # Apply filtering mask if provided (set filtered entities to very low score)
        if filter_mask is not None:
            all_scores = all_scores.masked_fill(filter_mask, -1e9)
        
        # Get the score of the true tail for each triple
        target_scores = all_scores.gather(1, t.unsqueeze(1)).squeeze(-1)
        
        # Worst-case tie breaking: count entities with score >= target
        # This gives the rank (1-based) directly without adding 1
        ranks = (all_scores >= target_scores.unsqueeze(1)).sum(dim=1)
        return ranks
        
    def get_regularization(self, triples):
        """
        Compute N3 regularization penalty for the embeddings.
        N3 (nuclear 3-norm) regularization helps prevent embeddings from growing too large
        and improves generalization.
        
        The penalty is the sum of cubed absolute values of all embeddings involved
        in the batch, normalized by batch size.
        
        Args:
            triples: Tensor of shape (batch_size, 3) with (head, relation, tail) indices
            
        Returns:
            penalty: Scalar tensor with regularization loss
        """
        h_idx, r_idx, t_idx = triples[:, 0], triples[:, 1], triples[:, 2]
        
        # Get embeddings for all components
        h_re = self.neural_model.entity_re(h_idx)
        h_im = self.neural_model.entity_im(h_idx)
        r_re = self.neural_model.relation_re(r_idx)
        r_im = self.neural_model.relation_im(r_idx)
        t_re = self.neural_model.entity_re(t_idx)
        t_im = self.neural_model.entity_im(t_idx)
        
        # N3 regularization: sum of cubed absolute values
        # Using absolute values before cubing is more stable
        penalty = (torch.abs(h_re)**3).sum() + (torch.abs(h_im)**3).sum() + \
                  (torch.abs(r_re)**3).sum() + (torch.abs(r_im)**3).sum() + \
                  (torch.abs(t_re)**3).sum() + (torch.abs(t_im)**3).sum()
                   
        return penalty / h_idx.shape[0]
    
    def compute_loss(self, pos_scores, neg_scores):
        """
        Compute self-adversarial negative sampling loss.
        
        This loss function:
        1. Encourages positive scores to be high (via log-sigmoid)
        2. Encourages negative scores to be low (via log-sigmoid of negative)
        3. Uses self-adversarial weighting to focus on hard negatives
           (negatives with higher scores get more weight)
        
        Args:
            pos_scores: Tensor of scores for positive triples
            neg_scores: Tensor of scores for negative triples
                       Can be (batch_size,) for 1 negative or
                       (batch_size * n_neg,) for multiple negatives
            
        Returns:
            loss: Scalar tensor with the combined loss
        """
        batch_size = pos_scores.shape[0]
        
        # Handle multiple negatives: reshape from flattened to (batch_size, n_neg)
        if neg_scores.dim() == 1 and neg_scores.shape[0] > batch_size:
            n_neg = neg_scores.shape[0] // batch_size
            # Important: negatives are ordered as [neg1 for all positives, neg2 for all positives, ...]
            # So reshape to (n_neg, batch_size) then transpose
            neg_scores = neg_scores.view(n_neg, batch_size).transpose(0, 1)
            
        # Self-adversarial weighting for multiple negatives
        if neg_scores.dim() == 2:
            alpha = 1.0  # Temperature for softmax (can be tuned)
            # Compute weights based on negative scores (higher score = higher weight)
            neg_weights = F.softmax(neg_scores * alpha, dim=-1).detach()  # Detach to avoid gradients
            
            # Positive loss: maximize log-sigmoid of positive scores
            pos_loss = -F.logsigmoid(pos_scores).mean()
            
            # Negative loss: weighted sum of log-sigmoid of negative scores
            # We want negative scores to be low, so we maximize log-sigmoid(-neg_scores)
            neg_loss_raw = -F.logsigmoid(-neg_scores)  # Shape: (batch_size, n_neg)
            neg_loss = (neg_weights * neg_loss_raw).sum(dim=-1).mean()
            
            # Combined loss
            loss = pos_loss + neg_loss
            
        # Simple case (1 negative per positive)
        else:
            pos_loss = -F.logsigmoid(pos_scores).mean()
            neg_loss = -F.logsigmoid(-neg_scores).mean()
            loss = pos_loss + neg_loss
            
        return loss
