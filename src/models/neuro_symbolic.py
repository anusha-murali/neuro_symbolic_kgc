"""
models/neuro_symbolic.py 

This module implements a neuro-symbolic knowledge graph completion model:
- Neural component: ComplEx (complex embeddings) for learning latent patterns
- Symbolic component: Rules handled externally via augment_data.py
- Includes evaluation with filtered metrics and self-adversarial loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# =============================================================================
# Mixed Precision Training Support
# =============================================================================
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
    Neuro-symbolic model for knowledge graph completion.
    
    Currently, symbolic rules are handled externally via data augmentation.
    This class focuses on the neural component with proper evaluation and loss functions.
    """
    
    def __init__(self, n_entities, n_relations, embedding_dim = 200, **kwargs):
        """
        Args:
            n_entities: Number of unique entities
            n_relations: Number of unique relations
            embedding_dim: Dimension of embeddings
            **kwargs: Additional arguments (for compatibility)
        """
        super().__init__()
        self.neural_model = ComplEx(n_entities, n_relations, embedding_dim)
        self.n_entities = n_entities
        self.n_relations = n_relations
        
    def set_rules(self, rules, relation2id):
        """
        Placeholder for rule integration.
        Rules are currently handled via augment_data.py which augments the training data
        with rule-derived triples rather than modifying scores directly.
        """
        pass  # Handled via augment_data.py
        
    @autocast()
    def forward(self, triples):
        """
        Forward pass through the model.
        
        Args:
            triples: Tensor of shape (batch_size, 3) with (head, relation, tail) indices
            
        Returns:
            Tuple of (combined_scores, neural_scores, symbolic_scores)
            For compatibility, we return neural scores for all three (symbolic placeholder)
        """
        neural_scores = self.neural_model(triples)
        empty_symbolic = torch.zeros_like(neural_scores)
        # Return neural scores for all outputs for compatibility with training loop
        return neural_scores, neural_scores, empty_symbolic
    
    @torch.no_grad()
    def evaluate_ranks(self, triples, filter_mask = None):
        """
        Evaluate ranks for triples with optional filtering.
        This is used during validation/testing to compute MRR and Hits@K.
        
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
        # unsqueeze(1) adds a dimension for gather, squeeze(-1) removes it after gathering
        target_scores = all_scores.gather(1, t.unsqueeze(1)).squeeze(-1)  # Shape: (batch_size,)
        
        # =====================================================================
        # WORST-CASE TIE BREAKING
        # =====================================================================
        # Standard KGC evaluation uses "worst-case" tie breaking:
        # If multiple entities have the same score as the true tail, we assume
        # the true tail is ranked after all of them (worst position).
        # This is achieved by using >= instead of > in the comparison.
        #
        # Example: Scores = [0.9, 0.9, 0.8, 0.7], true tail at index 0
        # Using >: rank = 1 (only strictly higher scores)
        # Using >=: rank = 2 (itself + the tie at index 1)
        # We use >= for worst-case evaluation.
        #
        # Also note: We DON'T add +1 because we want the count of entities
        # with score >= target, which gives the rank directly.
        ranks = (all_scores >= target_scores.unsqueeze(1)).sum(dim=1)  # Shape: (batch_size,)
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
        
        # =====================================================================
        # NUMERICALLY STABLE N3 REGULARIZATION
        # =====================================================================
        # Using absolute values before cubing is more stable than cubing first
        # (which could lead to very large numbers with negative signs)
        # Formula: penalty = (|h_re|³ + |h_im|³ + |r_re|³ + |r_im|³ + |t_re|³ + |t_im|³) / batch_size
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
        
        Args:
            pos_scores: Tensor of scores for positive triples
                       Shape can be (batch_size,) or (batch_size, 1)
            neg_scores: Tensor of scores for negative triples
                       Shape can be:
                       - (batch_size,) for 1 negative per positive
                       - (batch_size * n_neg,) for flattened multiple negatives
                       - (batch_size, n_neg) for already reshaped negatives
            
        Returns:
            loss: Scalar tensor with the combined loss
        """
        batch_size = pos_scores.shape[0]
        
        # =====================================================================
        # HANDLE MULTIPLE NEGATIVES (n_neg > 1)
        # =====================================================================
        # If neg_scores is 1D and longer than batch_size, it contains
        # n_neg * batch_size scores in a flattened format.
        # We need to reshape it to (batch_size, n_neg) for proper loss computation.
        if neg_scores.dim() == 1 and neg_scores.shape[0] > batch_size:
            n_neg = neg_scores.shape[0] // batch_size
            
            # IMPORTANT: The negative sampler returns negatives in a specific order:
            # [neg for batch item 0, neg for batch item 1, ...] * n_neg
            # So we need to reshape to (n_neg, batch_size) first, then transpose
            # to get (batch_size, n_neg) where each row corresponds to one positive
            neg_scores = neg_scores.view(n_neg, batch_size).transpose(0, 1)
            
        # =====================================================================
        # SELF-ADVERSARIAL WEIGHTING (for multiple negatives)
        # =====================================================================
        if neg_scores.dim() == 2:
            # Self-adversarial sampling: weight negatives by their score
            # Higher score = harder negative = higher weight
            alpha = 1.0  # Temperature for softmax (can be tuned)
            neg_weights = F.softmax(neg_scores * alpha, dim=-1).detach()  # Detach to avoid gradients
            
            # Positive loss: maximize log-sigmoid of positive scores
            pos_loss = -F.logsigmoid(pos_scores).mean()
            
            # Negative loss: weighted sum of log-sigmoid of negative scores
            # We want negative scores to be low, so we maximize log-sigmoid(-neg_scores)
            neg_loss_raw = -F.logsigmoid(-neg_scores)  # Shape: (batch_size, n_neg)
            neg_loss = (neg_weights * neg_loss_raw).sum(dim=-1).mean()
            
            # Combined loss
            loss = pos_loss + neg_loss
            
        # =====================================================================
        # SIMPLE CASE (1 negative per positive)
        # =====================================================================
        else:
            # Simple binary cross-entropy style loss
            pos_loss = -F.logsigmoid(pos_scores).mean()
            neg_loss = -F.logsigmoid(-neg_scores).mean()
            loss = pos_loss + neg_loss
            
        return loss
