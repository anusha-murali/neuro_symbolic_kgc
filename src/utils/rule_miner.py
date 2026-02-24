"""
utils/rule_miner.py

This module mines symbolic rules from the knowledge graph structure.
Rules are patterns that hold frequently in the data, such as:
- Inverse rules: If A interacts with B, then B interacts with A
- Symmetric rules: If A interacts with B, then B interacts with A (same relation)
- Chain rules: If A regulates B and B regulates C, then A regulates C
- Composition rules: If A interacts with B and A interacts with C, then B and C are related

These rules can be used to:
1. Augment the training data with rule-derived triples
2. Provide symbolic guidance to the neural model
3. Improve interpretability of predictions
"""

import numpy as np
from collections import defaultdict, Counter
from itertools import combinations


class BiologicalRuleMiner:
    """
    Mine biological rules from the knowledge graph.
    
    This class implements algorithms for discovering frequent patterns
    in the graph structure. It builds indices for efficient lookup and
    then mines different types of rules based on co-occurrence patterns.
    
    The mined rules include confidence scores and support counts,
    which can be used to filter low-quality rules.
    """
    
    def __init__(self, train_triples, n_relations, min_support=5, min_confidence=0.1):
        """
        Args:
            train_triples: numpy array of shape (n_triples, 3) with (head, relation, tail)
            n_relations: Number of unique relation types
            min_support: Minimum number of occurrences for a rule to be considered
            min_confidence: Minimum confidence threshold for rules
        """
        self.train_triples = train_triples
        self.n_relations = n_relations
        self.min_support = min_support
        self.min_confidence = min_confidence
        
        # Build indices for faster mining
        self._build_indices()
        
    def _build_indices(self):
        """
        Build indices for fast rule mining.
        
        Creates three indices:
        - triples_by_head: For each head entity, list of (relation, tail) pairs
        - triples_by_tail: For each tail entity, list of (head, relation) pairs
        - triples_by_pair: For each (head, tail) pair, set of relations
        
        These indices enable O(1) lookup of relevant triples during mining.
        """
        print("Building indices for rule mining...")
        
        # Index by head and tail
        self.triples_by_head = defaultdict(list)
        self.triples_by_tail = defaultdict(list)
        self.triples_by_pair = defaultdict(set)
        
        for h, r, t in self.train_triples:
            self.triples_by_head[h].append((r, t))
            self.triples_by_tail[t].append((h, r))
            self.triples_by_pair[(h, t)].add(r)
        
        print(f"  Found {len(self.triples_by_head)} unique heads")
        print(f"  Found {len(self.triples_by_tail)} unique tails")
        print(f"  Found {len(self.triples_by_pair)} unique (head, tail) pairs")
        
    def mine_inverse_rules(self):
        """
        Mine inverse rules: r1(X,Y) => r2(Y,X)
        
        Example: If protein A phosphorylates protein B, then protein B is
        phosphorylated by protein A. The relations might be different
        (e.g., 'phosphorylates' and 'phosphorylated_by').
        
        Returns:
            List of rule dictionaries with type 'inverse'
        """
        print("\nMining inverse rules...")
        
        # Count co-occurrences of (r1, r2) where (h,r1,t) and (t,r2,h) both exist
        inv_counts = defaultdict(lambda: defaultdict(int))
        
        # For each (head, tail) pair and its relations
        for (h, t), rels in self.triples_by_pair.items():
            for r1 in rels:
                # Check if the reverse pair (t, h) exists
                if (t, h) in self.triples_by_pair:
                    # For each relation in the reverse pair
                    for r2 in self.triples_by_pair[(t, h)]:
                        inv_counts[r1][r2] += 1
        
        # Convert counts to rules with confidence
        rules = []
        for r1 in inv_counts:
            for r2, count in inv_counts[r1].items():
                if count >= self.min_support:
                    # Calculate confidence: proportion of r1 occurrences that have r2 as inverse
                    # Note: This is a simplified confidence calculation
                    total_r1 = len(self.triples_by_pair.get((h, t), set()))
                    confidence = count / max(total_r1, 1)
                    
                    if confidence >= self.min_confidence:
                        rules.append({
                            'type': 'inverse',
                            'body': [('?X', r1, '?Y')],  # ?X and ?Y are variables
                            'head': ('?Y', r2, '?X'),
                            'confidence': float(confidence),
                            'support': int(count),
                            'source': 'mined'
                        })
        
        print(f"  Found {len(rules)} inverse rules")
        return rules
    
    def mine_symmetric_rules(self):
        """
        Mine symmetric rules: r(X,Y) => r(Y,X)
        
        Example: If protein A interacts with protein B, then protein B
        interacts with protein A (same relation).
        
        Returns:
            List of rule dictionaries with type 'symmetric'
        """
        print("\nMining symmetric rules...")
        
        # Count symmetric occurrences
        sym_counts = defaultdict(int)
        
        # For each (head, tail) pair
        for (h, t), rels in self.triples_by_pair.items():
            # Check if the reverse pair exists
            if (t, h) in self.triples_by_pair:
                # For each relation that appears in both directions
                for r in rels:
                    if r in self.triples_by_pair[(t, h)]:
                        sym_counts[r] += 1
        
        # Convert to rules
        rules = []
        for r, count in sym_counts.items():
            if count >= self.min_support:
                # Calculate confidence
                total = len(self.triples_by_pair.get((h, t), set()))
                confidence = count / max(total, 1)
                
                if confidence >= self.min_confidence:
                    rules.append({
                        'type': 'symmetric',
                        'body': [('?X', r, '?Y')],
                        'head': ('?Y', r, '?X'),
                        'confidence': float(confidence),
                        'support': int(count),
                        'source': 'mined'
                    })
        
        print(f"  Found {len(rules)} symmetric rules")
        return rules
    
    def mine_chain_rules(self, max_length=2):
        """
        Mine chain rules: r1(X,Y) & r2(Y,Z) => r3(X,Z)
        
        Example: If gene A regulates protein B and protein B participates in
        pathway C, then gene A is associated with pathway C.
        
        Args:
            max_length: Maximum length of chains (currently only 2 supported)
            
        Returns:
            List of rule dictionaries with type 'chain'
        """
        print(f"\nMining chain rules (max length={max_length})...")
        
        # Count chains of length 2
        chain_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # Find entities that serve as middle nodes in chains
        # These entities must have both incoming and outgoing edges
        for mid_entity in set(self.triples_by_head.keys()) & set(self.triples_by_tail.keys()):
            # Get outgoing edges from middle entity (r1, t)
            outgoing = self.triples_by_head[mid_entity]
            # Get incoming edges to middle entity (h, r2)
            incoming = self.triples_by_tail[mid_entity]
            
            # For each outgoing and incoming pair
            for r1, t in outgoing:
                for h, r2 in incoming:
                    # Check if there's a direct edge from h to t
                    if (h, t) in self.triples_by_pair:
                        # For each relation that connects h to t
                        for r3 in self.triples_by_pair[(h, t)]:
                            chain_counts[r1][r2][r3] += 1
        
        # Convert counts to rules
        rules = []
        for r1 in chain_counts:
            for r2 in chain_counts[r1]:
                for r3, count in chain_counts[r1][r2].items():
                    if count >= self.min_support:
                        # Approximate confidence (simplified)
                        confidence = count / (count + 1)
                        
                        if confidence >= self.min_confidence:
                            rules.append({
                                'type': 'chain',
                                'body': [('?X', r1, '?Y'), ('?Y', r2, '?Z')],
                                'head': ('?X', r3, '?Z'),
                                'confidence': float(confidence),
                                'support': int(count),
                                'source': 'mined'
                            })
        
        print(f"  Found {len(rules)} chain rules")
        return rules
    
    def mine_composition_rules(self):
        """
        Mine composition rules: r1(X,Y) & r2(X,Z) => r3(Y,Z)
        
        Example: If protein A interacts with protein B and protein A interacts with
        protein C, then protein B and protein C are functionally associated.
        
        Returns:
            List of rule dictionaries with type 'composition'
        """
        print("\nMining composition rules...")
        
        # Count compositions
        comp_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # For performance, we limit the number of entities to check
        # In a full implementation, you might want to sample or use all entities
        entities = list(set(self.triples_by_head.keys()))
        for h in entities[:1000]:  # Limit for performance
            if h in self.triples_by_head:
                # Get all outgoing edges from this head
                relations_tails = self.triples_by_head[h]
                
                # Check pairs of tails from the same head
                for i, (r1, t1) in enumerate(relations_tails):
                    for j, (r2, t2) in enumerate(relations_tails[i+1:], i+1):
                        # Check if there's a relation between t1 and t2
                        if (t1, t2) in self.triples_by_pair:
                            for r3 in self.triples_by_pair[(t1, t2)]:
                                comp_counts[r1][r2][r3] += 1
        
        # Convert counts to rules
        rules = []
        for r1 in comp_counts:
            for r2 in comp_counts[r1]:
                for r3, count in comp_counts[r1][r2].items():
                    if count >= self.min_support:
                        # Approximate confidence
                        confidence = count / (count + 1)
                        
                        if confidence >= self.min_confidence:
                            rules.append({
                                'type': 'composition',
                                'body': [('?X', r1, '?Y'), ('?X', r2, '?Z')],
                                'head': ('?Y', r3, '?Z'),
                                'confidence': float(confidence),
                                'support': int(count),
                                'source': 'mined'
                            })
        
        print(f"  Found {len(rules)} composition rules")
        return rules
    
    def mine_all_rules(self):
        """
        Mine all types of rules and combine them.
        
        This is the main entry point for rule mining. It runs all mining
        algorithms and returns a combined list of rules sorted by confidence.
        
        Returns:
            List of all mined rules (dictionaries with type, body, head, confidence, support)
        """
        print("\n" + "=" * 60)
        print("Mining all rules from knowledge graph...")
        print("=" * 60)
        
        all_rules = []
        
        # Mine each type of rule
        all_rules.extend(self.mine_inverse_rules())
        all_rules.extend(self.mine_symmetric_rules())
        all_rules.extend(self.mine_chain_rules())
        all_rules.extend(self.mine_composition_rules())
        
        # Sort by confidence (highest first)
        all_rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\nTotal rules mined: {len(all_rules)}")
        
        # Print statistics by rule type
        rule_types = {}
        for rule in all_rules:
            rt = rule['type']
            rule_types[rt] = rule_types.get(rt, 0) + 1
        
        print("\nMined rule type distribution:")
        for rt, count in sorted(rule_types.items()):
            print(f"  {rt}: {count}")
        
        return all_rules
