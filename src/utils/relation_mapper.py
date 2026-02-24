"""
utils/relation_mapper.py

This module provides utilities for mapping relation names from mined rules
to the actual relation names present in the BioKG dataset. This is necessary
because rule mining may produce relations with generic names (e.g., 'INTERACTS_WITH')
while the dataset uses specific naming conventions (e.g., 'PPI').

The mapper uses multiple strategies to find matches:
1. Direct lookup in a predefined mapping dictionary
2. Case-insensitive matching
3. Normalized matching (removing underscores, prefixes)
4. Partial word matching
"""

import re
import numpy as np
from collections import defaultdict


# =============================================================================
# Direct Mapping Dictionary
# =============================================================================
def get_direct_mapping():
    """
    Get a dictionary that maps generic relation names to actual BioKG relation names.
    
    This mapping is built based on domain knowledge of biological relations.
    For example, 'INTERACTS_WITH' maps to 'PPI' (Protein-Protein Interaction)
    because these represent the same underlying biological concept.
    
    Returns:
        dict: Mapping from generic relation names to BioKG relation names
    """
    return {
        # =====================================================================
        # Protein Interactions
        # =====================================================================
        'INTERACTS_WITH': 'PPI',           # Generic protein interaction
        'PPI': 'PPI',                       # Direct match
        'protein_interaction': 'PPI',       # Alternative phrasing
        
        # =====================================================================
        # Drug-Protein Interactions
        # =====================================================================
        'DRUG_TARGET': 'DRUG_TARGET',       # Direct match
        'TARGETS_PROTEIN': 'DRUG_TARGET',   # Drug targets a protein
        'targeted_by': 'DRUG_TARGET',       # Protein targeted by drug
        'drug_target': 'DRUG_TARGET',       # Common phrasing
        
        # =====================================================================
        # Drug-Related Relations
        # =====================================================================
        # Drug metabolism (enzymes that process drugs)
        'DRUG_ENZYME': 'DRUG_ENZYME',
        'METABOLIZES': 'DRUG_ENZYME',       # Drug metabolizes something
        'METABOLIZED_BY': 'DRUG_ENZYME',    # Drug metabolized by enzyme
        
        # Drug side effects
        'DRUG_SIDEEFFECT_ASSOCIATION': 'DRUG_SIDEEFFECT_ASSOCIATION',
        'HAS_SYMPTOM': 'DRUG_SIDEEFFECT_ASSOCIATION',  # Drug causes symptom
        'SIDE_EFFECT_SIMILAR': 'DRUG_SIDEEFFECT_ASSOCIATION',  # Similar side effects
        
        # Drug indications (what diseases a drug treats)
        'DRUG_INDICATION_ASSOCIATION': 'DRUG_INDICATION_ASSOCIATION',
        'INDICATION': 'DRUG_INDICATION_ASSOCIATION',
        
        # Drug carriers and transporters
        'DRUG_CARRIER': 'DRUG_CARRIER',
        'DRUG_TRANSPORTER': 'DRUG_TRANSPORTER',
        
        # =====================================================================
        # Disease-Related Relations
        # =====================================================================
        # Protein-disease associations
        'PROTEIN_DISEASE_ASSOCIATION': 'PROTEIN_DISEASE_ASSOCIATION',
        'DISEASE_ASSOCIATION': 'PROTEIN_DISEASE_ASSOCIATION',
        'ASSOCIATED_WITH_DISEASE': 'PROTEIN_DISEASE_ASSOCIATION',
        'MUTATED_IN': 'PROTEIN_DISEASE_ASSOCIATION',  # Protein mutated in disease
        
        # Disease-pathway associations
        'DISEASE_PATHWAY_ASSOCIATION': 'DISEASE_PATHWAY_ASSOCIATION',
        'PATHWAY_IN_DISEASE': 'DISEASE_PATHWAY_ASSOCIATION',
        
        # =====================================================================
        # Pathway-Related Relations
        # =====================================================================
        # Protein-pathway associations
        'PROTEIN_PATHWAY_ASSOCIATION': 'PROTEIN_PATHWAY_ASSOCIATION',
        'PARTICIPATES_IN': 'PROTEIN_PATHWAY_ASSOCIATION',  # Protein in pathway
        'INVOLVES_PROTEIN': 'PROTEIN_PATHWAY_ASSOCIATION',  # Pathway involves protein
        
        # Pathway hierarchy
        'HAS_PARENT_PATHWAY': 'HAS_PARENT_PATHWAY',
        'SUB_PATHWAY': 'HAS_PARENT_PATHWAY',
        'HAS_SUBPATHWAY': 'HAS_PARENT_PATHWAY',
        
        # Pathway categories
        'PATHWAY_CATEGORY': 'PATHWAY_CATEGORY',
        'CATEGORY': 'PATHWAY_CATEGORY',
        
        # =====================================================================
        # Gene Ontology (GO) Terms
        # =====================================================================
        'GO_BP': 'GO_BP',  # Biological Process
        'GO_CC': 'GO_CC',  # Cellular Component
        'GO_MF': 'GO_MF',  # Molecular Function
        'HAS_FUNCTION': 'GO_BP',      # Protein has biological function
        'HAS_PHENOTYPE': 'GO_CC',      # Protein has cellular location
        
        # =====================================================================
        # Protein Complex Relations
        # =====================================================================
        'MEMBER_OF_COMPLEX': 'MEMBER_OF_COMPLEX',
        'COMPLEX_MEMBER': 'MEMBER_OF_COMPLEX',
        'HAS_MEMBER': 'MEMBER_OF_COMPLEX',  # Complex has protein member
        
        # Complex-pathway relations
        'COMPLEX_IN_PATHWAY': 'COMPLEX_IN_PATHWAY',
        'COMPLEX_TOP_LEVEL_PATHWAY': 'COMPLEX_TOP_LEVEL_PATHWAY',
        
        # =====================================================================
        # Tissue Expression
        # =====================================================================
        'PROTEIN_EXPRESSED_IN': 'PROTEIN_EXPRESSED_IN',
        'EXPRESSED_IN': 'PROTEIN_EXPRESSED_IN',
        'PART_OF_TISSUE': 'PART_OF_TISSUE',
        
        # =====================================================================
        # Protein Domains and Sites
        # =====================================================================
        'DOMAIN': 'DOMAIN',                    # Protein domain
        'ACTIVE_SITE': 'ACTIVE_SITE',          # Active site
        'BINDING_SITE': 'BINDING_SITE',        # Binding site
        'CONSERVED_SITE': 'CONSERVED_SITE',    # Conserved site
        'PTM': 'PTM',                           # Post-translational modification
        'REPEAT': 'REPEAT',                     # Sequence repeat
        
        # =====================================================================
        # Protein Families and Homology
        # =====================================================================
        'FAMILY': 'FAMILY',                    # Protein family
        'HOMOLOGOUS_SUPERFAMILY': 'HOMOLOGOUS_SUPERFAMILY',
        'SIMILAR_TO': 'HOMOLOGOUS_SUPERFAMILY',  # Protein similar to another
        
        # =====================================================================
        # Drug-Drug Interactions
        # =====================================================================
        'DDI': 'DDI',
        'DRUG_DRUG_INTERACTION': 'DDI',
        
        # =====================================================================
        # Genetic Disorders
        # =====================================================================
        'DISEASE_GENETIC_DISORDER': 'DISEASE_GENETIC_DISORDER',
        'GENETIC_DISORDER': 'DISEASE_GENETIC_DISORDER',
        'RELATED_GENETIC_DISORDER': 'RELATED_GENETIC_DISORDER',
        
        # =====================================================================
        # Drug Classification
        # =====================================================================
        'DRUG_ATC_CODE': 'DRUG_ATC_CODE',      # Anatomical Therapeutic Chemical code
        'ATC_CODE': 'DRUG_ATC_CODE',
        
        # =====================================================================
        # Disease Classification
        # =====================================================================
        'DISEASE_SUPERGRP': 'DISEASE_SUPERGRP',  # Disease supergroup
    }


# =============================================================================
# Relation Mapper Class
# =============================================================================
class RelationMapper:
    """
    Enhanced mapper that uses multiple strategies to map relation names.
    
    This class handles the common problem in knowledge graph construction:
    different sources use different naming conventions for the same relation.
    For example, one source might use 'interacts_with' while another uses 'PPI'.
    
    The mapper tries multiple matching strategies in order of specificity:
    1. Direct lookup in predefined mapping
    2. Case-insensitive matching
    3. Normalized matching (removing underscores, prefixes)
    4. Partial word matching
    """
    
    def __init__(self, available_relations):
        """
        Args:
            available_relations: Set of relation names available in the dataset
        """
        self.available_relations = set(available_relations)
        
        # Separate string and integer relations
        # Some datasets use integer IDs instead of string names
        self.string_relations = {r for r in available_relations if isinstance(r, str)}
        self.int_relations = {r for r in available_relations if isinstance(r, (int, np.integer))}
        
        # Create normalized lookup for case-insensitive matching
        self.string_to_lower = {s.lower(): s for s in self.string_relations}
        
        # Build comprehensive mappings using multiple strategies
        self.relation_mappings = self._build_comprehensive_mappings()
        
    def _normalize_relation_name(self, name):
        """
        Normalize a relation name for comparison.
        
        This removes common variations that don't change the meaning:
        - Converts to lowercase
        - Removes underscores and hyphens
        - Removes common prefixes (has_, is_, etc.)
        
        Args:
            name: Relation name to normalize
            
        Returns:
            Normalized string
        """
        if not isinstance(name, str):
            return str(name)
        
        # Remove underscores and hyphens, convert to lowercase
        normalized = name.lower().replace('_', '').replace('-', '')
        
        # Remove common prefixes that don't affect meaning
        # e.g., 'has_function' -> 'function'
        prefixes = ['has_', 'is_', 'in_', 'of_', 'to_', 'by_', 'with_', 'for_']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                
        return normalized
    
    def _build_comprehensive_mappings(self):
        """
        Build comprehensive mappings between generic and actual relation names.
        
        This uses a predefined set of patterns and tries multiple matching strategies
        to find the best match in the available relations.
        
        Returns:
            dict: Mapping from generic relation names to actual dataset relation names
        """
        mappings = {}
        
        # =====================================================================
        # Predefined Patterns for Common Relation Names
        # =====================================================================
        # Each key is a generic relation name, and the value is a list of
        # patterns that might appear in the dataset
        direct_mappings = {
            # Protein interactions
            'INTERACTS_WITH': ['interacts_with', 'interaction', 'protein_interaction', 'ppi'],
            'INTERACTS': ['interacts_with', 'interaction', 'protein_interaction'],
            
            # Drug-target relationships
            'TARGETS': ['target', 'targets', 'drug_target', 'target_protein'],
            'TARGETED_BY': ['targeted_by', 'target_of', 'drug_target_of'],
            'TARGETS_PROTEIN': ['target', 'targets', 'drug_target', 'target_protein'],
            
            # Metabolism
            'METABOLIZES': ['metabolizes', 'metabolism', 'involved_in_metabolism', 'metabolic'],
            'METABOLIZED_BY': ['metabolized_by', 'metabolism_of', 'metabolic_process'],
            
            # Regulation
            'REGULATES': ['regulates', 'regulation', 'regulatory'],
            'REGULATED_BY': ['regulated_by', 'regulation_of', 'target_of_regulation'],
            'REGULATES_PATHWAY': ['regulates_pathway', 'pathway_regulation', 'regulates'],
            
            # Hierarchy
            'DIRECT_PARENT': ['parent', 'subclass_of', 'isa', 'subtype_of', 'broader'],
            'CELL_TYPE_PARENT': ['cell_type_parent', 'parent_cell_type', 'subclass_of'],
            'HAS_SUBPATHWAY': ['subpathway', 'part_of', 'has_part', 'contains_pathway'],
            'BELONGS_TO_FAMILY': ['family', 'protein_family', 'member_of_family', 'in_family'],
            
            # Similarity
            'SIMILAR_TO': ['similar', 'similar_to', 'homolog', 'homologous', 'ortholog', 'paralog'],
            'CHEMICAL_SIMILARITY': ['chemical_similarity', 'structurally_similar', 'similar_structure'],
            'STRUCTURALLY_SIMILAR': ['structurally_similar', 'structure_similar', 'similar_structure'],
            'SHARES_TARGET': ['shares_target', 'common_target', 'same_target'],
            'TARGET_SIMILAR': ['target_similar', 'similar_target', 'target_similarity'],
            'SHARES_PATHWAY': ['shares_pathway', 'common_pathway', 'same_pathway'],
            'SHARES_SIDE_EFFECT': ['shares_side_effect', 'common_side_effect', 'same_side_effect'],
            'SIDE_EFFECT_SIMILAR': ['side_effect_similar', 'similar_side_effect'],
            
            # Functions and phenotypes
            'HAS_FUNCTION': ['has_function', 'function', 'molecular_function', 'performs'],
            'HAS_PHENOTYPE': ['has_phenotype', 'phenotype', 'displays_phenotype'],
            'HAS_SYMPTOM': ['has_symptom', 'symptom', 'exhibits_symptom'],
            'HAS_GENETIC_BASIS': ['genetic_basis', 'has_genetic_basis', 'genetically_associated'],
            'ASSOCIATED_WITH_GENE': ['associated_with_gene', 'gene_associated', 'gene_link'],
            'ASSOCIATED_WITH_DISEASE': ['associated_with_disease', 'disease_associated', 'disease_link'],
            'MUTATED_IN': ['mutated_in', 'mutation_in', 'has_mutation'],
            
            # Pathway participation
            'PARTICIPATES_IN': ['participates_in', 'involved_in', 'part_of_pathway'],
            'INVOLVES_PROTEIN': ['involves_protein', 'protein_involved', 'has_protein'],
            'HAS_PARTICIPANT': ['has_participant', 'participant', 'involves'],
            'CONTAINS_REACTION': ['contains_reaction', 'has_reaction', 'reaction_in'],
            'CROSSTALK_WITH': ['crosstalk_with', 'pathway_crosstalk', 'crosstalk'],
            
            # Disease relationships
            'COMORBID_WITH': ['comorbid_with', 'comorbidity', 'associated_disease'],
            'OBSERVED_IN': ['observed_in', 'present_in', 'found_in', 'occurs_in'],
            'INHERITS_FROM': ['inherits_from', 'inherited_from', 'genetically_inherited'],
            'PHENOTYPICALLY_SIMILAR': ['phenotypically_similar', 'similar_phenotype'],
            
            # Protein relationships
            'SEQUENCE_SIMILARITY': ['sequence_similarity', 'sequence_similar', 'homolog'],
            'CO_EXPRESSED_WITH': ['co_expressed_with', 'coexpression', 'co_expressed'],
            'HAS_MEMBER': ['has_member', 'member', 'contains', 'subunit'],
            'FUNCTIONALLY_ASSOCIATED': ['functionally_associated', 'functional_association'],
            
            # Drug-specific
            'POTENTIAL_TREATMENT_FOR': ['treats', 'treatment_for', 'indicated_for', 'therapeutic_for'],
            
            # Others
            'SHARES_PROTEIN': ['shares_protein', 'common_protein', 'same_protein'],
        }
        
        # =====================================================================
        # Matching Strategy
        # =====================================================================
        # For each generic relation name and its patterns, try to find a match
        # in the available relations using increasingly lenient strategies
        for rule_rel, patterns in direct_mappings.items():
            for pattern in patterns:
                # Strategy 1: Exact match
                if pattern in self.string_relations:
                    mappings[rule_rel] = pattern
                    break
                
                # Strategy 2: Case-insensitive match
                pattern_lower = pattern.lower()
                for avail_rel in self.string_relations:
                    if avail_rel.lower() == pattern_lower:
                        mappings[rule_rel] = avail_rel
                        break
                if rule_rel in mappings:
                    break
                
                # Strategy 3: Normalized match (remove underscores, prefixes)
                norm_pattern = self._normalize_relation_name(pattern)
                for avail_rel in self.string_relations:
                    norm_avail = self._normalize_relation_name(avail_rel)
                    if (norm_pattern == norm_avail or 
                        norm_pattern in norm_avail or 
                        norm_avail in norm_pattern):
                        mappings[rule_rel] = avail_rel
                        break
                if rule_rel in mappings:
                    break
            
            # Strategy 4: If still not found, try partial word matching
            if rule_rel not in mappings:
                # Extract meaningful words from the pattern
                for pattern in patterns:
                    key_terms = pattern.lower().split('_')
                    for term in key_terms:
                        if len(term) > 3:  # Only consider words with >3 chars
                            for avail_rel in self.string_relations:
                                if term in avail_rel.lower():
                                    mappings[rule_rel] = avail_rel
                                    break
                        if rule_rel in mappings:
                            break
                    if rule_rel in mappings:
                        break
        
        return mappings

    def map_relation(self, relation_name):
        """
        Map a rule relation to an available dataset relation using multiple strategies.
        
        This method tries increasingly lenient matching strategies:
        1. Direct match in integer relations
        2. Direct match in string relations
        3. Pre-computed mappings
        4. Case-insensitive match
        5. Normalized match
        6. Partial word match
        
        Args:
            relation_name: The relation name from a rule (could be string or int)
            
        Returns:
            The mapped relation name if found, otherwise None
        """
        # =====================================================================
        # Handle Integer Relations
        # =====================================================================
        if isinstance(relation_name, (int, np.integer)):
            # Direct integer match
            if relation_name in self.int_relations:
                return relation_name
            
            # Try to find by value in string relations (if they're numeric strings)
            for avail_rel in self.string_relations:
                if str(relation_name) == avail_rel:
                    return avail_rel
                    
            # Return as-is if no mapping found (let the model handle it)
            return relation_name
        
        # =====================================================================
        # Handle String Relations
        # =====================================================================
        if isinstance(relation_name, str):
            # Strategy 1: Direct match
            if relation_name in self.string_relations:
                return relation_name
            
            # Strategy 2: Check pre-computed mappings
            if relation_name in self.relation_mappings:
                mapped = self.relation_mappings[relation_name]
                if mapped in self.string_relations:
                    return mapped
            
            # Strategy 3: Case-insensitive match
            rel_lower = relation_name.lower()
            if rel_lower in self.string_to_lower:
                return self.string_to_lower[rel_lower]
            
            # Strategy 4: Normalized match (remove underscores, prefixes)
            norm_rel = self._normalize_relation_name(relation_name)
            for avail_rel in self.string_relations:
                norm_avail = self._normalize_relation_name(avail_rel)
                if (norm_rel == norm_avail or 
                    norm_rel in norm_avail or 
                    norm_avail in norm_rel):
                    return avail_rel
            
            # Strategy 5: Partial word match
            # Extract individual words from the relation name
            words = re.findall(r'[a-z]+', relation_name.lower())
            for word in words:
                if len(word) > 3:  # Only consider meaningful words
                    for avail_rel in self.string_relations:
                        if word in avail_rel.lower():
                            return avail_rel
        
        # No mapping found
        return None
    
    def get_mapping_stats(self):
        """
        Get detailed statistics about the mapping success rate.
        
        Returns:
            dict: Statistics including counts of successful mappings
        """
        mapped_count = 0
        mapping_details = []
        
        for rule_rel, mapped_to in self.relation_mappings.items():
            if mapped_to in self.string_relations:
                mapped_count += 1
                mapping_details.append(f"{rule_rel} -> {mapped_to}")
        
        return {
            'total_rule_relations': len(self.relation_mappings),
            'successful_mappings': mapped_count,
            'available_string_relations': len(self.string_relations),
            'available_int_relations': len(self.int_relations),
            'mapping_details': mapping_details[:10]  # First 10 mappings as examples
        }
