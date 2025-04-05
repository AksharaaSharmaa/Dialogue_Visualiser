import spacy
import networkx as nx
import pandas as pd
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

class ArgumentMiner:
    """Class for extracting arguments and their relationships from text."""
    
    def __init__(self, use_transformers=True):
        """Initialize the argument miner with necessary models.
        
        Args:
            use_transformers: Whether to use Hugging Face transformers (more accurate but slower)
                             or basic rule-based extraction (faster but less accurate)
        """
        self.use_transformers = use_transformers
        
        # Load spaCy model for basic NLP tasks
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load models for argument mining if using transformers
        if use_transformers:
            try:
                # Set up claim detection model
                model_name = "roberta-base"  # In a real implementation, use specialized argument mining model
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.claim_classifier = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)
                
                # In a full implementation, we would also load:
                # - Premise detection model
                # - Stance classification model (support/attack relationships)
            except Exception as e:
                print(f"Error loading transformer models: {e}")
                self.use_transformers = False
                print("Falling back to rule-based approach")
    
    def _extract_claims_rule_based(self, text):
        """Extract potential claims using rule-based approach."""
        doc = self.nlp(text)
        claims = []
        
        # Simple heuristic: Sentences containing certain claim indicators
        claim_indicators = [
            "believe", "think", "argue", "claim", "suggest", "propose",
            "should", "must", "need to", "have to", "opinion", "view"
        ]
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            if any(indicator in sent_text for indicator in claim_indicators):
                claims.append({
                    "text": sent.text,
                    "confidence": 0.7,  # Arbitrary confidence for rule-based
                    "span": (sent.start_char, sent.end_char)
                })
        
        return claims
    
    def _extract_claims_transformers(self, text):
        """Extract potential claims using transformer model."""
        # In a real implementation, this would use a specialized argument mining model
        # For this prototype, we'll simulate the model's output
        
        doc = self.nlp(text)
        claims = []
        
        for sent in doc.sents:
            # Simulate classification - in reality, we would use:
            # result = self.claim_classifier(sent.text)
            
            # Simplified simulation:
            claim_indicators = [
                "believe", "think", "argue", "claim", "suggest", "propose",
                "should", "must", "need to", "have to", "opinion", "view"
            ]
            sent_text = sent.text.lower()
            
            confidence = 0.5
            if any(indicator in sent_text for indicator in claim_indicators):
                confidence = 0.8
            
            if confidence > 0.6:  # Confidence threshold
                claims.append({
                    "text": sent.text,
                    "confidence": confidence,
                    "span": (sent.start_char, sent.end_char)
                })
        
        return claims
    
    def extract_claims(self, text):
        """Extract claims from text."""
        if self.use_transformers:
            return self._extract_claims_transformers(text)
        else:
            return self._extract_claims_rule_based(text)
    
    def identify_stance(self, text1, text2):
        """Identify if text2 supports or attacks text1."""
        # In a real implementation, this would use a stance classification model
        # For this prototype, we'll use a simple heuristic approach
        
        support_indicators = ["agree", "support", "yes", "exactly", "right", "correct", "true"]
        attack_indicators = ["disagree", "but", "however", "though", "wrong", "false", "incorrect"]
        
        text2_lower = text2.lower()
        
        support_count = sum(1 for indicator in support_indicators if indicator in text2_lower)
        attack_count = sum(1 for indicator in attack_indicators if indicator in text2_lower)
        
        if support_count > attack_count:
            return "support", 0.6 + (0.3 * (support_count / (support_count + attack_count + 0.1)))
        elif attack_count > support_count:
            return "attack", 0.6 + (0.3 * (attack_count / (support_count + attack_count + 0.1)))
        else:
            return "neutral", 0.5
    
    def build_argument_graph(self, messages_df):
        """Build an argument graph from a DataFrame of messages.
        
        Args:
            messages_df: DataFrame with columns 'id', 'text', 'reply_to'
            
        Returns:
            NetworkX DiGraph where nodes are claims and edges represent support/attack relationships
        """
        G = nx.DiGraph()
        
        # Extract claims from each message
        message_claims = {}
        for _, row in messages_df.iterrows():
            claims = self.extract_claims(row['text'])
            if claims:
                message_claims[row['id']] = claims
                
                # Add nodes for each claim
                for i, claim in enumerate(claims):
                    claim_id = f"{row['id']}_{i}"
                    G.add_node(claim_id, 
                               text=claim['text'], 
                               message_id=row['id'],
                               confidence=claim['confidence'])
        
        # Add edges based on reply structure and stance
        for _, row in messages_df.iterrows():
            if pd.notna(row['reply_to']) and row['id'] in message_claims and row['reply_to'] in message_claims:
                parent_claims = message_claims[row['reply_to']]
                current_claims = message_claims[row['id']]
                
                for i, current_claim in enumerate(current_claims):
                    current_id = f"{row['id']}_{i}"
                    
                    # Find most likely claim in parent message that this claim responds to
                    max_similarity = 0
                    best_parent_id = None
                    stance_type = None
                    stance_confidence = 0
                    
                    for j, parent_claim in enumerate(parent_claims):
                        parent_id = f"{row['reply_to']}_{j}"
                        
                        # Determine stance
                        stance, confidence = self.identify_stance(parent_claim['text'], current_claim['text'])
                        
                        # Simple similarity based on stance confidence
                        similarity = confidence
                        
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_parent_id = parent_id
                            stance_type = stance
                            stance_confidence = confidence
                    
                    # Add edge if we found a connection
                    if best_parent_id and max_similarity > 0.6:
                        G.add_edge(best_parent_id, current_id, 
                                  relationship=stance_type,
                                  confidence=stance_confidence)
        
        return G