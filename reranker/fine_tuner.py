"""
Fine-tuning Module for Cross-Encoder Re-rankers.

This module provides functionality to fine-tune cross-encoder models
using hard negatives and margin ranking loss for Indonesian code search.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class FineTuningConfig:
    """Configuration for cross-encoder fine-tuning."""
    model_name: str = "sentence-transformers/ms-marco-MiniLM-L-12-v2-cross-encoder"
    learning_rate: float = 2e-5
    num_epochs: int = 10
    batch_size: int = 16
    warmup_steps: int = 100
    margin: float = 0.5
    max_seq_length: int = 512
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_path: str = "./full/models/cross-encoder-me5-base-full"
    gradient_accumulation_steps: int = 1


class TrainingDataset(Dataset):
    """Dataset for cross-encoder training with query-doc pairs."""
    
    def __init__(
        self,
        queries: List[str],
        positive_docs: List[str],
        negative_docs: List[str],
        tokenizer: AutoTokenizer,
        max_length: int = 512
    ):
        """
        Initialize training dataset.
        
        Args:
            queries: List of query strings
            positive_docs: List of positive (relevant) document strings
            negative_docs: List of negative (irrelevant) document strings
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.queries = queries
        self.positive_docs = positive_docs
        self.negative_docs = negative_docs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.queries)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query = self.queries[idx]
        pos_doc = self.positive_docs[idx]
        neg_doc = self.negative_docs[idx]
        
        # Encode positive pair
        pos_encoding = self.tokenizer(
            query,
            pos_doc,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Encode negative pair
        neg_encoding = self.tokenizer(
            query,
            neg_doc,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'pos_input_ids': pos_encoding['input_ids'].squeeze(0),
            'pos_attention_mask': pos_encoding['attention_mask'].squeeze(0),
            'neg_input_ids': neg_encoding['input_ids'].squeeze(0),
            'neg_attention_mask': neg_encoding['attention_mask'].squeeze(0),
        }


class CrossEncoderFineTuner:
    """
    Fine-tuner for cross-encoder models using margin ranking loss.
    
    This implements Method 2 from the research plan: Fine-tuned 
    Cross-Encoder with Hard Negatives.
    """
    
    def __init__(self, config: FineTuningConfig):
        """
        Initialize the fine-tuner.
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        
        logger.info(f"Initialized CrossEncoderFineTuner with config: {config}")
    
    def load_model(self):
        """Load the base model for fine-tuning."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name
        ).to(self.config.device)
        
        logger.info("Model loaded for fine-tuning")
    
    def create_dataset(
        self,
        queries: List[str],
        positive_docs: List[str],
        negative_docs: List[str]
    ) -> Dataset:
        """
        Create training dataset.
        
        Args:
            queries: List of query strings
            positive_docs: List of positive document strings
            negative_docs: List of negative document strings
            
        Returns:
            TrainingDataset
        """
        if self.tokenizer is None:
            self.load_model()
        
        return TrainingDataset(
            queries=queries,
            positive_docs=positive_docs,
            negative_docs=negative_docs,
            tokenizer=self.tokenizer,
            max_length=self.config.max_seq_length
        )
    
    def compute_loss(
        self,
        pos_scores: torch.Tensor,
        neg_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute margin ranking loss.
        
        Loss = max(0, margin - (pos_score - neg_score))
        
        Args:
            pos_scores: Scores for positive pairs
            neg_scores: Scores for negative pairs
            
        Returns:
            Loss value
        """
        margin = self.config.margin
        losses = torch.relu(margin - (pos_scores - neg_scores))
        return losses.mean()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            pos_input_ids = batch['pos_input_ids'].to(self.config.device)
            pos_attention_mask = batch['pos_attention_mask'].to(self.config.device)
            neg_input_ids = batch['neg_input_ids'].to(self.config.device)
            neg_attention_mask = batch['neg_attention_mask'].to(self.config.device)
            
            # Forward pass for positive pairs
            pos_outputs = self.model(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask
            )
            pos_scores = pos_outputs.logits.squeeze(-1)
            
            # Forward pass for negative pairs
            neg_outputs = self.model(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask
            )
            neg_scores = neg_outputs.logits.squeeze(-1)
            
            # Compute loss
            loss = self.compute_loss(pos_scores, neg_scores)
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(dataloader)
    
    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None
    ) -> Dict[str, List[float]]:
        """
        Train the cross-encoder model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Optional validation dataset
            
        Returns:
            Training history
        """
        if self.model is None:
            self.load_model()
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [] if val_dataset else None
        }
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader, optimizer, scheduler)
            history['train_loss'].append(train_loss)
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            
            # Save intermediate checkpoint
            checkpoint_dir = os.path.join(self.config.save_path, f"checkpoint-epoch-{epoch+1}")
            self.model.save_pretrained(checkpoint_dir)
            self.tokenizer.save_pretrained(checkpoint_dir)
            
            # Also update the 'latest' model in the main save_path
            self.model.save_pretrained(self.config.save_path)
            self.tokenizer.save_pretrained(self.config.save_path)
            
            logger.info(f"Epoch {epoch+1} complete. Model saved to {self.config.save_path}")
            
            # Validate if provided
            if val_dataset:
                val_loss = self.evaluate(val_dataset)
                history['val_loss'].append(val_loss)
                logger.info(f"Val Loss: {val_loss:.4f}")
        
        # Save model
        self.save_model()
        
        return history
    
    def evaluate(self, dataset: Dataset) -> float:
        """
        Evaluate the model on a dataset.
        
        Args:
            dataset: Evaluation dataset
            
        Returns:
            Average loss
        """
        self.model.eval()
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                pos_input_ids = batch['pos_input_ids'].to(self.config.device)
                pos_attention_mask = batch['pos_attention_mask'].to(self.config.device)
                neg_input_ids = batch['neg_input_ids'].to(self.config.device)
                neg_attention_mask = batch['neg_attention_mask'].to(self.config.device)
                
                pos_outputs = self.model(
                    input_ids=pos_input_ids,
                    attention_mask=pos_attention_mask
                )
                pos_scores = pos_outputs.logits.squeeze(-1)
                
                neg_outputs = self.model(
                    input_ids=neg_input_ids,
                    attention_mask=neg_attention_mask
                )
                neg_scores = neg_outputs.logits.squeeze(-1)
                
                loss = self.compute_loss(pos_scores, neg_scores)
                total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def save_model(self):
        """Save the fine-tuned model."""
        import os
        os.makedirs(self.config.save_path, exist_ok=True)
        
        self.model.save_pretrained(self.config.save_path)
        self.tokenizer.save_pretrained(self.config.save_path)
        
        logger.info(f"Model saved to {self.config.save_path}")
    
    def load_finetuned(self, model_path: str):
        """Load a fine-tuned model."""
        logger.info(f"Loading fine-tuned model from {model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path
        ).to(self.config.device)
        
        self._is_loaded = True
        logger.info("Fine-tuned model loaded")


def create_hard_negatives(
    queries: List[str],
    retrieved_docs: List[List[Dict[str, Any]]],
    relevance_labels: Optional[List[List[int]]] = None,
    top_k: int = 10
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create hard negatives from first-stage retrieval.
    
    This implements Method 2 from the research plan:
    Hard Negative Mining from first-stage retrieval.
    
    Args:
        queries: List of queries
        retrieved_docs: First-stage retrieved documents for each query
        relevance_labels: Optional ground truth relevance (for evaluation)
        top_k: Number of top negatives to use
        
    Returns:
        Tuple of (queries, positive_docs, negative_docs)
    """
    positive_queries = []
    positive_docs = []
    negative_docs = []
    
    for query, docs in zip(queries, retrieved_docs):
        if len(docs) < 2:
            continue
            
        # Assume first document is most relevant (or use labels if available)
        if relevance_labels is not None:
            # Find positive (relevant) and negative (irrelevant) docs
            for i, (doc, label) in enumerate(zip(docs, relevance_labels)):
                if label == 1:  # Positive
                    positive_queries.append(query)
                    positive_docs.append(doc.get('text', doc.get('content', '')))
                elif label == 0 and len(negative_docs) < top_k:  # Negative
                    negative_docs.append(doc.get('text', doc.get('content', '')))
        else:
            # Use top-1 as positive, rest as negatives
            positive_queries.append(query)
            positive_docs.append(docs[0].get('text', docs[0].get('content', '')))
            
            for doc in docs[1:top_k+1]:
                negative_docs.append(doc.get('text', doc.get('content', '')))
    
    return positive_queries, positive_docs, negative_docs
