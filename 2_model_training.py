"""
Aviation Question Generation System - Model Training Module (FIXED GENERATION)
Fine-tunes T5-small model on aviation corpus with improved generation.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from tqdm import tqdm
import os
from datetime import datetime


class AviationQADataset(Dataset):
    """Custom Dataset for aviation question generation."""
    
    def __init__(self, data_path, tokenizer, max_input_length=512, max_target_length=128):
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            item['input_text'],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            item['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Prepare labels (replace padding token id with -100)
        labels = target_encoding['input_ids'].squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


class AviationQuestionGenerator:
    """Trainer class for aviation question generation model."""
    
    def __init__(self, model_name='t5-small', output_dir='./models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device)
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.training_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epochs': 0
        }
    
    def train(self, train_path, val_path, epochs=3, batch_size=8, learning_rate=3e-4):
        """Train the model on aviation corpus."""
        print("\n" + "="*60)
        print("Starting Model Training (FIXED)")
        print("="*60)
        
        # Create datasets and dataloaders
        train_dataset = AviationQADataset(train_path, self.tokenizer)
        val_dataset = AviationQADataset(val_path, self.tokenizer)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Setup optimizer and scheduler - HIGHER LEARNING RATE
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        print(f"\nTraining Configuration:")
        print(f"  Training examples: {len(train_dataset)}")
        print(f"  Validation examples: {len(val_dataset)}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate} (INCREASED)")
        print(f"  Total steps: {total_steps}")
        print(f"  Device: {self.device}\n")
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_progress = tqdm(train_loader, desc="Training")
            
            for batch in train_progress:
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                # Update progress bar
                train_progress.set_postfix({'loss': f'{loss.item():.4f}'})
            
            avg_train_loss = train_loss / len(train_loader)
            self.training_stats['train_losses'].append(avg_train_loss)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            
            with torch.no_grad():
                val_progress = tqdm(val_loader, desc="Validation")
                for batch in val_progress:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    val_progress.set_postfix({'loss': f'{outputs.loss.item():.4f}'})
            
            avg_val_loss = val_loss / len(val_loader)
            self.training_stats['val_losses'].append(avg_val_loss)
            
            print(f"\nEpoch {epoch + 1} Summary:")
            print(f"  Average Train Loss: {avg_train_loss:.4f}")
            print(f"  Average Val Loss: {avg_val_loss:.4f}")
            
            # Test generation after each epoch
            if (epoch + 1) % 1 == 0:
                self.test_generation()
            
            # Save checkpoint
            self.save_checkpoint(epoch + 1)
        
        self.training_stats['epochs'] = epochs
        self.save_training_stats()
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
    
    def test_generation(self):
        """Test generation during training to monitor quality."""
        test_context = "The altimeter measures altitude by detecting atmospheric pressure changes."
        print(f"\n  Testing generation:")
        print(f"  Context: {test_context[:60]}...")
        
        questions = self.generate_question(test_context, num_return_sequences=2)
        for i, q in enumerate(questions, 1):
            print(f"    {i}. {q}")
    
    def save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        print(f"  ✓ Checkpoint saved to {checkpoint_dir}")
    
    def save_training_stats(self):
        """Save training statistics."""
        stats_path = os.path.join(self.output_dir, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        print(f"  ✓ Training stats saved to {stats_path}")
    
    def generate_question(self, context, max_length=80, num_return_sequences=1):
        """
        Generate questions from context with IMPROVED parameters.
        """
        self.model.eval()
        
        input_text = f"generate question: {context}"
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,  # INCREASED from 64
                min_length=5,  # ADDED: Minimum length
                num_return_sequences=num_return_sequences,
                num_beams=4,  # CHANGED: Use beam search instead of sampling
                length_penalty=1.0,  # ADDED: Encourage longer outputs
                no_repeat_ngram_size=2,  # ADDED: Prevent repetition
                early_stopping=True
            )
        
        questions = [
            self.tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        
        return questions


def main():
    """Main training function."""
    
    # Initialize trainer
    trainer = AviationQuestionGenerator(
        model_name='t5-small',
        output_dir='./aviation_qg_model'
    )
    
    # Train model with FIXED parameters
    trainer.train(
        train_path='aviation_train.json',
        val_path='aviation_val.json',
        epochs=7,  # Back to 3 epochs
        batch_size=8,
        learning_rate=3e-4  # INCREASED learning rate
    )
    
    # Test generation
    print("\n" + "="*60)
    print("Final Testing - Question Generation")
    print("="*60)
    
    test_contexts = [
        "The altimeter measures the height of an aircraft above sea level by detecting changes in atmospheric pressure.",
        "VOR stands for VHF Omnidirectional Range and is a radio navigation system.",
        "A stall occurs when the wing exceeds the critical angle of attack."
    ]
    
    for i, context in enumerate(test_contexts, 1):
        print(f"\nTest {i}:")
        print(f"Context: {context[:70]}...")
        print("Generated Questions:")
        questions = trainer.generate_question(context, num_return_sequences=3)
        for j, question in enumerate(questions, 1):
            print(f"  {j}. {question}")


if __name__ == "__main__":
    main()