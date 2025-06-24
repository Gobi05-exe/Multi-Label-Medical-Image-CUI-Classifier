
import os
import json
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from typing import Dict, Optional
import logging
from torch.utils.data import DataLoader
import torch.optim as optim


class MedicalImageTrainer:
    """Trainer class for medical image classification."""

    def __init__(
        self,
        model,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        logger: Optional[logging.Logger] = None,
        checkpoint_dir: str = "./checkpoints",
        threshold: float = 0.5,
        grad_accum_steps: int = 1,
        early_stop_patience: int = 3,
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.checkpoint_dir = checkpoint_dir
        self.threshold = threshold
        self.grad_accum_steps = grad_accum_steps
        self.early_stop_patience = early_stop_patience
        self.use_amp = use_amp and (device != 'cpu')  # AMP only on GPU

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=3e-5,
            weight_decay=0.01
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=len(train_loader) * 10  # Assuming 10 epochs
        )

        self.scaler = torch.amp.GradScaler() if self.use_amp else None

        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.best_f1 = 0.0

        self.early_stop_counter = 0

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        self.optimizer.zero_grad()

        pbar = tqdm(total=num_batches, desc="Training", leave=False)

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)

            if self.use_amp:
                with torch.amp.autocast(device_type=self.device):
                    outputs = self.model(pixel_values=images, labels=labels)
                    loss = outputs["loss"] / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(pixel_values=images, labels=labels)
                loss = outputs["loss"] / self.grad_accum_steps
                loss.backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0 or (batch_idx + 1) == num_batches:
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps  # multiply back for logging

            pbar.update(1)
            pbar.set_postfix({'Loss': f'{loss.item() * self.grad_accum_steps:.4f}'})

            if batch_idx % 10 == 0:
                self.logger.info(f'Batch {batch_idx}/{num_batches}, Loss: {loss.item() * self.grad_accum_steps:.4f}')

        pbar.close()
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(pixel_values=images, labels=labels)
                loss = outputs["loss"]

                total_loss += loss.item()

                preds = torch.sigmoid(outputs["logits"]) > self.threshold
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)

        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)

        metrics = {
            'val_loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        self.val_metrics.append(metrics)

        # Save checkpoint if F1 improved
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.save_checkpoint(metrics, is_best=True)
            self.logger.info(f"New best F1 score: {f1:.4f} - Model checkpoint saved")
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics
        }

        if is_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'latest_model.pt')

        torch.save(checkpoint, checkpoint_path)

    def save_training_history(self, filepath: str = "./training_history.json") -> None:
        """Save training history to JSON file."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_metrics': self.val_metrics,
            'best_f1_score': self.best_f1
        }

        with open(filepath, 'w') as f:
            json.dump(history, f, indent=2)

        self.logger.info(f"Training history saved to {filepath}")

    def train(self, num_epochs: int = 10) -> None:
        """Full training loop."""
        self.logger.info(f"Training on {self.device}")

        for epoch in range(num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{num_epochs}")
            self.logger.info("-" * 50)

            # Train
            train_loss = self.train_epoch()
            self.logger.info(f"Training Loss: {train_loss:.4f}")

            # Log current LR
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Learning Rate: {current_lr:.6f}")

            # Validate
            val_metrics = self.validate()
            self.logger.info(f"Validation Loss: {val_metrics['val_loss']:.4f}")
            self.logger.info(f"Precision: {val_metrics['precision']:.4f}")
            self.logger.info(f"Recall: {val_metrics['recall']:.4f}")
            self.logger.info(f"F1-Score: {val_metrics['f1_score']:.4f}")

            # Early stopping check
            if self.early_stop_counter >= self.early_stop_patience:
                self.logger.info(f"Early stopping triggered after {self.early_stop_patience} epochs with no improvement.")
                break

        # Save final training history
        self.save_training_history()