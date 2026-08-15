"""Neural Network model for depression detection using PyTorch."""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Tuple

from ...core.config import config


class DepressionNet(nn.Module):
    """Feedforward neural network for depression classification."""
    
    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.3):
        super(DepressionNet, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = config.model.nn_hidden_dims
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class NeuralNetModel:
    """PyTorch neural network wrapper."""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = DepressionNet(
            input_dim=input_dim,
            hidden_dims=config.model.nn_hidden_dims,
            dropout=config.model.nn_dropout,
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.model.nn_learning_rate
        )
        
        self.is_trained = False
        
        print(f"Using device: {self.device}")
        print(f"Model architecture:\n{self.model}")
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """Train the neural network."""
        print("\nTraining Neural Network...")
        
        # Prepare data loaders
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.model.nn_epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_acc = self._validate_epoch(val_loader)
                
                print(
                    f"Epoch [{epoch+1}/{config.model.nn_epochs}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
                )
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= config.model.nn_early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                print(
                    f"Epoch [{epoch+1}/{config.model.nn_epochs}] "
                    f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}"
                )
        
        self.is_trained = True
        print("✅ Neural Network training complete")
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        
        return predicted.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def _create_dataloader(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        shuffle: bool = False
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(
            dataset,
            batch_size=config.model.nn_batch_size,
            shuffle=shuffle
        )
        
        return loader
    
    def save(self, filepath: Path):
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
        }, filepath)
        print(f"Model saved to: {filepath}")
    
    def load(self, filepath: Path):
        """Load model from disk."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.input_dim = checkpoint['input_dim']
        self.model = DepressionNet(
            input_dim=self.input_dim,
            hidden_dims=config.model.nn_hidden_dims,
            dropout=config.model.nn_dropout,
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.is_trained = True
        
        print(f"Model loaded from: {filepath}")
