"""
HLA-based Clinical Predictors
---------------------------
Models for predicting clinical outcomes based on HLA alleles.
"""
import os
import logging
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed; prediction models will not be available")
    TORCH_AVAILABLE = False

class HLAPredictor:
    """Base class for HLA-based clinical predictors
    
    This class provides a foundation for building models that predict
    clinical outcomes based on HLA typing and optional clinical variables.
    """
    
    def __init__(
        self, 
        encoder,
        clinical_variables=None,
        model_type="mlp",
        freeze_encoder=True,
        device=None
    ):
        """Initialize predictor
        
        Args:
            encoder: HLAEncoder instance used to generate embeddings
            clinical_variables: List of clinical variable names to include
            model_type: Type of model architecture ('mlp', 'lstm', 'cnn')
            freeze_encoder: Whether to freeze encoder weights during training
            device: Device to run on ('cpu', 'cuda', or None for auto-detection)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed; cannot use HLAPredictor")
            
        self.encoder = encoder
        self.clinical_variables = clinical_variables or []
        self.model_type = model_type
        self.freeze_encoder = freeze_encoder
        
        # Set device
        self.device = device
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Initialize model
        if model_type == "mlp":
            self.model = self._create_mlp_model()
        elif model_type == "lstm":
            self.model = self._create_lstm_model()
        elif model_type == "cnn":
            self.model = self._create_cnn_model()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Move model to device
        self.model.to(self.device)
    
    def _create_mlp_model(self):
        """Create a simple MLP model
        
        Returns:
            PyTorch model
        """
        # Calculate input dimensions based on HLA and clinical variables
        hla_dim = 1024  # ProtBERT embedding size
        clinical_dim = len(self.clinical_variables)
        input_dim = hla_dim + clinical_dim
        
        # Define model architecture
        model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        return model
    
    def _create_lstm_model(self):
        """Create an LSTM-based model
        
        Returns:
            PyTorch model
        """
        # For LSTM, we'll use a sequence of embeddings as input
        # This could be multiple alleles per locus
        hla_dim = 1024  # ProtBERT embedding size
        clinical_dim = len(self.clinical_variables)
        
        class LSTMModel(nn.Module):
            def __init__(self, hla_dim, clinical_dim):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=hla_dim,
                    hidden_size=256,
                    num_layers=2,
                    batch_first=True,
                    dropout=0.3,
                    bidirectional=True
                )
                lstm_output_dim = 256 * 2  # bidirectional
                self.clinical_fc = nn.Linear(clinical_dim, 64) if clinical_dim > 0 else None
                final_dim = lstm_output_dim + (64 if clinical_dim > 0 else 0)
                self.classifier = nn.Sequential(
                    nn.Linear(final_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, hla_seq, clinical=None):
                # Process HLA sequence with LSTM
                lstm_out, _ = self.lstm(hla_seq)
                # Take the final output
                lstm_final = lstm_out[:, -1, :]
                
                if clinical is not None and self.clinical_fc is not None:
                    # Process clinical variables
                    clinical_out = self.clinical_fc(clinical)
                    clinical_out = torch.relu(clinical_out)
                    # Combine HLA and clinical
                    combined = torch.cat([lstm_final, clinical_out], dim=1)
                else:
                    combined = lstm_final
                
                # Final prediction
                out = self.classifier(combined)
                return out
        
        return LSTMModel(hla_dim, clinical_dim)
    
    def _create_cnn_model(self):
        """Create a CNN-based model
        
        Returns:
            PyTorch model
        """
        # For CNN, we'll treat the HLA embeddings as a 1D sequence
        hla_dim = 1024  # ProtBERT embedding size
        clinical_dim = len(self.clinical_variables)
        
        class CNNModel(nn.Module):
            def __init__(self, hla_dim, clinical_dim):
                super().__init__()
                # CNN for HLA embeddings
                self.cnn = nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Flatten()
                )
                # Calculate CNN output dimension
                cnn_output_dim = 128 * (hla_dim // 4)  # After two 2x pooling layers
                self.clinical_fc = nn.Linear(clinical_dim, 64) if clinical_dim > 0 else None
                final_dim = cnn_output_dim + (64 if clinical_dim > 0 else 0)
                self.classifier = nn.Sequential(
                    nn.Linear(final_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 1),
                    nn.Sigmoid()
                )
                
            def forward(self, hla_emb, clinical=None):
                # Add channel dimension for CNN
                hla_emb = hla_emb.unsqueeze(1)  # [batch, 1, embedding_dim]
                # Process HLA embedding with CNN
                cnn_out = self.cnn(hla_emb)
                
                if clinical is not None and self.clinical_fc is not None:
                    # Process clinical variables
                    clinical_out = self.clinical_fc(clinical)
                    clinical_out = torch.relu(clinical_out)
                    # Combine HLA and clinical
                    combined = torch.cat([cnn_out, clinical_out], dim=1)
                else:
                    combined = cnn_out
                
                # Final prediction
                out = self.classifier(combined)
                return out
        
        return CNNModel(hla_dim, clinical_dim)
    
    def prepare_input(self, hla_alleles, clinical_data=None):
        """Prepare model input from HLA typing and clinical data
        
        Args:
            hla_alleles: List of HLA alleles
            clinical_data: Dict of clinical variables (optional)
            
        Returns:
            Tuple of (hla_features, clinical_features) as PyTorch tensors
        """
        # Get embeddings for alleles
        try:
            # Get embeddings for all alleles
            embeddings = []
            for allele in hla_alleles:
                try:
                    embedding = self.encoder.get_embedding(allele)
                    embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Error getting embedding for {allele}: {e}")
            
            if not embeddings:
                raise ValueError("No valid embeddings found for any allele")
                
            # Average the embeddings
            hla_features = np.mean(embeddings, axis=0)
            
        except Exception as e:
            logger.error(f"Error preparing HLA features: {e}")
            raise
        
        # Prepare clinical features if available
        clinical_features = None
        if clinical_data and self.clinical_variables:
            try:
                clinical_features = np.array([
                    float(clinical_data.get(var, 0)) 
                    for var in self.clinical_variables
                ])
            except Exception as e:
                logger.error(f"Error preparing clinical features: {e}")
                clinical_features = np.zeros(len(self.clinical_variables))
        
        # Convert to PyTorch tensors
        hla_tensor = torch.tensor(hla_features, dtype=torch.float32).to(self.device)
        
        if clinical_features is not None:
            clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32).to(self.device)
            return hla_tensor, clinical_tensor
        else:
            return hla_tensor, None
    
    def predict(self, hla_alleles, clinical_data=None):
        """Make prediction for a single sample
        
        Args:
            hla_alleles: List of HLA alleles
            clinical_data: Dict of clinical variables (optional)
            
        Returns:
            Prediction score (0-1)
        """
        self.model.eval()
        with torch.no_grad():
            hla_features, clinical_features = self.prepare_input(hla_alleles, clinical_data)
            
            if self.model_type == "mlp":
                # Concatenate HLA and clinical features
                if clinical_features is not None:
                    features = torch.cat([hla_features, clinical_features])
                else:
                    features = hla_features
                output = self.model(features)
            else:
                # Add batch dimension for LSTM/CNN
                hla_features = hla_features.unsqueeze(0)
                if clinical_features is not None:
                    clinical_features = clinical_features.unsqueeze(0)
                output = self.model(hla_features, clinical_features)
                
        return output.item()
    
    def batch_predict(self, samples):
        """Make predictions for multiple samples
        
        Args:
            samples: List of (hla_alleles, clinical_data) tuples
            
        Returns:
            List of prediction scores
        """
        self.model.eval()
        predictions = []
        
        for hla_alleles, clinical_data in samples:
            try:
                pred = self.predict(hla_alleles, clinical_data)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Error predicting sample: {e}")
                predictions.append(None)
                
        return predictions
    
    def train(
        self, 
        train_data, 
        train_labels, 
        validation_data=None, 
        validation_labels=None,
        epochs=10, 
        batch_size=32, 
        learning_rate=1e-3,
        weight_decay=1e-4,
        early_stopping_patience=5
    ):
        """Train the model
        
        Args:
            train_data: List of (hla_alleles, clinical_data) tuples
            train_labels: List of target labels (0/1 for classification)
            validation_data: Optional validation data
            validation_labels: Optional validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            weight_decay: L2 regularization weight
            early_stopping_patience: Epochs to wait before early stopping
            
        Returns:
            Dict with training history
        """
        # Set model to training mode
        self.model.train()
        
        # Define optimizer and loss
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.BCELoss()
        
        # Track metrics
        history = {
            'train_loss': [],
            'train_auc': [],
            'val_loss': [],
            'val_auc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(epochs):
            # Train for one epoch
            self.model.train()
            train_losses = []
            train_preds = []
            train_true = []
            
            for i in range(0, len(train_data), batch_size):
                batch_data = train_data[i:i + batch_size]
                batch_labels = train_labels[i:i + batch_size]
                
                # Prepare batch inputs
                batch_inputs = []
                batch_clinical = []
                
                for hla_alleles, clinical_data in batch_data:
                    hla_tensor, clinical_tensor = self.prepare_input(hla_alleles, clinical_data)
                    batch_inputs.append(hla_tensor)
                    if clinical_tensor is not None:
                        batch_clinical.append(clinical_tensor)
                
                if not batch_inputs:
                    continue
                
                # Stack inputs into batches
                hla_batch = torch.stack(batch_inputs)
                clinical_batch = torch.stack(batch_clinical) if batch_clinical else None
                
                # Convert labels to tensor
                label_batch = torch.tensor(batch_labels, dtype=torch.float32).to(self.device)
                if len(label_batch.shape) == 1:
                    label_batch = label_batch.unsqueeze(1)
                
                # Forward pass
                optimizer.zero_grad()
                
                if self.model_type == "mlp":
                    # Concatenate HLA and clinical features
                    if clinical_batch is not None:
                        features = torch.cat([hla_batch, clinical_batch], dim=1)
                    else:
                        features = hla_batch
                    outputs = self.model(features)
                else:
                    # LSTM or CNN model
                    outputs = self.model(hla_batch, clinical_batch)
                
                # Compute loss
                loss = criterion(outputs, label_batch)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Record metrics
                train_losses.append(loss.item())
                train_preds.extend(outputs.detach().cpu().numpy().flatten())
                train_true.extend(label_batch.cpu().numpy().flatten())
            
            # Calculate epoch metrics
            epoch_train_loss = np.mean(train_losses)
            epoch_train_auc = roc_auc_score(train_true, train_preds) if len(set(train_true)) > 1 else 0.5
            
            history['train_loss'].append(epoch_train_loss)
            history['train_auc'].append(epoch_train_auc)
            
            # Validation
            if validation_data and validation_labels:
                self.model.eval()
                val_losses = []
                val_preds = []
                val_true = []
                
                with torch.no_grad():
                    for i in range(0, len(validation_data), batch_size):
                        batch_data = validation_data[i:i + batch_size]
                        batch_labels = validation_labels[i:i + batch_size]
                        
                        # Prepare batch inputs (similar to training)
                        batch_inputs = []
                        batch_clinical = []
                        
                        for hla_alleles, clinical_data in batch_data:
                            hla_tensor, clinical_tensor = self.prepare_input(hla_alleles, clinical_data)
                            batch_inputs.append(hla_tensor)
                            if clinical_tensor is not None:
                                batch_clinical.append(clinical_tensor)
                        
                        if not batch_inputs:
                            continue
                        
                        # Stack inputs into batches
                        hla_batch = torch.stack(batch_inputs)
                        clinical_batch = torch.stack(batch_clinical) if batch_clinical else None
                        
                        # Convert labels to tensor
                        label_batch = torch.tensor(batch_labels, dtype=torch.float32).to(self.device)
                        if len(label_batch.shape) == 1:
                            label_batch = label_batch.unsqueeze(1)
                        
                        # Forward pass
                        if self.model_type == "mlp":
                            # Concatenate HLA and clinical features
                            if clinical_batch is not None:
                                features = torch.cat([hla_batch, clinical_batch], dim=1)
                            else:
                                features = hla_batch
                            outputs = self.model(features)
                        else:
                            # LSTM or CNN model
                            outputs = self.model(hla_batch, clinical_batch)
                        
                        # Compute loss
                        loss = criterion(outputs, label_batch)
                        
                        # Record metrics
                        val_losses.append(loss.item())
                        val_preds.extend(outputs.cpu().numpy().flatten())
                        val_true.extend(label_batch.cpu().numpy().flatten())
                
                # Calculate validation metrics
                epoch_val_loss = np.mean(val_losses)
                epoch_val_auc = roc_auc_score(val_true, val_preds) if len(set(val_true)) > 1 else 0.5
                
                history['val_loss'].append(epoch_val_loss)
                history['val_auc'].append(epoch_val_auc)
                
                # Print progress
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {epoch_train_loss:.4f}, train_auc: {epoch_train_auc:.4f}, "
                          f"val_loss: {epoch_val_loss:.4f}, val_auc: {epoch_val_auc:.4f}")
                
                # Early stopping
                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                # No validation data
                logger.info(f"Epoch {epoch+1}/{epochs} - "
                          f"train_loss: {epoch_train_loss:.4f}, train_auc: {epoch_train_auc:.4f}")
        
        return history
    
    def save(self, filepath):
        """Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)
        
        save_dict = {
            'model_state': self.model.state_dict(),
            'model_type': self.model_type,
            'clinical_variables': self.clinical_variables,
        }
        
        torch.save(save_dict, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath, encoder):
        """Load a saved model
        
        Args:
            filepath: Path to saved model
            encoder: HLAEncoder instance
            
        Returns:
            Loaded HLAPredictor instance
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed; cannot load model")
            
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        # Load saved state
        save_dict = torch.load(filepath, map_location='cpu')
        
        # Create instance
        instance = cls(
            encoder=encoder,
            clinical_variables=save_dict['clinical_variables'],
            model_type=save_dict['model_type']
        )
        
        # Load state
        instance.model.load_state_dict(save_dict['model_state'])
        
        logger.info(f"Model loaded from {filepath}")
        return instance


class TransplantOutcomePredictor(HLAPredictor):
    """Predicts transplant outcomes based on HLA matching and clinical variables"""
    
    def __init__(
        self,
        encoder,
        clinical_variables=None,
        model_type="mlp",
        freeze_encoder=True,
        outcome_type="survival",  # survival, gvhd, rejection
        device=None
    ):
        """Initialize transplant outcome predictor
        
        Args:
            encoder: HLAEncoder instance
            clinical_variables: List of clinical variable names
            model_type: Model architecture type
            freeze_encoder: Whether to freeze encoder weights
            outcome_type: Type of outcome to predict
            device: Device to run on
        """
        self.outcome_type = outcome_type
        
        # Define default clinical variables based on outcome type
        if clinical_variables is None:
            if outcome_type == "survival":
                clinical_variables = ["recipient_age", "donor_age", "disease", "hct_ci", "donor_type"]
            elif outcome_type == "gvhd":
                clinical_variables = ["recipient_age", "donor_age", "gender_match", "conditioning"]
            elif outcome_type == "rejection":
                clinical_variables = ["recipient_age", "cell_dose", "conditioning", "prior_treatment"]
                
        # Initialize base class
        super().__init__(encoder, clinical_variables, model_type, freeze_encoder, device)
    
    def prepare_input(self, donor_alleles, recipient_alleles, clinical_data=None):
        """Prepare model input from donor and recipient HLA typing plus clinical data
        
        Args:
            donor_alleles: List of donor HLA alleles
            recipient_alleles: List of recipient HLA alleles
            clinical_data: Dict of clinical variables
            
        Returns:
            Tuple of (hla_features, clinical_features) as PyTorch tensors
        """
        # Get embeddings for donor and recipient
        donor_embeddings = {}
        recipient_embeddings = {}
        
        # Group alleles by locus
        for allele in donor_alleles:
            locus = allele.split('*')[0]
            if locus not in donor_embeddings:
                donor_embeddings[locus] = []
            try:
                donor_embeddings[locus].append(self.encoder.get_embedding(allele))
            except Exception as e:
                logger.warning(f"Error getting embedding for donor allele {allele}: {e}")
            
        for allele in recipient_alleles:
            locus = allele.split('*')[0]
            if locus not in recipient_embeddings:
                recipient_embeddings[locus] = []
            try:
                recipient_embeddings[locus].append(self.encoder.get_embedding(allele))
            except Exception as e:
                logger.warning(f"Error getting embedding for recipient allele {allele}: {e}")
        
        # Calculate differences between donor and recipient per locus
        diffs = []
        for locus in set(donor_embeddings.keys()) & set(recipient_embeddings.keys()):
            for d_emb in donor_embeddings[locus]:
                for r_emb in recipient_embeddings[locus]:
                    diff = np.abs(d_emb - r_emb)
                    diffs.append(diff)
        
        # Average the differences
        hla_features = np.mean(diffs, axis=0) if diffs else np.zeros(1024)  # ProtBERT dim
        
        # Prepare clinical features if available
        clinical_features = None
        if clinical_data and self.clinical_variables:
            try:
                clinical_features = np.array([
                    float(clinical_data.get(var, 0)) 
                    for var in self.clinical_variables
                ])
            except Exception as e:
                logger.error(f"Error preparing clinical features: {e}")
                clinical_features = np.zeros(len(self.clinical_variables))
        
        # Convert to PyTorch tensors
        hla_tensor = torch.tensor(hla_features, dtype=torch.float32).to(self.device)
        
        if clinical_features is not None:
            clinical_tensor = torch.tensor(clinical_features, dtype=torch.float32).to(self.device)
            return hla_tensor, clinical_tensor
        else:
            return hla_tensor, None
    
    def predict_outcome(self, donor_alleles, recipient_alleles, clinical_data=None):
        """Predict transplant outcome
        
        Args:
            donor_alleles: List of donor HLA alleles
            recipient_alleles: List of recipient HLA alleles
            clinical_data: Dict of clinical variables
            
        Returns:
            Outcome prediction score (0-1)
        """
        self.model.eval()
        with torch.no_grad():
            hla_features, clinical_features = self.prepare_input(
                donor_alleles, recipient_alleles, clinical_data
            )
            
            if self.model_type == "mlp":
                # Concatenate HLA and clinical features
                if clinical_features is not None:
                    features = torch.cat([hla_features, clinical_features])
                else:
                    features = hla_features
                output = self.model(features)
            else:
                # Add batch dimension for LSTM/CNN
                hla_features = hla_features.unsqueeze(0)
                if clinical_features is not None:
                    clinical_features = clinical_features.unsqueeze(0)
                output = self.model(hla_features, clinical_features)
                
        # Interpret output based on outcome type
        score = output.item()
        if self.outcome_type == "survival":
            return score  # Higher = better survival
        elif self.outcome_type == "gvhd":
            return score  # Higher = higher GVHD risk
        elif self.outcome_type == "rejection":
            return score  # Higher = higher rejection risk
        
    def get_outcome_explanation(self, score):
        """Get human-readable explanation of outcome score
        
        Args:
            score: Outcome prediction score
            
        Returns:
            Explanation string
        """
        if self.outcome_type == "survival":
            if score >= 0.8:
                return "Excellent predicted survival outcome"
            elif score >= 0.6:
                return "Good predicted survival outcome"
            elif score >= 0.4:
                return "Moderate predicted survival outcome"
            else:
                return "Poor predicted survival outcome"
        elif self.outcome_type == "gvhd":
            if score >= 0.8:
                return "Very high risk of GVHD"
            elif score >= 0.6:
                return "High risk of GVHD"
            elif score >= 0.4:
                return "Moderate risk of GVHD"
            else:
                return "Low risk of GVHD"
        elif self.outcome_type == "rejection":
            if score >= 0.8:
                return "Very high risk of rejection"
            elif score >= 0.6:
                return "High risk of rejection"
            elif score >= 0.4:
                return "Moderate risk of rejection"
            else:
                return "Low risk of rejection"
                
        return f"Prediction score: {score:.2f}"


class GVHDRiskPredictor(TransplantOutcomePredictor):
    """Specialized predictor for GVHD risk"""
    
    def __init__(
        self,
        encoder,
        clinical_variables=None,
        model_type="mlp",
        freeze_encoder=True,
        device=None
    ):
        """Initialize GVHD risk predictor
        
        Args:
            encoder: HLAEncoder instance
            clinical_variables: List of clinical variables (or None for defaults)
            model_type: Model architecture
            freeze_encoder: Whether to freeze encoder weights
            device: Device to run on
        """
        # Default clinical variables specific to GVHD
        if clinical_variables is None:
            clinical_variables = [
                "recipient_age", 
                "donor_age", 
                "gender_match", 
                "conditioning_intensity",
                "gvhd_prophylaxis", 
                "cell_source",
                "donor_type"
            ]
            
        super().__init__(
            encoder=encoder,
            clinical_variables=clinical_variables,
            model_type=model_type,
            freeze_encoder=freeze_encoder,
            outcome_type="gvhd",
            device=device
        )


class EngrafdmentPredictor(TransplantOutcomePredictor):
    """Specialized predictor for engraftment success"""
    
    def __init__(
        self,
        encoder,
        clinical_variables=None,
        model_type="mlp",
        freeze_encoder=True,
        device=None
    ):
        """Initialize engraftment predictor
        
        Args:
            encoder: HLAEncoder instance
            clinical_variables: List of clinical variables (or None for defaults)
            model_type: Model architecture
            freeze_encoder: Whether to freeze encoder weights
            device: Device to run on
        """
        # Default clinical variables specific to engraftment
        if clinical_variables is None:
            clinical_variables = [
                "recipient_age", 
                "cell_dose", 
                "cd34_dose",
                "conditioning_regimen",
                "prior_treatment", 
                "disease_stage",
                "donor_type"
            ]
            
        super().__init__(
            encoder=encoder,
            clinical_variables=clinical_variables,
            model_type=model_type,
            freeze_encoder=freeze_encoder,
            outcome_type="rejection",  # Use rejection mechanics for engraftment
            device=device
        )
        
    def predict_engraftment(self, donor_alleles, recipient_alleles, clinical_data=None):
        """Predict engraftment success
        
        Args:
            donor_alleles: List of donor HLA alleles
            recipient_alleles: List of recipient HLA alleles
            clinical_data: Dict of clinical variables
            
        Returns:
            Engraftment success probability (0-1)
        """
        # Invert rejection risk for engraftment success
        rejection_risk = self.predict_outcome(donor_alleles, recipient_alleles, clinical_data)
        engraftment_prob = 1.0 - rejection_risk
        return engraftment_prob
