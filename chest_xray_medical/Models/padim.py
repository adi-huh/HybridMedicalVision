import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import gc
from tqdm import tqdm
import numpy as np

class PaDiM(nn.Module):
    def __init__(self, backbone_name='resnet18'):
        super(PaDiM, self).__init__()
        
        # Load pre-trained model
        if backbone_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.features = nn.ModuleList([
                nn.Sequential(*list(model.children())[:5]),  # Up to first maxpool
                model.layer1,                               # Resolution: 56x56
                model.layer2,                              # Resolution: 28x28
            ])
            self.dims = [64, 64, 128]  # Feature dimensions for each layer
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        
        # Freeze the model
        for param in self.parameters():
            param.requires_grad = False
            
        self.gaussian = None
        self.means = None
        self.covs = None
        self.score_mean = None
        self.score_std = None
        
        # Calculate output sizes
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self.shapes = []
            x = dummy
            for layer in self.features:
                x = layer(x)
                self.shapes.append(x.shape)
                
        del dummy, x
        gc.collect()
        
    def forward(self, x):
        feature_maps = []
        
        # Get feature maps from different layers with memory cleanup
        out = x
        for i, layer in enumerate(self.features):
            out = layer(out)
            feature_maps.append(out.clone())
            
            # Clear intermediate tensors except the last output
            if i < len(self.features) - 1:
                del out
                torch.cuda.empty_cache()
                out = feature_maps[-1]
            
        return feature_maps
    
    def process_batch_embeddings(self, features, target_shape):
        reshaped_features = []
        
        for idx, feat in enumerate(features):
            # Interpolate to match target shape if necessary
            if feat.shape[2:] != target_shape:
                feat = F.interpolate(feat, size=target_shape, mode='bilinear', align_corners=False)
            
            # Reshape to (N, C, H*W)
            reshaped = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
            reshaped_features.append(reshaped)
            
            # Clear intermediate tensors
            del feat
        
        # Concatenate features along feature dimension
        embedding = torch.cat(reshaped_features, dim=1)
        del reshaped_features
        return embedding

    def compute_covariance_chunk(self, embeddings, chunk_size=500):  # Reduced chunk size for lower memory usage
        num_features = embeddings.size(1)
        cov_matrix = torch.zeros((num_features, num_features), device=self.device)
        
        # Process chunks to compute covariance
        for i in range(0, embeddings.size(0), chunk_size):
            end_idx = min(i + chunk_size, embeddings.size(0))
            chunk = embeddings[i:end_idx]
            
            # Center the chunk
            chunk = chunk - self.means
            
            # Update covariance matrix
            cov_matrix += torch.matmul(chunk.t(), chunk)
            
            del chunk
            gc.collect()
            
        # Normalize by N-1
        cov_matrix /= (embeddings.size(0) - 1)
        return cov_matrix
    
    def fit(self, train_loader, progress_bar=False):
        self.eval()  # Set to evaluation mode
        torch.cuda.empty_cache()
        gc.collect()
        
        # Initialize lists to store embeddings
        all_embeddings = []
        running_sum = None
        num_samples = 0
        
        # Setup progress bar if requested
        if progress_bar:
            pbar = tqdm(train_loader, desc="Processing images")
        else:
            pbar = train_loader
        
        with torch.no_grad():
            target_shape = self.shapes[0][2:]  # Use the shape of the first feature map
            
            for images, _ in pbar:
                images = images.to(self.device)
                features = self.forward(images)
                
                # Process batch embeddings
                embedding = self.process_batch_embeddings(features, target_shape)
                
                # Update running statistics
                if running_sum is None:
                    running_sum = torch.zeros(embedding.size(1), device=self.device)
                running_sum += embedding.sum(dim=0)
                num_samples += embedding.size(0)
                
                all_embeddings.append(embedding.cpu())  # Store on CPU to save GPU memory
                
                # Update progress bar before deleting variables
                if progress_bar:
                    pbar.set_postfix({"Batch Size": images.size(0)})
                    
                # Clear memory
                del images, features, embedding
                gc.collect()
        
        print("\nStep 2/2: Computing statistical parameters...")
        
        # Compute mean
        print("Computing mean...")
        self.means = running_sum / num_samples
        del running_sum
        gc.collect()
        
        # Compute covariance matrix in chunks
        print("Computing covariance matrix...")
        embeddings = torch.cat(all_embeddings, dim=0).to(self.device)
        self.covs = self.compute_covariance_chunk(embeddings)
        
        del embeddings, all_embeddings
        gc.collect()
        
        print("Applying numerical stability...")
        # Add small value to diagonal for numerical stability
        self.covs += torch.eye(self.covs.shape[0], device=self.device) * 1e-5
        
        print("Statistical parameters computed successfully!")

        # Second pass: compute training score normalization statistics
        print("\nComputing training score normalization statistics (second pass)...")
        running_score_mean = 0.0
        running_score_sq_mean = 0.0
        num_score_maps = 0

        with torch.no_grad():
            target_shape = self.shapes[0][2:]
            # Iterate again; optionally subsample to limit cost
            for batch_index, (images, _) in enumerate(train_loader):
                images = images.to(self.device)
                features = self.forward(images)
                embedding = self.process_batch_embeddings(features, target_shape)

                diff = (embedding - self.means).unsqueeze(1)
                inv_covs = torch.linalg.inv(self.covs)
                mahala_dist = torch.sum((torch.matmul(diff, inv_covs) * diff), dim=2)
                score = torch.sqrt(mahala_dist)

                h, w = target_shape
                score = score.reshape(h, w)

                # Accumulate mean and squared mean for variance
                current_mean = score.mean().item()
                current_sq_mean = (score ** 2).mean().item()

                running_score_mean += current_mean
                running_score_sq_mean += current_sq_mean
                num_score_maps += 1

                # Clear tensors
                del images, features, embedding, diff, inv_covs, mahala_dist, score
                gc.collect()

                # Optional light subsampling to speed up on very large sets
                if num_score_maps >= 200:
                    break

        if num_score_maps == 0:
            # Fallback to zeros if something went wrong
            self.score_mean = torch.tensor(0.0, device=self.device)
            self.score_std = torch.tensor(1.0, device=self.device)
        else:
            mean_val = running_score_mean / num_score_maps
            sq_mean_val = running_score_sq_mean / num_score_maps
            var_val = max(1e-6, sq_mean_val - (mean_val ** 2))
            self.score_mean = torch.tensor(mean_val, device=self.device)
            self.score_std = torch.tensor(var_val ** 0.5, device=self.device)

        print(f"Training score stats -> mean: {self.score_mean.item():.6f}, std: {self.score_std.item():.6f}")
        return self.means, self.covs
    
    def compute_score(self, image):
        self.eval()
        with torch.no_grad():
            image = image.to(self.device)
            features = self.forward(image)
            
            # Get the target shape from the first feature map
            target_shape = features[0].shape[2:]
            
            # Process features
            embedding = self.process_batch_embeddings(features, target_shape)
            
            # Calculate Mahalanobis distance
            diff = (embedding - self.means).unsqueeze(1)
            inv_covs = torch.linalg.inv(self.covs)
            
            # Calculate Mahalanobis distance efficiently
            mahala_dist = torch.sum((torch.matmul(diff, inv_covs) * diff), dim=2)
            
            # Calculate raw score as sqrt of Mahalanobis distance
            score = torch.sqrt(mahala_dist)
            
            # Reshape score to match feature map dimensions
            h, w = target_shape
            score = score.reshape(h, w)
            
            # Enhanced score normalization for better pneumonia detection
            if self.score_mean is None or self.score_std is None:
                temp_mean = score.mean()
                temp_std = score.std() + 1e-6
                score = (score - temp_mean) / temp_std
            else:
                # Use robust normalization
                score = (score - self.score_mean) / (self.score_std + 1e-6)
            
            # Apply non-linear transformation to enhance differences
            lower_bound = -4.0  # Increased range for better separation
            upper_bound = 4.0
            score = torch.clamp(score, min=lower_bound, max=upper_bound)
            score = (score - lower_bound) / (upper_bound - lower_bound)
            
            # Apply exponential scaling to emphasize anomalies
            score = torch.pow(score, 1.5)  # Increased power for better contrast
            
            # Apply additional emphasis to high-score regions
            high_score_mask = score > 0.6
            score[high_score_mask] = torch.pow(score[high_score_mask], 1.2)
            
            return score

def train_padim(train_loader, val_loader=None):
    print("Initializing PaDiM model...")
    model = PaDiM()
    
    # Clear GPU memory before training
    torch.cuda.empty_cache()
    gc.collect()
    
    print("Training started. This may take 15-25 minutes depending on your hardware.")
    print("Step 1/2: Feature extraction and embedding computation...")
    
    import time
    start_time = time.time()
    
    # Update the fit method to show progress
    model.fit(train_loader, progress_bar=True)
    
    end_time = time.time()
    training_time = end_time - start_time
    print(f"\nTraining completed in {training_time/60:.2f} minutes")
    
    return model

def save_model(model, path):
    try:
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model with torch.save
        state_dict = model.state_dict()
        checkpoint = {
            'means': model.means.cpu() if model.means is not None else None,
            'covs': model.covs.cpu() if model.covs is not None else None,
            'score_mean': model.score_mean.cpu() if model.score_mean is not None else None,
            'score_std': model.score_std.cpu() if model.score_std is not None else None,
            'state_dict': state_dict
        }
        torch.save(checkpoint, path)
        print(f"Model saved successfully to {path}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        raise