import os
import torch
import torch.nn as nn
import torch.optim as optim
import math
import yaml
import numpy as np
import random
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(
        self,
        model,
        config,
        device=None,
        checkpoint_path=None,
        experiment_name: str = "",
        logger_step: int = 100,
        output_dir: str = ""
    ):
        self.model = model
        self.config = config
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.best_val_loss = math.inf
        self.start_epoch = 0
        self.logger_step = logger_step
        
        self.beta_start = 1
        self.beta_end = self.config["training"]['beta']
        self.beta_warmup_epochs = self.config["training"]['beta_warmup_epochs'] 
    
        if self.config["training"]['optimizer'] == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config["training"]['learning_rate'],
                weight_decay=self.config["training"]['weight_decay']
            )
        elif self.config["training"]['optimizer'] == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config["training"]['learning_rate'],
                momentum=self.config["training"].get('momentum', 0),
                weight_decay=self.config["training"]['weight_decay'],
                nesterov=True
            )
        
        self.reduction_kl_fn = None

        if self.config["training"]['reduction'] == "mean":
            self.reduction_kl_fn = torch.mean
        elif self.config["training"]['reduction'] == "sum":
           self.reduction_kl_fn = torch.sum     

        if self.config['model']['activation'] == "sigmoid":
            self.reconstruction_criterion = nn.BCELoss(reduction=self.config["training"]['reduction'])
        else:
            self.reconstruction_criterion = nn.MSELoss(reduction=self.config["training"]['reduction'])
            
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            min_lr=5e-5
        )
        
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        else:
            timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')
        
            exp_name = experiment_name or self.config["experiment"]['name']
            version = self.config["experiment"]['version']
            
            self.exp_dir = os.path.join(
                output_dir,
                f"{exp_name}_V{version}_{timestamp}"
            )
            
            os.makedirs(self.exp_dir, exist_ok=True)
            
        self.logger = self.setup_logging(self.exp_dir)
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, 'tensorboard'))
        
        self.logger.info(f"Starting experiment: {self.config["experiment"]['name']}_V{self.config["experiment"]['version']}")        
        self.logger.info(f"Using optimizer: {self.config["training"]['optimizer']}, "
                f"lr: {self.config["training"]['learning_rate']}, "
                f"momentum: {self.config["training"].get('momentum', 0)}, "
                f"weight_decay: {self.config["training"]['weight_decay']}")
        
        self.logger.info(f"Using device: {self.device}")
        self.set_seed(self.config["experiment"]['seed'])
        self.logger.info(f"Set random seed: {self.config["experiment"]['seed']}")

        config_save_path = os.path.join(self.exp_dir, 'config.yml')
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f)
    
    def get_beta(self, epoch):
        """Get beta value for current epoch with warmup scheduling."""
        if epoch < self.beta_warmup_epochs:
            return self.beta_start + (self.beta_end - self.beta_start) * (epoch / self.beta_warmup_epochs)
        return self.beta_end
    
    def set_seed(self, seed):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def generate_samples(self, n, path):
        h, w = 28, 28 # hardcoded for MNIST
        self.model.eval()

        from PIL import Image

        with torch.no_grad():
            z = torch.randn((n, self.config['model']['latent_dim'])).to(self.device)

            images = self.model.decoder(z, self.config['model']['activation']).view(n, h, w)

            for image_idx in range(images.shape[0]):
                image = images[image_idx].cpu().detach().numpy()
                
                image = np.clip(image, 0, 1)
                image = (image * 255).astype(np.uint8)

                img = Image.fromarray(image, mode='L')
                img.save(os.path.join(path, f'generated_sample_{image_idx}.png'))
          

    def setup_logging(self, exp_dir):
        """Setup logging to file and console."""
        import logging

        logger = logging.getLogger('trainer')
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers if any
        if logger.handlers:
            logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(os.path.join(exp_dir, 'training.log'))
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def train_one_epoch(self, train_loader, epoch):
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0
        kl_div_total = 0
        reconstruction_loss_total = 0
        
        current_beta = self.get_beta(epoch)
        
        for batch_idx, (inputs, _ ) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()
        
            outputs, mu, logvar = self.model(inputs)
             
            reconstruction_loss = self.reconstruction_criterion(outputs.squeeze(), inputs.squeeze())
            
            kl_div = -0.5 * self.reduction_kl_fn(1 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction_loss + current_beta * kl_div  
         
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            reconstruction_loss_total += reconstruction_loss.item()
            kl_div_total += kl_div.item()

            if batch_idx % self.logger_step == 0 and batch_idx != 0:
                step = epoch * len(train_loader) + batch_idx    

                batch_size = self.config["training"]["batch_size"]
                batch_loss = self._get_per_batch_loss(loss.item(), batch_size)
                batch_reconstruction_loss = self._get_per_batch_loss(reconstruction_loss.item(), batch_size)
                batch_kl_div = self._get_per_batch_loss(kl_div.item(), batch_size)

                self.writer.add_scalar('Training/BatchLoss', batch_loss, step)
                self.writer.add_scalar('Training/BatchReconstructionLoss', batch_reconstruction_loss, step)
                self.writer.add_scalar('Training/BatchKLDivergence', batch_kl_div, step)                
                self.writer.add_scalar('Training/Beta', current_beta, step)
                
                log_msg = f'Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {batch_loss:.4f} | Loss_rec: {batch_reconstruction_loss:.4f} | kl_div: {batch_kl_div:.4f} | beta: {current_beta:.4f}'
                self.logger.info(log_msg)
        
        avg_loss = self._calculate_average_loss(total_loss, train_loader)
        avg_reconstruction_loss = self._calculate_average_loss(reconstruction_loss_total, train_loader)
        avg_kl_div = self._calculate_average_loss(kl_div_total, train_loader)

        self.writer.add_scalar('Training/EpochLoss', avg_loss, epoch)
        self.writer.add_scalar('Training/EpochReconstructionLoss', avg_reconstruction_loss, epoch)
        self.writer.add_scalar('Training/EpochKLDivergence', avg_kl_div, epoch)
        self.writer.add_scalar('Training/EpochBeta', current_beta, epoch)

        return avg_loss, avg_reconstruction_loss, avg_kl_div

    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        kl_div_total = 0
        reconstruction_loss_total = 0
        
        current_beta = self.get_beta(epoch)

        with torch.no_grad():
            for inputs, _ in val_loader:                
                inputs = inputs.to(self.device)
                outputs, mu, logvar = self.model(inputs)
            
                reconstruction_loss = self.reconstruction_criterion(outputs.squeeze(), inputs.squeeze())
                kl_div = -0.5 * self.reduction_kl_fn(1 + logvar - mu.pow(2) - logvar.exp())
                loss = reconstruction_loss + current_beta * kl_div
            
                total_loss += loss.item()
                reconstruction_loss_total += reconstruction_loss.item()
                kl_div_total += kl_div.item()

        avg_loss = self._calculate_average_loss(total_loss, val_loader)
        avg_reconstruction_loss = self._calculate_average_loss(reconstruction_loss_total, val_loader)
        avg_kl_div = self._calculate_average_loss(kl_div_total, val_loader)

        self.writer.add_scalar('Validation/Loss', avg_loss, epoch)
        self.writer.add_scalar('Validation/ReconstructionLoss', avg_reconstruction_loss, epoch)
        self.writer.add_scalar('Validation/KLDivergence', avg_kl_div, epoch)
    
        return avg_loss, avg_reconstruction_loss, avg_kl_div

    def save_checkpoint(self, epoch, val_loss):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        checkpoint_path = os.path.join(self.exp_dir, f'checkpoint_epoch_{epoch}.pkl')
        torch.save(checkpoint, checkpoint_path)     
        self.logger.info(f"Saved checkpoint at {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        if not os.path.exists(checkpoint_path):
            self.logger.info(f"Checkpoint {checkpoint_path} does not exist.")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_loss = checkpoint.get('val_loss', 0)
        
        loaded_config = checkpoint.get('config', None)
        if loaded_config:
            self.config = loaded_config
        
        self.exp_dir = os.path.dirname(checkpoint_path)
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resumed training from epoch {self.start_epoch}")
    
    def train(self, train_dataset, val_dataset, collate_fn=None):
        """Train model for multiple epochs."""
        
        self.logger.info("Starting training...")
        self.logger.info(f"Training dataset size: {len(train_dataset)}")
        self.logger.info(f"Validation dataset size: {len(val_dataset)}")

        batch_size = self.config["training"]["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(self.start_epoch, self.config["training"]["epochs"]):
            self.logger.info(f'\nEpoch {epoch+1}/{self.config["training"]["epochs"]}')

            train_loss, train_reconstruction_loss, train_kl_div = self.train_one_epoch(train_loader, epoch)

            val_loss, val_reconstruction_loss, val_kl_div = self.validate(val_loader, epoch)

            self.logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Train Reconstruction Loss: {train_reconstruction_loss:.4f} | Train KL Divergence: {train_kl_div:.4f}")
            self.logger.info(f"Epoch {epoch} | Valid Loss: {val_loss:.4f} | Valid Reconstruction Loss: {val_reconstruction_loss:.4f} | Valid KL Divergence: {val_kl_div:.4f}")

            self.scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)

            if epoch % 10 == 0 and epoch != 0:
                image_samples_dir = os.path.join(
                    self.exp_dir,
                    "generated_samples",
                    f"{epoch}_epoch"
                )
                os.makedirs(image_samples_dir, exist_ok=True)
                self.generate_samples(5, image_samples_dir)
                self.save_checkpoint(epoch, val_loss)    

            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Training/LearningRate', current_lr, epoch)
            self.logger.info(f'Current learning rate: {current_lr}')
        
        self.writer.close()
        self.logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.4f}")

        return self.best_val_loss
    
    def _calculate_average_loss(self, total_loss, data_loader):
        if self.config["training"]['reduction'] == "sum":
            # For sum reduction, divide by total number of samples
            return total_loss / len(data_loader.dataset)
        elif self.config["training"]['reduction'] == "mean":
            # For mean reduction, divide by number of batches
            return total_loss / len(data_loader)
        else:
            raise ValueError(f"Unsupported reduction type: {self.config['training']['reduction']}")
    
    def _get_per_batch_loss(self, batch_loss, batch_size):
        
        if self.config["training"]['reduction'] == "sum":
            # For sum reduction, divide by batch size to get per-sample loss
            return batch_loss / batch_size
        elif self.config["training"]['reduction'] == "mean":
            # For mean reduction, loss is already per-sample
            return batch_loss
        else:
            raise ValueError(f"Unsupported reduction type: {self.config['training']['reduction']}")
