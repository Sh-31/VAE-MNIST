import torch
import torchvision.datasets as datasets
from torchvision import transforms
from utils import load_config
from model_fnn import VariationalAutoEncoder
from trainer import Trainer

# Data preprocessing with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(
    root='data',
    train=True,
    transform=transform,
    target_transform=None,
    download=True
)

test_dataset = datasets.MNIST(
    root='data',
    train=False,
    transform=transform,
    target_transform=None,
    download=True
)

if __name__ == "__main__":
    
    config = load_config("config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VariationalAutoEncoder(
        input_dim=config['model']['input_dim'],
        latent_dim=config['model']['latent_dim'],
        activation=config['model']['activation'],
    )
    
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        checkpoint_path=None,
        experiment_name="VAE-MNIST",
        logger_step = 100,
        output_dir="outputs"
    )

    trainer.train(train_dataset=train_dataset, val_dataset=test_dataset)
