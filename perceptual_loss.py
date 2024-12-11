#Here's the set of steps and instructions, inclusive of code, for implementing perceptual loss. I didn't just use a rudimentary model, but accounted for a feature extractor that could be spectrogram or CLAP based (the code is generalized to acclimate for both).                                       
# Step 1 is defining the Perceptual Loss Module: Create Class for perceptual loss and as added functionality, provide support for any feature extractor. Create a new file named audio_diffusion_pytorch/perceptual_loss.py and the contents should be as follows:                                                 

import torch
import torch.nn as nn

class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor, layers=None, weight=1.0):
        """
        - "feature_extractor": A pre-trained model to extract features.
        - "layers": Specify layers of the extractor for comparison.
        - "weight": Controls the contribution of this loss term.
        """
        super().__init__()
        self.feature_extractor = feature_extractor
        self.layers = layers
        self.weight = weight

    def forward(self, generated, target):
        """
        - "generated": Model output.
        - "target": Ground truth.
        """
        #feature extraction from generated and target audio 
        with torch.no_grad():
            gen_features = self.feature_extractor(generated, layers=self.layers)
            target_features = self.feature_extractor(target, layers=self.layers)
        #determine mse
        loss = 0
        for gen_feat, target_feat in zip(gen_features, target_features):
            loss += torch.mean((gen_feat - target_feat) ** 2)

        return self.weight * loss

#Now comes Step 2: As y'all can see in the GitHub repo, there's a class called VDiffusion that handles diffusion loss. What we're doing now is changing it such that it accounts for perceptual loss and it calculates it in addition to the natural diffusion loss.                                                                  
# To do this we open audio_diffusion_pytorch/diffusion.py and place the following in the VDiffusion Class:                                                                       

from .perceptual_loss import PerceptualLoss

class VDiffusion(Diffusion):
    def __init__(self, net, sigma_distribution=UniformDistribution(), loss_fn=None, perceptual_loss=None):
        """
        Extends VDiffusion to include perceptual loss.
        - "net": The neural network model.
        - "sigma_distribution": Defines noise distribution.
        - "loss_fn": Standard loss function (e.g., MSE).
        - "perceptual_loss": Instance of PerceptualLoss.
        """
        super().__init__()
        self.net = net
        self.sigma_distribution = sigma_distribution
        self.loss_fn = loss_fn or F.mse_loss
        self.perceptual_loss = perceptual_loss

    def forward(self, x, target, **kwargs):
        """
        Computes combined diffusion and perceptual loss.
        """
        #adding noise
        batch_size, device = x.shape[0], x.device
        sigmas = self.sigma_distribution(num_samples=batch_size, device=device)
        sigmas_batch = extend_dim(sigmas, dim=x.ndim)
        noise = torch.randn_like(x)
        alphas, betas = self.get_alpha_beta(sigmas_batch)
        x_noisy = alphas * x + betas * noise
        v_target = alphas * noise - betas * x

        # Predict velocity and calculate standard loss
        v_pred = self.net(x_noisy, sigmas, **kwargs)
        loss = self.loss_fn(v_pred, v_target)

        if self.perceptual_loss:
            perceptual_loss = self.perceptual_loss(x_noisy, target)
            loss += perceptual_loss

        return loss

#This allows the model to optimize for both physical accuracy (via MSE) and perceptual similarity, improving overall performance.                               
# Now Step 3 although not necessary is to establish a feature extractor:                                                                                                                                
#Step 4 is to integrate perceptual loss into the training loop. So we're effectively adding the feature extractor and perceptual loss to the pipeline. This perceptual loss instance is passed onto the VDiffusion instance mentioned earlier.                                                                                 
# Here's the Training Script (As a bonus used dynamic weighting and conducted logging. FOR LOGGING PLEASE REMEMBER TO USE wandb or tensorboard):                                                                                                                                                                                                                     
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb 

from audio_diffusion_pytorch.diffusion import VDiffusion
from audio_diffusion_pytorch.models import DiffusionModel
from audio_diffusion_pytorch.perceptual_loss import PerceptualLoss
from your_dataset import YourDataset  # Replace with your dataset

wandb.init(project="perceptual-diffusion", name="training-run")

dataset = YourDataset(data_path="path/to/data", transform=None)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)

feature_extractor = PretrainedFeatureExtractor(pretrained=True)
perceptual_loss = PerceptualLoss(
    feature_extractor=feature_extractor,
    layers=None,
    weight=1.0
)

class DynamicWeighting:
    def __init__(self, initial_weight=1.0, max_weight=10.0, ramp_epochs=5):
        self.initial_weight = initial_weight
        self.max_weight = max_weight
        self.ramp_epochs = ramp_epochs

    def get_weight(self, current_epoch):
        # Linear ramp up over epochs
        if current_epoch >= self.ramp_epochs:
            return self.max_weight
        return self.initial_weight + (self.max_weight - self.initial_weight) * (current_epoch / self.ramp_epochs)

dynamic_weighting = DynamicWeighting()

model = DiffusionModel(
    net_t=YourModel,
    diffusion_t=lambda **kwargs: VDiffusion(
        net=YourModel(),
        loss_fn=nn.MSELoss(),
        perceptual_loss=perceptual_loss
    )
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 10
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    perceptual_loss_sum = 0

    perceptual_loss.weight = dynamic_weighting.get_weight(epoch)
    wandb.log({"perceptual_loss_weight": perceptual_loss.weight})

    for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
        inputs, targets = batch["input"].to(device), batch["target"].to(device)

        optimizer.zero_grad()

        loss = model(inputs, target=targets)

        #logploss
        with torch.no_grad():
            current_perceptual_loss = perceptual_loss(inputs, targets)
            perceptual_loss_sum += current_perceptual_loss.item()

       
        loss.backward()
        optimizer.step()

        #totalloss                                                                                                                                                                                                  
        epoch_loss += loss.item()
    #avgloss
    avg_loss = epoch_loss / len(dataloader)
    avg_perceptual_loss = perceptual_loss_sum / len(dataloader)

    # Log metrics to WandB
    wandb.log({
        "epoch": epoch + 1,
        "avg_loss": avg_loss,
        "avg_perceptual_loss": avg_perceptual_loss
    })

    print(f"Epoch {epoch + 1}: Avg Loss = {avg_loss:.4f}, Avg Perceptual Loss = {avg_perceptual_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), f"model_epoch_{epoch + 1}.pth")

print("Training complete!")
wandb.finish()