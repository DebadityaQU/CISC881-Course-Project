import torch
import wandb
import torch.nn as nn
from data import get_dataset
import torch.nn.functional as F
from autoencoder import Encoder, Decoder
from torchvision import models, transforms
from tqdm.auto import tqdm
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

def MMD(x, y, kernel="rbf", device="cuda:0"):
    """
    Code from: https://www.onurtunali.com/ml/2019/03/08/maximum-mean-discrepancy-in-machine-learning.html
    """
    
    x = torch.flatten(x, start_dim=1)
    y = torch.flatten(y, start_dim=1)
    
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)

class VGGLoss(nn.Module):
    """
    Code from: https://github.com/crowsonkb/vgg_loss/blob/master/vgg_loss.py
    """
    
    models = {'vgg16': models.vgg16, 'vgg19': models.vgg19}

    def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features[:layer+1]
        self.model.eval()
        self.model.requires_grad_(False)

    def get_features(self, input):
        return self.model(self.normalize(input))

    def train(self, mode=True):
        self.training = mode

    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)

def train_HER(device="cuda:0", NUM_EPOCHS=10000): 

    wandb.init(
        project="CISC881-courseproject",
        entity="debadityashome",
        id="Perceptual+MMD+L1 AE HER compression"
    )

    train_dataset = get_dataset(split="train")
    test_dataset = get_dataset(split="val")

    training_generator = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
    )

    test_generator = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=True
    )

    perceptual_loss = VGGLoss().to(device)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()],
        lr=1e-4,
    )

    ## TRAIN MODEL ##
    counter = 0

    psnr_train_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_train_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    psnr_test_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_test_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    ## Iterate across training loop
    for epoch in range(NUM_EPOCHS):
        
        encoder.train()
        decoder.train()
        
        running_loss = 0.0
        print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
        
        for (her_img, _) in tqdm(training_generator):
            
            optimizer.zero_grad()
            
            her_img = her_img.to(device)
            
            z = encoder(her_img)
            pred_her = decoder(z)
            
            loss = nn.L1Loss()(pred_her, her_img) + MMD(pred_her, her_img) + perceptual_loss(pred_her, her_img) #+ focal_freq_loss(pred_her, her_img) + edge_aware_loss(pred_her, her_img)
            
            psnr_train_metric.update(pred_her, her_img)
            ssim_train_metric.update(pred_her, her_img)
            
            wandb.log({'loss': loss})  # Log loss to wandb
            
            loss.backward()
            
            optimizer.step()
        
        print(f"\nPSNR on train set = {psnr_train_metric.compute().item()}\n")
        print(f"\nSSIM on train set = {ssim_train_metric.compute().item()}\n")
        
        counter += 1
        
        torch.save(encoder.state_dict(), "perceptualMMD_encoder_HER_L1.pth")
        torch.save(decoder.state_dict(), "perceptualMMD_decoder_HER_L1.pth")
        
        ## INFERENCE ##
        
        encoder.eval()
        decoder.eval()
        
        for (her_img, _) in tqdm(test_generator):
            
            her_img = her_img.to(device)
            
            z = encoder(her_img)
            pred_her = decoder(z)
            
            psnr_test_metric.update(pred_her, her_img)
            ssim_test_metric.update(pred_her, her_img)
        
        print(f"\nPSNR on test set = {psnr_test_metric.compute().item()}\n")
        print(f"\nSSIM on test set = {ssim_test_metric.compute().item()}\n")
        
        pred = pred_her.cpu().detach()
        her_img = her_img.cpu().detach()
        
        wandb.log(
            {
                "PSNR-train": psnr_train_metric.compute().item(),
                "PSNR-test": psnr_test_metric.compute().item(),
                "SSIM-train": ssim_train_metric.compute().item(),
                "SSIM-test": ssim_test_metric.compute().item()
            }
        )
        
        # save just one image per epoch
        wandb.log(
            {'pred-input': [wandb.Image(pred[0, :, :, :].permute(1, 2, 0).numpy()), wandb.Image(her_img[0, :, :, :].permute(1, 2, 0).numpy())]}
        )
        
        psnr_train_metric.reset()
        ssim_train_metric.reset()
        
        psnr_test_metric.reset()
        ssim_test_metric.reset()

def train_IHC(device="cuda:0", NUM_EPOCHS=10000): 

    wandb.init(
        project="CISC881-courseproject",
        entity="debadityashome",
        id="Perceptual+MMD+L1 AE IHC compression"
    )

    train_dataset = get_dataset(split="train")
    test_dataset = get_dataset(split="val")

    training_generator = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
    )

    test_generator = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=True
    )

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)

    optimizer = torch.optim.AdamW(
        [*encoder.parameters(), *decoder.parameters()],
        lr=1e-4,
    )

    ## TRAIN MODEL ##
    counter = 0

    psnr_train_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_train_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    psnr_test_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_test_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    ## Iterate across training loop
    for epoch in range(NUM_EPOCHS):
        
        encoder.train()
        decoder.train()
        
        running_loss = 0.0
        print('Epoch {}/{}'.format(epoch + 1, NUM_EPOCHS))
        
        for (_, ihc_img) in tqdm(training_generator):
            
            optimizer.zero_grad()
            
            ihc_img = ihc_img.to(device)
            
            z = encoder(ihc_img)
            pred_ihc = decoder(z)
            
            loss = nn.L1Loss()(pred_ihc, ihc_img) + MMD(pred_ihc, ihc_img) + perceptual_loss(pred_ihc, ihc_img)
            
            psnr_train_metric.update(pred_ihc, ihc_img)
            ssim_train_metric.update(pred_ihc, ihc_img)
            
            wandb.log({'loss': loss})  # Log loss to wandb
            
            loss.backward()
            
            optimizer.step()
        
        print(f"\nPSNR on train set = {psnr_train_metric.compute().item()}\n")
        print(f"\nSSIM on train set = {ssim_train_metric.compute().item()}\n")
        
        counter += 1
        
        torch.save(encoder.state_dict(), "perceptualMMD_encoder_IHC_L1.pth")
        torch.save(decoder.state_dict(), "perceptualMMD_decoder_IHC_L1.pth")
        
        ## INFERENCE ##
        
        encoder.eval()
        decoder.eval()
        
        for (_, ihc_img) in tqdm(test_generator):
            
            ihc_img = ihc_img.to(device)
            
            z = encoder(ihc_img)
            pred_ihc = decoder(z)
            
            psnr_test_metric.update(pred_ihc, ihc_img)
            ssim_test_metric.update(pred_ihc, ihc_img)
        
        print(f"\nPSNR on test set = {psnr_test_metric.compute().item()}\n")
        print(f"\nSSIM on test set = {ssim_test_metric.compute().item()}\n")
        
        pred = pred_ihc.cpu().detach()
        ihc_img = ihc_img.cpu().detach()
        
        wandb.log(
            {
                "PSNR-train": psnr_train_metric.compute().item(),
                "PSNR-test": psnr_test_metric.compute().item(),
                "SSIM-train": ssim_train_metric.compute().item(),
                "SSIM-test": ssim_test_metric.compute().item()
            }
        )
        
        # save just one image per batch
        wandb.log(
            {'pred-input': [wandb.Image(pred[0, :, :, :].permute(1, 2, 0).numpy()), wandb.Image(ihc_img[0, :, :, :].permute(1, 2, 0).numpy())]}
        )
        
        psnr_train_metric.reset()
        ssim_train_metric.reset()
        
        psnr_test_metric.reset()
        ssim_test_metric.reset()

if __name__ == "__main__":

    train_HER()
    train_IHC()