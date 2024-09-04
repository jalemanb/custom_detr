#!/usr/bin/env python3

import torch
from torch import nn
import torchvision.transforms as T
from presnet import PResNet
from hybrid_encoder import HybridEncoder
from rtdetrv2_decoder import RTDETRTransformerv2
from rtdetr_postprocessor import RTDETRPostProcessor
from matcher import HungarianMatcher
import cv2
import matplotlib.pyplot as plt
import time


def main():

    # Setup Feature Extractor
    presnet = PResNet(depth= 34,
                    variant = 'd',
                    freeze_at = -1,
                    return_idx = [1, 2, 3],
                    num_stages = 4,
                    freeze_norm = False,
                    pretrained = True )

    encoder = HybridEncoder(  in_channels = [128, 256, 512],
                            feat_strides = [8, 16, 32],
                            # intra
                            hidden_dim = 256,
                            use_encoder_idx = [2],
                            num_encoder_layers = 1,
                            nhead = 8,
                            dim_feedforward = 1024,
                            dropout = 0.,
                            enc_act = 'gelu' ,
                            # cross
                            expansion = 0.5,
                            depth_mult = 1,
                            act = 'silu')
    
    decoder = RTDETRTransformerv2(feat_channels = [256, 256, 256],
                                feat_strides = [8, 16, 32],
                                hidden_dim = 256,
                                num_levels = 3,
                                num_layers = 4,
                                num_queries = 300,
                                num_denoising = 100,
                                label_noise_ratio = 0.5,
                                box_noise_scale = 1.0, # 1.0 0.4
                                eval_idx = 2,
                                # NEW
                                num_points = [4, 4, 4], # [3,3,3] [2,2,2]
                                cross_attn_method = 'default', # default, discrete
                                query_select_method = 'default' # default, agnostic 
                                )
    
    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.backbone = presnet
            self.encoder = encoder
            self.decoder = decoder
            
        def forward(self, images, targets = None):
            features = self.backbone(images)
            features = self.encoder(features)
            out = self.decoder(features, targets)
            return out

    detr = Model()
    detr.train()
    detr.cuda()

    # Loading the dataset
    from dataset import PersonDataset
    from torch.utils.data import random_split, DataLoader

    root_dir = '/homes/jalemangallegos/datasets/person'
    sequence_list = [f'person-{i}' for i in range(1, 21)]
    dataset = PersonDataset(root_dir=root_dir, sequence_list=sequence_list, img_transform_size=(640, 640), template_transform_size=(256, 256), max_num_templates=10, max_detections = 300)

    # Define the lengths for training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # The rest for validation

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 6
    # Optionally, create DataLoader objects for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


    import torch.optim as optim
    from rtdetrv2_criterion import RTDETRCriterionv2
    from torch.cuda.amp import GradScaler, autocast

    # Defining the Optimizer
    learning_rate = 0.0001
    optimizer = optim.AdamW(detr.parameters(), lr=learning_rate)

    # Defining the criterion
    losses = ['vfl', 'boxes']
    weight_dict={'cost_class': 2, 'cost_bbox':5, 'cost_giou':2}
    matcher = HungarianMatcher(use_focal_loss=False, weight_dict = weight_dict)
    weight_dict = {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2}
    criterion = RTDETRCriterionv2(losses=losses, weight_dict=weight_dict, matcher=matcher)
    criterion.train()

    # Number of epochs
    num_epochs = 20

    # Initialize the GradScaler
    scaler = GradScaler()


    import wandb
    # Weights and biases config
    wandb.init(
        # set the wandb project where this run will be logged
        project="UltimateModel",

        # track hyperparameters and run metadata
        config={
        "learning_rate": learning_rate,
        "architecture": "RT-DETR",
        "dataset": "Lasot_Person",
        "epochs": num_epochs,
        }
    )


    # Define the paths for saving models
    checkpoint_path = "detr_checkpoint.pth"
    best_model_path = "detr_best_model.pth"

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training Phase
        detr.train()  # Set model to training mode
        running_loss = 0.0
        total_batches = len(train_loader)
        total_samples = len(train_loader.dataset)
        samples_processed = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch['img']  # Input images
            bounding_boxes_batch = batch['bounding_boxes']  # Target bounding boxes for the whole batch
            num_boxes_batch = batch['num_boxes']  # Number of valid bounding boxes for the whole batch

            batch_size = images.size(0)
            samples_processed += batch_size

            # Create a list of dictionaries for targets
            targets = []
            for i in range(len(images)):
                labels = torch.full((300,), 5, dtype=torch.long)  # Initialize all as 5 ("nothing" class)
                if num_boxes_batch[i].item() > 0:
                    labels[:num_boxes_batch[i]] = 1  # Mark the first num_boxes elements as 1 ("person" class)
                targets.append({'labels': labels, 'boxes': bounding_boxes_batch[i]})

            # Move data to GPU if available
            if torch.cuda.is_available():
                images = images.cuda()
                targets = [{'labels': target['labels'].cuda(), 'boxes': target['boxes'].cuda()} for target in targets]

            # Zero the parameter gradients
            optimizer.zero_grad()

            with autocast():  # Use automatic mixed precision
                # Forward pass
                outputs = detr(images, targets=targets)

                # Calculate the loss
                loss_dict = criterion(outputs, targets)
                loss: torch.Tensor = sum(loss_dict.values())
                loss_dict["total_loss"] = loss

            # Log the training loss
            wandb.log({f"train_{key}": value for key, value in loss_dict.items()})

            # Scale the loss before backpropagation
            scaler.scale(loss).backward()

            # Unscale the gradients and step the optimizer
            scaler.step(optimizer)
            scaler.update()

            # Accumulate loss
            running_loss += loss.item()

            # Calculate the percentage of samples processed
            percent_complete = 100 * (samples_processed / total_samples)

            # Print training progress
            print(f"Epoch [{epoch+1}/{num_epochs}] - Iteration [{batch_idx+1}/{total_batches}] "
                f"Loss: {loss.item():.4f} - Progress: {percent_complete:.2f}%")

        avg_train_loss = running_loss / total_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Training Loss: {avg_train_loss:.4f}")

        # Validation Phase
        detr.eval()  # Set model to evaluation mode
        val_running_loss = 0.0
        samples_processed = 0
        with torch.no_grad():  # Disable gradient computation for validation
            for batch_idx, batch in enumerate(val_loader):
                images = batch['img']
                bounding_boxes_batch = batch['bounding_boxes']
                num_boxes_batch = batch['num_boxes']

                batch_size = images.size(0)
                samples_processed += batch_size

                targets = []
                for i in range(len(images)):
                    labels = torch.full((300,), 5, dtype=torch.long)
                    if num_boxes_batch[i].item() > 0:
                        labels[:num_boxes_batch[i]] = 1
                    targets.append({'labels': labels, 'boxes': bounding_boxes_batch[i]})

                if torch.cuda.is_available():
                    images = images.cuda()
                    targets = [{'labels': target['labels'].cuda(), 'boxes': target['boxes'].cuda()} for target in targets]

                # Forward pass
                with autocast():
                    outputs = detr(images, targets=targets)
                    loss_dict = criterion(outputs, targets)
                    loss: torch.Tensor = sum(loss_dict.values())
                    loss_dict["total_loss"] = loss

                # Log the validation loss
                wandb.log({f"valid_{key}": value for key, value in loss_dict.items()})

                val_running_loss += loss.item()

        avg_val_loss = val_running_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Validation Loss: {avg_val_loss:.4f}")

        # Logging to wandb
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # Save the current checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': detr.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }, checkpoint_path)

        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(detr.state_dict(), best_model_path)
            print(f"Best model saved with validation loss: {best_val_loss:.4f}")

    print('Finished Training')
    wandb.finish()


if __name__ == "__main__":
    main()
