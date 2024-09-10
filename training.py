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
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from torch_ema import ExponentialMovingAverage
num_classes = 1

# Define your drawing function to draw bounding boxes and scores
def draw_predictions(image, boxes, scores, score_threshold=0.5):
    h, w = image.shape[:2]
    
    for i, score in enumerate(scores):
        if score > score_threshold:
            # Convert bbox format from (x, y, w, h) in unit scale to (x1, y1, x2, y2) in pixel scale
            x, y, bw, bh = boxes[i]
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
            
            # Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw the objectness score
            label = f"Objectness: {score[0]:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return image

# --- Warmup Scheduler Setup ---
def lr_lambda(current_step):
    warmup_steps = 2000  # Number of steps for the warmup phase
    start_factor = 0.001  # Start at 0.1% of the base learning rate
    
    # If the current step is in the warm-up phase, increase the LR linearly
    if current_step < warmup_steps:
        return start_factor + (1 - start_factor) * (current_step / warmup_steps)
    
    # After warm-up, keep the learning rate constant (or you could implement decay)
    return 1.0

def main():

    # Setup Feature Extractor
    presnet = PResNet(depth= 18,
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
                                num_layers = 3,
                                num_queries = 300,
                                num_denoising = 200,
                                label_noise_ratio = 0.5,
                                box_noise_scale = 1.0, # 1.0 0.4
                                eval_idx = -1,
                                # NEW
                                num_points = [4, 4, 4], # [3,3,3] [2,2,2]
                                cross_attn_method = 'default', # default, discrete
                                query_select_method = 'agnostic', # default, agnostic 
                                num_classes=1
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

    root_dir = '/media/enrique/Extreme SSD/person'
    sequence_list = [f'person-{i}' for i in range(1, 21)]
    sequence_list = ["person-9"] # Overfit on just one subdataset

    dataset = PersonDataset(root_dir=root_dir, sequence_list=sequence_list, img_transform_size=(640, 640), template_transform_size=(256, 256), max_num_templates=10, max_detections = 300)

    # Define the lengths for training and validation sets
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # The rest for validation

    # Split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 4
    # Optionally, create DataLoader objects for the training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    import torch.optim as optim
    from rtdetrv2_criterion import RTDETRCriterionv2
    from torch.cuda.amp import GradScaler, autocast
    from torch.optim.lr_scheduler import LambdaLR

    # Defining the Optimizer
    learning_rate = 0.0001
    optimizer = optim.AdamW(detr.parameters(), lr=learning_rate, weight_decay=0.0001)

    # Defining the criterion
    losses = ['vfl', 'boxes']
    weight_dict={'cost_class': 2, 'cost_bbox':5, 'cost_giou':2}
    matcher = HungarianMatcher(use_focal_loss=False, weight_dict = weight_dict, alpha = 0.75)
    weight_dict = {'loss_vfl': 1, 'loss_bbox': 5, 'loss_giou': 2}
    criterion = RTDETRCriterionv2(losses=losses, weight_dict=weight_dict, matcher=matcher, num_classes=num_classes)
    criterion.train()
    ema = ExponentialMovingAverage(detr.parameters(), decay=0.9999)

    # Number of epochs
    num_epochs = 100

    # Initialize the GradScaler
    scaler = GradScaler()
    scaler = None

    max_norm = 0.1

    # Create the LambdaLR scheduler
    lr_warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    exp_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

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
    checkpoint_path = "detr_checkpoint_desk.pth"
    output_image_path = "output_images"
    N = 100  # Save images every N iterations

    best_val_loss = float('inf')
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure the output directory exists
    os.makedirs(output_image_path, exist_ok=True)

    for epoch in range(num_epochs):
        # Training Phase
        detr.train()  # Set model to training mode
        running_loss = 0.0
        total_batches = len(dataloader)
        total_samples = len(dataloader.dataset)
        samples_processed = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch['img']  # Input images
            bounding_boxes_batch = batch['bounding_boxes']  # Target bounding boxes for the whole batch
            num_boxes_batch = batch['num_boxes']  # Number of valid bounding boxes for the whole batch

            batch_size = images.size(0)
            samples_processed += batch_size

            # Create a list of dictionaries for targets
            targets = []
            for i in range(len(images)):
                labels = torch.full((num_boxes_batch[i],), 0, dtype=torch.long) 
                targets.append({'labels': labels, 'boxes': bounding_boxes_batch[i][:num_boxes_batch[i],:]})

            # Move data to GPU if available
            if torch.cuda.is_available():
                images = images.cuda()
                targets = [{'labels': target['labels'].cuda(), 'boxes': target['boxes'].cuda()} for target in targets]

            if scaler is not None:
                with torch.autocast(device_type=str(device_str), cache_enabled=True):
                    outputs = detr(images, targets=targets)
                
                with torch.autocast(device_type=str(device_str), enabled=False):
                    loss_dict = criterion(outputs, targets)

                loss: torch.Tensor = sum(loss_dict.values()); loss_dict["total_loss"] = loss
                scaler.scale(loss).backward()

                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(detr.parameters(), max_norm)

                # Unscale the gradients and step the optimizer
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            else:
                outputs = detr(images, targets=targets)

                loss_dict = criterion(outputs, targets)

                loss: torch.Tensor = sum(loss_dict.values()); loss_dict["total_loss"] = loss
                optimizer.zero_grad()
                loss.backward()

                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(detr.parameters(), max_norm)

                optimizer.step()

            # Save output images every N iterations
            if batch_idx % N == 0:
                print("TIME TO DRAW SOME IMAGES")
                print("TIME TO DRAW SOME IMAGES")
                print("TIME TO DRAW SOME IMAGES")

                pred_logits = outputs["pred_logits"]  # [batch, max_detections, num_classes]
                pred_boxes = outputs["pred_boxes"]  # [batch, max_detections, 4]

                
                # Iterate over the batch
                for i in range(pred_logits.size(0)):  # Iterate through batch
                    image_np = images[i].permute(1, 2, 0).cpu().numpy()  # Convert image to numpy
                    image_np = (image_np * 255).astype(np.uint8)  # Scale to [0, 255]
                    
                    print("pred_logits[i]", pred_logits[i].shape)

                    # Apply sigmoid to get objectness score
                    scores = torch.sigmoid(pred_logits[i])

                    topk_logits = torch.topk(scores.squeeze(), 5)
                    print("Top Scores:", topk_logits)  

                    scores = scores.cpu().detach().numpy()  # Objectness scores in [0, 1]
                    boxes = pred_boxes[i].cpu().detach().numpy()  # Bounding boxes in unit scale


                    if torch.any(topk_logits.values > 0.5):
                        # Draw predictions on the image
                        image_with_boxes = draw_predictions(image_np.copy(), boxes, scores)

                        # Save the image with bounding boxes
                        output_file = f"{output_image_path}/epoch_{epoch}_batch_{batch_idx}_img_{i}.jpg"
                        cv2.imwrite(output_file, image_with_boxes)
                        print(f"Saved image: {output_file}")
                    else:
                        print("NO PREDICTION IS GOOD ENOUGH!!!!!")
                    
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

            if exp_scheduler is not None:
                exp_scheduler.step()

            if ema is not None:
                ema.update()

            # Log the training loss
            wandb.log({f"train_{key}": value for key, value in loss_dict.items()})

            # Accumulate loss
            running_loss += loss.item()

            # Calculate the percentage of samples processed
            percent_complete = 100 * (samples_processed / total_samples)

            # Print training progress
            print(f"Epoch [{epoch+1}/{num_epochs}] - Iteration [{batch_idx+1}/{total_batches}] "
                f"Loss: {loss.item():.4f} - Progress: {percent_complete:.2f}%")
            
        avg_train_loss = running_loss / total_batches
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Training Loss: {avg_train_loss:.4f}")

        # Logging to wandb
        wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss})

        # Save the current checkpoint
        if scaler is not None:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': detr.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': avg_train_loss,
            }, checkpoint_path)
        else:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': detr.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
            }, checkpoint_path)


    print('Finished Training')
    wandb.finish()

if __name__ == "__main__":
    main()
