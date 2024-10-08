import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Define the custom transformation to add random black patches
class RandomBlackPatches(object):
    def __init__(self, max_num_patches=20, max_patch_size=30, min_patch_size=10):
        self.max_num_patches = max_num_patches
        self.max_patch_size = max_patch_size
        self.min_patch_size = min_patch_size

    def __call__(self, img):
        # Convert image to numpy array
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        # Apply random black patches
        for _ in range(random.randint(1, self.max_num_patches + 1)):
            patch_size = random.randint(self.min_patch_size, self.max_patch_size + 1)
            x = random.randint(0, w - patch_size - 1)
            y = random.randint(0, h - patch_size - 1)
            img_np[y:y+patch_size, x:x+patch_size] = 0

        # Convert back to PIL image
        return Image.fromarray(img_np)
    
def rescale_bbox(normalized_bbox, image_size):
    """
    Rescale the normalized bounding box to the given image size.

    Args:
        normalized_bbox (tuple): Normalized bounding box in format (x, y, width, height),
                                 where values are between 0 and 1.
        image_size (tuple): Desired image size in format (height, width).

    Returns:
        bbox (tuple): Rescaled bounding box in format (x, y, width, height).
    """
    img_h, img_w = image_size
    
    x, y, width, height = normalized_bbox
    
    # Rescale the bounding box
    x = int(x * img_w)
    y = int(y * img_h)
    width = int(width * img_w)
    height = int(height * img_h)
    
    return x, y, width, height

def unitscale_bbox( bbox, image_size):
    """
    Normalize the bounding box to be independent of image size.

    Args:
        bbox (tuple): Bounding box in format (x, y, width, height).
        image_size (tuple): Image size in format (height, width).

    Returns:
        normalized_bbox (tuple): Normalized bounding box in format (x, y, width, height)
                                with values between 0 and 1.
    """
    img_h, img_w = image_size
    
    x, y, width, height = bbox
    
    # Normalize the bounding box
    x = x / img_w
    y = y / img_h
    width = width / img_w
    height = height / img_h
    
    return x, y, width, height


class PersonDataset(Dataset):
    def __init__(self, root_dir, sequence_list, img_transform_size = (640, 640), template_transform_size = (256, 256), max_num_templates=10, max_detections = 300):
        """
        Args:
            root_dir (string): Root directory path where the "person" class folder is located.
            sequence_list (list): List of sequence names (e.g., ['person-1', 'person-2']).
            transform (callable, optional): Optional transform to be applied on a sample.
            num_templates (int, optional): Number of random templates to load from the gallery.
        """
        self.root_dir = root_dir
        self.sequence_list = sequence_list
        self.img_transform_size = img_transform_size
        self.template_transform_size = template_transform_size

        self.max_num_templates = max_num_templates
        self.max_detections = max_detections
        self.data = self._load_data()

        self.img_transform = transforms.Compose([
            transforms.Resize(img_transform_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0., 0., 0.],  std=[1, 1, 1])

        ])

        self.template_transform = transforms.Compose([
            transforms.Resize(template_transform_size),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),  # Apply Gaussian blur
            RandomBlackPatches(),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize(mean=[0., 0., 0.],  std=[1, 1, 1])
        ])



    def _load_data(self):
        data = []
        for seq in self.sequence_list:
            img_dir = os.path.join(self.root_dir, seq, 'img')
            det_dir = os.path.join(self.root_dir, seq, 'detections')
            gallery_dir = os.path.join(self.root_dir, seq, 'gallery')
            groundtruth_file = os.path.join(self.root_dir, seq, 'groundtruth.txt')

            # Load images
            image_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

            # Load target bounding boxes (ground truth)
            with open(groundtruth_file, 'r') as f:
                target_boxes = [list(map(int, line.strip().split(','))) for line in f]


            # Ensure the image files and target_boxes are aligned
            assert len(image_files) == len(target_boxes), "Mismatch between images and target boxes"

            for idx, image_file in enumerate(image_files):
                # Full paths
                img_path = os.path.join(img_dir, image_file)
                det_file = os.path.join(det_dir, os.path.splitext(image_file)[0] + '.txt')

                # Load bounding boxes from the detection file
                if os.path.exists(det_file):
                    with open(det_file, 'r') as f:
                        bounding_boxes = [list(map(int, line.strip().split(','))) for line in f]
                else:
                    bounding_boxes = []
                
                # inserting the target object into the distractor objects
                # The first bounding box now represents target object and the rest the distractor objs
                bounding_boxes.insert(0, target_boxes[idx])
                # Converting from x_top_left, y_top_left, w_h, to x_center, y_center, w, h
                bounding_boxes = [[int(x + w / 2), int(y + h / 2), w, h] for x, y, w, h in bounding_boxes]

                data.append({
                    'img_path': img_path,
                    'bounding_boxes': bounding_boxes,
                    'templates_path': gallery_dir
                })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data[idx]

        # Getting the bounding boxes of the distractor objects
        bounding_boxes_list = sample['bounding_boxes'].copy()

        # Load the image
        image = Image.open(sample['img_path']).convert('RGB')

        # Get the original Img Size
        orig_w, orig_h = image.size

        # Apply transforms to the image if provided
        if self.img_transform:
            image = self.img_transform(image)
        
        # Select a random number of templates from 1 up to the maximum allowed
        num_templates = random.randint(1, self.max_num_templates)
        template_files = os.listdir(sample['templates_path'])

        # Load templates from the gallery directory
        templates = random.sample(template_files, min(num_templates, len(template_files)))
        template_paths = [os.path.join(sample['templates_path'], tpl) for tpl in templates]

        # Load templates and apply transforms
        templates_imgs = [self.template_transform(Image.open(tpl).convert('RGB')) if self.template_transform else Image.open(tpl).convert('RGB') for tpl in template_paths]

        # Create a Tensor with multiple template imgs
        templates_imgs = torch.stack(templates_imgs)

        # Unit scale the bounding boxes (distractors + target)
        bounding_boxes_list_unit = [unitscale_bbox(bbox, (orig_h, orig_w)) for bbox in bounding_boxes_list]

        # Convert bounding boxes to tensors
        bounding_boxes_tensor = torch.tensor(bounding_boxes_list_unit, dtype=torch.float32)

        # Initialize a tensor of zeros with shape [max_boxes, 4]
        num_boxes = bounding_boxes_tensor.shape[0]
        padded_boxes = torch.zeros((self.max_detections, 4), dtype=bounding_boxes_tensor.dtype)

        num_templates = templates_imgs.shape[0]
        padded_templates = torch.zeros((self.max_num_templates, templates_imgs.shape[1], templates_imgs.shape[2], templates_imgs.shape[3]))

        if not(bounding_boxes_tensor.shape == torch.Size([0])):
            # Fill in the existing bounding boxes
            padded_boxes[:num_boxes] = bounding_boxes_tensor

        padded_templates[:num_templates, :, :, :] = templates_imgs

        # Return a dictionary with the data
        return {
            'img': image,
            'bounding_boxes': padded_boxes,
            'templates': padded_templates,
            'num_boxes': num_boxes,
            'num_templates': num_templates,
            'orig_dim': (orig_h, orig_w),
            'img_path': sample['img_path']
        }