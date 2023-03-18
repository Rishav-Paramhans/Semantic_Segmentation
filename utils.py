import torch
import torchvision
from dataset import A2D2SegmentationDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T



def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    data_dir,
    data_maskdir,
    annotation_csv_file,
    batch_size,
    num_workers=4,
    pin_memory=True,
    transform=None
):
    data_ds = A2D2SegmentationDataset(
        image_dir=data_dir,
        mask_dir=data_maskdir,
        csv_file=annotation_csv_file,
        transform=transform
    )

    data_loader = DataLoader(
        data_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return data_loader


def check_accuracy(preds, targets,model, device= 'cuda:0'):
    num_correct = 0
    num_pixels = 0


    with torch.no_grad():
        preds = preds.to(device)
        targets = targets.to(device)
        preds= torch.nn.functional.softmax(preds, dim= 1)
        preds= torch.argmax(preds, dim=1)

        num_correct += (preds == targets).sum()
        num_pixels += torch.numel(preds)

    #print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    accuracy = (num_correct/num_pixels)
    model.train()
    return accuracy

def class_map_to_one_hot_predictions(input_img, num_classes=22):
    BS, H, W = input_img.shape
    one_hot = np.zeros((BS, num_classes, H, W))

    class_ids = np.unique(input_img)
    for class_id in class_ids:
        one_hot[:, int(class_id), :, :] = np.where(input_img[:, :, :] == class_id, 1, one_hot[:, int(class_id), :, :])

    return one_hot



def mean_dice_score(preds, targets, num_classes=22, smooth=1):
    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)
    preds = np.asarray(torch.Tensor.cpu(preds))
    targets = np.asarray(torch.Tensor.cpu(targets))

    preds = class_map_to_one_hot_predictions(preds, num_classes=22)
    targets = class_map_to_one_hot_predictions(targets, num_classes=22)

    intersection = (np.logical_and(preds, targets).astype(int))
    union = (np.logical_or(preds, targets).astype(int))

    intersection = intersection.sum(axis=3).sum(axis=2).sum(axis=0)  # sum across batches
    union = union.sum(axis=3).sum(axis=2).sum(axis=0)

    dice_score = (2 * intersection + smooth) / (union + intersection + smooth)

    return torch.Tensor(dice_score)


def mean_IOU_score(preds, targets, num_classes=20, smooth=1):
    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)
    preds = np.asarray(torch.Tensor.cpu(preds))
    targets = np.asarray(torch.Tensor.cpu(targets))

    preds = class_map_to_one_hot_predictions(preds, num_classes=22)
    targets = class_map_to_one_hot_predictions(targets, num_classes=22)

    intersection = (np.logical_and(preds, targets).astype(int))
    union = (np.logical_or(preds, targets).astype(int))

    intersection = intersection.sum(axis=3).sum(axis=2).sum(axis=0)  # sum across batches
    union = union.sum(axis=3).sum(axis=2).sum(axis=0)

    iou_score = (intersection + smooth) / (union + smooth)

    return iou_score

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda:0"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():

            preds = F.softmax(model(x), dim= 1)

            preds = torch.argmax(preds, dim=1)

            preds= preds.float()
            preds= preds[:1,:,:]
            print(preds.shape)
        #torchvision.utils.save_image(
        #    preds, f"{folder}/pred_{idx}.png"
        #)
        #torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")
        transform = T.ToPILImage()
        img = transform(preds)
        img.show()
    model.train()