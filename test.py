import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)
import os

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 200
NUM_WORKERS = 0
IMAGE_HEIGHT = 1208  # 1280 originally
IMAGE_WIDTH = 1920  # 1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = r"E:\Rishav_Thesis\A2D2_dataset\images"  # Have to set this
TRAIN_MASK_DIR = r"E:\Rishav_Thesis\A2D2_dataset\seg_label"
TRAIN_CSV_FILE = r"E:\Rishav_Thesis\A2D2_dataset\train.csv"


def train_fn(loader, model, optimizer, loss_fn, scaler, training_df, ep):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        # print(targets.shape)
        # print(targets)
        # forward

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            predictions = model(data)
            predictions = predictions.float()
            # print(predictions.shape)

            loss = loss_fn(predictions, targets)

        # backward
        loss.backward()
        optimizer.step()

        training_df = training_df.append({"epoch": ep, "train_loss": loss.item()}, ignore_index=True)
        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=22).to(DEVICE)
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    TRAIN_IMG_DIR,
    TRAIN_MASK_DIR,
    TRAIN_CSV_FILE,
    BATCH_SIZE,
    train_transform,
    NUM_WORKERS,
    PIN_MEMORY
    train_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        TRAIN_CSV_FILE,
        BATCH_SIZE,
        train_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    # check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    training_df = pd.DataFrame(columns=['epoch', 'Training_loss'])
    for epoch in range(NUM_EPOCHS):

        train_fn(train_loader, model, optimizer, loss_fn, scaler, training_df, ep=epoch)
        training_df.to_csv('training.csv')

        if epoch / 5 == 0:

            dir = r"E:\Rishav_Thesis\Baseline\Semantic_Segmentation\Saved_model"
            PATH = os.path.join(dir, str(epoch))
            if not os.path.exists(PATH):
                os.makedirs(PATH)
            model_PATH = os.path.join(PATH, 'model_checkpoint.pth')
            optimizer_PATH = os.path.join(PATH, 'optimizer_checkpoint.pth')

            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint['model'], model_PATH)
            torch.save(checkpoint['optimizer'], optimizer_PATH)


if __name__ == "__main__":
    main()