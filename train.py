import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (
    load_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    mean_IOU_score,
    mean_dice_score
)
import os
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import pandas as pd


# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print('Using device:', DEVICE)
#DEVICE= 'cpu'
BATCH_SIZE = 4
NUM_EPOCHS = 150
NUM_WORKERS = 0
IMAGE_HEIGHT = 608 # 1280 originally
IMAGE_WIDTH = 960  # 1920 originally
PIN_MEMORY = True
LOAD_MODEL = False
IMG_DIR = r"E:\Thesis-Rishav\A2D2_dataset\images"    #Have to set this
MASK_DIR = r"E:\Thesis-Rishav\A2D2_dataset\seg_label"
TRAIN_CSV_FILE= r"E:\Thesis-Rishav\A2D2_dataset\train.csv"
TEST_CSV_FILE= r"E:\Thesis-Rishav\A2D2_dataset\test.csv"
WRITER = True  # Controlling the tensorboard


def train_fn(train_loader, model, optimizer,loss_fn, scaler, epoch):
    loop = tqdm(train_loader)
    mean_loss=[]
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        #print('target', targets)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data.float())
            predictions = predictions.float()
            #print('predictions',predictions.shape)

            loss = loss_fn(predictions, targets)

            mean_loss.append(loss.item())
        # backward

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        #checking accuracy
        accuracy= check_accuracy(predictions, targets, model)

        # update tqdm loop
        loop.set_postfix(epoch= epoch, loss=loss.item())

    #Saving model and optimizer
    model_name= "/model_{epoch_num}.pth".format(epoch_num= epoch)
    opt_name = "/optimizer_{epoch_num}.pth".format(epoch_num=epoch)
    model_PATH= "E:\Thesis-Rishav\Baselines\Semantic_Segmentation\Trained_parameters" + model_name
    optimizer_PATH = "E:\Thesis-Rishav\Baselines\Semantic_Segmentation\Trained_parameters" + opt_name

    torch.save(model.state_dict(), model_PATH)
    torch.save(optimizer.state_dict(), optimizer_PATH)

    mean_loss_value = sum(mean_loss) / len(mean_loss)
    print('MEAN_LOSS_VALUE:',mean_loss_value)
    return mean_loss_value, accuracy

def eval_fn(test_loader,model, loss_fn, scaler):
    loop = tqdm(test_loader)
    mean_loss=[]
    IOU_score= []
    Dice_score = []
    model.eval()
    for batch_idx, (data, targets) in enumerate(loop):

        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        with torch.no_grad():
            predictions = model(data.float())
            predictions= predictions.float()
            #print(predictions.shape)
            loss = loss_fn(predictions, targets)

            iou_score=  mean_IOU_score(predictions, targets)
            dice_score= mean_dice_score(predictions, targets)
            mean_loss.append(loss.item())
            IOU_score.append(iou_score)
            Dice_score.append(dice_score)

        loop.set_postfix(loss=loss.item())
        accuracy= check_accuracy(predictions,targets,model)
    mean_loss_value = sum(mean_loss) / len(mean_loss)
    mean_IOU= sum(IOU_score)/len(IOU_score)
    mean_DICE= sum(Dice_score)/len(Dice_score)
    model.train()
    return mean_loss_value, mean_IOU, mean_DICE,accuracy


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),

            ToTensorV2(),
        ],
    )
    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),

            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=22).to(DEVICE)

    loss_fn= nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if WRITER:
        train_writer = SummaryWriter("tensorboard/train")
        test_writer = SummaryWriter("tensorboard/test")


    train_loader = get_loaders(
        IMG_DIR,
        MASK_DIR,
        TRAIN_CSV_FILE,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        transform= train_transform
    )

    test_loader = get_loaders(
        IMG_DIR,
        MASK_DIR,
        TEST_CSV_FILE,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        transform= test_transform
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(0,NUM_EPOCHS):
        print(f"EPOCH : {epoch}")

        avg_train_loss, train_accuracy = train_fn(train_loader, model, optimizer,loss_fn, scaler, epoch)
        if WRITER:
            train_writer.add_scalar('Loss', avg_train_loss, global_step=epoch)
            train_writer.add_scalar('Accuracy', train_accuracy, global_step=epoch)

        if epoch % 5 == 0:
            avg_test_loss, mean_test_IOU, mean_test_DICE, test_accuracy = eval_fn(test_loader, model, loss_fn, scaler)
            if WRITER:
                test_writer.add_scalar('Loss', avg_test_loss, global_step=epoch)
                test_writer.add_scalar('Accuracy', test_accuracy, global_step=epoch)
                #test_writer.add_scalars("Test_IOU", mean_test_IOU, global_step=epoch)
                #test_writer.add_scalars("Test_DICE", mean_test_DICE, global_step=epoch)



if __name__ == "__main__":
    main()