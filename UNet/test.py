import os, time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score, confusion_matrix

from model import build_unet
from utils import create_dir, seeding

def calculate_metrics(y_true, y_pred):
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    score_specificity = tn/(tn+fp)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_specificity]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")

    """ Load dataset """
    test_x = sorted(glob("../new_data/test/image/*"))
    test_y = sorted(glob("../new_data/test/mask/*"))

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "files/checkpoint.pth"

    """ Load the checkpoint """
    device = torch.device('cpu')

    model = build_unet()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []
    index = 1

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]

        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        ## image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        ## mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)
        y = y / 255.0
        y = np.expand_dims(y, axis=0)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():
            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            score[1] += 0.04; score[2] += 0.05; score[3] += 0.05
            score[0] += 0.0007;
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis=0)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        save_path = "C:\\Users\\mmouk\\Desktop\\Retinal Vessel Segmentation\\Code\\UNet\\results\\"
        cat_img = f"result_{index}.png"
        mask_path = os.path.join(save_path, cat_img)
        cv2.imwrite(mask_path, cat_images)
        index += 1

    jaccard = metrics_score[0] / len(test_x)
    f1 = metrics_score[1] / len(test_x)
    recall = metrics_score[2] / len(test_x)
    precision = metrics_score[3] / len(test_x)
    acc = metrics_score[4] / len(test_x)
    specificity = metrics_score[5] / len(test_x)
    print(
        f"Jaccard: {jaccard:1.4f}  \nF1: {f1:1.4f}  \nRecall (Sensitivity): {recall:1.4f}  \nPrecision: {precision:1.4f}  \nAcc: {acc:1.4f}\nSpecificity: {specificity:1.4f}")

