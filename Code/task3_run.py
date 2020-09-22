# Load data loading and processing functions
from task3_utils import *

# Load model definition
from models import BiggishConvNet 

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import classification_report

import argparse
import os
import pandas as pd
import sys

# Absolute path to dataset root folder
DATASET_DEFAULT_PATH = "/media/ntnuerc/39828D065F4B0A9F/CORSMAL/Dataset/"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', 
        default=DATASET_DEFAULT_PATH,
        help='Absolute path to dataset root.')
parser.add_argument('--input-csv',
        default='default.csv',
        help='The input CSV to modify. Loads `default.csv` by default.')
parser.add_argument('--output-csv',
        default='task3_test_preds.csv',
        help='Output filename of predictions CSV.')
parser.add_argument('--pred', action='store_true')
parser.add_argument('--model-weights', default=None)

def train():
    # Load dataset sequences from training data
    print("[task3_run]: Loading dataset...")
    data, targets = load_depth_roi_dataset(args.dataset)
    print(len(data))

    # Construct PyTorch data loaders for training and validation
    print("[task3_run]: Performing train/val split and constructing PyTorch data"
          "loader")
    train_loader, val_loader = construct_pytorch_dataset(data, targets, 
            test_size=0.15, batch_size=8)

    n_train_samples = len(train_loader)
    n_test_samples = len(val_loader)
    print(f"[task3_run]: Loaded total of {n_train_samples} batches of 8 for training "
          f"and {n_test_samples} batches of 1 for testing.")

    # Instantiate model
    model = BiggishConvNet().cuda()
    # Adam optimzier with small learning rate
    optimizer = optim.Adam(model.parameters(), lr=0.00025)
    criterion = nn.MSELoss()

    PRINT_INTERVAL = 15

    # Get names of the filling types from the filling type dict
    target_names = list(fi_dict.keys())

    # Train for 500 epochs
    for epoch in range(500):
        print(f"Epoch {epoch+1}")
        running_loss = 0.0
        model.train() # Train mode
        for i_batch, batch in enumerate(train_loader):
            # Load data batch
            (x_img, x_roi_info), y_volume = batch[0], batch[1].cuda()
            x_img = x_img.cuda()
            x_roi_info = x_roi_info.cuda()

            assert len(x_img) == len(y_volume)

            # Reset gradients
            optimizer.zero_grad()
            
            # Get model predictions
            pred_y_volume = model(x_img, x_roi_info)
            
            # Compute loss
            loss = criterion(pred_y_volume, y_volume)

            running_loss += loss.item()
            
            # Backpropagate
            loss.backward()
            # Update weights
            optimizer.step()
            
            if i_batch % PRINT_INTERVAL == PRINT_INTERVAL-1:
                running_loss += running_loss
                print(f"[{epoch+1}, {i_batch+1}]: Loss: {running_loss/PRINT_INTERVAL:.4f}")
                running_loss = 0.0
                
        if epoch % 10 == 9:
            test_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                model.eval()
                for batch in val_loader:
                    (x_img, x_roi_info), y_volume = batch[0], batch[1].cuda()
                    x_img = x_img.cuda()
                    x_roi_info = x_roi_info.cuda()
                    
                    loss = criterion(pred_y_volume, y_volume)
                    test_loss += loss.item()
                    total_samples += 1
                    
                test_loss = test_loss / total_samples
                
                print(f"Test Loss: {test_loss:.5f}")
                print("#"*40)

    print("[task3_run]: Finished training.")
    torch.save(model.state_dict(), 'weights/task3.weights')
    print("[task3_run]: Saved model state to `weights/task3.weights`.")

    return model

def generate_test_preds(model):
    print("[task3_run]: Generating predictions for test set.")
    # Iterate over test set, get predictions
    prediction_list = []
    for obj_id in range(10, 12):
        for i in range(84):
            i_str = str(i)
            _str = '0' * (4 - len(i_str)) + i_str
            sample_frames = natsorted(glob.glob(os.path.join(args.dataset, 
                                        f"{obj_id}/depth/{_str}/c3/*.png")))

            # Run predictions through trained model
            pred = getModelPrediction(model, sample_frames)
            if pred is not None:
                prediction_list.append(pred)
            else: # Couldn't get prediction from this sample
                prediction_list.append(-1)

    for i in range(60):
        i_str = str(i)
        _str = '0' * (4 - len(i_str)) + i_str
        sample_frames = natsorted(glob.glob(os.path.join(args.dataset, 
                                    f"12/depth/{_str}/c3/*.png")))

        # Run predictions through trained model
        pred = getModelPrediction(model, sample_frames)
        if pred is not None:
            prediction_list.append(pred)
        else: # Couldn't get prediction from this sample
            prediction_list.append(-1)

    # Open reference csv
    df = pd.read_csv(args.input_csv, index_col=0)
    # Write predictions to 'Container Capacity' column'
    df['Container Capacity'] = prediction_list
    df.to_csv(args.output_csv)
    print(f"[task3_run]: Saving predictions to `{args.output_csv}`")
    print("Finished.")

if __name__ == "__main__":
    args = parser.parse_args()

    if not args.pred:
        # Train and generate test predictions
        trained_model = train()
        generate_test_preds(trained_model)
    else:
        if args.model_weights == None:
            print("Please specify `--model-weights` when using `--pred`.")
            sys.exit(1)

        # Load model weights and generate predictions
        model = BiggishConvNet()
        print(f"Loading `{args.model_weights}`.")
        model.load_state_dict(torch.load(args.model_weights))
        model.cuda()
        model.eval()
        generate_test_preds(model)
