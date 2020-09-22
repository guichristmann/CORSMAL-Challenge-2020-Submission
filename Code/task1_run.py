# Load data loading and processing functions
from task1_utils import *

# Load model definition
from models import SimpleConvNet

import torch
import torch.optim as optim
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
        default='task1_test_preds.csv',
        help='Output filename of predictions CSV.')
parser.add_argument('--pred', action='store_true')
parser.add_argument('--model-weights', default=None)

def train():
    # Load dataset sequences from training data
    print("[task1_run]: Loading dataset...")
    data, labels = load_mfcc_training_dataset(args.dataset)
    sequence_length = data[0].shape[0]
    print(f"[task1_run]: Finished loading. Sequence length: {sequence_length}")

    # Construct PyTorch data loaders for training and validation
    print("[task1_run]: Performing train/val split and constructing PyTorch data"
          "loader")
    train_loader, val_loader, class_weights = construct_pytorch_dataset(data, labels, 
            test_size=0.1, batch_size=16)

    n_train_samples = len(train_loader)
    n_test_samples = len(val_loader)
    print(f"[task1_run]: Loaded total of {n_train_samples} batches of 16 for training"
          f"and {n_test_samples} batches of 16 for testing.")

    # Instantiate model
    model = SimpleConvNet().cuda()
    # SGD optimzier with small learning rate
    optimizer = optim.SGD(model.parameters(), lr=0.00025, momentum=0.9)
    # Cross entropy loss with class weights
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).cuda())

    PRINT_INTERVAL = 35

    # Get names of the filling types from the filling type dict
    target_names = list(fi_dict.keys())

    # Train for 500 epochs
    for epoch in range(500):
        print(f"Epoch {epoch+1}")
        running_loss = 0.0
        model.train() # Train mode
        for i_batch, batch in enumerate(train_loader):
            # Load data batch
            x, y = batch[0].cuda(), batch[1].cuda()
            x = x.unsqueeze(1) # Add channel dimension 

            # Reset gradients
            optimizer.zero_grad()
            
            # Get model predictions
            pred_y = model(x)
            
            # Compute loss
            loss = criterion(pred_y, y)
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
            correct = 0
            total = 0
            y_true = []
            y_pred = []
            with torch.no_grad():
                model.eval() # Eval mode
                for batch in val_loader:
                    x, y = batch[0].cuda(), batch[1].cuda()
                    x = x.unsqueeze(1)
                    
                    y_true += list(y.cpu().numpy())
                    
                    pred_y = model(x)
                    _, predicted = torch.max(pred_y.data, 1)
                    y_pred += list(predicted.cpu().numpy())
                    
                    total += y.size(0)
                    correct += (predicted == y).sum().item()
                    
            print(classification_report(y_true, y_pred, target_names=target_names))
                    
            acc = 100 * correct / total
            print(f"Test Acc: {acc:.3f}%")

    print("[task1_run]: Finished training.")
    torch.save(model.state_dict(), 'weights/task1.weights')
    print("[task1_run]: Saved model state to `weights/task1.weights`.")

    return model

def generate_test_preds(model):
    print("[task1_run]: Generating predictions for test set.")
    # Iterate over test set, get predictions
    prediction_list = []
    for i in range(10,13,1):
        path = os.path.join(args.dataset, str(i)+'/audio/*')
        test_list = natsorted(glob.glob(path))
        for tl in test_list:
            # Sequence lenght should be 1501
            prediction_list.append(getModelPrediction(model, tl, sequence_length=1501))

    # Open reference csv
    df = pd.read_csv(args.input_csv, index_col=0)
    # Write predictions to 'Filling type' column'
    df['Filling type'] = prediction_list
    df.to_csv(args.output_csv)
    print(f"[task1_run]: Saving predictions to `{args.output_csv}`")
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
        model = SimpleConvNet()
        print(f"Loading `{args.model_weights}`.")
        model.load_state_dict(torch.load(args.model_weights))
        model.cuda()
        generate_test_preds(model)
