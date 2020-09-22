import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from data_utils import *

N_MFCC = 40

def load_and_extract_mfcc(path, window_size=20.0/1000, max_length=30):
    ''' Given a path to an audio sample, will load the audio data and
        extract and normalize the MFCCs from it.

    '''

    # Load audio data from file
    audio, sample_rate = librosa.load(path, res_type='kaiser_fast')
    
    length = audio.shape[0]
    step_size = int(window_size * sample_rate) # In samples    
    
    # Retrieve the sequence of MFCCs
    sequence = librosa.feature.mfcc(y=audio[:max_length*sample_rate], sr=sample_rate, n_mfcc=N_MFCC,
                                    hop_length=step_size)
    sequence = np.transpose(sequence)
    
    # Normalize the sequence according to its own data
    ### Normalization for each MFCC individually
    _mean = np.mean(sequence, axis=0)
    _std = np.std(sequence, axis=0)
    
    # Returned normalized MFCC data
    return (sequence - _mean) / _std

def load_mfcc_training_dataset(dataset_root):
    ''' Load the dataset for Task 1, the MFCC sequence from each audio file.

    '''

    data = []
    labels = []
    # Iterate over dataset combinations to load audio files
    for obj_id in range(1, 10):
        print(f"Extracting data from object id: `{obj_id}`")
        for sit in s_dict.keys():
            for fi in fi_dict.keys():
                for fu in fu_dict.keys():
                    for b in b_dict.keys():
                        for l in l_dict.keys():
                            try:
                                # Grab sample path
                                sample = retrieve_data(dataset_root, obj_id, s=sit, fi=fi, fu=fu, b=b, l=l)

                                if sample != -1:
                                    # Get MFCCs from this sample
                                    seq_data = load_and_extract_mfcc(sample['audio'])
                                    # Convert to Tensor and append to dataset
                                    data.append(torch.Tensor(seq_data))
                                    # Save label (as integer)
                                    labels.append(fi_dict[fi])

                            except Exception as e:
                                print(f"There's no sample with combination: {(obj_id, sit, fi, fu, b, l)}")


    # Padding sequences to maximum length
    data = nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

    return data, labels

def construct_pytorch_dataset(data, labels, test_size, batch_size, plot_graphs=False):
    ''' Given the list of MFCCs sequence (data) and its labels, constructs
        PyTorch Dataset objects for training and validation. Performs train
        and validation split as well as computes the weight of each class
        fed to the cross entropy loss to counteract class imbalance.
        
    '''

    # Analyze class distribution
    _classes, counts = np.unique(labels, return_counts=True)
    if plot_graphs:
        print(_classes, counts)
        plt.bar(_classes, counts)
        plt.title("Whole dataset")
        plt.show()

    # Split in train and test
    X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                       test_size=0.1)

    _classes, counts = np.unique(y_train, return_counts=True)
    n_train_samples = len(y_train)
    if plot_graphs:
        print(_classes, counts)
        plt.bar(_classes, counts)
        plt.title("Train dataset")
        plt.show()

    # Compute the class weights used for training. The weight of each class is 
    # inversely proportional to the number of samples in the class
    class_weights = np.array([c / n_train_samples for c in counts])
    class_weights = 1.0 / class_weights
    class_weights = class_weights / class_weights.sum()

    _classes, counts = np.unique(y_test, return_counts=True)
    if plot_graphs:
        print(_classes, counts)
        plt.bar(_classes, counts)
        plt.title("Test dataset")
        plt.show()

    # Build Dataset object
    class AudioDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            
            assert len(self.x) == len(self.y)
            
        def __len__(self):
            return len(self.x)
        
        def __getitem__(self, idx):
            return torch.Tensor(self.x[idx]), self.y[idx]
        
    train_dataset = AudioDataset(X_train, y_train)
    test_dataset = AudioDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True,
                              num_workers=0)

    return train_loader, test_loader, class_weights

def getModelPrediction(model, path, sequence_length):
    ''' Given a (trained) model and path to a filename, returns the model 
        prediction for the filling type present in the data

    '''

    rev = {0: 'nothing', 1: 'pasta', 2: 'rice', 3: 'water'}

    # Loading audio file and extracting MFCC feature
    features = load_and_extract_mfcc(path)
    
    # Padding the sequence
    pad_length = sequence_length - features.shape[0]
    features = np.concatenate([features, np.zeros((pad_length, N_MFCC))])

    # Make sure shape dims are all nice and good for pytorch
    feat_tensor = torch.Tensor(features)
    feat_tensor = feat_tensor.unsqueeze(0)
    feat_tensor = feat_tensor.unsqueeze(0)

    with torch.no_grad():
        model.eval()

        # Run data through the model
        out = model(feat_tensor.cuda())

        # Run softmax to get class probs
        scores = F.softmax(out, dim=1).cpu().numpy()

        # Pick class with largest prob as the prediction
        _class = np.argmax(scores)
        score = np.max(scores)

    return _class
