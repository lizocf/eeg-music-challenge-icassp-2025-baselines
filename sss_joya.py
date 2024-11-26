import os
import mne
import json
import numpy as np
from tqdm import tqdm
import src.config as config
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from src.eeg_transforms import RandomCrop, ToTensor, Standardize, ImageAugmentation, augment_images
import pywt
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import torch

import torch.nn as nn
import torchvision.models as models
from types import SimpleNamespace
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
from sklearn import metrics

np.random.seed(42)

mne.set_log_level("ERROR")

LABEL_NAMES = ["joy", "relief", "sadness", "anger"]

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def create_confusion_matrix(y_true, y_pred, labels=LABEL_NAMES, title="Confusion Matrix", cmap=plt.cm.Blues):
    # Compute confusion matrix
    confmat = confusion_matrix(y_true, y_pred)

    # Plot the matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(confmat, interpolation='nearest', cmap=cmap)
    # ax.figure.colorbar(im, ax=ax)

    # Show labels on axes
    n_labels = len(labels)
    ax.set(xticks=np.arange(n_labels),
           yticks=np.arange(n_labels),
           xticklabels=labels, yticklabels=labels,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate the tick labels for x-axis
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations in each cell
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(j, i, format(confmat[i, j], 'd'),
                    ha="center", va="center",
                    color="black")

    plt.tight_layout()
    plt.show()



sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="0104d50f3acb45e789c24dcb0af5fbe8",
                                                           client_secret="2fb2248e08574ac99b345580ff478f15"))   


AUDIO_FEATURES_FILE = 'spotify_audio_features.csv'
if os.path.exists(AUDIO_FEATURES_FILE):
    audio_features_df = pd.read_csv(AUDIO_FEATURES_FILE, index_col="track_id")
else:
    audio_features_df = pd.DataFrame(columns=[
        'track_id', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo'
    ])
    audio_features_df.set_index('track_id', inplace=True)

def get_audio_features(track_id, max_retries=5):
    # Check if track_id is already in DataFrame
    if track_id in audio_features_df.index:
        # Return features from DataFrame if available
        return audio_features_df.loc[track_id].tolist()

    # Otherwise, try fetching from Spotify API
    attempt = 0
    while attempt < max_retries:
        try:
            features = sp.audio_features(track_id)[0]
            if features:  # Check if data is retrieved successfully
                # Extract desired keys
                keys = ['danceability', 'energy', 'key', 'loudness', 'mode',
                        'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo']
                audio_features = {key: features[key] for key in keys}
                # audio_features['track_id'] = track_id

                # Append to DataFrame and save
                # breakpoint()
                audio_features_df.loc[track_id] = list(audio_features.values())
                audio_features_df.to_csv(AUDIO_FEATURES_FILE)  # Save after each addition
                return list(audio_features.values())  # Return the features as a list

        except Exception as e:
            print(f"Error fetching audio features for track {track_id} on attempt {attempt + 1}: {e}")
            attempt += 1
            time.sleep(2 ** attempt)  # Exponential backoff

    # If all retries fail, return None
    return None


# https://towardsdatascience.com/multiple-time-series-classification-by-using-continuous-wavelet-transformation-d29df97c0442

def generate_scalogram(eeg_signal, wavelet='morl', sampling_rate=128, output_size=(128, 128), save_dir='scalogram_images', filename=None):
    """
    Generate or load a scalogram for a given EEG signal. If the image already exists, load it; otherwise, generate and save it.

    Args:
        eeg_signal (numpy.array): 1D array of EEG data.
        wavelet (str): Wavelet type for CWT.
        sampling_rate (int): Sampling rate of the EEG signal.
        output_size (tuple): Size of the output scalogram image.
        save_dir (str): Directory to save/load the scalogram images.
        filename (str): Name of the file for the scalogram image (without extension).

    Returns:
        torch.Tensor: Scaled scalogram image.
    """
    # resolution = 128
    resolution = 64
    # Ensure the directory exists
    # save_dir = os.path.join(save_dir, str(resolution))
    os.makedirs(save_dir, exist_ok=True)

    # If filename is None, generate it based on a hash or sample ID
    if filename is None:
        filename = f"{hash(eeg_signal.tostring())}.png"
    
    # Check if the image already exists
    img_path = os.path.join(save_dir, filename)
    if os.path.exists(img_path):
        # Load the existing image
        image = Image.open(img_path).convert('RGB').resize(output_size)
    else:
        # Generate the scalogram
        widths = np.arange(1, resolution)  # Adjust based on the signal resolution
        cwt_matrix, freqs = pywt.cwt(eeg_signal, widths, wavelet, sampling_period=1/sampling_rate)
        
        # Normalize and create an image
        plt.figure(figsize=(4, 4))
        plt.imshow(np.abs(cwt_matrix), aspect='auto', cmap='viridis', origin='lower')
        plt.axis('off')
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        image = Image.open(buffer).convert('RGB').resize(output_size)
        buffer.close()
        plt.close()
        
        # Save the image
        image.save(img_path)

    # Convert to tensor
    return T.ToTensor()(image)

def generate_spectrogram(eeg_signal, sampling_rate=128, output_size=(128, 128), save_dir='spectrogram_images', filename=None):
    """
    Generate or load a spectrogram for a given EEG signal.
    If the image already exists, load it; otherwise, generate and save it.

    Args:
        eeg_signal (numpy.array): 1D array of EEG data.
        sampling_rate (int): Sampling rate of the EEG signal.
        output_size (tuple): Size of the output spectrogram image.
        save_dir (str): Directory to save/load the spectrogram images.
        filename (str): Name of the file for the spectrogram image (without extension).

    Returns:
        torch.Tensor: Scaled spectrogram image.
    """
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        filename = f"{hash(eeg_signal.tostring())}_spec.png"
    
    img_path = os.path.join(save_dir, filename)
    if os.path.exists(img_path):
        try:
            image = Image.open(img_path).convert('RGB').resize(output_size)
        except:
            plt.figure(figsize=(4, 4))
            plt.specgram(eeg_signal, Fs=sampling_rate, cmap='viridis')
            plt.axis('off')
            buffer = BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
            buffer.seek(0)
            image = Image.open(buffer).convert('RGB').resize(output_size)
            buffer.close()
            plt.close()
        
            # Save the image
            image.save(img_path)
    else:
        # Generate spectrogram
        plt.figure(figsize=(4, 4))
        plt.specgram(eeg_signal, Fs=sampling_rate, cmap='viridis')
        plt.axis('off')

        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
        buffer.seek(0)
        image = Image.open(buffer).convert('RGB').resize(output_size)
        buffer.close()
        plt.close()
        
        # Save the image
        image.save(img_path)

    return T.ToTensor()(image)

def get_splits(splits, split):
  id_split = [split["id"] for split in splits[split]]
  label_split = [split["label"] for split in splits[split]]

  return id_split, label_split


def load_images_for_split(split_ids, split, base_dir='image_time_series'):
#   for id in split_ids:
    id = str(split_ids)

    if split == "train" or split == "val_trial" or split == "val_subject":
        split = "train"
    image_path = os.path.join(base_dir, split, f"{id}_eeg.png")
    if os.path.exists(image_path):
        with Image.open(image_path) as img:
            image_array = np.array(img.convert('L'), dtype=np.float32)
            # Normalize pixel values to be between 0 and 1
            image_array /= 255.0
            image_array = np.expand_dims(image_array, axis=-1)
    else:
        print(f"Image not found: {image_path}")

    return T.ToTensor()(image_array)


class EEGEmotionClassifier(nn.Module):
    def __init__(self, eeg_channels, scalogram_channels, spectrogram_channels, num_classes):
        super(EEGEmotionClassifier, self).__init__()

        # self.scalogram_branch = nn.Sequential(
        #     nn.Conv2d(scalogram_channels * 3, 8, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Flatten(),
        #     nn.Linear(8 * 64 * 64, 32),  # Adjust based on input size
        #     nn.ReLU()
        # )

        # Spectrogram feature extractor (ResNet18)
        self.scalogram_branch = models.resnet18(weights=False)
        self.scalogram_branch.conv1 = nn.Conv2d(
            scalogram_channels * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.scalogram_branch.fc = nn.Linear(self.scalogram_branch.fc.in_features, num_classes)
        

        # # Simplified Spectrogram branch
        # self.spectrogram_branch = nn.Sequential(
        #     nn.Conv2d(spectrogram_channels * 3, 8, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Flatten(),
        #     nn.Linear(8 * 64 * 64, 32),  # Adjust based on input size
        #     nn.ReLU()
        # )

        # self.spectrogram_branch = models.resnet18(weights=False)
        # self.spectrogram_branch.conv1 = nn.Conv2d(
        #     spectrogram_channels * 3, 64, kernel_size=7, stride=2, padding=3, bias=False
        # )
        # self.spectrogram_branch.fc = nn.Linear(self.spectrogram_branch.fc.in_features, num_classes)
        

        # # Simplified Brainplot branch
        # self.brainplot_branch = nn.Sequential(
        #     nn.Conv2d(32, 8, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2),
        #     nn.Flatten(),
        #     nn.Linear(8 * 3 * 5, 16),  # Adjust based on input size
        #     nn.ReLU()
        # )

        # # Classifier
        # self.classifier = nn.Sequential(
        #     # nn.Linear(32 + 32 + 16, 64),  # Combined features from all branches
        #     nn.Linear(32 + 32, 64),  # Combined features from all branches
        #     # nn.Linear(32, 64),  # Combined features from all branches
        #     nn.ReLU(),
        #     nn.Linear(64, num_classes),
        #     nn.Softmax(dim=1)
        # )



        # self.eeg_branch.add_module("dropout", nn.Dropout(0.5))
        # self.scalogram_branch.add_module("dropout", nn.Dropout(0.5))
        # self.spectrogram_branch.add_module("dropout", nn.Dropout(0.5))
        
    def forward(self, eeg, scalogram, spectrogram, song_features, brainplot):
        # EEG features
        # eeg_features = self.eeg_branch(eeg)

        # Song features
        # song_features = self.song_branch(song_features)
        
        # Reshape scalogram and spectrogram tensors
        # breakpoint()

        scalogram = scalogram.view(scalogram.size(0), -1, scalogram.size(3), scalogram.size(4))
        # spectrogram = spectrogram.view(spectrogram.size(0), -1, spectrogram.size(3), spectrogram.size(4))

        # brainplot = brainplot.squeeze(2)

        # print(scalogram.size(), spectrogram.size(), brainplot.size())

        # Features from branches
        scalogram_features = self.scalogram_branch(scalogram).view(scalogram.size(0), -1)
        # spectrogram_features = self.spectrogram_branch(spectrogram).view(spectrogram.size(0), -1)

        # brainplot = brainplot.squeeze(2)
        # brainplot_features = self.brainplot_branch(brainplot).view(brainplot.size(0), -1)

        # Concatenate features
        # features = torch.cat((scalogram_features, spectrogram_features), dim=1)
        # features = torch.cat((scalogram_features, spectrogram_features, brainplot_features), dim=1)
        # features = torch.cat((eeg_features, song_features, scalogram_features, spectrogram_features), dim=1)
        # return self.classifier(features)
        return scalogram_features


class EremusDataset(Dataset):
    def __init__(self, subdir, split_dir, split="train", task="subject_identification", ext="fif", transform=None, prefix=""):
        
        self.dataset_dir = config.get_attribute("dataset_path", prefix=prefix)
        self.subdir = os.path.join(subdir, split) if "test" in split else os.path.join(subdir, "train")
        self.split_dir = split_dir
        self.transform = transform
        self.split = split
        self.label_name = "subject_id" if task == "subject_identification" else "label"
        self.ext = ext
        
        splits = json.load(open(os.path.join(split_dir, f"resampled_merged_splits_{task}.json")))
        self.samples = splits[split]
        
        files = []
        for sample in self.samples:
            path = os.path.join(self.dataset_dir, self.subdir, f"{sample['id']}_eeg.{self.ext}")
            files.append(path)
        files = list(set(files))

        if self.ext == "npy":
            self.files = {f: np.load(f) for f in tqdm(files)}
        elif self.ext == "fif":
            self.files = {f: mne.io.read_raw_fif(f, preload=True).get_data() for f in tqdm(files)}
        else:
            raise ValueError(f"Extension {ext} not recognized")

        ## get spotify features
        self.spotify_data = {}
        for sample in self.samples:
            track_info = {
                "spotify_track_id": sample.get("spotify_track_id"),
                "song_title": sample.get("song_title"),
                "emotion": sample.get("emotion"),
                "session_type": sample.get("session_type"),
                "song_author": sample.get("song_author"),
            }
            self.spotify_data[sample["id"]] = track_info
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = self.files[os.path.join(self.dataset_dir, self.subdir, f"{sample['id']}_eeg.{self.ext}")]

        # Generate scalograms
        scalograms = []
        output_size = (128, 128)
        for ch in range(data.shape[0]):
            filename = f"scalogram_{sample['id']}_ch{ch}.png"
            scalograms.append(generate_scalogram(data[ch], filename=filename, output_size=output_size))
        scalograms = torch.stack(scalograms)  # Shape: (channels, height, width)

        # Generate spectrograms
        spectrograms = []
        for ch in range(data.shape[0]):
            filename = f"spectrogram_{sample['id']}_ch{ch}.png"
            spectrograms.append(generate_spectrogram(data[ch], filename=filename, output_size=output_size))
        spectrograms = torch.stack(spectrograms)  # Shape: (channels, height, width)

        brainplots = []
        for ch in range(data.shape[0]):
            filename = f"{sample['id']}_eeg.png"
            brainplots.append(load_images_for_split(sample['id'], self.split))
        brainplots = torch.stack(brainplots)    # Shape: (channels, height, width)

        # Augment images
        # scalograms = augment_images(scalograms)
        # spectrograms = augment_images(spectrograms)

        sample = {
            "id": sample['id'],
            "eeg": data,
            "scalogram": scalograms,
            "spectrogram": spectrograms,
            "brainplot": brainplots,
            "song_features": np.array(get_audio_features(sample['spotify_track_id'])).astype(np.float32),
            "label": sample[self.label_name] if "test" not in self.split else -1,
        }


        # breakpoint()
        
        if self.transform:
            sample = self.transform(sample)

        return sample


      
def get_loaders(args):
    
    if args.task == "subject_identification":
        splits = ["train", "val_trial"]
    elif args.task == "emotion_recognition":
        splits = ["train", "val_trial", "val_subject"]
    else:
        raise ValueError(f"Task {args.task} not recognized")
    
    # Define transforms
    train_transforms = T.Compose([
        RandomCrop(args.crop_size),
        ToTensor(label_interface="long"),
        Standardize()
    ])
    
    test_transforms = T.Compose([
        RandomCrop(args.crop_size),
        ToTensor(label_interface="long"),
        Standardize()
    ])

    # Select dataset
    subdir = args.data_type
    if args.data_type == "raw":
        ext = "fif"
    elif args.data_type == "pruned":
        ext = "fif"
    else:
        ext = "npy"

    datasets = {
        split: EremusDataset(
            subdir=subdir,
            split_dir=args.split_dir,
            split=split,
            ext = ext,
            task = args.task,
            transform=train_transforms if split == "train" else test_transforms
        )
        for split in splits
    }
    
    
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size if split == "train" else 1,
            shuffle=True if split == "train" else False,
            num_workers=args.num_workers
        )
        for split, dataset in datasets.items()
    }

    return loaders, args

def get_test_loader(args):
    
    if args.task == "subject_identification":
        splits = ["test_trial"]
    elif args.task == "emotion_recognition":
        # splits = ["test_trial", "test_subject"]
        splits = ["val_trial", "val_subject"]
    else:
        raise ValueError(f"Task {args.task} not recognized")
    
    # Define transforms
    test_transforms = T.Compose([
        RandomCrop(args.crop_size),
        ToTensor(label_interface="long"),
        Standardize()
    ])

    # Select dataset
    subdir = args.data_type
    if args.data_type == "raw":
        ext = "fif"
    elif args.data_type == "pruned":
        ext = "fif"
    else:
        ext = "npy"

    datasets = {
        split: EremusDataset(
        subdir=subdir,
        split_dir=args.split_dir,
        split=split,
        ext = ext,
        task = args.task,
        transform=test_transforms
        ) for split in splits
    }
    
    datasets_no_transform = {
        split: EremusDataset(
        subdir=subdir,
        split_dir=args.split_dir,
        split=split,
        ext = ext,
        task = args.task,
        transform=None
        ) for split in splits
    }
    
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers
        )
        for split, dataset in datasets.items()
    }

    return datasets_no_transform, loaders, args



# DEFINE ARGS

args = SimpleNamespace(
    task="emotion_recognition",
    data_type="preprocessed",
    split_dir= str("data/splits"),
    batch_size=32,
    crop_size=1024,
    num_workers=4,
    epochs=20,
    learning_rate=1e-5
)


loaders, args = get_loaders(args)

train_loader = loaders["train"]
val_loader_s = loaders["val_subject"]
val_loader_t = loaders["val_trial"]

eeg_channels = 32
scalogram_channels = 32
spectrogram_channels = 32
num_classes = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EEGEmotionClassifier(eeg_channels, scalogram_channels, spectrogram_channels, num_classes).to(device)

def train(model, train_loader, val_loader, val_loader_2, args, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    best_val_accuracy = 0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        correct, total = 0, 0
        
        # Training loop
        for batch in train_loader:
            
            # breakpoint()
            eeg = batch["eeg"].to(device)
            spectrogram = batch["spectrogram"].to(device)
            scalogram = batch["scalogram"].to(device)
            labels = batch["label"].to(device)
            song_features = batch["song_features"].to(device)
            brainplot = batch["brainplot"].to(device)
            # breakpoint()

            # plt.imshow(scalogram[0].permute(1, 2, 0).cpu().numpy())
            # plt.show()

            toutputs = model(eeg, scalogram, spectrogram, song_features, brainplot)
            loss = criterion(toutputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = toutputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        train_accuracy = correct / total
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss / len(train_loader):.4f}, Accuracy: {train_accuracy:.4f}")
        
        # Validation loop
        model.eval()
        val_loss_s, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:

                # breakpoint()
                eeg = batch["eeg"].to(device)
                scalogram = batch["scalogram"].to(device)
                spectrogram = batch["spectrogram"].to(device)
                labels = batch["label"].to(device)
                song_features = batch["song_features"].to(device)
                brainplot = batch["brainplot"].to(device)
                
                outputs = model(eeg, scalogram, spectrogram, song_features, brainplot)
                # print(outputs, toutputs)
                val_loss_s += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            
        val_s_accuracy = correct / total
        val_loss_t, correct, total = 0, 0, 0
        print(f"val_subject Loss: {val_loss_s / len(val_loader):.4f}, val_subject accuracy: {val_s_accuracy:.4f}")

        with torch.no_grad():    
            for batch in val_loader_2:
                # breakpoint()
                eeg = batch["eeg"].to(device)
                scalogram = batch["scalogram"].to(device)
                spectrogram = batch["spectrogram"].to(device)
                labels = batch["label"].to(device)
                song_features = batch["song_features"].to(device)
                brainplot = batch["brainplot"].to(device)

                outputs = model(eeg, scalogram, spectrogram, song_features, brainplot)
                # print(outputs, toutputs)
                val_loss_t += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_t_accuracy = correct / total
        print(f"val_trial Loss: {val_loss_t / len(val_loader_2):.4f}, val_trial Accuracy: {val_t_accuracy:.4f}")

        avg_val_accuracy = (val_s_accuracy + val_t_accuracy) / 2
        if avg_val_accuracy >= best_val_accuracy:
            # best_val_accuracy = max(val_s_accuracy, val_t_accuracy)
            best_val_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), f"ckpts/scale/best_model_{best_val_accuracy:.4f}_{epoch}.pth")
            print(f"Saving best model: best_model_{best_val_accuracy:.4f}_{epoch}.pth")


# train(model, train_loader, val_loader_s, val_loader_t, args, device)

# save the model
# torch.save(model.state_dict(), "model.pth")
# print("Model saved!")
# torch.load('sssj_ckpts/best_model_0.3320_31.pth')

model.load_state_dict(torch.load('ckpts/wave/best_model_0.2659_2.pth'))

def test(model, test_loader, args, device, val=False):
    model.eval()
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch["eeg"].to(device)
            spectrogram = batch["spectrogram"].to(device)
            scalogram = batch["scalogram"].to(device)
            song_features = batch["song_features"].to(device)
            brainplot = batch["brainplot"].to(device)

            if val:
                labels.append(batch["label"].item())
            outputs = model(eeg, scalogram, spectrogram, song_features, brainplot)
            _, predicted = outputs.max(1)
            predictions.append(predicted.item())
    
    return predictions, labels

datasets_no_transform, test_loader, args = get_test_loader(args)
predictions, labels = test(model, test_loader["val_subject"], args, device, val=True)
create_confusion_matrix(labels, predictions, title="Confusion Matrix - val_subject")
breakpoint()