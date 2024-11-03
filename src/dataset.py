import os
import mne
import json
import numpy as np
from tqdm import tqdm
import src.config as config
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from src.eeg_transforms import RandomCrop, ToTensor, Standardize

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time

# probably unsafe L
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="0104d50f3acb45e789c24dcb0af5fbe8",
                                                           client_secret="2fb2248e08574ac99b345580ff478f15"))   

mne.set_log_level("ERROR")

# Load or initialize a DataFrame to store audio features
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

# Function to get audio features
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
                audio_features['track_id'] = track_id

                # Append to DataFrame and save
                audio_features_df.loc[track_id] = list(audio_features.values())
                audio_features_df.to_csv(AUDIO_FEATURES_FILE)  # Save after each addition
                return list(audio_features.values())  # Return the features as a list

        except Exception as e:
            print(f"Error fetching audio features for track {track_id} on attempt {attempt + 1}: {e}")
            attempt += 1
            time.sleep(2 ** attempt)  # Exponential backoff

    # If all retries fail, return None
    return None
    
class EremusDataset(Dataset):
    def __init__(self, subdir, split_dir, split="train", task="subject_identification", ext="fif", transform=None, prefix=""):
        
        self.dataset_dir = config.get_attribute("dataset_path", prefix=prefix)
        self.subdir = os.path.join(subdir, split) if "test" in split else os.path.join(subdir, "train")
        self.split_dir = split_dir
        self.transform = transform
        self.split = split
        self.label_name = "subject_id" if task == "subject_identification" else "label"
        self.ext = ext
        
        splits = json.load(open(os.path.join(split_dir, f"splits_{task}.json")))
        self.samples = splits[split]
        
        files = []
        for sample in self.samples:
            #path = os.path.join(self.dataset_dir, self.subdir, sample['filename_preprocessed'])
            path = os.path.join(self.dataset_dir, self.subdir, f"{sample['id']}_eeg.{self.ext}")
            files.append(path)
        files = list(set(files))
        #self.files = {f: np.load(f)['arr_0'] for f in files}
        if self.ext == "npy":
            self.files = {f: np.load(f) for f in tqdm(files)}
        elif self.ext == "fif":
            self.files = {f: mne.io.read_raw_fif(f, preload=True).get_data() for f in tqdm(files)}
        else:
            raise ValueError(f"Extension {ext} not recognized")

        ## Load Spotify audio features ##
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
        spotify_info = self.spotify_data.get(sample["id"], {})

        sample = {
            "id": sample['id'],
            "eeg": data,
            "label": sample[self.label_name] if "test" not in self.split else -1,
            "song_features": np.array(get_audio_features(sample['spotify_track_id']))
        }
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
        splits = ["test_trial", "test_subject"]
    else:
        raise ValueError(f"Task {args.task} not recognized")
    
    # Define transforms
    test_transforms = T.Compose([
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