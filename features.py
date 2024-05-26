import librosa
import numpy as np

def extract_features(file_name):
    """
    Extracts features from an audio file using librosa.
    
    Parameters:
        file_name (str): Path to the audio file.
    
    Returns:
        np.ndarray: An array of extracted features.
    """
    y, sr = librosa.load(file_name, sr=None)
    #mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    #chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    #contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    #tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    #features = np.vstack((mfcc, mel))
    features = mel
    
    return features

def average_features(features, n_frames=10):
    """
    Calculates the average of each feature vector over n_frames.
    
    Parameters:
        features (np.ndarray): Array of features.
        n_frames (int): Number of frames over which to average the features.
    
    Returns:
        np.ndarray: Array of averaged feature vectors.
    """
    num_features = features.shape[0]
    frame_length = features.shape[1] // n_frames
    averaged_features = np.zeros((num_features, n_frames))
    
    for i in range(n_frames):
        start = i * frame_length
        end = start + frame_length
        averaged_features[:, i] = np.mean(features[:, start:end], axis=1)
    
    return averaged_features
