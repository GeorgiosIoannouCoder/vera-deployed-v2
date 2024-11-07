########################################################################
import librosa  # Music and audio analysis.
import numpy as np  # Data wrangling.

from statistics import mode  # Find the most likely predicted emotion.
import tensorflow as tf
import joblib  # Load StandardScaler.

import boto3  # AWS SDK to manage AWS services.
import streamlit as st
import s3fs

########################################################################
def load_model_and_scaler():
    # Session states.
    if "bucket_name" not in st.session_state:
        st.session_state["bucket_name"] = st.secrets["bucket_name"]

    # Create the connection to S3.
    if "s3" not in st.session_state:
        st.session_state["s3"] = s3fs.S3FileSystem(anon=False)

    if "client" not in st.session_state:
        st.session_state["client"] = boto3.client("s3")

    if "model_path" not in st.session_state:
        st.session_state["model_path"] = st.secrets["model_path"]

    if "standard_scaler_path" not in st.session_state:
        st.session_state["standard_scaler_path"] = st.secrets["standard_scaler_path"]

    if "standard_scaler" not in st.session_state:
        st.session_state["client"].download_file(
            st.session_state["bucket_name"],
            st.session_state["standard_scaler_path"],
            "standard_scaler.save",
        )
        st.session_state["standard_scaler"] = joblib.load("standard_scaler.save")

    if "model" not in st.session_state:
        st.session_state["client"].download_file(
            st.session_state["bucket_name"],
            st.session_state["model_path"],
            "model.h5",
        )
        # Load the model.
        st.session_state["model"] = tf.keras.models.load_model("model.h5")


# List of emotions the model was trained on.
emotions_classes = [
    "angry",
    "calm",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]


########################################################################
# Next 4 functions are Audio Data Augmentation:
# Noise Injection.
def inject_noise(data, sampling_rate=0.035, threshold=0.075, random=False):
    if random:
        sampling_rate = np.random.random() * threshold
    noise_amplitude = sampling_rate * np.random.uniform() * np.amax(data)
    augmented_data = data + noise_amplitude * np.random.normal(size=data.shape[0])
    return augmented_data


# Pitching.
def pitching(data, sampling_rate, pitch_factor=0.7, random=False):
    if random:
        pitch_factor = np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)


# Zero crossing rate.
def zero_crossing_rate(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(
        y=data, frame_length=frame_length, hop_length=hop_length
    )
    return np.squeeze(zcr)


# Root mean square.
def root_mean_square(data, frame_length=2048, hop_length=512):
    rms = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rms)


# Mel frequency cepstral coefficients.
def mel_frequency_cepstral_coefficients(
    data, sampling_rate, frame_length=2048, hop_length=512, flatten: bool = True
):
    mfcc = librosa.feature.mfcc(y=data, sr=sampling_rate)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)


# Combined audio data feature extraction.
def feature_extraction(data, sampling_rate, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack(
        (
            result,
            zero_crossing_rate(data, frame_length, hop_length),
            root_mean_square(data, frame_length, hop_length),
            mel_frequency_cepstral_coefficients(
                data, sampling_rate, frame_length, hop_length
            ),
        )
    )
    return result


# Duration and offset act as placeholders because there is no audio in start and the ending of
# each audio file is normally below three seconds.
# Combine audio data augmentation and audio data feature extraction.
def get_features(file_path, duration=2.5, offset=0.6):
    data, sampling_rate = librosa.load(file_path, duration=duration, offset=offset)

    # No audio data augmentation.
    audio_1 = feature_extraction(data, sampling_rate)
    audio = np.array(audio_1)

    # Inject Noise.
    noise_audio = inject_noise(data, random=True)
    audio_2 = feature_extraction(noise_audio, sampling_rate)
    audio = np.vstack((audio, audio_2))

    # Pitching.
    pitch_audio = pitching(data, sampling_rate, random=True)
    audio_3 = feature_extraction(pitch_audio, sampling_rate)
    audio = np.vstack((audio, audio_3))

    # Pitching and Inject Noise.
    pitch_audio_1 = pitching(data, sampling_rate, random=True)
    pitch_noise_audio = inject_noise(pitch_audio_1, random=True)
    audio_4 = feature_extraction(pitch_noise_audio, sampling_rate)
    audio = np.vstack((audio, audio_4))

    audio_features = audio

    return audio_features


# Increase ndarray dimensions to [4,2376].
def increase_ndarray_size(features_test):
    tmp = np.zeros([4, 2377])
    offsets = [0, 1]
    insert_here = tuple(
        [
            slice(offsets[dim], offsets[dim] + features_test.shape[dim])
            for dim in range(features_test.ndim)
        ]
    )

    tmp[insert_here] = features_test
    features_test = tmp
    features_test = np.delete(features_test, 0, axis=1)

    return features_test


# Determine if ndarray needs to be increase in size.
def increase_array_size(audio_features):
    if audio_features.shape[1] < 2376:
        audio_features = increase_ndarray_size(audio_features)
    return audio_features


# Make the prediction.
def predict(audio_features):
    audio_features = st.session_state["standard_scaler"].transform(audio_features)
    audio_features = np.expand_dims(audio_features, axis=2)

    y_pred = st.session_state["model"].predict(audio_features)
    y_pred = np.argmax(y_pred, axis=1)

    try:
        # Model debugging.
        # print("\nPredicted emotion for each and every feature extraction.\n\n", y_pred)
        # print("\nAvailable emotions_classes = ", emotions_classes)
        # print("\nModel predicted emotion: ", emotions_classes[mode(y_pred)])
        return emotions_classes[mode(y_pred)]
    except:
        return emotions_classes[y_pred[0]]
