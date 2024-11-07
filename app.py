########################################################################
import random  # Random variable generators.
import streamlit as st  # Streamlit.

from audio_processing import load_model_and_scaler, get_features, increase_array_size, predict

import uuid
import os

########################################################################
load_model_and_scaler()
########################################################################
# Use local CSS.
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


########################################################################

# Load CSS.
local_css("./static/style.css")

# Prompts used in training data.
prompts = [
    "Kids are talking by the door",
    "Dogs are sitting by the door",
    "It's eleven o'clock",
    "That is exactly what happened",
    "I'm on my way to the meeting",
    "I wonder what this is about",
    "The airplane is almost full",
    "Maybe tomorrow it will be cold",
    "I think I have a doctor's appointment",
    "Say the word apple",
]

emotion_dict = {
    "angry": "angry 😡",
    "calm": "calm 😌",
    "disgust": "disgusted 🤢",
    "fear": "scared 😨",
    "happy": "happy 😆",
    "neutral": "neutral 🙂",
    "sad": "sad 😢",
    "surprise": "surprised 😳",
}

# Session states.

if "audio_value" not in st.session_state:
    st.session_state["audio_value"] = ""

if "prompt" not in st.session_state:
    st.session_state["prompt"] = ""

if "emotion" not in st.session_state:
    st.session_state["emotion"] = ""

if "is_prompt" not in st.session_state:
    st.session_state["is_prompt"] = False

if "is_emotion" not in st.session_state:
    st.session_state["is_emotion"] = False

if "is_first_time_prompt" not in st.session_state:
    st.session_state["is_first_time_prompt"] = True


def make_grid(rows, cols):
    grid = [0] * rows
    for i in range(rows):
        with st.container():
            grid[i] = st.columns(cols)
    return grid


# Title.
title = f"""<p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.3rem;">
            Voice Emotion Recognition on Audio</p>"""
st.markdown(title, unsafe_allow_html=True)

# Image.
image = "./static/waveform.jpg"
st.image(image, use_container_width=True)

# Header.
header = f"""<p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 1.7rem;">
            Click to generate a random prompt and emotion:</p>"""
st.markdown(header, unsafe_allow_html=True)


########################################################################
# Prompt button.
def prompt_btn():
    prompt = '"' + random.choice(prompts) + '"'
    st.session_state["prompt"] = prompt

    st.markdown(
        f"""
            <p align="center" style="font-family: monospace; color: #ffffff; font-size: 2rem;">
            {st.session_state["prompt"]}</p>
        """,
        unsafe_allow_html=True,
    )

    if not (st.session_state["is_first_time_prompt"]):
        st.markdown(
            f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                Try to sound {emotion_dict.get(st.session_state["emotion"])}</p>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                Please generate an emotion!</p>
            """,
            unsafe_allow_html=True,
        )


# Emotion button.
def emotion_btn():
    st.session_state["is_first_time_prompt"] = False

    emotion = random.choice(list(emotion_dict))
    partition = emotion_dict.get(emotion).split(" ")
    emotion = partition[0]
    st.session_state["emotion"] = emotion

    if st.session_state["emotion"] == "disgusted":
        st.session_state["emotion"] = "disgust"

    if st.session_state["emotion"] == "scared":
        st.session_state["emotion"] = "fear"

    if st.session_state["emotion"] == "surprised":
        st.session_state["emotion"] = "surprise"

    st.markdown(
        f"""
            <p align="center" style="font-family: monospace; color: #ffffff; font-size: 2rem;">
            {st.session_state["prompt"]}</p>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
            Try to sound {emotion_dict.get(st.session_state["emotion"])}</p>
        """,
        unsafe_allow_html=True,
    )


def save_audio_file(audio_data, filename="audio.wav"):
    """Save audio file to disk"""
    try:
        unique_id = str(uuid.uuid4())
        filename = f"audio_{unique_id}.wav"
        
        with open(filename, "wb") as f:
            f.write(audio_data.getvalue())
        
        # Store the filename in session state for later use
        st.session_state["current_audio_file"] = filename
        
        return True
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        return False


# Record button.
def record_btn():
    if st.session_state["audio_value"] is not None:
        if save_audio_file(st.session_state["audio_value"]):
            st.success("Recording saved successfully!")
            st.session_state["recording_saved"] = True
    else:
        st.warning("Please record audio first!")


# Play button.
def play_btn():  # Play the recorded audio.
    st.markdown(
        f"""
            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2rem;">
            {st.session_state["prompt"]}</p>
        """,
        unsafe_allow_html=True,
    )
    if not (st.session_state["is_first_time_prompt"]):
        st.markdown(
            f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                Try to sound {emotion_dict.get(st.session_state["emotion"])}</p>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                Please generate a prompt and an emotion!</p>
                <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">Then, record your audio.</p>
            """,
            unsafe_allow_html=True,
        )
    try:  # Load audio file.
        # audio_file = open("audio.wav", "rb")
        # audio_bytes = audio_file.read()
        # st.audio(audio_bytes)
        audio_file = open(st.session_state["current_audio_file"], "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        audio_file.close()
    except:
        st.write("Please record and save sound first.")


# Classify button.
def classify_btn():
    try:
        if "current_audio_file" not in st.session_state:
            st.write("Please record and save sound first.")
            return
        
        wav_path = st.session_state["current_audio_file"]
        audio_features = get_features(wav_path)

        audio_features = increase_array_size(audio_features)
        emotion = predict(audio_features)

        if emotion == "disgust":
            emotion = "disgusted"

        if emotion == "fear":
            emotion = "scared"

        if emotion == "surprise":
            emotion = "surprised"

        if st.session_state["emotion"] == "disgust":
            st.session_state["emotion"] = "disgusted"

        if st.session_state["emotion"] == "fear":
            st.session_state["emotion"] = "scared"

        if st.session_state["emotion"] == "surprise":
            st.session_state["emotion"] = "surprised"

        if st.session_state["emotion"] != "":
            if emotion in st.session_state["emotion"]:
                st.markdown(
                    f"""
                            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                            You tried to sound {st.session_state["emotion"].upper()} and you sounded {emotion.upper()}</p>
                            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">Well done!👍</p>
                        """,
                    unsafe_allow_html=True,
                )

                try:  # Load audio file.
                    audio_file = open("audio.wav", "rb")
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes)

                except:
                    st.write("Please record sound first.")
                st.balloons()

            else:
                st.markdown(
                    f"""
                            <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;">
                            You tried to sound {st.session_state["emotion"].upper()} however you sounded {emotion.upper()}👎</p>
                        """,
                    unsafe_allow_html=True,
                )

                try:  # Load audio file.
                    audio_file = open(st.session_state["current_audio_file"], "rb")
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes)
                    audio_file.close()
                    os.remove(wav_path)
                    del st.session_state["current_audio_file"]

                except:
                    st.write("Please record sound first.")
        else:
            st.markdown(
                f"""
                    <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 2.5rem;"> Please generate a prompt and an emotion.</p>
                    """,
                unsafe_allow_html=True,
            )
    except Exception as e:
        st.write(e)
        st.write("Something went wrong. Please try again.")
        if "current_audio_file" in st.session_state:
            try:
                os.remove(st.session_state["current_audio_file"])
                del st.session_state["current_audio_file"]
            except:
                pass


########################################################################

# Create custom grid.
grid1 = make_grid(3, (12, 12, 4))

# Prompt Button.
prompt = grid1[0][0].button("Prompt")
if prompt or st.session_state["is_prompt"]:
    st.session_state["is_emotion"] = False
    prompt_btn()

# Emotion Button.
emotion = grid1[0][2].button("Emotion")
if emotion or st.session_state["is_emotion"]:
    st.session_state["is_prompt"] = False
    emotion_btn()

st.markdown("### Record Your Voice")
st.session_state["audio_value"] = st.audio_input(
    label="Record audio: Keep audio recording to 3 seconds for better results!",
    help="Enable microphone access in your web browser!",
    key="audio_recorder",
)

# Create custom grid.
grid2 = make_grid(3, (12, 12, 4))

# Record Button.
record = grid2[0][0].button("Save Recording")
if record:
    record_btn()

# Play Button.
play = grid2[0][1].button("Play")
if play:
    play_btn()

# Classify Button.
classify = grid2[0][2].button("Classify")
if classify:
    classify_btn()

st.markdown(
    """
    <style>
        .stButton button {
            padding: 0px 2px;
            border: 3px solid #e5bf4d;
            border-radius: 10px;
    </style>
""",
    unsafe_allow_html=True,
)

# GitHub repository of project.
st.markdown(
    f"""
        <p align="center" style="font-family: monospace; color: #FAF9F6; font-size: 1rem;"><b> Check out our
        <a href="https://github.com/GeorgiosIoannouCoder/vera-deployed-v2" style="color: #FAF9F6;"> GitHub repository</a></b>
        </p>
   """,
    unsafe_allow_html=True,
)
