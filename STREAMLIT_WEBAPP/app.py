import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import soundfile as sf
import io

# Set page configuration
st.set_page_config(page_title="Audio Denoising App", layout="wide", initial_sidebar_state="expanded")

# Custom CSS styling for improved UI
st.markdown(
    """
    <style>
    /* General body background */
    .reportview-container {
        background: #f0f2f6;
    }
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2E7BCF, #1B4F72);
        color: white;
    }
    /* Title styling */
    .title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        color: #2E7BCF;
        margin-top: 20px;
    }
    /* Subheader styling */
    .subheader {
        font-size: 24px;
        font-weight: bold;
        color: #2E7BCF;
    }
    /* Instruction text styling */
    .instructions {
        font-size: 18px;
        color: #333333;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model (cached to avoid reloading on every interaction)
@st.cache_resource
def load_model():
    model_path = 'model.h5'  # Ensure this path is correct
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={'mse': tf.keras.losses.MeanSquaredError()}
    )
    return model

def load_audio(file, sr=16000):
    """Load audio file from file-like object and return the signal."""
    audio, _ = librosa.load(file, sr=sr)
    return audio

def compute_spectrogram(audio, n_fft=512, hop_length=256):
    """Compute magnitude and phase of the spectrogram."""
    spectrogram = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(spectrogram)
    phase = np.angle(spectrogram)
    return magnitude, phase

def preprocess_data(audio, n_fft=512, hop_length=256):
    """Preprocess audio by computing its spectrogram."""
    magnitude, phase = compute_spectrogram(audio, n_fft, hop_length)
    return magnitude, phase

def denoise_audio(model, noisy_audio, n_fft=512, hop_length=256):
    """Denoise noisy audio using the trained model."""
    noisy_magnitude, noisy_phase = preprocess_data(noisy_audio, n_fft, hop_length)
    noisy_magnitude = noisy_magnitude.T[..., np.newaxis]  # Reshape for model input
    denoised_magnitude = model.predict(noisy_magnitude)
    denoised_magnitude = denoised_magnitude.squeeze().T  # Reshape back to original dimensions
    denoised_spectrogram = denoised_magnitude * np.exp(1j * noisy_phase)
    denoised_audio = librosa.istft(denoised_spectrogram, hop_length=hop_length)
    return denoised_audio

def main():
    # Title and introduction
    st.markdown('<div class="title">Audio Denoising App</div>', unsafe_allow_html=True)
    st.markdown('<div class="instructions">Upload your noisy audio file to denoise it using our advanced deep learning model.</div>', unsafe_allow_html=True)
    
    # Sidebar with instructions and file uploader
    st.sidebar.markdown("## Instructions")
    st.sidebar.markdown(
        """
        - **Upload** a noisy audio file (WAV, MP3, or OGG).
        - The model will process the file and output the denoised audio.
        - **Listen** to the original and denoised versions.
        - **Download** the denoised audio if you like the result.
        """
    )
    
    # Load the model
    with st.spinner("Loading model..."):
        model = load_model()
    st.sidebar.success("Model loaded successfully!")
    
    # File uploader placed in the sidebar for clarity
    uploaded_file = st.sidebar.file_uploader("Choose a noisy audio file", type=["wav", "mp3", "ogg"])
    
    if uploaded_file is not None:
        # Layout: display original and denoised audio in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Original Audio")
            st.audio(uploaded_file, format='audio/wav')
            noisy_audio = load_audio(uploaded_file)
        
        with st.spinner("Denoising audio, please wait..."):
            denoised_audio = denoise_audio(model, noisy_audio)
        st.success("Audio denoised successfully!")
        
        # Convert the denoised audio to WAV format in memory
        wav_io = io.BytesIO()
        sf.write(wav_io, denoised_audio, 16000, format='WAV')
        wav_data = wav_io.getvalue()
        
        with col2:
            st.markdown("### Denoised Audio")
            st.audio(wav_data, format='audio/wav')
            st.download_button(
                label="Download Denoised Audio",
                data=wav_data,
                file_name="denoised_audio.wav",
                mime="audio/wav"
            )
    else:
        st.info("Please upload an audio file from the sidebar to get started.")

if __name__ == "__main__":
    main()
