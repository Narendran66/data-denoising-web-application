# Efficient Seismic Data Denoising via Deep Learning with Improved AutoEncoder
Here's a detailed explanation of both the code files you provided ("app.py" and "Inference-Code.ipynb") and step-by-step setup instructions to run this on your desktop.

## 1. Code Explanation

### A. `app.py` (Streamlit Web App)

**Goal:** Provide an interactive web interface for uploading noisy audio, denoising it using a pre-trained deep learning model, and listening/downloading the result.

#### Key Components

- **Imports:**
  - `streamlit as st`: Used to build the web interface.
  - `numpy`, `librosa`, `tensorflow`, `soundfile`, `io`: Libraries for audio processing and deep learning.

- **Streamlit Setup:**
  - `st.set_page_config(...)` sets the page's title and design/layout.

- **Model Loading:**
  - `@st.cache_resource` decorator ensures the model is loaded only once and reused (improving app performance).
  - `load_model()` loads the pre-trained Keras model from "model.h5".

- **Audio functions:**
  - `load_audio(file, sr=16000)`: Loads an uploaded audio file and resamples to 16kHz.
  - `compute_spectrogram(audio, n_fft=512, hop_length=256)`: Computes the Short-Time Fourier Transform (STFT), returning magnitude and phase (needed for reconstructing audio).
  - `preprocess_data(audio, n_fft=512, hop_length=256)`: Wrapper to get both magnitude and phase.
  - `denoise_audio(model, noisy_audio, ...)`: Main function. Takes in noisy audio, computes its spectrogram, reshapes for the model, predicts the denoised magnitude spectrogram, reconstructs the waveform using inverse STFT and original phase.

- **Main App Logic:**
  - The actual interface `main()` is not fully shown but would typically:
    - Load model via `load_model()`
    - Allow user to upload audio
    - Call `denoise_audio()` to process uploaded file
    - Provide output for playback/download

### B. `Inference-Code.ipynb` (Jupyter Notebook Inference Script)

**Goal:** Provide a script to denoise an audio file using the same DDAE model in a local (non-web) environment.

#### Main Steps

- **Imports:** Same core libraries as above.

- **Model Loading:**
  - Loads the model with `tf.keras.models.load_model(...)`, specifying the custom loss 'mse'.

- **Function Definitions:**
  - `load_audio(file_path, sr=16000)`: Loads WAV file.
  - `compute_spectrogram`, `preprocess_data`, `denoise_audio`: Same ideas as in the web app, adjusted to script form.

- **Inference Pipeline:**
  1. Set path to noisy WAV file: `noisy_audio_path = 'p232_011.wav'`.
  2. Load audio with `load_audio`.
  3. Denoise with `denoise_audio`, passing the loaded model.
  4. Save denoised audio to new WAV file: `new_denoised_audio.wav`.

## 2. How To Set Up and Run On Your Desktop

### **A. Prerequisites**

- **Python** (Recommend version 3.8+)
- **pip** (Python package installer)
- **Required packages:**
  - streamlit
  - numpy
  - librosa
  - tensorflow (install tensorflow==2.x for compatibility)
  - soundfile

### **B. Installation Steps**

#### 1. Install Python and pip
If not already installed:
- Download from [python.org](https://www.python.org/downloads/)
- On Windows: Select “Add Python to PATH” during installation.

#### 2. Create a new directory for the project

```bash
mkdir audio-denoising-app
cd audio-denoising-app
```

#### 3. Place your files

- Place `app.py`, `Inference-Code.ipynb`, and your trained model (e.g., `model.h5` or `ddae_model.h5`) in this directory.

#### 4. Install required packages

You can make a file called `requirements.txt` with the following content (recommended):

```
streamlit
numpy
librosa
tensorflow
soundfile
```

Then run:

```bash
pip install -r requirements.txt
```

Or, install each directly:

```bash
pip install streamlit numpy librosa tensorflow soundfile
```

#### 5. Running the Streamlit Web App

```bash
streamlit run app.py
```
- This will open the web application in your browser. 
- You’ll be able to upload an audio file and denoise it using the model.

#### 6. Running the Inference Script (Jupyter Notebook)

- Make sure Jupyter is installed:
  ```bash
  pip install notebook
  ```

- Launch Jupyter Notebook (or Jupyter Lab):

  ```bash
  jupyter notebook
  ```

- Open `Inference-Code.ipynb` from the browser tab that opens.
- Set the file path variables (`model_path`, `noisy_audio_path`) to your local file names.
- Run each cell in order to process and save the denoised audio file.

## **Tips and Troubleshooting**
- Ensure your model file (`model.h5` or `ddae_model.h5`) matches the one your code expects.
- For the web app, audio files must be in a format supported by `librosa` (e.g., .wav).
- If GPU is available, TensorFlow will use it for faster inference.
- If you encounter port errors with Streamlit, try a different port:  
  `streamlit run app.py --server.port 8502`

This setup will let you run both a web-based UI for quick demos and a notebook/script for in-depth processing or batch jobs, leveraging your pretrained audio denoising neural network.
### If error Comes Try this...
The error message "Error: Invalid value: File does not exist: app.py" means that the file app.py is not present in the directory from which you are running the streamlit command.

### How to Fix It

#### 1. Check Your Current Folder
- Make sure you are in the same folder where your `app.py` file is saved.

#### 2. Move to Correct Directory
- Use the `cd` (change directory) command in your terminal to navigate to the folder containing `app.py`, for example:

```bash
cd C:\Users\NARENDRAN\audio-denoising-app
```
*(Replace "audio-denoising-app" with your project folder name where app.py is saved.)*

#### 3. List Files to Confirm
- Check if `app.py` is present:

```bash
dir
```
or
```bash
ls
```
- If you see `app.py` in the output, proceed to the next step.

#### 4. Run Streamlit Again

```bash
streamlit run app.py
```

#### 5. If You Don't Have `app.py`
- Make sure you have created and saved `app.py` in the correct folder.
- If you need the template, save the code from your project into a new file called `app.py` in the working directory.

***

**Summary:**  
Make sure `app.py` exists in the folder where you are running your command. Navigate to that folder and retry your command.
