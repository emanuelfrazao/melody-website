import streamlit as st

import numpy as np

from io import BytesIO
from scipy.io import wavfile
import pretty_midi


import tempfile


from converter import convert_to_midi
# from generation import MelodyGenerator


st.set_page_config(page_title="MIDI File Player", layout="wide")
st.title("Melody Tragic")
st.write("Upload your chords, download your masterpiece!")

# 1. Upload audio file (must be wav, mp3, or midi)
uploaded_file = st.file_uploader("Upload your chords", type=["wav"])
if uploaded_file is not None:
    format = uploaded_file.name.split(".")[-1]
    st.audio(uploaded_file, format=f"audio/{format}", start_time=0)

    # 2. Convert the wav to midi format
    midi = convert_to_midi(uploaded_file)
    instrument = midi.instruments[0]

    # 3. Play the midi file, letting the user choose the instrument
    st.markdown("Choose an instrument to play your chords")
    st.select_slider()
    audio_data = midi.fluidsynth()
    audio_data = np.int16(
        audio_data / np.max(np.abs(audio_data)) * 32767 * 0.9
    )  # -- Normalize for 16 bit audio https://github.com/jkanner/streamlit-audio/blob/main/helper.py

    virtual_file = BytesIO()
    wavfile.write(virtual_file, 44100, audio_data)

    st.audio(virtual_file)
    st.markdown("Download the audio by right-clicking on the media player")

    # 3. Preprocess the midi file for modelling
    #rhythm = build_rhythm_array_from_track(instrument)

    # 4. Get melody generation from model
    #melody_gen = MelodyGenerator(encoder_weights_path='./weights/encoder/', decoder_weights_path='./weights/decoder/')
    #generated_melody = melody_gen.generate(rhythm, stop_time=10)
    #generated_midi = array_to_midi(generated_melody)

    # 5. Join model generation with original MIDI track
    # combine_midi_rhythm_melody(midi, melody_midi_file, output_file='combined_output.mid')


    # 6. Let the user download the MIDI file

    # 7. Have an audio player for the new MIDI track
    # file_temp = tempfile.NamedTemporaryFile()
    # generated_midi.write(file_temp)

    # st.audio(uploaded_file, format=f"audio/mid", start_time=0)
# 1. Upload MIDI file
# uploaded_file = st.file_uploader("Upload a MIDI file", type=["mid"])
# if uploaded_file is not None:
#     # 2. Convert MIDI to WAV using pydub
#     midi_data = uploaded_file.read()
#     midi_audio = AudioSegment.from_file(BytesIO(midi_data), format="mid")

#     # Create a temporary WAV file
#     wav_temp_path = "temp.wav"
#     midi_audio.export(wav_temp_path, format="wav")

#     # Provide a link to play the WAV audio
#     st.audio(wav_temp_path, format="audio/wav", start_time=0)

#     # Remove the temporary WAV file after playing
#     os.remove(wav_temp_path)
#     st.success("Temporary WAV file deleted.")
