import streamlit as st

import numpy as np
from io import BytesIO

from preprocessing import (
    instrument_to_dataframe,
    process_rhythm_dataframe,
    preprocess_rhythm_subsequence
)

from model import generate

from postprocessing import (
    postprocess_melody_encoded, 
    dataframe_to_instrument
)


from converter import convert_to_midi
# from generation import MelodyGenerator


st.set_page_config(page_title="MIDI File Player", layout="wide")
st.title("Melody Magic")
st.write("Upload your chords, download your masterpiece!")

# Initialization = True

# 1. Upload audio file (must be wav, mp3, or midi)
uploaded_file = st.file_uploader("Upload your chords", type=["wav"])
if uploaded_file is not None:
    format = uploaded_file.name.split(".")[-1]
    st.audio(uploaded_file, format=f"audio/{format}", start_time=0)


    
    if st.button("Generate"):
        # 2. Convert the wav to midi format
        with st.spinner('Converting your chords to MIDI...'):
            midi = convert_to_midi(uploaded_file)
            instrument = midi.instruments[0]

            
        with st.spinner('Generating your melody...'):
        # 3. Preprocess the midi file for modelling
            instrument_df = instrument_to_dataframe(instrument)
            processed_df = process_rhythm_dataframe(instrument_df)
            x_num, x_cat = preprocess_rhythm_subsequence(processed_df)

            X_num = np.expand_dims(x_num, axis=0)
            X_cat = np.expand_dims(x_cat, axis=0)

        # 4. Get melody generation from model
            melody = generate(x_num, x_cat, temperature=0.8)
            melody_df = postprocess_melody_encoded(melody)
            melody_instrument = dataframe_to_instrument(melody_df)
            st.success("Melody generated!")

        # 5. Join model generation with original MIDI track
            midi.instruments.append(melody_instrument)
            st.success("Melody joined with chords!")
        
        # 6. Let the user download the MIDI file
        midi_data = BytesIO()
        midi.write(midi_data)
        midi_data.seek(0)  # Reset the file pointer to the beginning

        st.download_button(
            label="Download your masterpiece!",
            data=midi_data,
            file_name='chords_and_melody.mid',
            mime="audio/mid"
        )
        