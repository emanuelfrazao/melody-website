import numpy as np
import pandas as pd
import pretty_midi

def postprocess_melody_encoded(melody_encoded: np.ndarray) -> pd.DataFrame:    
    melody_encoded = melody_encoded.copy()

    # De-normalize
    melody_df = pd.DataFrame(melody_encoded, columns=['velocity', 'step', 'duration', 'octave', 'semitone'])
    melody_df['velocity'] *= 127
    melody_df['step'] *= 10
    melody_df['octave'] *= 10
    melody_df[['velocity', 'octave', 'semitone']] = melody_df[['velocity', 'octave', 'semitone']].astype(int)

    # Restore start and end times
    melody_df['start'] = melody_df['step'].cumsum()
    melody_df['end'] = melody_df['start'] + melody_df['duration']

    # Restore pitch
    melody_df['pitch'] = melody_df['octave'] * 12 + (melody_df['semitone'] - 1)

    # Keep only the relevant columns
    melody_df = melody_df[['start', 'end', 'velocity', 'pitch']]
    
    return melody_df

def row_to_note(note_row: pd.Series) -> pretty_midi.Note:
    return pretty_midi.Note(
        velocity=int(note_row['velocity']),
        pitch=int(note_row['pitch']),
        start=note_row['start'],
        end=note_row['end']
    )

def dataframe_to_instrument(notes_df: pd.DataFrame, program: int=25) -> pretty_midi.Instrument:
    instrument = pretty_midi.Instrument(program=program)
    notes = notes_df.apply(row_to_note, axis=1).tolist()
    instrument.notes = notes
    return instrument
