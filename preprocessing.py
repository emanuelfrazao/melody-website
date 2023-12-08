import pretty_midi
import numpy as np
import pandas as pd
from keras.utils import to_categorical


def note_to_dict(note: pretty_midi.Note) -> dict[str, float]:
    return {
        'start': note.start,
        'end': note.end,
        'pitch': note.pitch,
        'velocity': note.velocity
    }

def instrument_to_dataframe(instrument: pretty_midi.Instrument) -> pd.DataFrame:
    return pd.DataFrame([note_to_dict(note) for note in instrument.notes])


# Unfold a chord
def unfold_rhythm_chord(notes_df: pd.DataFrame, notes_per_chord: int=6) -> pd.DataFrame:
    """Unfold a chord into multiple columns.
       Assumes that the DataFrame contains `start` and `pitch` columns.
    """
    pitches_by_start = notes_df.groupby('start')['pitch'].apply(np.array).apply(pd.Series)
    return pitches_by_start.reindex(labels=range(notes_per_chord), axis=1).fillna(0).astype(int)


# Extract note name and octave
@np.vectorize
def get_semitone(note_number: int) -> str:
    return 1 + (note_number % 12) if note_number > 0 else 0 # 0 is silence

@np.vectorize
def get_octave(note_number: int, default: int=-1) -> int:
    return note_number // 12 - 1 if note_number > 0 else -1


# Extract octave min and max
def get_octaves_min_max(octaves_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame with the minimum and maximum octave for each row."""
    return pd.DataFrame({
            'min_octave': octaves_df.apply(lambda x: x[x >= 0].min(), axis=1),
            'max_octave': octaves_df.apply(lambda x: x[x >= 0].max(), axis=1),
        })


# Get notes metrics
def get_notes_metrics(notes_df: pd.DataFrame) -> pd.DataFrame:
    """Returns a DataFrame with the "metrics" of each note aggregated by start time:
        - start
        - end
        - velocity
        - step
        - duration
    """
    metrics = notes_df.groupby('start').agg({
        'start': 'first',
        'end': 'max',
        'velocity': 'max',
    }).reset_index(drop=True)

    # 4. Add steps and duration
    metrics['step'] = metrics['start'].diff().fillna(0.)
    metrics['duration'] = metrics['end'] - metrics['start']

    return metrics


def process_rhythm_dataframe(rhythm_df: pd.DataFrame) -> pd.DataFrame:
    """Process a rhythm instrument into a DataFrame with the following columns:
        - start
        - end
        - step
        - duration
        - velocity
        - min_octave
        - max_octave
        - note_0
        - note_1
        - note_2
    """
    # 1. Unfold chords
    unfolded_chords = unfold_rhythm_chord(rhythm_df)
        # Get octaves min and max
    notes_octaves = unfolded_chords.apply(get_octave)
    notes_octaves_min_max = get_octaves_min_max(notes_octaves)
        # Get notes names
    notes_names = unfolded_chords.apply(get_semitone)
        # Merge both
    notes_octaves_names = notes_octaves_min_max.join(notes_names)

    # 2. Aggregate notes by start time
    notes_metrics = get_notes_metrics(rhythm_df)

    # 3. Merge notes metrics with notes octaves and names, and return
    notes = notes_metrics.merge(notes_octaves_names, left_on='start', right_index=True)
    return notes

def preprocess_rhythm_subsequence(rhythm_subsequence: pd.DataFrame) -> np.ndarray:
    rhythm_subsequence = rhythm_subsequence.copy()
    rhythm_subsequence.drop(columns=['start', 'end'], inplace=True)
    rhythm_subsequence['velocity'] /= 127
    rhythm_subsequence['step'] /= 10
    rhythm_subsequence['min_octave'] /= 10
    rhythm_subsequence['max_octave'] /= 10

    num_cols = ['velocity', 'step', 'duration', 'min_octave', 'max_octave']
    num = rhythm_subsequence[num_cols].values
    cat = rhythm_subsequence.drop(columns=num_cols).values
    cat_one_hot = to_categorical(cat, 13).any(axis=1).astype(int)
    return num, cat_one_hot