from basic_pitch.inference import predict
import tempfile
from streamlit import cache_data
from pretty_midi import PrettyMIDI

@cache_data
def convert_to_midi(audio_file) -> PrettyMIDI:
    # create a temporary file (basic-pitch needs it)
    with tempfile.NamedTemporaryFile() as file_path:
        # write contents
        file_path.write(audio_file.getvalue())

        # convert to MIDI by referencing the temporary file
        _, midi, _ = predict(file_path.name)
    return midi
