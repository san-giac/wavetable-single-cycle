import streamlit as st
import os
import io
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import zipfile

st.set_page_config(layout="wide")

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def midi_to_freq(midi_note):
    """Convert a MIDI note number to frequency (A4=440 Hz)."""
    A440 = 440.0
    return A440 * 2 ** ((midi_note - 69) / 12.0)

def get_fractional_zero_cross(y, integer_cross):
    """Return fractional sample index of the zero crossing between integer_cross and integer_cross+1."""
    if integer_cross < 0 or integer_cross >= len(y) - 1:
        return float(integer_cross)
    y1, y2 = y[integer_cross], y[integer_cross + 1]
    denom = y2 - y1
    if abs(denom) < 1e-12:
        return float(integer_cross)
    alpha = -y1 / denom
    alpha = max(0, min(1, alpha))
    return integer_cross + alpha

def refined_zero_cross_index(y, integer_cross, window=10):
    """
    Search +/- 'window' samples around 'integer_cross' for the largest slope crossing.
    Then return fractional crossing using get_fractional_zero_cross.
    """
    t_start = max(0, integer_cross - window)
    t_end = min(len(y) - 2, integer_cross + window)

    best_t = integer_cross
    best_slope_mag = 0.0
    for t in range(t_start, t_end + 1):
        slope = y[t + 1] - y[t]
        if abs(slope) > best_slope_mag:
            best_slope_mag = abs(slope)
            best_t = t

    return get_fractional_zero_cross(y, best_t)

def fractional_slice(y, start_frac, end_frac, num_samples=2048):
    """Interpolates a segment of y between start_frac and end_frac into num_samples points."""
    if end_frac < start_frac:
        start_frac, end_frac = end_frac, start_frac
    x = np.linspace(start_frac, end_frac, num_samples, endpoint=False)
    return np.interp(x, np.arange(len(y)), y)

def normalize_to_minus_6dbfs(signal):
    """Normalize signal to around -6 dBFS."""
    peak = np.max(np.abs(signal))
    if peak < 1e-12:
        return signal
    target_linear = 10 ** (-0.5 / 20.0)
    scale = target_linear / peak
    return signal * scale

def generate_playback_wave(single_cycle, duration=2.0, freq=261.63, sr=44100):
    """
    Generate a wave of 'duration' seconds at 'freq' using 'single_cycle' as the wavetable.
    We'll treat single_cycle (2048 samples) as one period, stepping through it at a rate
    that yields freq Hz at the given sample rate.
    """
    wave_len = len(single_cycle)
    phase_inc = (wave_len * freq) / sr
    total_samples = int(sr * duration)

    out = np.zeros(total_samples, dtype=np.float32)
    phase = 0.0
    for i in range(total_samples):
        idx = int(phase) % wave_len
        out[i] = single_cycle[idx]
        phase += phase_inc

    return out

def chunk_dict_items(data_dict, chunk_size=4):
    """Group items from 'data_dict' in sub-lists of size 'chunk_size'."""
    items = list(data_dict.items())
    for i in range(0, len(items), chunk_size):
        yield items[i:i+chunk_size]

# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------

st.title("Wavetable Single Cycle Extractor")

st.write("""
This thing allows you to extract single cycles from any audio file given it original pitch. These can subsequently be used in wavetables.
""")

# Session state
if "analysis_data" not in st.session_state:
    # We'll store per-file data in a dict
    st.session_state.analysis_data = {}

analysis_data = st.session_state.analysis_data

# File uploader
uploaded_files = st.file_uploader(
    "Upload audio files (WAV, MP3, FLAC, OGG, AIFF, etc.)",
    type=["wav", "mp3", "flac", "ogg", "aiff", "aif"],
    accept_multiple_files=True
)

# MIDI note input
midi_note = st.number_input("Enter MIDI Note:", min_value=0, max_value=127, value=60, step=1)
fundamental_freq = midi_to_freq(midi_note)

# Analyze button
if st.button("Analyze Files"):
    if not uploaded_files:
        st.warning("Please upload at least one audio file.")
    else:
        # Clear old analysis
        st.session_state.analysis_data = {}
        analysis_data = st.session_state.analysis_data

        for upfile in uploaded_files:
            file_name = upfile.name
            y, sr = librosa.load(upfile, sr=None, mono=True)
            zc = np.where(np.diff(np.sign(y)))[0]
            period = sr / fundamental_freq

            analysis_data[file_name] = {
                "audio": y,
                "sr": sr,
                "zero_crossings": zc,
                "current_zc_index": 0,
                "inverted": False,
                "offset": 0.0,
                "period": period
            }

        st.success("Analysis complete! See results below.")

# ---------------------------------------------------------------------
# Display side-by-side in chunks of 4
# ---------------------------------------------------------------------
if analysis_data:
    st.header("Analysis Results (Side by Side)")
    for chunk in chunk_dict_items(analysis_data, 4):
        columns = st.columns(len(chunk))
        for col, (file_name, data) in zip(columns, chunk):
            with col:

                st.subheader(file_name)

                y = data["audio"]
                sr = data["sr"]
                zc = data["zero_crossings"]
                zc_idx = data["current_zc_index"]
                offset = data["offset"]
                inverted = data["inverted"]
                period = data["period"]

                if len(zc) == 0:
                    st.warning("No zero-crossings found!")
                    continue

                r1c1, r1c2, r1c3, r1c4 = st.columns(4)
                with r1c1:
                    if st.button("Prev ZC", key=f"prev_{file_name}"):
                        data["current_zc_index"] = max(0, zc_idx - 1)

                with r1c2:
                    if st.button("Next ZC", key=f"next_{file_name}"):
                        data["current_zc_index"] = min(zc_idx + 1, len(zc) - 1)

                with r1c3:
                    st.markdown(f"**ZC**: {data['current_zc_index']}")

                with r1c4:
                    if st.button("Invert", key=f"invert_{file_name}"):
                        data["inverted"] = not data["inverted"]

                r2c1, r2c2, r2c3, r2c4 = st.columns(4)
                with r2c1:
                    if st.button("Shift -", key=f"shift_minus_{file_name}"):
                        data["offset"] -= selected_shift
                with r2c2:
                    if st.button("Shift +", key=f"shift_plus_{file_name}"):
                        data["offset"] += selected_shift
                with r2c3:
                    " "
                with r2c4:
                    shift_amounts = [0.5, 10, 100]
                    selected_shift = st.selectbox(
                        "Shift Step",
                        shift_amounts,
                        key=f"shift_select_{file_name}"
                )

                zc_idx = data["current_zc_index"]
                offset = data["offset"]
                inverted = data["inverted"]

                sign = -1.0 if inverted else 1.0
                start_int = zc[zc_idx]
                refined_start = refined_zero_cross_index(sign * y, start_int, window=10)
                start_frac = refined_start + offset
                start_frac = np.clip(start_frac, 0, len(y) - 1)
                end_guess = start_frac + period

                next_cross_idx = None
                for cross in zc:
                    frac_c = refined_zero_cross_index(sign * y, cross, window=10)
                    if frac_c > end_guess:
                        next_cross_idx = frac_c
                        break

                if next_cross_idx is not None:
                    end_frac = next_cross_idx
                else:
                    end_frac = min(end_guess, len(y) - 1)

                extracted_cycle = fractional_slice(sign * y, start_frac, end_frac, num_samples=2048)
                extracted_cycle = normalize_to_minus_6dbfs(extracted_cycle)

                fig, ax = plt.subplots(figsize=(7,3)) 
                ax.plot(extracted_cycle, linewidth=1.0)
                ax.set_title(f"ZC={zc_idx}, off={offset:.1f}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(True, linestyle=':')
                st.pyplot(fig)

                wave_2s = generate_playback_wave(extracted_cycle, duration=2.0, freq=261.63, sr=44100)
                wave_2s = normalize_to_minus_6dbfs(wave_2s)

                audio_bytes = io.BytesIO()
                sf.write(audio_bytes, wave_2s, 44100, format="WAV")

                st.audio(audio_bytes.getvalue(), format="audio/wav")

                single_cycle_io = io.BytesIO()
                sf.write(single_cycle_io, extracted_cycle, sr, format="WAV")
                st.download_button(
                    label="Download Single-Cycle WAV",
                    data=single_cycle_io.getvalue(),
                    file_name=f"{os.path.splitext(file_name)[0]}_zc{zc_idx}.wav",
                    mime="audio/wav"
                )

    st.write("---")
    if st.button("Export All Single-Cycles as ZIP"):
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, mode="w") as zf:
            for file_name, data in analysis_data.items():
                zc = data["zero_crossings"]
                if len(zc) == 0:
                    continue
                y = data["audio"]
                sr = data["sr"]
                offset = data["offset"]
                zc_idx = data["current_zc_index"]
                inverted = data["inverted"]
                period = data["period"]
                if zc_idx < 0 or zc_idx >= len(zc):
                    continue

                sign = -1.0 if inverted else 1.0
                start_int = zc[zc_idx]
                refined_start = refined_zero_cross_index(sign * y, start_int, window=10)
                start_frac = refined_start + offset
                start_frac = np.clip(start_frac, 0, len(y) - 1)
                end_guess = start_frac + period

                next_cross_idx = None
                for cross in zc:
                    frac_c = refined_zero_cross_index(sign * y, cross, window=10)
                    if frac_c > end_guess:
                        next_cross_idx = frac_c
                        break

                if next_cross_idx is not None:
                    end_frac = next_cross_idx
                else:
                    end_frac = min(end_guess, len(y) - 1)

                extracted_cycle = fractional_slice(sign * y, start_frac, end_frac, num_samples=2048)
                extracted_cycle = normalize_to_minus_6dbfs(extracted_cycle)

                bio = io.BytesIO()
                sf.write(bio, extracted_cycle, sr, format="WAV")
                cycle_name = f"{os.path.splitext(file_name)[0]}_zc{zc_idx}.wav"
                zf.writestr(cycle_name, bio.getvalue())

        st.download_button(
            label="Download ZIP of All Single-Cycles",
            data=zip_buf.getvalue(),
            file_name="all_segments.zip",
            mime="application/zip"
        )
