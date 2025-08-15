# muse-receiver (Windows)

A minimal, dark-themed desktop application to:

- Find/resolve Muse LSL streams
- Connect and display incoming raw data (EEG/ACC/GYRO/PPG)
- Visualize time series, spectrogram, topography
- Show basic signal quality per electrode
- Compute and visualize canonical EEG band powers (delta/theta/alpha/beta/gamma)

Font: Inter (place TTFs in `assets/`)  •  Theme: Black background, white text

## Setup (Windows)

1) Install Python 3.11+ and Visual C++ Build Tools if needed.
2) Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3) Install dependencies:

```bash
pip install -r requirements.txt
```

4) (Optional) Install muselsl if you plan to discover/connect via BLE:

```bash
pip install muselsl
```

5) Place Inter font files (e.g., `Inter-Regular.ttf`, `Inter-Medium.ttf`, `Inter-SemiBold.ttf`) into `assets/`.

## Run

```bash
python app/main.py
```

- Click "Find Muse" to populate available EEG LSL streams
- Choose a stream by name and click "Connect"
- Switch tabs for Stream, Spectral, and Topography views

To start a Muse LSL stream from the same machine:

```bash
muselsl list
muselsl stream -n "Muse-XXXX" --acc --gyro --ppg
```

## Notes

- Spectrogram, topography, heart rate, and quality metrics are basic prototypes designed to be scientifically reasonable and computationally light. You can refine them in `app/processing.py`.
- The topography uses a nearest-neighbor interpolation over a unit head map; supply your electrode coordinates for better fidelity.
- All plots are light on labels for a minimal aesthetic; adjust in `app/main.py` as needed.
