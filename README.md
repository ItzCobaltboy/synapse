# synapse

Pipeline for real-time EMG gesture classification using MATLAB feature extraction and a Python FastAPI inference server.

## What this repository contains
- Real-time EMG signal ingestion (serial stream from ESP32/EMG hardware)
- Feature extraction in MATLAB (EMG toolbox + FFT/wavelet features)
- SVM-based gesture classification in Python (scikit-learn, One-vs-One)
- Optional personal calibration to remap predicted classes and normalize channels
- Keyboard control output (`W/A/S/D/Space`) via prediction results

## Repository structure
```text
synapse/
  calibration/
    calibration_collect_raw.m
    build_calibration.m
    calibration_raw.mat
    user_calibration.json
  matlab_realtime/
    int_feature.m
    realtime_predict.m
    EMG-Feature-Extraction-Toolbox/
  model/
    model.py
    svm_quadratic_python.joblib
  server/
    main.py
  docs/
    PROJECT_STRUCTURE.md
    SETUP_AND_RUN.md
  README.md
```

## End-to-end flow
1. MATLAB reads EMG samples from serial (`<ch1,ch2>` format).
2. MATLAB extracts features per window (`WINDOW=100`) and sends feature vectors to FastAPI `/predict`.
3. Python server scales features and runs SVM inference.
4. Prediction is optionally remapped using user calibration (`calibration/user_calibration.json`).
5. Server returns gesture ID and emits mapped keypress.

## Quick start
### 1) Python environment
```bash
python -m venv .venv
.venv\Scripts\activate
pip install fastapi "uvicorn[standard]" numpy scikit-learn joblib pydantic pynput
```

### 2) Start prediction server
```bash
python server/main.py
```
Server runs at `http://127.0.0.1:8000`.

### 3) (Optional) Build personal calibration
1. In MATLAB, run `calibration/calibration_collect_raw.m`.
2. Then run `calibration/build_calibration.m`.
3. This creates `calibration/user_calibration.json`.

### 4) Run real-time predictor in MATLAB
Run `matlab_realtime/realtime_predict.m`.

## API
### `POST /predict`
Request body:
```json
{
  "features": [0.12, -0.03, 1.44]
}
```
Response:
```json
{
  "gesture": 3
}
```

### `GET /latest`
Returns latest predicted gesture or `"none"`.

## Configuration notes
- Serial port is currently set to `COM5` in MATLAB scripts. Change it for your machine.
- Gesture-key mapping is defined in `server/main.py`.
- Model path and calibration paths are currently absolute in scripts. Update these paths if your project location differs.

## Showcase checklist
- Include one short demo GIF/video in the GitHub README.
- Add hardware details (sensor, board, sampling rate) in `docs/SETUP_AND_RUN.md`.
- Add one architecture diagram for quick reviewer understanding.

## License

Code written specifically for the Synapse real-time EMG inference pipeline is released under the **MIT License**.

This repository also includes third-party components:

- `matlab_realtime/EMG-Feature-Extraction-Toolbox/`  
  Licensed under the **BSD 3-Clause License** by its original author.

- Portions of implementation derived from work associated with  
  Fuentes-Aguilar et al. (2024), licensed under **CC BY-NC**.

All third-party code remains under its respective original license.
Users should review those licenses before reuse or redistribution.

## References and Attribution

This project integrates signal processing utilities and concepts derived from the following sources.

### EMG Feature Extraction Toolbox
Feature extraction functions included in:

`matlab_realtime/EMG-Feature-Extraction-Toolbox/`

originate from the **EMG Feature Extraction Toolbox** repository by JingweiToo.

Original repository:  
https://github.com/JingweiToo/EMG-Feature-Extraction-Toolbox

This toolbox is licensed under the **BSD 3-Clause License**.

### Research dataset and implementation
This project also draws from work presented in:

Fuentes-Aguilar, R. Q., Llorente-Vidrio, D., Campos-Macías, L., & Morales-Vargas, E. (2024).  
*Surface electromyography dataset from different movements of the hand using a portable and a non-portable device.*  
Data in Brief, 57, 111079.  
https://doi.org/10.1016/j.dib.2024.111079

Portions of the code from the authors’ implementation were adapted for use in this repository.  
Their implementation is distributed under **Creative Commons Attribution–NonCommercial (CC BY-NC)**.

This repository integrates these components into a real-time EMG gesture classification pipeline for research and educational purposes.