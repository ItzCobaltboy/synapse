# Setup and Run

## Prerequisites
- Python 3.10+
- MATLAB with required signal/wavelet toolboxes used by `int_feature.m`
- EMG hardware streaming serial lines in format `<value1,value2>`. This can be done on ESP32 or Arduino or any other Microcontroller
- It is recommended to apply smoothening to recorded EMG values at MCU level before publishing on Serial

## Python setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install fastapi "uvicorn[standard]" numpy scikit-learn joblib pydantic pynput
python server/main.py
```

## MATLAB setup
1. Open the repository in MATLAB.
2. Ensure `matlab_realtime/EMG-Feature-Extraction-Toolbox` is on MATLAB path.
3. Update `port = 'COM5'` in:
   - `calibration/calibration_collect_raw.m`
   - `matlab_realtime/realtime_predict.m`

## Calibration flow
1. Run `calibration/calibration_collect_raw.m`.
2. Perform each prompted gesture (`rest`, `g1`..`g4`).
3. Run `calibration/build_calibration.m`.
4. Confirm `calibration/user_calibration.json` exists.

## Realtime inference flow
1. Start Python server (`python server/main.py`).
2. Run MATLAB script `matlab_realtime/realtime_predict.m`.
3. Verify server logs predictions and keypress behavior.

## Troubleshooting
- `Expected N, got M` error on `/predict`:
  - Feature vector length does not match training configuration.
- No predictions received:
  - Confirm FastAPI server is running on `127.0.0.1:8000`.
  - Check serial parsing format from hardware.
- Calibration not applied:
  - Ensure `calibration/user_calibration.json` is valid JSON.
