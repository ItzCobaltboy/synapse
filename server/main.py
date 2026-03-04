import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import numpy as np
import json

from pynput.keyboard import Controller, Key

app = FastAPI()

# ============================
# Resolve repo paths
# ============================
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "model" / "svm_quadratic_python.joblib"
CALIB_PATH = BASE_DIR / "calibration" / "user_calibration.json"
CALIB_RAW = BASE_DIR / "calibration" / "calibration_raw.mat"

# ============================
# Load trained pipeline
# ============================
pipe = load(MODEL_PATH)
scaler = pipe["scaler"]
model = pipe["model"]


def is_calibration_mode():
    return CALIB_RAW.exists() and not CALIB_PATH.exists()


def load_calibration():
    if CALIB_PATH.exists():
        with CALIB_PATH.open() as f:
            return json.load(f)["class_map"]

    print("WARNING: calibration not found. Using identity mapping.")
    return {}


class_map = load_calibration()
last_pred = None


# Hot reload before every prediction
def remap(pred):
    global class_map
    if CALIB_PATH.exists():
        try:
            new_map = load_calibration()
            if new_map != class_map:
                print("Calibration reloaded.")
                class_map = new_map
        except Exception:
            pass

    return int(class_map.get(str(pred), pred))


keyboard = Controller()

gesture_map = {
    1: "w",
    2: "a",
    3: "s",
    4: "d",
    5: Key.space,
}


def press_key(key):
    keyboard.press(key)
    keyboard.release(key)


class FeaturePacket(BaseModel):
    features: list[float]


@app.post("/predict")
def predict(fp: FeaturePacket):
    try:
        x = np.array(fp.features, dtype=np.float64).reshape(1, -1)

        expected = scaler.n_features_in_
        if x.shape[1] != expected:
            raise HTTPException(400, f"Expected {expected}, got {x.shape[1]}")

        x_scaled = scaler.transform(x)
        raw_pred = int(model.predict(x_scaled)[0])
        print("Raw prediction:", raw_pred)

        final_pred = remap(raw_pred)
        print("Final:", final_pred)

        if is_calibration_mode():
            print("Calibration mode active: NOT sending keypresses.")
        else:
            if final_pred in gesture_map:
                press_key(gesture_map[final_pred])

        global last_pred
        last_pred = final_pred

        return {"gesture": final_pred}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error: {str(e)}")


@app.get("/latest")
def latest():
    if last_pred is None:
        return {"gesture": "none"}
    return {"gesture": last_pred}


if __name__ == "__main__":
    import uvicorn

    time.sleep(3)
    print("Starting EMG Predict Server...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
