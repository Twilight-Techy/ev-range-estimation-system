# EV Range Estimation — Edge AI for Electric Tricycles

An end-to-end machine learning pipeline that predicts the remaining driving range of a
"Keke Maruwa" electric tricycle from live telemetry, then runs that model **on-device** on an
ESP32 microcontroller.

The interesting part is not the model — it is getting a time-series deep learning model to fit
and run inside a microcontroller's memory budget.

---

## Why an LSTM

Range estimation is a time-series problem, not a snapshot problem. A tricycle driven hard for
the last minute has a very different remaining range than one driven smoothly, even when their
instantaneous state of charge and battery temperature are identical.

The model therefore takes a **60-second rolling window of 10 telemetry features** —
`(60, 10)` — rather than a single reading:

speed, acceleration, passenger load, road slope, auxiliary load, voltage, current,
state of charge, battery temperature, and state of health.

## Pipeline

| Stage | What happens |
|---|---|
| **Data generation** | `generate_data.py` simulates 10 trips at 1 Hz across seasons, driving profiles, and battery aging, with realistic sensor noise injected so the network learns robust representations instead of memorizing clean synthetic curves. |
| **Training** | `Kaggle_Maruwa_Training.ipynb` trains the LSTM on GPU. **KerasTuner (Bayesian optimization)** searches LSTM units (32–256), dropout (0.1–0.4), and learning rate (1e-2 to 1e-4). |
| **Quantization** | Full-integer **post-training quantization** to INT8 via a representative dataset. Cuts the model footprint by **over 75%** and strips GPU/CuDNN-only ops so the graph is portable to Xtensa/ARM. |
| **Deployment** | `esp32_firmware/` loads the `.tflite` model through TensorFlow Lite Micro and runs inference once per second. |

## On-device inference

The firmware keeps a **ring buffer** in RAM holding the trailing 60 seconds of sensor readings.
Each second it scales the raw inputs using the `StandardScaler` constants exported from
training, quantizes them into the input tensor, and invokes `tflite::MicroInterpreter`.

Everything has to fit inside a **120 KB tensor arena** — that constraint is what drives the
INT8 quantization and the architecture search bounds.

A **Wokwi** simulation proves the hardware path without physical soldering: four potentiometers
map to speed, SoC, temperature, and load, and an I2C SSD1306 OLED renders the live prediction
as you turn the knobs.

---

## Repo layout

```
generate_data.py               synthetic telemetry generator
generate_notebook.py           builds the training notebook
Kaggle_Maruwa_Training.ipynb   training + tuning + quantization
esp32_firmware/
  esp32_firmware.ino           ring buffer, scaling, inference loop
  model.h / model.cpp          quantized model as a C array
PROJECT_REPORT.md              full write-up
```

## Running it

```bash
pip install -r requirements.txt
python generate_data.py          # regenerates maruwa_synthetic_data.csv
```

Then open `Kaggle_Maruwa_Training.ipynb` on Kaggle or Colab with a GPU runtime and run through
to the quantization cell, which emits the `.tflite` payload and the C array for the firmware.

For the hardware side, open `esp32_firmware/` in the Arduino IDE or PlatformIO (see
`libraries.txt` for dependencies), or load it in Wokwi to run simulated.

## Status and limitations

This is a working prototype, not a production system. The training data is **synthetic** —
generated from a physics-based consumption model rather than collected from real vehicles.
The natural next step is replacing it with CAN-bus logs from an actual tricycle, followed by
OTA model updates so a fleet can be retrained and pushed to in the field.
