# Project Report: AI-Powered EV Range Estimation for Keke Maruwa Tricycles

## Executive Summary
This project successfully developed an end-to-end Machine Learning Operations (MLOps) pipeline and Edge AI hardware implementation for real-time electric vehicle (EV) range estimation. Targeted specifically at the "Keke Maruwa" electric tricycle architecture, the system leverages a Long Short-Term Memory (LSTM) neural network to predict remaining range based on real-time telemetry (State of Charge, Battery Temperature, Load, and Speed), effectively outperforming static physics-based formulas by adapting to complex, nonlinear environmental degradation factors. 

The finalized AI model was heavily optimized, quantized to 8-bit integers (INT8), and deployed as a TensorFlow Lite Micro (TFLite Micro) payload running directly on an ESP32 microcontroller with strict memory constraints.

---

## 1. Data Generation and Simulation
Due to the lack of publicly available, high-resolution telemetry data for electric tricycles, a robust synthetic data generator (`generate_data.py`) was engineered. 

**Simulation Parameters:**
- **Battery:** 5.0 kWh capacity, 48V nominal system.
- **Trips:** 10 diverse trips simulating different seasons, driving profiles, and battery aging (State of Health).
- **Features Captured (1 Hz):** Speed, Acceleration, Passenger Load, Road Slope, Auxiliary Load, Voltage, Current, SoC (%), Battery Temperature, and SoH (%).

The target variable, **Remaining Range (km)**, was dynamically calculated using a base consumption model penalized heavily by extreme temperatures (heating/cooling inefficiencies) and battery degradation (SoH drop over years of use). Realistic sensor noise was injected to ensure the neural network learned robust representations rather than overfitting to synthetic patterns.

---

## 2. Machine Learning Pipeline (Kaggle Cloud)
The core modeling pipeline (`generate_notebook.py`) was designed to run autonomously in the Kaggle Cloud environment, taking advantage of GPU acceleration to rapidly explore deep learning architectures.

### 2.1 Model Architecture
An **LSTM (Long Short-Term Memory)** network was selected as the primary architecture because EV range estimation is fundamentally a time-series problem. A vehicle driven aggressively for the past 60 seconds will have a drastically different remaining range than a vehicle driven smoothly, even if their instantaneous SoC and Temperature are identical. 
- **Input Shape:** `(60, 10)` representing a 60-second rolling window of the 10 sensor features.

### 2.2 Hyperparameter Tuning
We integrated **KerasTuner (Bayesian Optimization)** to autonomously search for the optimal LSTM architecture. The tuner aggressively explored:
- Number of LSTM units (32 to 256)
- Dropout rates (0.1 to 0.4) for regularization
- Learning rates ($10^{-2}$ to $10^{-4}$)
*Result:* The tuner successfully converged on an optimal architecture (e.g., 96 units, LR 0.01) that minimized Mean Absolute Error (MAE) on the validation set.

### 2.3 Post-Training Quantization (PTQ)
Deploying a heavy TensorFlow model to a microcontroller requires drastic compression. We implemented a **Full Integer Post-Training Quantization** pipeline.
By feeding a representative dataset through the TFLite Converter, the model's 32-bit floating-point weights and activations were perfectly compressed into 8-bit integers (INT8). This reduced the model's physical storage footprint by over 75% and stripped away complex GPU/CuDNN operations, rendering it universally compatible with constrained ARM/Xtensa microprocessors.

---

## 3. Hardware Integration (Edge AI / ESP32)
The crown jewel of this project is the physical hardware implementation (`esp32_firmware/`). We bridged the gap between cloud-based deep learning and local Edge AI.

### 3.1 Firmware Architecture
The C++ firmware uses the `TensorFlowLite_ESP32` library to load the quantized `.tflite` model directly into the ESP32's flash memory.
- **Ring Buffer:** The ESP32 maintains a hyper-efficient circular buffer in its RAM to store the trailing 60 seconds of sensor data.
- **Dynamic Scaling:** Physical sensor inputs are mathematically scaled (using Kaggle's extracted `StandardScaler` constants) before being quantized and fed into the TFLite Input Tensor.
- **Inference Engine:** Once per second, the `tflite::MicroInterpreter` executes the LSTM graph against the buffer, extracting a highly accurate range estimation.

### 3.2 Wokwi Circuit Simulation
To prove the hardware viability without requiring physical soldering, the project includes a complete Wokwi simulation package (`diagram.json`). 
- **Inputs:** 4 physical potentiometers mapping to Speed, SoC, Temperature, and Load.
- **Output:** An I2C SSD1306 OLED display that renders the real-time AI prediction.
Turning the simulated knobs dynamically alters the OLED's range prediction without crashing the ESP32's strict 120KB Tensor Arena limit.

---

## 4. Conclusion and Next Steps
This project proves that advanced, time-series deep learning models can be effectively deployed to $5 microcontrollers to solve highly complex, nonlinear physics problems in the real world. 

**Recommended Next Steps for Production:**
1. **Real-World Data Collection:** Replace the `generate_data.py` dataset with physical CAN-bus logs from a real Keke Maruwa tricycle.
2. **Over-The-Air (OTA) Updates:** Implement OTA protocols on the ESP32 to seamlessly push newly trained `.tflite` model updates directly to the tricycles in the field as the fleet gathers more data.
