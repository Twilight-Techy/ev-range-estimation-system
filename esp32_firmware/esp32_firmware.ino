#include <TensorFlowLite_ESP32.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include "model.h"

// --- Hardware Pins ---
#define POT_SPEED 34
#define POT_SOC 35
#define POT_TEMP 32
#define POT_LOAD 33

// --- OLED Settings ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// --- TFLite Globals ---
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Allocate memory for TFLite
constexpr int kTensorArenaSize = 120 * 1024; // 120 KB is usually safe for an LSTM
uint8_t tensor_arena[kTensorArenaSize];

// --- Ring Buffer for LSTM (60 timesteps x 10 features) ---
const int WINDOW_SIZE = 60;
const int NUM_FEATURES = 10;
float ring_buffer[WINDOW_SIZE][NUM_FEATURES];
int current_step = 0;
bool buffer_full = false;

// --- Data Scalers (Must match Kaggle StandardScaler) ---
// Approximate values used for simulation. Replace with actual `scaler.mean_` and `scaler.scale_` from Kaggle.
float scaler_means[10] = {22.0, 0.0, 112.5, 0.0, 100.0, 46.0, 20.0, 50.0, 20.0, 90.0};
float scaler_scales[10] = {10.0, 0.5, 80.0, 1.0, 40.0, 4.0, 10.0, 25.0, 10.0, 5.0};

void setup() {
  Serial.begin(115200);

  // 1. Initialize OLED
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    for(;;);
  }
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.println("Initializing TFLite...");
  display.display();

  // 2. Load the TFLite Model
  model = tflite::GetModel(lstm_model_quantized_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model provided is schema version not equal to supported version!");
    return;
  }

  // 3. Set up the Op Resolver (AllOps is safe, but consumes more flash)
  static tflite::AllOpsResolver resolver;

  // 4. Instantiate the Interpreter
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // 5. Allocate Tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  
  // Verify input shape is [1, 60, 10]
  if (input->dims->size != 3 || input->dims->data[1] != WINDOW_SIZE || input->dims->data[2] != NUM_FEATURES) {
    Serial.println("Error: Input shape mismatch!");
    return;
  }

  // Initialize ring buffer with zeros
  for (int i=0; i<WINDOW_SIZE; i++) {
    for (int j=0; j<NUM_FEATURES; j++) {
      ring_buffer[i][j] = 0.0f;
    }
  }

  display.println("Ready!");
  display.display();
  delay(1000);
}

void loop() {
  // 1. Read Potentiometers & Map to physical bounds
  int raw_speed = analogRead(POT_SPEED);
  int raw_soc = analogRead(POT_SOC);
  int raw_temp = analogRead(POT_TEMP);
  int raw_load = analogRead(POT_LOAD);

  float speed_kmh = map(raw_speed, 0, 4095, 0, 45); // 0 to 45 km/h
  float soc = map(raw_soc, 0, 4095, 0, 100);        // 0 to 100%
  float temp = map(raw_temp, 0, 4095, 0, 50);       // 0 to 50 C
  float load = map(raw_load, 0, 4095, 0, 225);      // 0 to 225 kg

  // Estimate derived parameters to fill the 10 features
  float accel = 0.0; // Assume steady state for simplicity
  float slope = 0.0; // Assume flat road
  float aux = 100.0; // 100W constant lights/electronics
  float voltage = 42.0 + (soc/100.0) * 12.6; // 48V battery rough voltage curve
  float current = (speed_kmh * 2.0) + (load * 0.1); // Rough current assumption
  float soh = 95.0; // Assume 95% Battery Health

  float raw_features[10] = {speed_kmh, accel, load, slope, aux, voltage, current, soc, temp, soh};

  // 2. Scale features and push to ring buffer
  for(int i=0; i<NUM_FEATURES; i++) {
    float scaled = (raw_features[i] - scaler_means[i]) / scaler_scales[i];
    ring_buffer[current_step][i] = scaled;
  }

  current_step++;
  if(current_step >= WINDOW_SIZE) {
    current_step = 0;
    buffer_full = true;
  }

  // 3. If we don't have 60 seconds of data yet, wait.
  if(!buffer_full) {
    display.clearDisplay();
    display.setCursor(0, 0);
    display.println("Collecting Data...");
    display.print(current_step); display.println("/60s");
    display.display();
    delay(1000);
    return;
  }

  // 4. Flatten the unrolled ring buffer into the TFLite input tensor
  // The input type is typically INT8 since we used INT8 PTQ Quantization
  // We must quantize our float inputs to the input tensor's scale and zero_point
  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;
  
  int tensor_idx = 0;
  for(int i=0; i<WINDOW_SIZE; i++) {
    int buffer_idx = (current_step + i) % WINDOW_SIZE; // Unroll circular buffer chronologically
    for(int j=0; j<NUM_FEATURES; j++) {
      float val = ring_buffer[buffer_idx][j];
      
      if(input->type == kTfLiteInt8) {
         int8_t quantized = (int8_t)(val / input_scale + input_zero_point);
         input->data.int8[tensor_idx++] = quantized;
      } else {
         input->data.f[tensor_idx++] = val; // Fallback if Float32
      }
    }
  }

  // 5. Run Inference!
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // 6. Dequantize output
  float predicted_range_km = 0.0;
  if(output->type == kTfLiteInt8) {
    float output_scale = output->params.scale;
    int output_zero_point = output->params.zero_point;
    predicted_range_km = (output->data.int8[0] - output_zero_point) * output_scale;
  } else {
    predicted_range_km = output->data.f[0];
  }

  // 7. Update OLED Display
  display.clearDisplay();
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("Spd:"); display.print(speed_kmh, 1); display.println("km/h");
  display.print("SoC:"); display.print(soc, 1); display.print("% ");
  display.print("T:"); display.print(temp, 1); display.println("C");
  display.println("-----------------");
  
  display.setTextSize(2);
  display.print(predicted_range_km, 1); display.println(" km");
  display.display();

  delay(1000); // 1Hz sampling rate
}
