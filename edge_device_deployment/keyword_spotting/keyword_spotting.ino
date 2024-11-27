#include "Arduino.h"
#include <TensorFlowLite.h>
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include <PDM.h> // For capturing audio using the onboard microphone
#include "compute_mfcc.h"
#include "model.h" // Include the quantized model

// Audio settings
#define AUDIO_BUFFER_SIZE 16000 // 100ms of audio at 16kHz
int16_t audioBuffer[AUDIO_BUFFER_SIZE];
volatile int audioBufferIndex = 0;

// TensorFlow Lite globals
constexpr int tensorArenaSize = 140 * 1024; // Adjust based on available RAM
uint8_t tensorArena[tensorArenaSize];
tflite::ErrorReporter *errorReporter;
tflite::MicroInterpreter *interpreter;
TfLiteTensor *inputTensor;

// MFCC buffer
#define MAX_FRAMES 99 // (16000/8000)*(1000ms/20ms) - Adjust based on available RAM
float mfccFeatures[MAX_FRAMES][NUM_MFCC_FEATURES];

// Flag to indicate audio readiness
volatile bool isAudioReady = false;

// PDM callback
void onPDMData()
{
  int bytesAvailable = PDM.available();
  if (bytesAvailable > 0)
  {
    // int bytesToRead = min(bytesAvailable, (AUDIO_BUFFER_SIZE - audioBufferIndex) * sizeof(int16_t));
    int bytesToRead = min(bytesAvailable, (AUDIO_BUFFER_SIZE - audioBufferIndex) * 2);
    // PDM.read((uint8_t *)&audioBuffer[audioBufferIndex], bytesToRead);
    PDM.read(audioBuffer, bytesToRead);
    // audioBufferIndex += bytesToRead / sizeof(int16_t);
    audioBufferIndex += bytesToRead / 2;
    if (audioBufferIndex >= AUDIO_BUFFER_SIZE)
    {
      isAudioReady = true;
      audioBufferIndex = 0;
    }
  }
}

void setup()
{
  Serial.begin(115200);
  while (!Serial)
    ;

  // Initialize PDM microphone
  Serial.println("Initializing mic...");
  PDM.onReceive(onPDMData);
  if (!PDM.begin(1, INPUT_SAMPLE_RATE))
  {
    Serial.println("Failed to start PDM microphone!");
    while (1)
      ;
  }
  Serial.println("Initialized mic.");
  

  // Initialize MFCC precomputed data
  initialize_mfcc();

  // Initialize TensorFlow Lite
  static tflite::MicroErrorReporter microErrorReporter;
  errorReporter = &microErrorReporter;

  const tflite::Model *model = tflite::GetModel(model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION)
  {
    errorReporter->Report("Model schema mismatch!");
    while (1)
      ;
  }

  // static tflite::MicroMutableOpResolver resolver;
  static tflite::micro::ops::AllOpsResolver resolver;

  // // Register the operators used in your model
  // resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
  //                     tflite::ops::micro::Register_CONV_2D());
  // resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
  //                     tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  // resolver.AddBuiltin(tflite::BuiltinOperator_RELU,
  //                     tflite::ops::micro::Register_RELU());
  // resolver.AddBuiltin(tflite::BuiltinOperator_ADD,
  //                     tflite::ops::micro::Register_ADD());
  // resolver.AddBuiltin(tflite::BuiltinOperator_AVERAGE_POOL_2D,
  //                     tflite::ops::micro::Register_AVERAGE_POOL_2D());
  // resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
  //                     tflite::ops::micro::Register_FULLY_CONNECTED());


  static tflite::MicroInterpreter staticInterpreter(
      model, resolver, tensorArena, tensorArenaSize, errorReporter);
  interpreter = &staticInterpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    errorReporter->Report("Failed to allocate tensors!");
    while (1)
      ;
  }

  inputTensor = interpreter->input(0);
  if (inputTensor->dims->data[1] != MAX_FRAMES || inputTensor->dims->data[2] != NUM_MFCC_FEATURES)
  {
    errorReporter->Report("Unexpected input tensor shape!");
    while (1)
      ;
  }

  Serial.println("Setup complete.");
}

void loop()
{
  if (!isAudioReady) {
    // Serial.println("Audio not ready.");
    return;
  }

  // Reset the flag
  isAudioReady = false;

  // Compute MFCC features
  Serial.println("Computing MFCCs.");
  computeMFCC(audioBuffer, AUDIO_BUFFER_SIZE, mfccFeatures);

  Serial.println("Copying MFCCs to input tensor.");
  // Copy MFCC features to input tensor
  for (int i = 0; i < MAX_FRAMES; i++)
  {
    for (int j = 0; j < NUM_MFCC_FEATURES; j++)
    {
      inputTensor->data.f[i * NUM_MFCC_FEATURES + j] = mfccFeatures[i][j];
    }
  }

  Serial.println("Performing inference.");
  // Perform inference
  if (interpreter->Invoke() != kTfLiteOk)
  {
    errorReporter->Report("Invoke failed!");
    return;
  }

  // Get predictions
  TfLiteTensor *outputTensor = interpreter->output(0);
  float *predictions = outputTensor->data.f;
  int numPredictions = outputTensor->dims->data[1];
  Serial.println("Getting prediction.");

  // Find the best prediction
  int maxIndex = 0;
  float maxScore = predictions[0];
  for (int i = 1; i < numPredictions; i++)
  {
    if (predictions[i] > maxScore)
    {
      maxScore = predictions[i];
      maxIndex = i;
    }
  }

  Serial.print("Detected class: ");
  Serial.print(maxIndex);
  Serial.print(" with confidence: ");
  Serial.println(maxScore);
}
