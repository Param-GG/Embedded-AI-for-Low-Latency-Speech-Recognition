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

// Global Variables for TensorFlow Lite
constexpr int kTensorArenaSize = 120 * 1024; // Adjust based on model size
uint8_t tensor_arena[kTensorArenaSize] = {0};

tflite::MicroInterpreter *interpreter = nullptr;
tflite::ErrorReporter *error_reporter = nullptr;
// TfLiteTensor* input = nullptr;
// TfLiteTensor* output = nullptr;
// float* input_buffer = nullptr;

// Audio capture globals
const int kSampleRate = 16000;                                    // Sampling rate in Hz
const int kAudioCaptureDuration = 1;                              // Capture 1 second of audio
const int kAudioBufferSize = kSampleRate * kAudioCaptureDuration; // 16,000 samples for 1 second
int16_t audio_buffer[kAudioBufferSize];
volatile int audio_buffer_index = 0;
float mfcc_features[99][NUM_MFCC_FEATURES];

// Function to Capture Audio using onboard PDM Mic
void captureAudio()
{
    while (PDM.available())
    {
        int16_t sample;
        PDM.read(&sample, sizeof(int16_t));
        if (audio_buffer_index < kAudioBufferSize)
        {
            audio_buffer[audio_buffer_index++] = sample;
        }
        Serial.print("Sample captured: ");
        Serial.println(sample);
    }
    Serial.print("Audio buffer index: ");
    Serial.println(audio_buffer_index); // Print the buffer index periodically
}

void setup()
{
    Serial.begin(9600);
    while (!Serial)
        ;
    Serial.println("Started");

    // TensorFlow Lite Initialization
    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    const tflite::Model *model = tflite::GetModel(model_data);
    Serial.print("Model size in bytes: ");
    Serial.println(model_data_len);
    if (model->version() != TFLITE_SCHEMA_VERSION)
    {
        error_reporter->Report("Model is schema version: %d\nSupported schema version is: %d", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }
    error_reporter->Report("got model");

    static tflite::MicroMutableOpResolver resolver;
    resolver.AddBuiltin(tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
                        tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
    resolver.AddBuiltin(tflite::BuiltinOperator_RELU,
                        tflite::ops::micro::Register_RELU());
    resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                        tflite::ops::micro::Register_SOFTMAX());

    // static tflite::ops::micro::AllOpsResolver resolver;
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk)
    {
        error_reporter->Report("Tensor allocation failed!");
        return;
    }
    error_reporter->Report("Allocated memory");

    // // Obtain pointers to the model's input and output tensors.
    // input = interpreter->input(0);
    // output = interpreter->output(0);
    // error_reporter->Report( "Results" );

    // input_buffer = input->data.f;

    Serial.println("Model loaded successfully!");

    // Initialize PDM (Mic)
    if (!PDM.begin(1, kSampleRate))
    { // 1 channel, 16kHz
        Serial.println("Failed to initalize PDM Mic");
        return;
    }
    Serial.println("Capturing audio...");
    PDM.onReceive(captureAudio);
}

void loop()
{
    if (audio_buffer_index >= kAudioBufferSize)
    {
        Serial.println("Audio captured. Processing...");
        // Reset buffer index for next capture
        audio_buffer_index = 0;

        // Extract MFCC features (implement or call a function here)
        computeMFCC(audio_buffer, kAudioBufferSize, mfcc_features);

        // Run inference
        float *input_tensor = interpreter->input(0)->data.f;
        memcpy(input_tensor, mfcc_features, sizeof(mfcc_features));

        TfLiteStatus invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk)
        {
            error_reporter->Report("Invoke failed");
            return;
        }

        // Get the prediction
        float *result = interpreter->output(0)->data.f;
        int predicted_class = std::distance(result, std::max_element(result, result + 36)); // There are 36 classes
        Serial.print("Predicted Keyword: ");
        Serial.println(predicted_class);
    }
}