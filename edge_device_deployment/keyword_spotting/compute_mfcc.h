#ifndef COMPUTE_MFCC_H
#define COMPUTE_MFCC_H

#include <stdint.h>

// MFCC parameters
#define INPUT_SAMPLE_RATE 16000      // Sampling rate of the audio input (16 kHz)
#define MFCC_SAMPLE_RATE 8000        // Sampling rate for MFCC computation (8 kHz)
#define FRAME_SIZE_MS 32             // Frame size in milliseconds
#define FRAME_STRIDE_MS 20           // Frame stride in milliseconds
#define FFT_SIZE 256                 // FFT size
#define NUM_MEL_BINS 40              // Number of Mel filterbanks
#define NUM_MFCC_FEATURES 12         // Number of MFCC coefficients

// Derived parameters for MFCC sample rate
#define FRAME_SIZE (MFCC_SAMPLE_RATE * FRAME_SIZE_MS / 1000)  // Frame size in samples
#define FRAME_STRIDE (MFCC_SAMPLE_RATE * FRAME_STRIDE_MS / 1000)  // Frame stride in samples

// Function prototypes
void initialize_mfcc(); // Precompute data for MFCC extraction
void computeMFCC(const int16_t* audio_buffer, int buffer_size, float mfcc_features[][NUM_MFCC_FEATURES]);
void downsample_audio(const int16_t* input_buffer, int input_size, int16_t* output_buffer, int output_size);

#endif  // COMPUTE_MFCC_H
