#include "compute_mfcc.h"
#include "arm_math.h"  // CMSIS-DSP functions
// #include "CMSIS/DSP/Include/arm_const_structs.h"
#include <math.h>

// Buffers for precomputed data
static float hamming_window[FRAME_SIZE];
static float mel_filterbank[NUM_MEL_BINS][FFT_SIZE / 2 + 1];
static float dct_matrix[NUM_MFCC_FEATURES][NUM_MEL_BINS];

// Precompute data for MFCC extraction
void initialize_mfcc() {
    // Hamming window
    for (int i = 0; i < FRAME_SIZE; i++) {
        hamming_window[i] = 0.54 - 0.46 * cos(2 * M_PI * i / (FRAME_SIZE - 1));
    }

    // Mel filterbank
    float mel_low = 0.0;  // Lowest Mel scale frequency
    float mel_high = MFCC_SAMPLE_RATE / 2;  // Highest Mel scale frequency
    float mel_points[NUM_MEL_BINS + 2];  // Mel bin boundaries (including edges)

    // Compute Mel points (equally spaced in Mel scale)
    for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
        mel_points[i] = mel_low + (mel_high - mel_low) * i / (NUM_MEL_BINS + 1);
    }

    // Convert Mel points to Hz
    float hz_points[NUM_MEL_BINS + 2];
    for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
        hz_points[i] = 700 * (pow(10, mel_points[i] / 2595) - 1);  // Convert from Mel scale to Hz
    }

    // Convert Hz to FFT bin indices
    int bin_indices[NUM_MEL_BINS + 2];
    for (int i = 0; i < NUM_MEL_BINS + 2; i++) {
        bin_indices[i] = floor((FFT_SIZE + 1) * hz_points[i] / MFCC_SAMPLE_RATE);  // FFT bin index
    }

    // Compute the filterbank weights
    for (int i = 0; i < NUM_MEL_BINS; i++) {
        for (int j = 0; j < FFT_SIZE / 2 + 1; j++) {
            if (j < bin_indices[i]) {
                mel_filterbank[i][j] = 0.0;  // Left of the triangle
            } else if (j <= bin_indices[i + 1]) {
                mel_filterbank[i][j] = (float)(j - bin_indices[i]) / (bin_indices[i + 1] - bin_indices[i]);  // Rising slope
            } else if (j <= bin_indices[i + 2]) {
                mel_filterbank[i][j] = (float)(bin_indices[i + 2] - j) / (bin_indices[i + 2] - bin_indices[i + 1]);  // Falling slope
            } else {
                mel_filterbank[i][j] = 0.0;  // Right of the triangle
            }
        }
    }

    // DCT matrix
    for (int i = 0; i < NUM_MFCC_FEATURES; i++) {
        for (int j = 0; j < NUM_MEL_BINS; j++) {
            dct_matrix[i][j] = cos(i * M_PI / NUM_MEL_BINS * (j + 0.5));
        }
    }
}

void downsample_audio(const int16_t* input_buffer, int input_size, int16_t* output_buffer, int output_size) {
    int decimation_factor = INPUT_SAMPLE_RATE / MFCC_SAMPLE_RATE;

    // // Optional: Apply a basic low-pass filter (simple moving average)
    // for (int i = 0; i < input_size - decimation_factor; i++) {
    //     input_buffer[i] = (input_buffer[i] + input_buffer[i + 1]) / 2;
    // }

    // Downsample: Keep every decimation_factor-th sample
    for (int i = 0, j = 0; i < output_size && j < input_size; i++, j += decimation_factor) {
        output_buffer[i] = input_buffer[j];
    }
}

// Compute MFCC features
void computeMFCC(const int16_t* audio_buffer, int buffer_size, float mfcc_features[][NUM_MFCC_FEATURES]) {
    // Downsampled audio buffer
    int downsampled_size = buffer_size / 2;  // 16 kHz -> 8 kHz
    int16_t downsampled_audio[downsampled_size];

    // Downsample audio
    downsample_audio(audio_buffer, buffer_size, downsampled_audio, downsampled_size);

    int num_frames = (downsampled_size - FRAME_SIZE) / FRAME_STRIDE + 1;

    // Buffers for processing
    float frame[FRAME_SIZE];
    float fft_input[FFT_SIZE];
    float fft_output[FFT_SIZE / 2 + 1];
    float mel_energies[NUM_MEL_BINS];

    // FFT instance
    arm_rfft_fast_instance_f32 fft_instance;
    arm_rfft_fast_init_f32(&fft_instance, FFT_SIZE);

    // Process each frame
    for (int frame_idx = 0; frame_idx < num_frames; frame_idx++) {
        // Extract frame
        for (int i = 0; i < FRAME_SIZE; i++) {
            frame[i] = (float)downsampled_audio[frame_idx * FRAME_STRIDE + i];
        }

        // Apply Hamming window
        for (int i = 0; i < FRAME_SIZE; i++) {
            frame[i] *= hamming_window[i];
        }

        // Zero-pad and FFT
        for (int i = 0; i < FRAME_SIZE; i++) {
            fft_input[i] = frame[i];
        }
        for (int i = FRAME_SIZE; i < FFT_SIZE; i++) {
            fft_input[i] = 0.0f;
        }
        arm_rfft_fast_f32(&fft_instance, fft_input, fft_output, 0);

        // Power spectrum
        for (int i = 0; i < FFT_SIZE / 2 + 1; i++) {
            fft_output[i] *= fft_output[i];
        }

        // Mel filtering
        for (int i = 0; i < NUM_MEL_BINS; i++) {
            mel_energies[i] = 0.0f;
            for (int j = 0; j < FFT_SIZE / 2 + 1; j++) {
                mel_energies[i] += fft_output[j] * mel_filterbank[i][j];
            }
            mel_energies[i] = logf(mel_energies[i] + 1e-6f);  // Log scale
        }

        // DCT
        for (int i = 0; i < NUM_MFCC_FEATURES; i++) {
            mfcc_features[frame_idx][i] = 0.0f;
            for (int j = 0; j < NUM_MEL_BINS; j++) {
                mfcc_features[frame_idx][i] += mel_energies[j] * dct_matrix[i][j];
            }
        }
    }
}