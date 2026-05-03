#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <omp.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <cstdio>

#define Q_GPU 8380417
#define MAX_GUESS_BATCH 100000 

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            fprintf(stderr, "CUBLAS Error at %s:%d\n", __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

__device__ __forceinline__ int32_t reduce_np_fast(int64_t a) {
    int32_t t = (int32_t)a * 58728449;       
    int64_t m = (int64_t)t * Q_GPU;
    return (int32_t)((a - m) >> 32); 
}

// [UNMASKED FIX]: Hardware-accelerated Hamming Weight of the raw 32-bit integer
__device__ __forceinline__ float compute_leakage_hw(int32_t val_s32) {
    return (float)__popc((unsigned int)val_s32);
}

// Standard Z-Score Normalization for Unmasked Traces
__global__ void normalize_traces_T(const float* traces, int N_phys, int T, int round, int K_ROWS, float* T_norm) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    
    int N_virt = N_phys * K_ROWS;
    float sum = 0.0f, sumsq = 0.0f;
    
    for (int i = 0; i < N_phys; ++i) {
        for (int k = 0; k < K_ROWS; ++k) {
            int src_idx = i * (K_ROWS * 256) * T + (k * 256 + round) * T + t;
            float val = traces[src_idx];
            sum += val;
            sumsq += val * val;
        }
    }
    
    float mean = sum / (float)N_virt;
    float var = sumsq - (sum * sum) / (float)N_virt;
    float inv_std = (var > 1e-6f) ? rsqrtf(var) : 0.0f;
    
    for (int i = 0; i < N_phys; ++i) {
        for (int k = 0; k < K_ROWS; ++k) {
            int src_idx = i * (K_ROWS * 256) * T + (k * 256 + round) * T + t;
            T_norm[(i * K_ROWS + k) * T + t] = (traces[src_idx] - mean) * inv_std;
        }
    }
}

__global__ void generate_L_norm_cs1_unmasked(const int32_t* full_labels, int N, int round, int start_g, int count_g, float* L_norm) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= count_g) return;

    int32_t guess = start_g + g_idx;
    float h_sum = 0.0f, h_sumsq = 0.0f;
    
    for(int i = 0; i < N; ++i) {
        float leak = compute_leakage_hw(reduce_np_fast((int64_t)full_labels[i * 256 + round] * guess));
        h_sum += leak;
        h_sumsq += leak * leak;
    }

    float mean = h_sum / (float)N;
    float var = h_sumsq - (h_sum * h_sum) / (float)N;
    float inv_std = (var > 1e-6f) ? rsqrtf(var) : 0.0f;

    for(int i = 0; i < N; ++i) {
        float leak = compute_leakage_hw(reduce_np_fast((int64_t)full_labels[i * 256 + round] * guess));
        L_norm[g_idx * N + i] = (leak - mean) * inv_std;
    }
}

__global__ void generate_L_norm_ay_unmasked(const int32_t* c_ntt, const int32_t* z_ntt, const int32_t* a_col, 
                                            int N_phys, int round, int start_g, int count_g, int K_ROWS, float* L_norm) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= count_g) return;

    int32_t guess = start_g + g_idx;
    int N_virt = N_phys * K_ROWS; 
    
    float h_sum = 0.0f, h_sumsq = 0.0f;
    for (int i = 0; i < N_phys; ++i) {
        int32_t cs1_contam = reduce_np_fast((int64_t)c_ntt[i * 256 + round] * guess);
        int64_t cs1_scaled = ((int64_t)cs1_contam * 4193792LL) % Q_GPU;
        if (cs1_scaled < 0) cs1_scaled += Q_GPU;
        
        int32_t y_ntt = (z_ntt[i * 256 + round] - (int32_t)cs1_scaled) % Q_GPU;
        if (y_ntt < 0) y_ntt += Q_GPU;
        
        for (int k = 0; k < K_ROWS; ++k) {
            float leak = compute_leakage_hw(reduce_np_fast((int64_t)a_col[k] * y_ntt)); 
            h_sum += leak;
            h_sumsq += leak * leak;
        }
    }
    
    float mean = h_sum / (float)N_virt;
    float var = h_sumsq - (h_sum * h_sum) / (float)N_virt;
    float inv_std = (var > 1e-6f) ? rsqrtf(var) : 0.0f;
    
    for (int i = 0; i < N_phys; ++i) {
        int32_t cs1_contam = reduce_np_fast((int64_t)c_ntt[i * 256 + round] * guess);
        int64_t cs1_scaled = ((int64_t)cs1_contam * 4193792LL) % Q_GPU;
        if (cs1_scaled < 0) cs1_scaled += Q_GPU;
        
        int32_t y_ntt = (z_ntt[i * 256 + round] - (int32_t)cs1_scaled) % Q_GPU;
        if (y_ntt < 0) y_ntt += Q_GPU;
        
        for (int k = 0; k < K_ROWS; ++k) {
            float leak = compute_leakage_hw(reduce_np_fast((int64_t)a_col[k] * y_ntt));
            L_norm[g_idx * N_virt + (i * K_ROWS + k)] = (leak - mean) * inv_std;
        }
    }
}

__global__ void compute_standalone_scores(
    const float* __restrict__ R_cs1, int T_cs1,
    const float* __restrict__ R_ay, int T_ay,
    int count_g,
    float* __restrict__ out_cs1, float* __restrict__ out_ay) 
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= count_g) return;

    float max_c1 = 0.0f;
    for(int t = 0; t < T_cs1; t++) {
        float c = fabsf(R_cs1[g * T_cs1 + t]); 
        if(c > max_c1) max_c1 = c;
    }

    float max_c2 = 0.0f;
    for(int t = 0; t < T_ay; t++) {
        float c = fabsf(R_ay[g * T_ay + t]); 
        if(c > max_c2) max_c2 = c;
    }

    out_cs1[g] = max_c1;
    out_ay[g] = max_c2;
}

thread_local cublasHandle_t cublas_handle = nullptr;

thread_local float *d_cs1_traces = nullptr, *d_cs1_T_norm = nullptr, *d_cs1_L_norm = nullptr, *d_cs1_R = nullptr;
thread_local int32_t *d_cs1_labels = nullptr;

thread_local float *d_ay_traces = nullptr, *d_ay_T_norm = nullptr, *d_ay_L_norm = nullptr, *d_ay_R = nullptr;
thread_local int32_t *d_ay_c_ntt = nullptr, *d_ay_z_ntt = nullptr, *d_ay_a_col = nullptr;

thread_local float *d_out_cs1 = nullptr, *d_out_ay = nullptr;

extern "C" void gpu_solve_round_unmasked(
    bool is_new_instance,
    const std::vector<float>& inst_cs1_traces, const std::vector<int32_t>& c_labels,
    int N_cs1, int max_N_cs1, int T_cs1,
    const std::vector<float>& inst_ay_traces,
    const std::vector<int32_t>& c_ntt, const std::vector<int32_t>& z_ntt, const std::vector<int32_t>& a_col,
    int N_ay_phys, int max_N_ay_phys, int T_ay, int K_ROWS,
    int total_guesses, int current_round,
    std::vector<float>& out_cs1, std::vector<float>& out_ay
) {
    int num_gpus = 0; CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    
    if (num_gpus <= 0) {
    fprintf(stderr, "No CUDA device found.\n");
    exit(1);
    }

    int guesses_per_gpu = (total_guesses + num_gpus - 1) / num_gpus;


    #pragma omp parallel num_threads(num_gpus)
    {
        int gpu_id = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(gpu_id));
        if (cublas_handle == nullptr) { CUBLAS_CHECK(cublasCreate(&cublas_handle)); }

        int start_g = gpu_id * guesses_per_gpu;
        int end_g = std::min(start_g + guesses_per_gpu, total_guesses);
        int total_gpu_guesses = end_g - start_g;

        if (total_gpu_guesses > 0) {
            int max_N_ay_virt = max_N_ay_phys * K_ROWS; 
            int N_ay_virt = N_ay_phys * K_ROWS;

            if (is_new_instance || d_cs1_traces == nullptr) {
                if (d_cs1_traces != nullptr) {
                    cudaFree(d_cs1_traces); cudaFree(d_cs1_labels);
                    cudaFree(d_cs1_T_norm); cudaFree(d_cs1_L_norm); cudaFree(d_cs1_R);
                    cudaFree(d_ay_traces); cudaFree(d_ay_c_ntt); cudaFree(d_ay_z_ntt); cudaFree(d_ay_a_col);
                    cudaFree(d_ay_T_norm); cudaFree(d_ay_L_norm); cudaFree(d_ay_R);
                    cudaFree(d_out_cs1); cudaFree(d_out_ay);
                }
                
                CUDA_CHECK(cudaMalloc(&d_cs1_traces, (size_t)max_N_cs1 * 256 * T_cs1 * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_cs1_labels, (size_t)max_N_cs1 * 256 * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(&d_cs1_T_norm, (size_t)max_N_cs1 * T_cs1 * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_cs1_L_norm, (size_t)MAX_GUESS_BATCH * max_N_cs1 * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_cs1_R, (size_t)MAX_GUESS_BATCH * T_cs1 * sizeof(float)));
                
                CUDA_CHECK(cudaMalloc(&d_ay_traces, (size_t)inst_ay_traces.size() * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_ay_c_ntt, (size_t)c_ntt.size() * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(&d_ay_z_ntt, (size_t)z_ntt.size() * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(&d_ay_a_col, K_ROWS * sizeof(int32_t))); 
                CUDA_CHECK(cudaMalloc(&d_ay_T_norm, (size_t)max_N_ay_virt * T_ay * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_ay_L_norm, (size_t)MAX_GUESS_BATCH * max_N_ay_virt * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_ay_R, (size_t)MAX_GUESS_BATCH * T_ay * sizeof(float)));

                CUDA_CHECK(cudaMalloc(&d_out_cs1, (size_t)MAX_GUESS_BATCH * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_out_ay, (size_t)MAX_GUESS_BATCH * sizeof(float)));

                CUDA_CHECK(cudaMemcpy(d_cs1_traces, inst_cs1_traces.data(), (size_t)max_N_cs1 * 256 * T_cs1 * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_cs1_labels, c_labels.data(), (size_t)max_N_cs1 * 256 * sizeof(int32_t), cudaMemcpyHostToDevice));

                CUDA_CHECK(cudaMemcpy(d_ay_traces, inst_ay_traces.data(), (size_t)inst_ay_traces.size() * sizeof(float), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ay_c_ntt, c_ntt.data(), (size_t)c_ntt.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_ay_z_ntt, z_ntt.data(), (size_t)z_ntt.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
            }
            
            CUDA_CHECK(cudaMemcpy(d_ay_a_col, a_col.data(), K_ROWS * sizeof(int32_t), cudaMemcpyHostToDevice));

            int threads = 256; 
            int blocks_T_cs1 = (T_cs1 + threads - 1) / threads;
            int blocks_T_ay = (T_ay + threads - 1) / threads;

            // Notice we treat CS1 as K_ROWS=1 for the normalization logic to reuse the kernel cleanly
            normalize_traces_T<<<blocks_T_cs1, threads>>>(d_cs1_traces, max_N_cs1, T_cs1, current_round, 1, d_cs1_T_norm);
            normalize_traces_T<<<blocks_T_ay, threads>>>(d_ay_traces, max_N_ay_phys, T_ay, current_round, K_ROWS, d_ay_T_norm);
            CUDA_CHECK(cudaDeviceSynchronize());

            for (int b_offset = 0; b_offset < total_gpu_guesses; b_offset += MAX_GUESS_BATCH) {
                int b_count = std::min((int)MAX_GUESS_BATCH, total_gpu_guesses - b_offset);
                int current_start_g = start_g + b_offset;
                int blocks_L = (b_count + threads - 1) / threads;

                generate_L_norm_cs1_unmasked<<<blocks_L, threads>>>(d_cs1_labels, N_cs1, current_round, current_start_g, b_count, d_cs1_L_norm);
                generate_L_norm_ay_unmasked<<<blocks_L, threads>>>(d_ay_c_ntt, d_ay_z_ntt, d_ay_a_col, N_ay_phys, current_round, current_start_g, b_count, K_ROWS, d_ay_L_norm);
                CUDA_CHECK(cudaDeviceSynchronize());

                float alpha = 1.0f, beta = 0.0f;
                CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         T_cs1, b_count, N_cs1,
                                         &alpha, d_cs1_T_norm, T_cs1,
                                         d_cs1_L_norm, N_cs1,
                                         &beta, d_cs1_R, T_cs1));

                CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         T_ay, b_count, N_ay_virt,
                                         &alpha, d_ay_T_norm, T_ay,
                                         d_ay_L_norm, N_ay_virt,
                                         &beta, d_ay_R, T_ay));
                CUDA_CHECK(cudaDeviceSynchronize());

                compute_standalone_scores<<<blocks_L, threads>>>(d_cs1_R, T_cs1, d_ay_R, T_ay, b_count, d_out_cs1, d_out_ay);
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(out_cs1.data() + current_start_g, d_out_cs1, (size_t)b_count * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(out_ay.data() + current_start_g, d_out_ay, (size_t)b_count * sizeof(float), cudaMemcpyDeviceToHost));
            }
        }
    }
}