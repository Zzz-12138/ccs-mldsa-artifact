#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <omp.h>
#include <vector>

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
    int32_t t = static_cast<int32_t>(a) * 58728449;
    int64_t m = static_cast<int64_t>(t) * Q_GPU;
    return static_cast<int32_t>((a - m) >> 32);
}

__device__ __forceinline__ float compute_leakage(int32_t val_s32, int hw_model_type) {
    (void)hw_model_type;

    if (val_s32 < 0) val_s32 += Q_GPU;
    if (val_s32 > Q_GPU / 2) val_s32 -= Q_GPU;

    return static_cast<float>(val_s32 < 0 ? -val_s32 : val_s32);
}

__global__ void combine_and_normalize_cs1_T(
    const float* full_s0,
    const float* full_s1,
    int max_N,
    int N,
    int T,
    int round,
    float* T_norm) {
    (void)max_N;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    float sum_s0 = 0.0f;
    float sum_s1 = 0.0f;

    for (int i = 0; i < N; ++i) {
        int idx = i * 256 * T + round * T + t;
        sum_s0 += full_s0[idx];
        sum_s1 += full_s1[idx];
    }

    float mean_s0 = sum_s0 / static_cast<float>(N);
    float mean_s1 = sum_s1 / static_cast<float>(N);

    float sum_cp = 0.0f;
    float sumsq_cp = 0.0f;

    for (int i = 0; i < N; ++i) {
        int idx = i * 256 * T + round * T + t;
        float cp = (full_s0[idx] - mean_s0) * (full_s1[idx] - mean_s1);
        sum_cp += cp;
        sumsq_cp += cp * cp;
    }

    float mean_cp = sum_cp / static_cast<float>(N);
    float var_cp = sumsq_cp - (sum_cp * sum_cp) / static_cast<float>(N);
    float inv_std_cp = (var_cp > 1e-6f) ? rsqrtf(var_cp) : 0.0f;

    for (int i = 0; i < N; ++i) {
        int idx = i * 256 * T + round * T + t;
        float cp = (full_s0[idx] - mean_s0) * (full_s1[idx] - mean_s1);
        T_norm[i * T + t] = (cp - mean_cp) * inv_std_cp;
    }
}

__global__ void generate_L_norm_cs1(
    const int32_t* full_labels,
    int N,
    int round,
    int start_g,
    int count_g,
    float* L_norm) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= count_g) return;

    int32_t guess = start_g + g_idx;

    float h_sum = 0.0f;
    float h_sumsq = 0.0f;

    for (int i = 0; i < N; ++i) {
        float leak = compute_leakage(
            reduce_np_fast(static_cast<int64_t>(full_labels[i * 256 + round]) * guess),
            2);
        h_sum += leak;
        h_sumsq += leak * leak;
    }

    float mean = h_sum / static_cast<float>(N);
    float var = h_sumsq - (h_sum * h_sum) / static_cast<float>(N);
    float inv_std = (var > 1e-6f) ? rsqrtf(var) : 0.0f;

    for (int i = 0; i < N; ++i) {
        float leak = compute_leakage(
            reduce_np_fast(static_cast<int64_t>(full_labels[i * 256 + round]) * guess),
            2);
        L_norm[g_idx * N + i] = (leak - mean) * inv_std;
    }
}

__global__ void combine_and_normalize_ay_T(
    const float* full_s0,
    const float* full_s1,
    int max_N_phys,
    int N_phys,
    int T,
    int round,
    int K_ROWS,
    float* T_norm) {
    (void)max_N_phys;

    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;

    int N_virt = N_phys * K_ROWS;

    float sum_s0 = 0.0f;
    float sum_s1 = 0.0f;

    for (int i = 0; i < N_phys; ++i) {
        for (int k = 0; k < K_ROWS; ++k) {
            int src_idx = i * (K_ROWS * 256) * T + (k * 256 + round) * T + t;
            sum_s0 += full_s0[src_idx];
            sum_s1 += full_s1[src_idx];
        }
    }

    float mean_s0 = sum_s0 / static_cast<float>(N_virt);
    float mean_s1 = sum_s1 / static_cast<float>(N_virt);

    float sum_cp = 0.0f;
    float sumsq_cp = 0.0f;

    for (int i = 0; i < N_phys; ++i) {
        for (int k = 0; k < K_ROWS; ++k) {
            int src_idx = i * (K_ROWS * 256) * T + (k * 256 + round) * T + t;
            float cp = (full_s0[src_idx] - mean_s0) * (full_s1[src_idx] - mean_s1);
            sum_cp += cp;
            sumsq_cp += cp * cp;
        }
    }

    float mean_cp = sum_cp / static_cast<float>(N_virt);
    float var_cp = sumsq_cp - (sum_cp * sum_cp) / static_cast<float>(N_virt);
    float inv_std = (var_cp > 1e-6f) ? rsqrtf(var_cp) : 0.0f;

    for (int i = 0; i < N_phys; ++i) {
        for (int k = 0; k < K_ROWS; ++k) {
            int src_idx = i * (K_ROWS * 256) * T + (k * 256 + round) * T + t;
            float cp = (full_s0[src_idx] - mean_s0) * (full_s1[src_idx] - mean_s1);
            T_norm[(i * K_ROWS + k) * T + t] = (cp - mean_cp) * inv_std;
        }
    }
}

__global__ void generate_L_norm_ay(
    const int32_t* c_ntt,
    const int32_t* z_ntt,
    const int32_t* a_col,
    int N_phys,
    int round,
    int start_g,
    int count_g,
    int K_ROWS,
    float* L_norm) {
    int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (g_idx >= count_g) return;

    int32_t guess = start_g + g_idx;
    int N_virt = N_phys * K_ROWS;

    float h_sum = 0.0f;
    float h_sumsq = 0.0f;

    for (int i = 0; i < N_phys; ++i) {
        int32_t cs1_contam = reduce_np_fast(static_cast<int64_t>(c_ntt[i * 256 + round]) * guess);
        int64_t cs1_scaled = (static_cast<int64_t>(cs1_contam) * 4193792LL) % Q_GPU;
        if (cs1_scaled < 0) cs1_scaled += Q_GPU;

        int32_t y_ntt = (z_ntt[i * 256 + round] - static_cast<int32_t>(cs1_scaled)) % Q_GPU;
        if (y_ntt < 0) y_ntt += Q_GPU;

        for (int k = 0; k < K_ROWS; ++k) {
            float leak = compute_leakage(
                reduce_np_fast(static_cast<int64_t>(a_col[k]) * y_ntt),
                2);
            h_sum += leak;
            h_sumsq += leak * leak;
        }
    }

    float mean = h_sum / static_cast<float>(N_virt);
    float var = h_sumsq - (h_sum * h_sum) / static_cast<float>(N_virt);
    float inv_std = (var > 1e-6f) ? rsqrtf(var) : 0.0f;

    for (int i = 0; i < N_phys; ++i) {
        int32_t cs1_contam = reduce_np_fast(static_cast<int64_t>(c_ntt[i * 256 + round]) * guess);
        int64_t cs1_scaled = (static_cast<int64_t>(cs1_contam) * 4193792LL) % Q_GPU;
        if (cs1_scaled < 0) cs1_scaled += Q_GPU;

        int32_t y_ntt = (z_ntt[i * 256 + round] - static_cast<int32_t>(cs1_scaled)) % Q_GPU;
        if (y_ntt < 0) y_ntt += Q_GPU;

        for (int k = 0; k < K_ROWS; ++k) {
            float leak = compute_leakage(
                reduce_np_fast(static_cast<int64_t>(a_col[k]) * y_ntt),
                2);
            L_norm[g_idx * N_virt + (i * K_ROWS + k)] = (leak - mean) * inv_std;
        }
    }
}

__global__ void compute_standalone_scores(
    const float* __restrict__ R_cs1,
    int T_cs1,
    const float* __restrict__ R_ay,
    int T_ay,
    int count_g,
    float* __restrict__ out_cs1,
    float* __restrict__ out_ay) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= count_g) return;

    float max_c1 = 0.0f;
    for (int t = 0; t < T_cs1; ++t) {
        float c = fabsf(R_cs1[g * T_cs1 + t]);
        if (c > max_c1) max_c1 = c;
    }

    float max_c2 = 0.0f;
    for (int t = 0; t < T_ay; ++t) {
        float c = fabsf(R_ay[g * T_ay + t]);
        if (c > max_c2) max_c2 = c;
    }

    out_cs1[g] = max_c1;
    out_ay[g] = max_c2;
}

__global__ void compute_single_scores(
    const float* __restrict__ R,
    int T,
    int count_g,
    float* __restrict__ out) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= count_g) return;

    float best = 0.0f;
    for (int t = 0; t < T; ++t) {
        float c = fabsf(R[g * T + t]);
        if (c > best) best = c;
    }

    out[g] = best;
}

thread_local cublasHandle_t cublas_handle = nullptr;

thread_local float* d_cs1_full_s0 = nullptr;
thread_local float* d_cs1_full_s1 = nullptr;
thread_local float* d_cs1_T_norm = nullptr;
thread_local float* d_cs1_L_norm = nullptr;
thread_local float* d_cs1_R = nullptr;
thread_local int32_t* d_cs1_labels = nullptr;

thread_local float* d_ay_full_s0 = nullptr;
thread_local float* d_ay_full_s1 = nullptr;
thread_local float* d_ay_T_norm = nullptr;
thread_local float* d_ay_L_norm = nullptr;
thread_local float* d_ay_R = nullptr;
thread_local int32_t* d_ay_c_ntt = nullptr;
thread_local int32_t* d_ay_z_ntt = nullptr;
thread_local int32_t* d_ay_a_col = nullptr;

thread_local float* d_out_cs1 = nullptr;
thread_local float* d_out_ay = nullptr;

thread_local float* d_single_s0 = nullptr;
thread_local float* d_single_s1 = nullptr;
thread_local float* d_single_T_norm = nullptr;
thread_local float* d_single_L_norm = nullptr;
thread_local float* d_single_R = nullptr;
thread_local float* d_single_out = nullptr;
thread_local int32_t* d_single_labels = nullptr;

extern "C" void gpu_solve_round_cublas_fusion(
    bool is_new_instance,
    const std::vector<float>& inst_cs1_s0,
    const std::vector<float>& inst_cs1_s1,
    const std::vector<int32_t>& c_labels,
    int N_cs1,
    int max_N_cs1,
    int T_cs1,
    const std::vector<float>& inst_ay_s0,
    const std::vector<float>& inst_ay_s1,
    const std::vector<int32_t>& c_ntt,
    const std::vector<int32_t>& z_ntt,
    const std::vector<int32_t>& a_col,
    int N_ay_phys,
    int max_N_ay_phys,
    int T_ay,
    int K_ROWS,
    int total_guesses,
    int current_round,
    std::vector<float>& out_cs1,
    std::vector<float>& out_ay,
    std::vector<float>& out_m1,
    std::vector<float>& out_m2,
    std::vector<float>& out_m3,
    std::vector<float>& out_m4) {
    (void)out_m1;
    (void)out_m2;
    (void)out_m3;
    (void)out_m4;

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus <= 0) {
        fprintf(stderr, "No CUDA device found.\n");
        exit(1);
    }

    int guesses_per_gpu = (total_guesses + num_gpus - 1) / num_gpus;

    #pragma omp parallel num_threads(num_gpus)
    {
        int gpu_id = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(gpu_id));

        if (cublas_handle == nullptr) {
            CUBLAS_CHECK(cublasCreate(&cublas_handle));
        }

        int start_g = gpu_id * guesses_per_gpu;
        int end_g = std::min(start_g + guesses_per_gpu, total_guesses);
        int total_gpu_guesses = end_g - start_g;

        if (total_gpu_guesses > 0) {
            int max_N_ay_virt = max_N_ay_phys * K_ROWS;
            int N_ay_virt = N_ay_phys * K_ROWS;

            if (is_new_instance || d_cs1_full_s0 == nullptr) {
                if (d_cs1_full_s0 != nullptr) {
                    cudaFree(d_cs1_full_s0);
                    cudaFree(d_cs1_full_s1);
                    cudaFree(d_cs1_labels);
                    cudaFree(d_cs1_T_norm);
                    cudaFree(d_cs1_L_norm);
                    cudaFree(d_cs1_R);

                    cudaFree(d_ay_full_s0);
                    cudaFree(d_ay_full_s1);
                    cudaFree(d_ay_c_ntt);
                    cudaFree(d_ay_z_ntt);
                    cudaFree(d_ay_a_col);
                    cudaFree(d_ay_T_norm);
                    cudaFree(d_ay_L_norm);
                    cudaFree(d_ay_R);

                    cudaFree(d_out_cs1);
                    cudaFree(d_out_ay);
                }

                CUDA_CHECK(cudaMalloc(&d_cs1_full_s0, static_cast<size_t>(max_N_cs1) * 256 * T_cs1 * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_cs1_full_s1, static_cast<size_t>(max_N_cs1) * 256 * T_cs1 * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_cs1_labels, static_cast<size_t>(max_N_cs1) * 256 * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(&d_cs1_T_norm, static_cast<size_t>(max_N_cs1) * T_cs1 * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_cs1_L_norm, static_cast<size_t>(MAX_GUESS_BATCH) * max_N_cs1 * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_cs1_R, static_cast<size_t>(MAX_GUESS_BATCH) * T_cs1 * sizeof(float)));

                CUDA_CHECK(cudaMalloc(&d_ay_full_s0, static_cast<size_t>(inst_ay_s0.size()) * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_ay_full_s1, static_cast<size_t>(inst_ay_s1.size()) * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_ay_c_ntt, static_cast<size_t>(c_ntt.size()) * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(&d_ay_z_ntt, static_cast<size_t>(z_ntt.size()) * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(&d_ay_a_col, static_cast<size_t>(K_ROWS) * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(&d_ay_T_norm, static_cast<size_t>(max_N_ay_virt) * T_ay * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_ay_L_norm, static_cast<size_t>(MAX_GUESS_BATCH) * max_N_ay_virt * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_ay_R, static_cast<size_t>(MAX_GUESS_BATCH) * T_ay * sizeof(float)));

                CUDA_CHECK(cudaMalloc(&d_out_cs1, static_cast<size_t>(MAX_GUESS_BATCH) * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_out_ay, static_cast<size_t>(MAX_GUESS_BATCH) * sizeof(float)));

                CUDA_CHECK(cudaMemcpy(
                    d_cs1_full_s0,
                    inst_cs1_s0.data(),
                    static_cast<size_t>(max_N_cs1) * 256 * T_cs1 * sizeof(float),
                    cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(
                    d_cs1_full_s1,
                    inst_cs1_s1.data(),
                    static_cast<size_t>(max_N_cs1) * 256 * T_cs1 * sizeof(float),
                    cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(
                    d_cs1_labels,
                    c_labels.data(),
                    static_cast<size_t>(max_N_cs1) * 256 * sizeof(int32_t),
                    cudaMemcpyHostToDevice));

                CUDA_CHECK(cudaMemcpy(
                    d_ay_full_s0,
                    inst_ay_s0.data(),
                    static_cast<size_t>(inst_ay_s0.size()) * sizeof(float),
                    cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(
                    d_ay_full_s1,
                    inst_ay_s1.data(),
                    static_cast<size_t>(inst_ay_s1.size()) * sizeof(float),
                    cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(
                    d_ay_c_ntt,
                    c_ntt.data(),
                    static_cast<size_t>(c_ntt.size()) * sizeof(int32_t),
                    cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(
                    d_ay_z_ntt,
                    z_ntt.data(),
                    static_cast<size_t>(z_ntt.size()) * sizeof(int32_t),
                    cudaMemcpyHostToDevice));
            }

            CUDA_CHECK(cudaMemcpy(
                d_ay_a_col,
                a_col.data(),
                static_cast<size_t>(K_ROWS) * sizeof(int32_t),
                cudaMemcpyHostToDevice));

            int threads = 256;
            int blocks_T_cs1 = (T_cs1 + threads - 1) / threads;
            int blocks_T_ay = (T_ay + threads - 1) / threads;

            combine_and_normalize_cs1_T<<<blocks_T_cs1, threads>>>(
                d_cs1_full_s0,
                d_cs1_full_s1,
                max_N_cs1,
                N_cs1,
                T_cs1,
                current_round,
                d_cs1_T_norm);

            combine_and_normalize_ay_T<<<blocks_T_ay, threads>>>(
                d_ay_full_s0,
                d_ay_full_s1,
                max_N_ay_phys,
                N_ay_phys,
                T_ay,
                current_round,
                K_ROWS,
                d_ay_T_norm);

            CUDA_CHECK(cudaDeviceSynchronize());

            for (int b_offset = 0; b_offset < total_gpu_guesses; b_offset += MAX_GUESS_BATCH) {
                int b_count = std::min(static_cast<int>(MAX_GUESS_BATCH), total_gpu_guesses - b_offset);
                int current_start_g = start_g + b_offset;
                int blocks_L = (b_count + threads - 1) / threads;

                generate_L_norm_cs1<<<blocks_L, threads>>>(
                    d_cs1_labels,
                    N_cs1,
                    current_round,
                    current_start_g,
                    b_count,
                    d_cs1_L_norm);

                generate_L_norm_ay<<<blocks_L, threads>>>(
                    d_ay_c_ntt,
                    d_ay_z_ntt,
                    d_ay_a_col,
                    N_ay_phys,
                    current_round,
                    current_start_g,
                    b_count,
                    K_ROWS,
                    d_ay_L_norm);

                CUDA_CHECK(cudaDeviceSynchronize());

                float alpha = 1.0f;
                float beta = 0.0f;

                CUBLAS_CHECK(cublasSgemm(
                    cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    T_cs1,
                    b_count,
                    N_cs1,
                    &alpha,
                    d_cs1_T_norm,
                    T_cs1,
                    d_cs1_L_norm,
                    N_cs1,
                    &beta,
                    d_cs1_R,
                    T_cs1));

                CUBLAS_CHECK(cublasSgemm(
                    cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    T_ay,
                    b_count,
                    N_ay_virt,
                    &alpha,
                    d_ay_T_norm,
                    T_ay,
                    d_ay_L_norm,
                    N_ay_virt,
                    &beta,
                    d_ay_R,
                    T_ay));

                CUDA_CHECK(cudaDeviceSynchronize());

                compute_standalone_scores<<<blocks_L, threads>>>(
                    d_cs1_R,
                    T_cs1,
                    d_ay_R,
                    T_ay,
                    b_count,
                    d_out_cs1,
                    d_out_ay);

                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(
                    out_cs1.data() + current_start_g,
                    d_out_cs1,
                    static_cast<size_t>(b_count) * sizeof(float),
                    cudaMemcpyDeviceToHost));

                CUDA_CHECK(cudaMemcpy(
                    out_ay.data() + current_start_g,
                    d_out_ay,
                    static_cast<size_t>(b_count) * sizeof(float),
                    cudaMemcpyDeviceToHost));
            }
        }
    }
}

extern "C" void gpu_solve_round_cublas(
    bool is_new_instance,
    const std::vector<float>& inst_s0,
    const std::vector<float>& inst_s1,
    const std::vector<int32_t>& inst_labels,
    int total_guesses,
    int N,
    int max_N,
    int T,
    int current_round,
    std::vector<float>& global_results,
    std::vector<int>& global_best_t,
    int hw_model_type) {
    (void)global_best_t;
    (void)hw_model_type;

    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    if (num_gpus <= 0) {
        fprintf(stderr, "No CUDA device found.\n");
        exit(1);
    }

    int guesses_per_gpu = (total_guesses + num_gpus - 1) / num_gpus;

    #pragma omp parallel num_threads(num_gpus)
    {
        int gpu_id = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(gpu_id));

        if (cublas_handle == nullptr) {
            CUBLAS_CHECK(cublasCreate(&cublas_handle));
        }

        int start_g = gpu_id * guesses_per_gpu;
        int end_g = std::min(start_g + guesses_per_gpu, total_guesses);
        int total_gpu_guesses = end_g - start_g;

        if (total_gpu_guesses > 0) {
            if (is_new_instance || d_single_s0 == nullptr) {
                if (d_single_s0 != nullptr) {
                    cudaFree(d_single_s0);
                    cudaFree(d_single_s1);
                    cudaFree(d_single_labels);
                    cudaFree(d_single_T_norm);
                    cudaFree(d_single_L_norm);
                    cudaFree(d_single_R);
                    cudaFree(d_single_out);
                }

                CUDA_CHECK(cudaMalloc(&d_single_s0, static_cast<size_t>(max_N) * 256 * T * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_single_s1, static_cast<size_t>(max_N) * 256 * T * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_single_labels, static_cast<size_t>(max_N) * 256 * sizeof(int32_t)));
                CUDA_CHECK(cudaMalloc(&d_single_T_norm, static_cast<size_t>(max_N) * T * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_single_L_norm, static_cast<size_t>(MAX_GUESS_BATCH) * max_N * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_single_R, static_cast<size_t>(MAX_GUESS_BATCH) * T * sizeof(float)));
                CUDA_CHECK(cudaMalloc(&d_single_out, static_cast<size_t>(MAX_GUESS_BATCH) * sizeof(float)));

                CUDA_CHECK(cudaMemcpy(
                    d_single_s0,
                    inst_s0.data(),
                    static_cast<size_t>(max_N) * 256 * T * sizeof(float),
                    cudaMemcpyHostToDevice));

                CUDA_CHECK(cudaMemcpy(
                    d_single_s1,
                    inst_s1.data(),
                    static_cast<size_t>(max_N) * 256 * T * sizeof(float),
                    cudaMemcpyHostToDevice));

                CUDA_CHECK(cudaMemcpy(
                    d_single_labels,
                    inst_labels.data(),
                    static_cast<size_t>(max_N) * 256 * sizeof(int32_t),
                    cudaMemcpyHostToDevice));
            }

            int threads = 256;
            int blocks_T = (T + threads - 1) / threads;

            combine_and_normalize_cs1_T<<<blocks_T, threads>>>(
                d_single_s0,
                d_single_s1,
                max_N,
                N,
                T,
                current_round,
                d_single_T_norm);

            CUDA_CHECK(cudaDeviceSynchronize());

            for (int b_offset = 0; b_offset < total_gpu_guesses; b_offset += MAX_GUESS_BATCH) {
                int b_count = std::min(static_cast<int>(MAX_GUESS_BATCH), total_gpu_guesses - b_offset);
                int current_start_g = start_g + b_offset;
                int blocks_L = (b_count + threads - 1) / threads;

                generate_L_norm_cs1<<<blocks_L, threads>>>(
                    d_single_labels,
                    N,
                    current_round,
                    current_start_g,
                    b_count,
                    d_single_L_norm);

                CUDA_CHECK(cudaDeviceSynchronize());

                float alpha = 1.0f;
                float beta = 0.0f;

                CUBLAS_CHECK(cublasSgemm(
                    cublas_handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_N,
                    T,
                    b_count,
                    N,
                    &alpha,
                    d_single_T_norm,
                    T,
                    d_single_L_norm,
                    N,
                    &beta,
                    d_single_R,
                    T));

                CUDA_CHECK(cudaDeviceSynchronize());

                compute_single_scores<<<blocks_L, threads>>>(
                    d_single_R,
                    T,
                    b_count,
                    d_single_out);

                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaMemcpy(
                    global_results.data() + current_start_g,
                    d_single_out,
                    static_cast<size_t>(b_count) * sizeof(float),
                    cudaMemcpyDeviceToHost));
            }
        }
    }
}
