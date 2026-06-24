//! CUDA kernel source code — equivalent to metal/shaders.rs.
//! Compiled to PTX at runtime via cudarc::nvrtc.

/// All kernel names that get loaded into the CUDA module.
pub const KERNEL_NAMES: &[&str] = &[
    "matmul_tiled",
    "matmul_tiled_trans_b",
    "matmul_trans_a_tiled",
    "batched_matmul_tiled",
    "batched_matmul_tiled_trans_b",
    "batched_matmul_tiled_trans_a",
    "batched_matmul_tiled_fp32",
    "batched_matmul_tiled_trans_b_fp32",
    "batched_matmul_tiled_trans_a_fp32",
    "matmul_tiled_f16",
    "matmul_tiled_trans_b_f16",
    "matmul_trans_a_tiled_f16",
    "batched_matmul_tiled_f16",
    "batched_matmul_tiled_trans_b_f16",
    "batched_matmul_tiled_trans_a_f16",
    "softmax",
    "rms_norm",
    "rms_norm_residual",
    "rope",
    "rope_backward",
    "rope_yarn",
    "rope_yarn_backward",
    "add_kernel",
    "add_inplace",
    "mul_kernel",
    "scale_kernel",
    "fill_kernel",
    "copy_kernel",
    "silu",
    "silu_gate",
    "cross_entropy",
    "reduce_sum",
    "kl_divergence",
    "adamw_update",
    "embedding_lookup",
    "cast_f32_to_f16",
    "cast_f16_to_f32",
    "cast_f32_to_bf16",
    "transpose_perm_forward",
    "transpose_perm_backward",
    "transpose_2d",
    "causal_mask",
    "compact_strided_copy",
    "strided_batch_copy",
    "buffer_copy",
    "silu_backward",
    "silu_gate_backward",
    "rms_norm_backward",
    "softmax_backward",
    "embedding_backward",
    "l2_norm_check",
    "argmax",
    "temperature_scale",
    "gradient_mask",
    "zero_rows",
    "ema_update",
    "repeat_kv",
    "repeat_kv_backward",
    "causal_doc_mask",
    "relu",
    "relu_backward",
    "exp_kernel",
    "axpy",
    "scale_rows",
    "row_dot_reduce",
    "broadcast_rows",
    "slice_cols",
    "concat_cols",
    "block_mean_keys",
    "block_sparse_topk_mask",
    "gather_blocks",
    "gather_blocks_backward",
    "gather_causal_mask",
    "muon_frob_normalize",
    "inv_sqrt_bc",
    "cautious_mask",
    "cautious_scale",
    "matmul_tiled_fp32",
    "matmul_tiled_trans_b_fp32",
    "moe_gather",
    "moe_scatter_add",
    "matmul_tiled_bf16",
    "ternary_matmul",
    "ternary_absmean",
    "ternary_pack",
    "adamw_8bit_update",
    "flash_attention_forward",
    "flash_attn_precompute_d",
    "flash_attention_backward",
    "lion_update",
    "sophia_update",
    "logsumexp",
    "compute_inv_rms",
    "causal_mask_window",
];

/// All CUDA kernels in a single compilation unit.
pub const ALL_KERNELS: &str = r#"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================
// Tiled Matrix Multiply: C = A @ B
// A: [M, K], B: [K, N], C: [M, N]
// Tile: 32x32, 64 threads per block, 4x4 sub-tile per thread
// ============================================================

struct MatmulParams {
    unsigned int M, N, K;
};

#define TILE 32
#define THREAD_TILE 4

extern "C" __global__ void matmul_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    int local_row = threadIdx.x / 8;
    int local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * TILE;
    int tile_col = blockIdx.x * TILE;

    __shared__ __half As[TILE][TILE];
    __shared__ __half Bs[TILE][TILE];
    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    for (int k_block = 0; k_block < K; k_block += TILE) {
        // Load A tile
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gr = tile_row + r, gc = k_block + c;
            float val = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            As[r][c] = __float2half(fminf(fmaxf(val, -65504.0f), 65504.0f));
        }
        // Load B tile
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gr = k_block + r, gc = tile_col + c;
            float val = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            Bs[r][c] = __float2half(fminf(fmaxf(val, -65504.0f), 65504.0f));
        }
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            __half a_vals[THREAD_TILE], b_vals[THREAD_TILE];
            for (int i = 0; i < THREAD_TILE; i++)
                a_vals[i] = As[local_row * THREAD_TILE + i][k];
            for (int j = 0; j < THREAD_TILE; j++)
                b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            for (int i = 0; i < THREAD_TILE; i++)
                for (int j = 0; j < THREAD_TILE; j++)
                    acc[i][j] += __half2float(__hmul(a_vals[i], b_vals[j]));
        }
        __syncthreads();
    }

    for (int i = 0; i < THREAD_TILE; i++)
        for (int j = 0; j < THREAD_TILE; j++) {
            int gr = tile_row + local_row * THREAD_TILE + i;
            int gc = tile_col + local_col * THREAD_TILE + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

// ============================================================
// Matmul with B transposed: C = A @ B^T
// ============================================================
extern "C" __global__ void matmul_tiled_trans_b(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    int local_row = threadIdx.x / 8;
    int local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * TILE;
    int tile_col = blockIdx.x * TILE;

    __shared__ __half As[TILE][TILE];
    __shared__ __half Bs[TILE][TILE];
    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    for (int k_block = 0; k_block < K; k_block += TILE) {
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gr = tile_row + r, gc = k_block + c;
            float val = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
            As[r][c] = __float2half(fminf(fmaxf(val, -65504.0f), 65504.0f));
        }
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gk = k_block + r, gn = tile_col + c;
            float val = (gk < K && gn < N) ? B[gn * K + gk] : 0.0f;
            Bs[r][c] = __float2half(fminf(fmaxf(val, -65504.0f), 65504.0f));
        }
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            __half a_vals[THREAD_TILE], b_vals[THREAD_TILE];
            for (int i = 0; i < THREAD_TILE; i++)
                a_vals[i] = As[local_row * THREAD_TILE + i][k];
            for (int j = 0; j < THREAD_TILE; j++)
                b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            for (int i = 0; i < THREAD_TILE; i++)
                for (int j = 0; j < THREAD_TILE; j++)
                    acc[i][j] += __half2float(__hmul(a_vals[i], b_vals[j]));
        }
        __syncthreads();
    }

    for (int i = 0; i < THREAD_TILE; i++)
        for (int j = 0; j < THREAD_TILE; j++) {
            int gr = tile_row + local_row * THREAD_TILE + i;
            int gc = tile_col + local_col * THREAD_TILE + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

// FP32 twin of matmul_tiled_trans_b (the precise, no-fp16-cast path). C = A @ B^T.
extern "C" __global__ void matmul_tiled_trans_b_fp32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    int local_row = threadIdx.x / 8;
    int local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * TILE;
    int tile_col = blockIdx.x * TILE;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    for (int k_block = 0; k_block < K; k_block += TILE) {
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gr = tile_row + r, gc = k_block + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gk = k_block + r, gn = tile_col + c;
            Bs[r][c] = (gk < K && gn < N) ? B[gn * K + gk] : 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            float a_vals[THREAD_TILE], b_vals[THREAD_TILE];
            for (int i = 0; i < THREAD_TILE; i++)
                a_vals[i] = As[local_row * THREAD_TILE + i][k];
            for (int j = 0; j < THREAD_TILE; j++)
                b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            for (int i = 0; i < THREAD_TILE; i++)
                for (int j = 0; j < THREAD_TILE; j++)
                    acc[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < THREAD_TILE; i++)
        for (int j = 0; j < THREAD_TILE; j++) {
            int gr = tile_row + local_row * THREAD_TILE + i;
            int gc = tile_col + local_col * THREAD_TILE + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

// ============================================================
// Matmul A^T @ B: C = A^T @ B
// ============================================================
extern "C" __global__ void matmul_trans_a_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M, unsigned int K, unsigned int N
) {
    int local_row = threadIdx.x / 8;
    int local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * TILE;
    int tile_col = blockIdx.x * TILE;

    __shared__ __half As[TILE][TILE];
    __shared__ __half Bs[TILE][TILE];
    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    for (int m_block = 0; m_block < M; m_block += TILE) {
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gk = tile_row + r, gm = m_block + c;
            float val = (gk < K && gm < M) ? A[gm * K + gk] : 0.0f;
            As[r][c] = __float2half(fminf(fmaxf(val, -65504.0f), 65504.0f));
        }
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gm = m_block + r, gn = tile_col + c;
            float val = (gm < M && gn < N) ? B[gm * N + gn] : 0.0f;
            Bs[r][c] = __float2half(fminf(fmaxf(val, -65504.0f), 65504.0f));
        }
        __syncthreads();

        for (int m = 0; m < TILE; m++) {
            __half a_vals[THREAD_TILE], b_vals[THREAD_TILE];
            for (int i = 0; i < THREAD_TILE; i++)
                a_vals[i] = As[local_row * THREAD_TILE + i][m];
            for (int j = 0; j < THREAD_TILE; j++)
                b_vals[j] = Bs[m][local_col * THREAD_TILE + j];
            for (int i = 0; i < THREAD_TILE; i++)
                for (int j = 0; j < THREAD_TILE; j++)
                    acc[i][j] += __half2float(__hmul(a_vals[i], b_vals[j]));
        }
        __syncthreads();
    }

    for (int i = 0; i < THREAD_TILE; i++)
        for (int j = 0; j < THREAD_TILE; j++) {
            int gr = tile_row + local_row * THREAD_TILE + i;
            int gc = tile_col + local_col * THREAD_TILE + j;
            if (gr < K && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

// ============================================================
// Softmax: output[row] = softmax(input[row])
// ============================================================
extern "C" __global__ void softmax(
    const float* input, float* output,
    unsigned int rows, unsigned int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    __shared__ float shared_max[256];
    float local_max = -1e30f;
    for (int c = tid; c < cols; c += nthreads)
        local_max = fmaxf(local_max, row_in[c]);
    shared_max[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        __syncthreads();
    }
    float row_max = shared_max[0];

    __shared__ float shared_sum[256];
    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += nthreads) {
        float val = expf(row_in[c] - row_max);
        row_out[c] = val;
        local_sum += val;
    }
    shared_sum[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / shared_sum[0];
    for (int c = tid; c < cols; c += nthreads)
        row_out[c] *= inv_sum;
}

// ============================================================
// RMS Norm: output = (x / rms(x)) * weight
// ============================================================
extern "C" __global__ void rms_norm(
    const float* input, const float* weight, float* output,
    unsigned int rows, unsigned int cols, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const float* row_in = input + row * cols;
    float* row_out = output + row * cols;

    __shared__ float shared_ss[256];
    float local_ss = 0.0f;
    for (int c = tid; c < cols; c += nthreads) {
        float v = row_in[c];
        local_ss += v * v;
    }
    shared_ss[tid] = local_ss;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) shared_ss[tid] += shared_ss[tid + s];
        __syncthreads();
    }
    float rms = sqrtf(shared_ss[0] / (float)cols + eps);
    float inv_rms = 1.0f / rms;
    for (int c = tid; c < cols; c += nthreads)
        row_out[c] = row_in[c] * inv_rms * weight[c];
}

// ============================================================
// Element-wise kernels
// ============================================================
extern "C" __global__ void add_kernel(const float* a, const float* b, float* c, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) c[i] = a[i] + b[i];
}

extern "C" __global__ void add_inplace(float* a, const float* b, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) a[i] += b[i];
}

extern "C" __global__ void mul_kernel(const float* a, const float* b, float* c, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) c[i] = a[i] * b[i];
}

extern "C" __global__ void scale_kernel(float* data, unsigned int size, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) data[i] *= scale;
}

extern "C" __global__ void fill_kernel(float* data, unsigned int size, float value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) data[i] = value;
}

extern "C" __global__ void copy_kernel(const float* src, float* dst, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) dst[i] = src[i];
}

extern "C" __global__ void silu_gate(const float* gate, const float* up, float* output, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float x = gate[i];
        float silu_x = x / (1.0f + expf(-x));
        output[i] = silu_x * up[i];
    }
}

// ============================================================
// RoPE (Rotary Position Embedding)
// ============================================================
extern "C" __global__ void rope(
    float* data, unsigned int total_rows, unsigned int seq_len,
    unsigned int head_dim, unsigned int offset, float theta
) {
    int row = blockIdx.x;
    int pos = blockIdx.y;
    int pair = threadIdx.x;
    if (row >= total_rows || pos >= seq_len || pair >= head_dim / 2) return;

    float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)head_dim);
    float angle = (float)(pos + offset) * freq;
    float cos_val, sin_val;
    sincosf(angle, &sin_val, &cos_val);

    int base = row * seq_len * head_dim + pos * head_dim;
    int i0 = base + pair * 2;
    int i1 = base + pair * 2 + 1;
    float x0 = data[i0], x1 = data[i1];
    data[i0] = x0 * cos_val - x1 * sin_val;
    data[i1] = x0 * sin_val + x1 * cos_val;
}

extern "C" __global__ void rope_backward(
    const float* grad_out, float* grad_in,
    unsigned int total_rows, unsigned int seq_len,
    unsigned int head_dim, unsigned int offset, float theta
) {
    int row = blockIdx.x;
    int pos = blockIdx.y;
    int pair = threadIdx.x;
    if (row >= total_rows || pos >= seq_len || pair >= head_dim / 2) return;

    float freq = 1.0f / powf(theta, (float)(2 * pair) / (float)head_dim);
    float angle = (float)(pos + offset) * freq;
    float cos_val, sin_val;
    sincosf(angle, &sin_val, &cos_val);

    int base = row * seq_len * head_dim + pos * head_dim;
    int i0 = base + 2 * pair;
    int i1 = i0 + 1;
    float g0 = grad_out[i0], g1 = grad_out[i1];
    grad_in[i0] = g0 * cos_val + g1 * sin_val;
    grad_in[i1] = -g0 * sin_val + g1 * cos_val;
}

// YaRN NTK-by-parts RoPE frequency: extrapolate high-frequency dims, interpolate low-frequency
// dims (mirrors the Metal fused_transpose_rope_yarn shader). yarn_scale==1.0 => plain RoPE.
__device__ __forceinline__ float yarn_rope_freq(
    int pair, unsigned int head_dim, float theta, float yarn_scale, float yarn_orig_max
) {
    float inv_extrap = 1.0f / powf(theta, (float)(2 * pair) / (float)head_dim);
    if (yarn_scale == 1.0f) return inv_extrap;
    float inv_interp = inv_extrap / yarn_scale;
    const float two_pi = 6.28318530718f;
    float log_base = logf(theta);
    float low = floorf((float)head_dim * logf(yarn_orig_max / (32.0f * two_pi)) / (2.0f * log_base));
    float high = ceilf((float)head_dim * logf(yarn_orig_max / two_pi) / (2.0f * log_base));
    low = fmaxf(low, 0.0f);
    high = fminf(high, (float)(head_dim / 2 - 1));
    float ramp = fminf(fmaxf(((float)pair - low) / fmaxf(high - low, 1e-3f), 0.0f), 1.0f);
    float extrap_mask = 1.0f - ramp;
    return inv_interp * (1.0f - extrap_mask) + inv_extrap * extrap_mask;
}

extern "C" __global__ void rope_yarn(
    float* data, unsigned int total_rows, unsigned int seq_len,
    unsigned int head_dim, unsigned int offset, float theta,
    float yarn_scale, float yarn_orig_max
) {
    int row = blockIdx.x;
    int pos = blockIdx.y;
    int pair = threadIdx.x;
    if (row >= total_rows || pos >= seq_len || pair >= head_dim / 2) return;

    float freq = yarn_rope_freq(pair, head_dim, theta, yarn_scale, yarn_orig_max);
    float angle = (float)(pos + offset) * freq;
    float cos_val, sin_val;
    sincosf(angle, &sin_val, &cos_val);

    int base = row * seq_len * head_dim + pos * head_dim;
    int i0 = base + pair * 2;
    int i1 = base + pair * 2 + 1;
    float x0 = data[i0], x1 = data[i1];
    data[i0] = x0 * cos_val - x1 * sin_val;
    data[i1] = x0 * sin_val + x1 * cos_val;
}

extern "C" __global__ void rope_yarn_backward(
    const float* grad_out, float* grad_in,
    unsigned int total_rows, unsigned int seq_len,
    unsigned int head_dim, unsigned int offset, float theta,
    float yarn_scale, float yarn_orig_max
) {
    int row = blockIdx.x;
    int pos = blockIdx.y;
    int pair = threadIdx.x;
    if (row >= total_rows || pos >= seq_len || pair >= head_dim / 2) return;

    float freq = yarn_rope_freq(pair, head_dim, theta, yarn_scale, yarn_orig_max);
    float angle = (float)(pos + offset) * freq;
    float cos_val, sin_val;
    sincosf(angle, &sin_val, &cos_val);

    int base = row * seq_len * head_dim + pos * head_dim;
    int i0 = base + 2 * pair;
    int i1 = i0 + 1;
    float g0 = grad_out[i0], g1 = grad_out[i1];
    grad_in[i0] = g0 * cos_val + g1 * sin_val;
    grad_in[i1] = -g0 * sin_val + g1 * cos_val;
}

// ============================================================
// Cross-entropy loss (fused softmax + NLL)
// ============================================================
extern "C" __global__ void cross_entropy(
    const float* logits, const unsigned int* targets,
    float* losses, float* grad_logits,
    unsigned int batch, unsigned int vocab
) {
    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const float* row_in = logits + row * vocab;
    float* row_grad = grad_logits + row * vocab;
    unsigned int target = targets[row];

    // Max for stability
    __shared__ float shared_max[256];
    float local_max = -1e30f;
    for (int c = tid; c < vocab; c += nthreads)
        local_max = fmaxf(local_max, row_in[c]);
    shared_max[tid] = local_max;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        __syncthreads();
    }
    float row_max = shared_max[0];

    // Sum of exp
    __shared__ float shared_sum[256];
    float local_sum = 0.0f;
    for (int c = tid; c < vocab; c += nthreads)
        local_sum += expf(row_in[c] - row_max);
    shared_sum[tid] = local_sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) shared_sum[tid] += shared_sum[tid + s];
        __syncthreads();
    }
    float log_sum_exp = logf(shared_sum[0]) + row_max;

    // Loss + gradient
    if (tid == 0)
        losses[row] = -(row_in[target] - log_sum_exp);

    float inv_sum = 1.0f / shared_sum[0];
    // grad = softmax - onehot (unscaled). The /batch mean-scaling is applied by the
    // gpu_cross_entropy wrapper via gpu_scale — the in-kernel `batch` param proved unreliable
    // in this expression under nvrtc (grid/loss saw 4, but 1.0f/(float)batch evaluated to 1.0).
    for (int c = tid; c < vocab; c += nthreads) {
        float prob = expf(row_in[c] - row_max) * inv_sum;
        row_grad[c] = prob - (c == target ? 1.0f : 0.0f);
    }
}

// ============================================================
// Reduce sum
// ============================================================
extern "C" __global__ void reduce_sum(const float* input, float* output, unsigned int size) {
    __shared__ float shared[256];
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (int i = tid; i < size; i += blockDim.x)
        local_sum += input[i];
    shared[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid == 0) output[0] = shared[0];
}

// ============================================================
// AdamW optimizer update (fused)
// ============================================================
extern "C" __global__ void adamw_update(
    float* param, const float* grad, float* m, float* v,
    unsigned int size, float lr, float beta1, float beta2,
    float eps, float weight_decay, int step, float update_clip
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    float g = grad[i];
    float m_val = beta1 * m[i] + (1.0f - beta1) * g;
    float v_val = beta2 * v[i] + (1.0f - beta2) * g * g;
    m[i] = m_val;
    v[i] = v_val;

    float m_hat = m_val / (1.0f - powf(beta1, (float)step));
    float v_hat = v_val / (1.0f - powf(beta2, (float)step));

    // Normalized Adam update; update_clip bounds it at the source (0 = disabled).
    float update = m_hat / (sqrtf(v_hat) + eps);
    if (update_clip > 0.0f) update = fmaxf(-update_clip, fminf(update_clip, update));
    param[i] = param[i] * (1.0f - lr * weight_decay) - lr * update;
}

// ============================================================
// Embedding lookup
// ============================================================
extern "C" __global__ void embedding_lookup(
    const float* table, const unsigned int* tokens, float* output,
    unsigned int seq_len, unsigned int dim
) {
    int pos = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (pos >= seq_len || d >= dim) return;
    output[pos * dim + d] = table[tokens[pos] * dim + d];
}

// ============================================================
// Causal mask
// ============================================================
extern "C" __global__ void causal_mask(
    float* scores, unsigned int batch_heads, unsigned int seq_q,
    unsigned int seq_k, unsigned int offset
) {
    int bh = blockIdx.x;
    int q = blockIdx.y;
    int k = blockIdx.z * blockDim.x + threadIdx.x;
    if (bh >= batch_heads || q >= seq_q || k >= seq_k) return;
    if (k > q + offset)
        scores[bh * seq_q * seq_k + q * seq_k + k] = -1e9f;
}

// ============================================================
// FP16 cast kernels
// ============================================================
// The "f16" buffer is a CudaSlice<f32> (4-byte elements); accessing it as __half* (2-byte stride)
// does not round-trip through cudarc. Pack TWO halves per 4-byte word and access it as unsigned int*
// (element size matches the f32 buffer exactly). One thread per packed word (= 2 source floats).
extern "C" __global__ void cast_f32_to_f16(const float* input, unsigned int* output, unsigned int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // packed-word index
    unsigned int n_words = (size + 1) / 2;
    if (i >= n_words) return;
    unsigned int lo_idx = 2 * i, hi_idx = 2 * i + 1;
    unsigned short lo = __half_as_ushort(__float2half(fminf(fmaxf(input[lo_idx], -65504.0f), 65504.0f)));
    unsigned short hi = 0;
    if (hi_idx < size) hi = __half_as_ushort(__float2half(fminf(fmaxf(input[hi_idx], -65504.0f), 65504.0f)));
    output[i] = ((unsigned int)hi << 16) | (unsigned int)lo;
}

extern "C" __global__ void cast_f16_to_f32(const unsigned int* input, float* output, unsigned int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // packed-word index
    unsigned int n_words = (size + 1) / 2;
    if (i >= n_words) return;
    unsigned int packed = input[i];
    output[2 * i] = __half2float(__ushort_as_half((unsigned short)(packed & 0xFFFF)));
    if (2 * i + 1 < size) output[2 * i + 1] = __half2float(__ushort_as_half((unsigned short)(packed >> 16)));
}

// f32 -> bf16, two bf16 packed little-endian per 32-bit word (element 2i in the low half), matching
// the contiguous __nv_bfloat16 layout cublasGemmEx reads. No clamp: bf16 shares fp32's exponent
// range, only the mantissa narrows to 7 bits (round-to-nearest-even via __float2bfloat16).
extern "C" __global__ void cast_f32_to_bf16(const float* input, unsigned int* output, unsigned int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; // packed-word index
    unsigned int n_words = (size + 1) / 2;
    if (i >= n_words) return;
    unsigned int lo_idx = 2 * i, hi_idx = 2 * i + 1;
    unsigned short lo = __bfloat16_as_ushort(__float2bfloat16(input[lo_idx]));
    unsigned short hi = 0;
    if (hi_idx < size) hi = __bfloat16_as_ushort(__float2bfloat16(input[hi_idx]));
    output[i] = ((unsigned int)hi << 16) | (unsigned int)lo;
}

// ============================================================
// L2 norm check (for gradient clipping)
// ============================================================
extern "C" __global__ void l2_norm_check(const float* data, float* output, unsigned int size) {
    // Multi-block grid-stride reduction. Each block reduces its slice in shared memory, then
    // atomically accumulates into a PRE-ZEROED output ([sum_sq, nan_flag]). This saturates the
    // whole GPU; the old single-block launch (launch_cfg(tpg,1)) serialized multi-million-element
    // gradient norms onto one SM and cost ~50% of CUDA training time on a 4090.
    __shared__ float shared_ss[256];
    __shared__ float shared_nan[256];
    int tid = threadIdx.x;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    float local_ss = 0.0f;
    float local_nan = 0.0f;
    for (unsigned int i = gid; i < size; i += stride) {
        float v = data[i];
        if (isnan(v) || isinf(v)) local_nan = 1.0f;
        else local_ss += v * v;
    }
    shared_ss[tid] = local_ss;
    shared_nan[tid] = local_nan;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_ss[tid] += shared_ss[tid + s];
            shared_nan[tid] = fmaxf(shared_nan[tid], shared_nan[tid + s]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&output[0], shared_ss[0]);
        if (shared_nan[0] > 0.0f) output[1] = 1.0f; // benign race: writers only ever store 1
    }
}

// ============================================================
// Buffer copy with offset
// ============================================================
extern "C" __global__ void buffer_copy(
    const float* src, float* dst,
    unsigned int src_offset, unsigned int dst_offset, unsigned int count
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) dst[dst_offset + i] = src[src_offset + i];
}

// ============================================================
// Backward kernels
// ============================================================
extern "C" __global__ void silu_backward(
    const float* input, const float* grad_out, float* grad_in, unsigned int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float x = input[i];
    float sig = 1.0f / (1.0f + expf(-x));
    grad_in[i] = grad_out[i] * (sig + x * sig * (1.0f - sig));
}

extern "C" __global__ void silu_gate_backward(
    const float* gate, const float* up, const float* grad_out,
    float* grad_gate, float* grad_up, unsigned int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    float x = gate[i];
    float sig = 1.0f / (1.0f + expf(-x));
    float silu_x = x * sig;
    grad_gate[i] = grad_out[i] * up[i] * (sig + x * sig * (1.0f - sig));
    grad_up[i] = grad_out[i] * silu_x;
}

extern "C" __global__ void rms_norm_backward(
    const float* input, const float* weight, const float* grad_out,
    float* grad_input, float* grad_weight,
    unsigned int rows, unsigned int cols, float eps, unsigned int clamp_on
) {
    // One block per row; blockDim is a power of 2 (>= a divisor sweep of cols via grid-stride).
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x, nt = blockDim.x;
    const float* x = input + row * cols;
    const float* go = grad_out + row * cols;
    float* gi = grad_input + row * cols;

    __shared__ float sh[256];

    // sum of squares -> inv_rms
    float local_ss = 0.0f;
    for (int c = tid; c < cols; c += nt) local_ss += x[c] * x[c];
    sh[tid] = local_ss;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    float inv_rms = rsqrtf(sh[0] / (float)cols + eps);
    // Collapsed-row guard: bound inv_rms so the inv_rms^3 correction can't explode (mean_sq->0).
    if (clamp_on) inv_rms = fminf(inv_rms, 31.62f);
    __syncthreads();

    // dot_sum = sum_c(grad_out[c] * weight[c] * x[c])
    float local_dot = 0.0f;
    for (int c = tid; c < cols; c += nt) local_dot += go[c] * weight[c] * x[c];
    sh[tid] = local_dot;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    float dot_sum = sh[0];

    // grad_input = grad_out*weight*inv_rms - x * (dot_sum * inv_rms^3 / cols)
    float correction = dot_sum * inv_rms * inv_rms * inv_rms / (float)cols;
    for (int c = tid; c < cols; c += nt) {
        float g = go[c] * weight[c] * inv_rms - x[c] * correction;
        gi[c] = clamp_on ? fmaxf(-1.0e3f, fminf(1.0e3f, g)) : g;
        atomicAdd(&grad_weight[c], go[c] * x[c] * inv_rms);
    }
}

extern "C" __global__ void softmax_backward(
    const float* softmax_out, const float* grad_out, float* grad_in,
    unsigned int rows, unsigned int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    const float* s = softmax_out + row * cols;
    const float* go = grad_out + row * cols;
    float* gi = grad_in + row * cols;

    __shared__ float shared[256];
    float local_dot = 0.0f;
    for (int c = tid; c < cols; c += nthreads)
        local_dot += go[c] * s[c];
    shared[tid] = local_dot;
    __syncthreads();
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    float dot_sum = shared[0];

    for (int c = tid; c < cols; c += nthreads)
        gi[c] = s[c] * (go[c] - dot_sum);
}

extern "C" __global__ void embedding_backward(
    const unsigned int* tokens, const float* grad_out, float* grad_table,
    unsigned int seq_len, unsigned int dim
) {
    int pos = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (pos >= seq_len || d >= dim) return;
    atomicAdd(&grad_table[tokens[pos] * dim + d], grad_out[pos * dim + d]);
}

// ============================================================
// BATCHED MATMUL: C[b] = A[b] @ B[b]
// ============================================================
extern "C" __global__ void batched_matmul_tiled(
    const float* A, const float* B, float* C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int batch
) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) return;
    int local_row = threadIdx.x / 8, local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * 32, tile_col = blockIdx.x * 32;

    const float* A_b = A + batch_idx * M * K;
    const float* B_b = B + batch_idx * K * N;
    float* C_b = C + batch_idx * M * N;

    __shared__ __half As[32][32], Bs[32][32];
    float acc[4][4] = {{0.0f}};

    for (int kb = 0; kb < K; kb += 32) {
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x * 16 + i, r = f / 32, c = f % 32;
            int gr = tile_row + r, gc = kb + c;
            float v = (gr < M && gc < K) ? A_b[gr*K+gc] : 0.0f;
            As[r][c] = __float2half(fminf(fmaxf(v, -65504.f), 65504.f));
        }
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x * 16 + i, r = f / 32, c = f % 32;
            int gr = kb + r, gc = tile_col + c;
            float v = (gr < K && gc < N) ? B_b[gr*N+gc] : 0.0f;
            Bs[r][c] = __float2half(fminf(fmaxf(v, -65504.f), 65504.f));
        }
        __syncthreads();
        for (int k = 0; k < 32; k++) {
            __half av[4], bv[4];
            for (int i=0;i<4;i++) av[i] = As[local_row*4+i][k];
            for (int j=0;j<4;j++) bv[j] = Bs[k][local_col*4+j];
            for (int i=0;i<4;i++) for (int j=0;j<4;j++)
                acc[i][j] += __half2float(__hmul(av[i], bv[j]));
        }
        __syncthreads();
    }
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) {
        int gr = tile_row + local_row*4+i, gc = tile_col + local_col*4+j;
        if (gr < M && gc < N) C_b[gr*N+gc] = acc[i][j];
    }
}

// Batched C[b] = A[b] @ B[b]^T
extern "C" __global__ void batched_matmul_tiled_trans_b(
    const float* A, const float* B, float* C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int batch
) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) return;
    int local_row = threadIdx.x / 8, local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * 32, tile_col = blockIdx.x * 32;

    const float* A_b = A + batch_idx * M * K;
    const float* B_b = B + batch_idx * N * K;
    float* C_b = C + batch_idx * M * N;

    __shared__ __half As[32][32], Bs[32][32];
    float acc[4][4] = {{0.0f}};

    for (int kb = 0; kb < K; kb += 32) {
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x*16+i, r = f/32, c = f%32;
            int gr = tile_row+r, gc = kb+c;
            float v = (gr<M&&gc<K) ? A_b[gr*K+gc] : 0.f;
            As[r][c] = __float2half(fminf(fmaxf(v,-65504.f),65504.f));
        }
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x*16+i, r = f/32, c = f%32;
            int gk = kb+r, gn = tile_col+c;
            float v = (gk<K&&gn<N) ? B_b[gn*K+gk] : 0.f;
            Bs[r][c] = __float2half(fminf(fmaxf(v,-65504.f),65504.f));
        }
        __syncthreads();
        for (int k=0;k<32;k++) {
            __half av[4],bv[4];
            for(int i=0;i<4;i++) av[i]=As[local_row*4+i][k];
            for(int j=0;j<4;j++) bv[j]=Bs[k][local_col*4+j];
            for(int i=0;i<4;i++) for(int j=0;j<4;j++)
                acc[i][j]+=__half2float(__hmul(av[i],bv[j]));
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
        int gr=tile_row+local_row*4+i, gc=tile_col+local_col*4+j;
        if(gr<M&&gc<N) C_b[gr*N+gc]=acc[i][j];
    }
}

// Batched C[b] = A[b]^T @ B[b]
extern "C" __global__ void batched_matmul_tiled_trans_a(
    const float* A, const float* B, float* C,
    unsigned int M, unsigned int K, unsigned int N, unsigned int batch
) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) return;
    int local_row = threadIdx.x / 8, local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * 32, tile_col = blockIdx.x * 32;

    const float* A_b = A + batch_idx * M * K;
    const float* B_b = B + batch_idx * M * N;
    float* C_b = C + batch_idx * K * N;

    __shared__ __half As[32][32], Bs[32][32];
    float acc[4][4] = {{0.0f}};

    for (int mb = 0; mb < M; mb += 32) {
        for (int i=0;i<16;i++) {
            int f=threadIdx.x*16+i, r=f/32, c=f%32;
            int gk=tile_row+r, gm=mb+c;
            float v=(gk<K&&gm<M)?A_b[gm*K+gk]:0.f;
            As[r][c]=__float2half(fminf(fmaxf(v,-65504.f),65504.f));
        }
        for (int i=0;i<16;i++) {
            int f=threadIdx.x*16+i, r=f/32, c=f%32;
            int gm=mb+r, gn=tile_col+c;
            float v=(gm<M&&gn<N)?B_b[gm*N+gn]:0.f;
            Bs[r][c]=__float2half(fminf(fmaxf(v,-65504.f),65504.f));
        }
        __syncthreads();
        for(int m=0;m<32;m++){
            __half av[4],bv[4];
            for(int i=0;i<4;i++) av[i]=As[local_row*4+i][m];
            for(int j=0;j<4;j++) bv[j]=Bs[m][local_col*4+j];
            for(int i=0;i<4;i++) for(int j=0;j<4;j++)
                acc[i][j]+=__half2float(__hmul(av[i],bv[j]));
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++){
        int gr=tile_row+local_row*4+i, gc=tile_col+local_col*4+j;
        if(gr<K&&gc<N) C_b[gr*N+gc]=acc[i][j];
    }
}

// FP32 twins of the batched matmuls — the precise path (the __half versions above are the
// fp16 fast path). float tiles + float multiply, no __float2half clamp.
extern "C" __global__ void batched_matmul_tiled_fp32(
    const float* A, const float* B, float* C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int batch
) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) return;
    int local_row = threadIdx.x / 8, local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * 32, tile_col = blockIdx.x * 32;
    const float* A_b = A + batch_idx * M * K;
    const float* B_b = B + batch_idx * K * N;
    float* C_b = C + batch_idx * M * N;
    __shared__ float As[32][32], Bs[32][32];
    float acc[4][4] = {{0.0f}};
    for (int kb = 0; kb < K; kb += 32) {
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x*16+i, r = f/32, c = f%32;
            int gr = tile_row+r, gc = kb+c;
            As[r][c] = (gr<M&&gc<K) ? A_b[gr*K+gc] : 0.0f;
        }
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x*16+i, r = f/32, c = f%32;
            int gr = kb+r, gc = tile_col+c;
            Bs[r][c] = (gr<K&&gc<N) ? B_b[gr*N+gc] : 0.0f;
        }
        __syncthreads();
        for (int k=0;k<32;k++) {
            float av[4], bv[4];
            for(int i=0;i<4;i++) av[i]=As[local_row*4+i][k];
            for(int j=0;j<4;j++) bv[j]=Bs[k][local_col*4+j];
            for(int i=0;i<4;i++) for(int j=0;j<4;j++) acc[i][j]+=av[i]*bv[j];
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
        int gr=tile_row+local_row*4+i, gc=tile_col+local_col*4+j;
        if(gr<M&&gc<N) C_b[gr*N+gc]=acc[i][j];
    }
}

extern "C" __global__ void batched_matmul_tiled_trans_b_fp32(
    const float* A, const float* B, float* C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int batch
) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) return;
    int local_row = threadIdx.x / 8, local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * 32, tile_col = blockIdx.x * 32;
    const float* A_b = A + batch_idx * M * K;
    const float* B_b = B + batch_idx * N * K;
    float* C_b = C + batch_idx * M * N;
    __shared__ float As[32][32], Bs[32][32];
    float acc[4][4] = {{0.0f}};
    for (int kb = 0; kb < K; kb += 32) {
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x*16+i, r = f/32, c = f%32;
            int gr = tile_row+r, gc = kb+c;
            As[r][c] = (gr<M&&gc<K) ? A_b[gr*K+gc] : 0.0f;
        }
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x*16+i, r = f/32, c = f%32;
            int gk = kb+r, gn = tile_col+c;
            Bs[r][c] = (gk<K&&gn<N) ? B_b[gn*K+gk] : 0.0f;
        }
        __syncthreads();
        for (int k=0;k<32;k++) {
            float av[4],bv[4];
            for(int i=0;i<4;i++) av[i]=As[local_row*4+i][k];
            for(int j=0;j<4;j++) bv[j]=Bs[k][local_col*4+j];
            for(int i=0;i<4;i++) for(int j=0;j<4;j++) acc[i][j]+=av[i]*bv[j];
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
        int gr=tile_row+local_row*4+i, gc=tile_col+local_col*4+j;
        if(gr<M&&gc<N) C_b[gr*N+gc]=acc[i][j];
    }
}

extern "C" __global__ void batched_matmul_tiled_trans_a_fp32(
    const float* A, const float* B, float* C,
    unsigned int M, unsigned int K, unsigned int N, unsigned int batch
) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) return;
    int local_row = threadIdx.x / 8, local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * 32, tile_col = blockIdx.x * 32;
    const float* A_b = A + batch_idx * M * K;
    const float* B_b = B + batch_idx * M * N;
    float* C_b = C + batch_idx * K * N;
    __shared__ float As[32][32], Bs[32][32];
    float acc[4][4] = {{0.0f}};
    for (int mb = 0; mb < M; mb += 32) {
        for (int i=0;i<16;i++) {
            int f=threadIdx.x*16+i, r=f/32, c=f%32;
            int gk=tile_row+r, gm=mb+c;
            As[r][c]=(gk<K&&gm<M)?A_b[gm*K+gk]:0.0f;
        }
        for (int i=0;i<16;i++) {
            int f=threadIdx.x*16+i, r=f/32, c=f%32;
            int gm=mb+r, gn=tile_col+c;
            Bs[r][c]=(gm<M&&gn<N)?B_b[gm*N+gn]:0.0f;
        }
        __syncthreads();
        for(int m=0;m<32;m++){
            float av[4],bv[4];
            for(int i=0;i<4;i++) av[i]=As[local_row*4+i][m];
            for(int j=0;j<4;j++) bv[j]=Bs[m][local_col*4+j];
            for(int i=0;i<4;i++) for(int j=0;j<4;j++) acc[i][j]+=av[i]*bv[j];
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++){
        int gr=tile_row+local_row*4+i, gc=tile_col+local_col*4+j;
        if(gr<K&&gc<N) C_b[gr*N+gc]=acc[i][j];
    }
}

// ============================================================
// FP16-INPUT MATMUL VARIANTS (half* inputs, float output)
// ============================================================
extern "C" __global__ void matmul_tiled_f16(
    const __half* A, const __half* B, float* C,
    unsigned int M, unsigned int N, unsigned int K
) {
    int local_row=threadIdx.x/8, local_col=threadIdx.x%8;
    int tile_row=blockIdx.y*32, tile_col=blockIdx.x*32;
    __shared__ __half As[32][32], Bs[32][32];
    float acc[4][4]={{0.f}};
    for(int kb=0;kb<K;kb+=32){
        for(int i=0;i<16;i++){
            int f=threadIdx.x*16+i,r=f/32,c=f%32;
            int gr=tile_row+r,gc=kb+c;
            As[r][c]=(gr<M&&gc<K)?A[gr*K+gc]:__float2half(0.f);
        }
        for(int i=0;i<16;i++){
            int f=threadIdx.x*16+i,r=f/32,c=f%32;
            int gr=kb+r,gc=tile_col+c;
            Bs[r][c]=(gr<K&&gc<N)?B[gr*N+gc]:__float2half(0.f);
        }
        __syncthreads();
        for(int k=0;k<32;k++){
            __half av[4],bv[4];
            for(int i=0;i<4;i++) av[i]=As[local_row*4+i][k];
            for(int j=0;j<4;j++) bv[j]=Bs[k][local_col*4+j];
            for(int i=0;i<4;i++) for(int j=0;j<4;j++)
                acc[i][j]+=__half2float(__hmul(av[i],bv[j]));
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++){
        int gr=tile_row+local_row*4+i,gc=tile_col+local_col*4+j;
        if(gr<M&&gc<N) C[gr*N+gc]=acc[i][j];
    }
}

extern "C" __global__ void matmul_tiled_trans_b_f16(
    const __half* A, const __half* B, float* C,
    unsigned int M, unsigned int N, unsigned int K
) {
    int local_row=threadIdx.x/8, local_col=threadIdx.x%8;
    int tile_row=blockIdx.y*32, tile_col=blockIdx.x*32;
    __shared__ __half As[32][32], Bs[32][32];
    float acc[4][4]={{0.f}};
    for(int kb=0;kb<K;kb+=32){
        for(int i=0;i<16;i++){
            int f=threadIdx.x*16+i,r=f/32,c=f%32;
            int gr=tile_row+r,gc=kb+c;
            As[r][c]=(gr<M&&gc<K)?A[gr*K+gc]:__float2half(0.f);
        }
        for(int i=0;i<16;i++){
            int f=threadIdx.x*16+i,r=f/32,c=f%32;
            int gk=kb+r,gn=tile_col+c;
            Bs[r][c]=(gk<K&&gn<N)?B[gn*K+gk]:__float2half(0.f);
        }
        __syncthreads();
        for(int k=0;k<32;k++){
            __half av[4],bv[4];
            for(int i=0;i<4;i++) av[i]=As[local_row*4+i][k];
            for(int j=0;j<4;j++) bv[j]=Bs[k][local_col*4+j];
            for(int i=0;i<4;i++) for(int j=0;j<4;j++)
                acc[i][j]+=__half2float(__hmul(av[i],bv[j]));
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++){
        int gr=tile_row+local_row*4+i,gc=tile_col+local_col*4+j;
        if(gr<M&&gc<N) C[gr*N+gc]=acc[i][j];
    }
}

extern "C" __global__ void matmul_trans_a_tiled_f16(
    const __half* A, const __half* B, float* C,
    unsigned int M, unsigned int K, unsigned int N
) {
    int local_row=threadIdx.x/8, local_col=threadIdx.x%8;
    int tile_row=blockIdx.y*32, tile_col=blockIdx.x*32;
    __shared__ __half As[32][32], Bs[32][32];
    float acc[4][4]={{0.f}};
    for(int mb=0;mb<M;mb+=32){
        for(int i=0;i<16;i++){
            int f=threadIdx.x*16+i,r=f/32,c=f%32;
            int gk=tile_row+r,gm=mb+c;
            As[r][c]=(gk<K&&gm<M)?A[gm*K+gk]:__float2half(0.f);
        }
        for(int i=0;i<16;i++){
            int f=threadIdx.x*16+i,r=f/32,c=f%32;
            int gm=mb+r,gn=tile_col+c;
            Bs[r][c]=(gm<M&&gn<N)?B[gm*N+gn]:__float2half(0.f);
        }
        __syncthreads();
        for(int m=0;m<32;m++){
            __half av[4],bv[4];
            for(int i=0;i<4;i++) av[i]=As[local_row*4+i][m];
            for(int j=0;j<4;j++) bv[j]=Bs[m][local_col*4+j];
            for(int i=0;i<4;i++) for(int j=0;j<4;j++)
                acc[i][j]+=__half2float(__hmul(av[i],bv[j]));
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++){
        int gr=tile_row+local_row*4+i,gc=tile_col+local_col*4+j;
        if(gr<K&&gc<N) C[gr*N+gc]=acc[i][j];
    }
}

// Batched FP16-input variants (half* A,B → float C). Identical tiling/compute to the f32 batched
// kernels (which cast f32→half in-tile), but read pre-packed halves directly → bit-identical results,
// half the input bandwidth. Inputs come from cast_to_f16 (attention) or gpu_cast_f32_to_f16 (tests).
extern "C" __global__ void batched_matmul_tiled_f16(
    const __half* A, const __half* B, float* C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int batch
) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) return;
    int local_row = threadIdx.x / 8, local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * 32, tile_col = blockIdx.x * 32;
    const __half* A_b = A + batch_idx * M * K;
    const __half* B_b = B + batch_idx * K * N;
    float* C_b = C + batch_idx * M * N;
    __shared__ __half As[32][32], Bs[32][32];
    float acc[4][4] = {{0.0f}};
    for (int kb = 0; kb < K; kb += 32) {
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x*16+i, r = f/32, c = f%32;
            int gr = tile_row+r, gc = kb+c;
            As[r][c] = (gr<M&&gc<K) ? A_b[gr*K+gc] : __float2half(0.f);
        }
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x*16+i, r = f/32, c = f%32;
            int gr = kb+r, gc = tile_col+c;
            Bs[r][c] = (gr<K&&gc<N) ? B_b[gr*N+gc] : __float2half(0.f);
        }
        __syncthreads();
        for (int k = 0; k < 32; k++) {
            __half av[4], bv[4];
            for (int i=0;i<4;i++) av[i] = As[local_row*4+i][k];
            for (int j=0;j<4;j++) bv[j] = Bs[k][local_col*4+j];
            for (int i=0;i<4;i++) for (int j=0;j<4;j++)
                acc[i][j] += __half2float(__hmul(av[i], bv[j]));
        }
        __syncthreads();
    }
    for (int i=0;i<4;i++) for (int j=0;j<4;j++) {
        int gr = tile_row+local_row*4+i, gc = tile_col+local_col*4+j;
        if (gr<M&&gc<N) C_b[gr*N+gc] = acc[i][j];
    }
}

extern "C" __global__ void batched_matmul_tiled_trans_b_f16(
    const __half* A, const __half* B, float* C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int batch
) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) return;
    int local_row = threadIdx.x / 8, local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * 32, tile_col = blockIdx.x * 32;
    const __half* A_b = A + batch_idx * M * K;
    const __half* B_b = B + batch_idx * N * K;
    float* C_b = C + batch_idx * M * N;
    __shared__ __half As[32][32], Bs[32][32];
    float acc[4][4] = {{0.0f}};
    for (int kb = 0; kb < K; kb += 32) {
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x*16+i, r = f/32, c = f%32;
            int gr = tile_row+r, gc = kb+c;
            As[r][c] = (gr<M&&gc<K) ? A_b[gr*K+gc] : __float2half(0.f);
        }
        for (int i = 0; i < 16; i++) {
            int f = threadIdx.x*16+i, r = f/32, c = f%32;
            int gk = kb+r, gn = tile_col+c;
            Bs[r][c] = (gk<K&&gn<N) ? B_b[gn*K+gk] : __float2half(0.f);
        }
        __syncthreads();
        for (int k=0;k<32;k++) {
            __half av[4],bv[4];
            for(int i=0;i<4;i++) av[i]=As[local_row*4+i][k];
            for(int j=0;j<4;j++) bv[j]=Bs[k][local_col*4+j];
            for(int i=0;i<4;i++) for(int j=0;j<4;j++)
                acc[i][j]+=__half2float(__hmul(av[i],bv[j]));
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) {
        int gr=tile_row+local_row*4+i, gc=tile_col+local_col*4+j;
        if(gr<M&&gc<N) C_b[gr*N+gc]=acc[i][j];
    }
}

extern "C" __global__ void batched_matmul_tiled_trans_a_f16(
    const __half* A, const __half* B, float* C,
    unsigned int M, unsigned int K, unsigned int N, unsigned int batch
) {
    int batch_idx = blockIdx.z;
    if (batch_idx >= batch) return;
    int local_row = threadIdx.x / 8, local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * 32, tile_col = blockIdx.x * 32;
    const __half* A_b = A + batch_idx * M * K;
    const __half* B_b = B + batch_idx * M * N;
    float* C_b = C + batch_idx * K * N;
    __shared__ __half As[32][32], Bs[32][32];
    float acc[4][4] = {{0.0f}};
    for (int mb = 0; mb < M; mb += 32) {
        for (int i=0;i<16;i++) {
            int f=threadIdx.x*16+i, r=f/32, c=f%32;
            int gk=tile_row+r, gm=mb+c;
            As[r][c]=(gk<K&&gm<M)?A_b[gm*K+gk]:__float2half(0.f);
        }
        for (int i=0;i<16;i++) {
            int f=threadIdx.x*16+i, r=f/32, c=f%32;
            int gm=mb+r, gn=tile_col+c;
            Bs[r][c]=(gm<M&&gn<N)?B_b[gm*N+gn]:__float2half(0.f);
        }
        __syncthreads();
        for(int m=0;m<32;m++){
            __half av[4],bv[4];
            for(int i=0;i<4;i++) av[i]=As[local_row*4+i][m];
            for(int j=0;j<4;j++) bv[j]=Bs[m][local_col*4+j];
            for(int i=0;i<4;i++) for(int j=0;j<4;j++)
                acc[i][j]+=__half2float(__hmul(av[i],bv[j]));
        }
        __syncthreads();
    }
    for(int i=0;i<4;i++) for(int j=0;j<4;j++){
        int gr=tile_row+local_row*4+i, gc=tile_col+local_col*4+j;
        if(gr<K&&gc<N) C_b[gr*N+gc]=acc[i][j];
    }
}

// ============================================================
// TRANSPOSE PERMUTATION (for attention reshape)
// ============================================================
extern "C" __global__ void transpose_perm_forward(
    const float* input, float* output,
    unsigned int batch, unsigned int seq_len, unsigned int n_heads, unsigned int head_dim
) {
    // [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_heads * seq_len * head_dim;
    if (idx >= total) return;

    int hd = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % n_heads;
    int b = idx / (head_dim * seq_len * n_heads);

    int src = (b * seq_len + s) * (n_heads * head_dim) + h * head_dim + hd;
    output[idx] = input[src];
}

extern "C" __global__ void transpose_perm_backward(
    const float* grad_out, float* grad_in,
    unsigned int batch, unsigned int seq_len, unsigned int n_heads, unsigned int head_dim
) {
    // [batch*n_heads, seq, head_dim] → [batch*seq, n_heads*head_dim]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_heads * seq_len * head_dim;
    if (idx >= total) return;

    int hd = idx % head_dim;
    int s = (idx / head_dim) % seq_len;
    int h = (idx / (head_dim * seq_len)) % n_heads;
    int b = idx / (head_dim * seq_len * n_heads);

    int dst = (b * seq_len + s) * (n_heads * head_dim) + h * head_dim + hd;
    grad_in[dst] = grad_out[idx];
}

// ============================================================
// RMS NORM RESIDUAL (fused add + norm)
// ============================================================
extern "C" __global__ void rms_norm_residual(
    const float* a, const float* b, const float* weight,
    float* output, float* sum_out,
    unsigned int rows, unsigned int cols, float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int tid = threadIdx.x, nt = blockDim.x;

    const float* ra = a + row * cols;
    const float* rb = b + row * cols;
    float* ro = output + row * cols;
    float* rs = sum_out + row * cols;

    // Compute sum and sum-of-squares
    __shared__ float shared_ss[256];
    float local_ss = 0.0f;
    for (int c = tid; c < cols; c += nt) {
        float s = ra[c] + rb[c];
        rs[c] = s;
        local_ss += s * s;
    }
    shared_ss[tid] = local_ss;
    __syncthreads();
    for (int s = nt/2; s > 0; s >>= 1) {
        if (tid < s) shared_ss[tid] += shared_ss[tid+s];
        __syncthreads();
    }
    float inv_rms = rsqrtf(shared_ss[0] / (float)cols + eps);
    for (int c = tid; c < cols; c += nt)
        ro[c] = rs[c] * inv_rms * weight[c];
}

// ============================================================
// KV CACHE OPERATIONS
// ============================================================
extern "C" __global__ void compact_strided_copy(
    const float* src, float* dst,
    unsigned int src_stride, unsigned int dst_stride,
    unsigned int copy_len, unsigned int n_rows
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= n_rows || col >= copy_len) return;
    dst[row * dst_stride + col] = src[row * src_stride + col];
}

extern "C" __global__ void strided_batch_copy(
    const float* src, float* dst,
    unsigned int src_offset, unsigned int dst_offset,
    unsigned int copy_len, unsigned int src_stride, unsigned int dst_stride,
    unsigned int n_rows
) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (row >= n_rows || col >= copy_len) return;
    dst[row * dst_stride + dst_offset + col] = src[row * src_stride + src_offset + col];
}

// ============================================================
// KL DIVERGENCE (for distillation)
// ============================================================
extern "C" __global__ void kl_divergence(
    const float* teacher, const float* student,
    float* losses, float* grad_student,
    unsigned int batch, unsigned int vocab, float temperature
) {
    int row = blockIdx.x;
    if (row >= batch) return;
    int tid = threadIdx.x, nt = blockDim.x;
    float inv_T = 1.0f / temperature;

    const float* t_row = teacher + row * vocab;
    const float* s_row = student + row * vocab;
    float* g_row = grad_student + row * vocab;

    // Teacher max + sum
    __shared__ float sh_tmax[256], sh_tsum[256];
    float t_max = -1e30f;
    for (int c = tid; c < vocab; c += nt) t_max = fmaxf(t_max, t_row[c] * inv_T);
    sh_tmax[tid] = t_max;
    __syncthreads();
    for (int s=nt/2;s>0;s>>=1) { if(tid<s) sh_tmax[tid]=fmaxf(sh_tmax[tid],sh_tmax[tid+s]); __syncthreads(); }
    t_max = sh_tmax[0];
    float t_sum = 0.0f;
    for (int c = tid; c < vocab; c += nt) t_sum += expf(t_row[c]*inv_T - t_max);
    sh_tsum[tid] = t_sum;
    __syncthreads();
    for (int s=nt/2;s>0;s>>=1) { if(tid<s) sh_tsum[tid]+=sh_tsum[tid+s]; __syncthreads(); }
    float t_log_sum = logf(sh_tsum[0]) + t_max;

    // Student max + sum
    __shared__ float sh_smax[256], sh_ssum[256];
    float s_max = -1e30f;
    for (int c = tid; c < vocab; c += nt) s_max = fmaxf(s_max, s_row[c] * inv_T);
    sh_smax[tid] = s_max;
    __syncthreads();
    for (int s=nt/2;s>0;s>>=1) { if(tid<s) sh_smax[tid]=fmaxf(sh_smax[tid],sh_smax[tid+s]); __syncthreads(); }
    s_max = sh_smax[0];
    float s_sum = 0.0f;
    for (int c = tid; c < vocab; c += nt) s_sum += expf(s_row[c]*inv_T - s_max);
    sh_ssum[tid] = s_sum;
    __syncthreads();
    for (int s=nt/2;s>0;s>>=1) { if(tid<s) sh_ssum[tid]+=sh_ssum[tid+s]; __syncthreads(); }
    float s_log_sum = logf(sh_ssum[0]) + s_max;

    // KL divergence + gradient
    __shared__ float sh_kl[256];
    float local_kl = 0.0f;
    float inv_batch = 1.0f / (float)batch;
    for (int c = tid; c < vocab; c += nt) {
        float p = expf(t_row[c]*inv_T - t_max) / sh_tsum[0];
        float q = expf(s_row[c]*inv_T - s_max) / sh_ssum[0];
        if (p > 1e-10f) local_kl += p * (logf(p) - logf(q + 1e-10f));
        g_row[c] = inv_T * (q - p) * inv_batch;
    }
    sh_kl[tid] = local_kl;
    __syncthreads();
    for (int s=nt/2;s>0;s>>=1) { if(tid<s) sh_kl[tid]+=sh_kl[tid+s]; __syncthreads(); }
    if (tid == 0) losses[row] = sh_kl[0];
}

// ============================================================
// UTILITY KERNELS
// ============================================================
// argmax / temperature_scale / gradient_mask / zero_rows are defined once further down
// (the "Sampling / generation" + "Training utilities" sections) with the signatures the Rust
// launch code in compute.rs actually calls (e.g. temperature_scale takes inv_temperature, and
// gradient_mask takes (grad, mask, total, vocab_size)). Duplicate defs here made NVRTC fail with
// "function already defined", which broke EVERY GPU kernel — so they were removed.

extern "C" __global__ void transpose_2d(
    const float* input, float* output,
    unsigned int rows, unsigned int cols
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    int r = i / cols, c = i % cols;
    output[c * rows + r] = input[r * cols + c];
}

extern "C" __global__ void silu(const float* input, float* output, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

// EMA update: ema = ema*decay + src*(1-decay)
extern "C" __global__ void ema_update(float* ema, const float* src, unsigned int size, float decay) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) ema[i] = ema[i] * decay + src[i] * (1.0f - decay);
}

// GQA: expand [n_kv_total, seq, hd] -> [n_kv_total*group_size, seq, hd] by repeating each KV head.
extern "C" __global__ void repeat_kv(const float* input, float* output,
    unsigned int n_kv_total, unsigned int group_size, unsigned int seq_len, unsigned int head_dim) {
    unsigned int head_block = seq_len * head_dim;
    unsigned int total = n_kv_total * group_size * head_block;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    unsigned int out_head = tid / head_block;
    unsigned int in_head = out_head / group_size;
    output[tid] = input[in_head * head_block + (tid % head_block)];
}

// GQA backward: sum the group_size expanded-head gradients back into each KV head.
extern "C" __global__ void repeat_kv_backward(const float* out_grad, float* kv_grad,
    unsigned int n_kv_total, unsigned int group_size, unsigned int seq_len, unsigned int head_dim) {
    unsigned int head_block = seq_len * head_dim;
    unsigned int total = n_kv_total * head_block;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    unsigned int kv_head = tid / head_block;
    unsigned int off = tid % head_block;
    float sum = 0.0f;
    for (unsigned int g = 0; g < group_size; g++)
        sum += out_grad[(kv_head * group_size + g) * head_block + off];
    kv_grad[tid] = sum;
}

// Seq-packing: mask future (k>q) and cross-document (seg_ids differ) positions.
extern "C" __global__ void causal_doc_mask(float* scores, const unsigned int* seg_ids,
    unsigned int batch_heads, unsigned int seq, unsigned int n_heads) {
    unsigned int bh = blockIdx.x;
    unsigned int q = blockIdx.y;
    unsigned int k = blockIdx.z * blockDim.x + threadIdx.x;
    if (bh >= batch_heads || q >= seq || k >= seq) return;
    unsigned int base = (bh / n_heads) * seq;
    if (k > q || seg_ids[base + q] != seg_ids[base + k])
        scores[bh * seq * seq + q * seq + k] = __int_as_float(0xff800000); // -inf (nvrtc lacks INFINITY)
}

extern "C" __global__ void relu(const float* input, float* output, unsigned int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) output[i] = fmaxf(input[i], 0.0f);
}
extern "C" __global__ void relu_backward(const float* input, const float* grad_out, float* grad_in, unsigned int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) grad_in[i] = input[i] > 0.0f ? grad_out[i] : 0.0f;
}
extern "C" __global__ void exp_kernel(const float* input, float* output, unsigned int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) output[i] = expf(input[i]);
}
extern "C" __global__ void axpy(float* y, const float* x, unsigned int size, float alpha) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) y[i] += alpha * x[i];
}
extern "C" __global__ void scale_rows(const float* input, const float* scales, float* output, unsigned int rows, unsigned int cols) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    output[idx] = input[idx] * scales[idx / cols];
}
extern "C" __global__ void row_dot_reduce(const float* a, const float* b, float* output, unsigned int rows, unsigned int cols) {
    unsigned int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    float s = 0.0f;
    for (unsigned int c = 0; c < cols; c++) s += a[r * cols + c] * b[r * cols + c];
    output[r] = s;
}
extern "C" __global__ void broadcast_rows(const float* vec, float* out, unsigned int rows, unsigned int cols) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows * cols) out[i] = vec[i % cols];
}
extern "C" __global__ void slice_cols(const float* src, float* dst, unsigned int rows, unsigned int src_cols, unsigned int dst_cols, unsigned int col_offset) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * dst_cols) return;
    unsigned int r = idx / dst_cols, c = idx % dst_cols;
    dst[idx] = src[r * src_cols + col_offset + c];
}
extern "C" __global__ void concat_cols(const float* src, float* dst, unsigned int rows, unsigned int src_cols, unsigned int dst_cols, unsigned int col_offset) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * src_cols) return;
    unsigned int r = idx / src_cols, c = idx % src_cols;
    dst[r * dst_cols + col_offset + c] = src[idx];
}

// ===== Block-sparse attention (MoBA/NSA): block-mean routing + gather/scatter + masks =====

// Mean of keys within each block: K[bh,seq,hd] -> out[bh,nb,hd]. One thread per output element.
extern "C" __global__ void block_mean_keys(const float* k, float* out,
    unsigned int bh, unsigned int seq, unsigned int hd, unsigned int nb, unsigned int block_size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = bh * nb * hd;
    if (idx >= total) return;
    unsigned int d = idx % hd;
    unsigned int t = idx / hd;
    unsigned int blk = t % nb;
    unsigned int bh_i = t / nb;
    unsigned int start = blk * block_size;
    unsigned int end = start + block_size; if (end > seq) end = seq;
    float s = 0.0f; unsigned int cnt = 0;
    for (unsigned int i = start; i < end; i++) { s += k[bh_i * seq * hd + i * hd + d]; cnt++; }
    out[bh_i * nb * hd + blk * hd + d] = cnt > 0 ? s / (float)cnt : 0.0f;
}

// Top-k block-sparse mask (own block + top-k past blocks by score), masks dense scores[bh,seq,seq]
// in place; includes the causal mask. block_scores: [bh,seq,nb]. (Non-training fallback path.)
extern "C" __global__ void block_sparse_topk_mask(float* scores, const float* block_scores,
    unsigned int bh, unsigned int seq, unsigned int nb, unsigned int block_size, unsigned int top_k) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = bh * seq * seq;
    if (gid >= total) return;
    unsigned int k = gid % seq;
    unsigned int t = gid / seq;
    unsigned int q = t % seq;
    unsigned int b = t / seq;
    unsigned int idx = b * seq * seq + q * seq + k;
    if (k > q) { scores[idx] = __int_as_float(0xff800000); return; } // causal
    unsigned int qb = q / block_size, kb = k / block_size;
    if (kb == qb) return; // own block always attended
    float my = block_scores[b * seq * nb + q * nb + kb];
    unsigned int better = 0;
    for (unsigned int j = 0; j < qb; j++) {
        if (block_scores[b * seq * nb + q * nb + j] > my) better++;
    }
    if (better >= top_k) scores[idx] = __int_as_float(0xff800000); // outside top-k
}

// Gather selected K/V blocks into compact [bh*nb, k_sel*block, hd]. sel: [bh*nb*k_sel] u32 indices
// (sentinel >= nb -> zero-filled padding). One thread per output element.
extern "C" __global__ void gather_blocks(const float* src, const unsigned int* sel, float* out,
    unsigned int bh, unsigned int nb, unsigned int seq, unsigned int hd, unsigned int block, unsigned int k_sel) {
    unsigned int sel_w = k_sel * block;
    unsigned int total = bh * nb * sel_w * hd;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    unsigned int d = gid % hd;
    unsigned int rem = gid / hd;
    unsigned int pos = rem % sel_w;
    unsigned int bnq = rem / sel_w;
    unsigned int slot = pos / block;
    unsigned int w = pos % block;
    unsigned int bh_idx = bnq / nb;
    unsigned int block_idx = sel[bnq * k_sel + slot];
    if (block_idx >= nb) { out[gid] = 0.0f; return; }
    unsigned int src_row = block_idx * block + w;
    if (src_row >= seq) { out[gid] = 0.0f; return; }
    out[gid] = src[bh_idx * seq * hd + src_row * hd + d];
}

// Backward (scatter-add transpose) of gather_blocks. d_src MUST be pre-zeroed (the wrapper does it);
// multiple query-blocks may select the same source block, so accumulate atomically.
extern "C" __global__ void gather_blocks_backward(const float* d_out, const unsigned int* sel, float* d_src,
    unsigned int bh, unsigned int nb, unsigned int seq, unsigned int hd, unsigned int block, unsigned int k_sel) {
    unsigned int sel_w = k_sel * block;
    unsigned int total = bh * nb * sel_w * hd;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    unsigned int d = gid % hd;
    unsigned int rem = gid / hd;
    unsigned int pos = rem % sel_w;
    unsigned int bnq = rem / sel_w;
    unsigned int slot = pos / block;
    unsigned int w = pos % block;
    unsigned int bh_idx = bnq / nb;
    unsigned int block_idx = sel[bnq * k_sel + slot];
    if (block_idx >= nb) return;             // sentinel gathered 0 -> no source
    unsigned int src_row = block_idx * block + w;
    if (src_row >= seq) return;
    atomicAdd(&d_src[bh_idx * seq * hd + src_row * hd + d], d_out[gid]);
}

// Causal mask for gathered scores [bh*nb, block, k_sel*block]: -inf where the gathered key column is
// padding or violates causality (key_global > query_global). One thread per score element.
extern "C" __global__ void gather_causal_mask(float* scores, const unsigned int* sel,
    unsigned int bh_nb, unsigned int nb, unsigned int block, unsigned int k_sel) {
    unsigned int sel_w = k_sel * block;
    unsigned int total = bh_nb * block * sel_w;
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    unsigned int c = gid % sel_w;
    unsigned int t = gid / sel_w;
    unsigned int r = t % block;
    unsigned int bnq = t / block;
    unsigned int qb = bnq % nb;
    unsigned int q_global = qb * block + r;
    unsigned int slot = c / block;
    unsigned int w = c % block;
    unsigned int block_idx = sel[bnq * k_sel + slot];
    unsigned int idx = bnq * block * sel_w + r * sel_w + c;
    bool masked;
    if (block_idx >= nb) { masked = true; }
    else { unsigned int k_global = block_idx * block + w; masked = k_global > q_global; }
    if (masked) scores[idx] = __int_as_float(0xff800000);
}

// ===== Muon optimizer: Frobenius normalize (for Newton-Schulz) + NorMuon per-neuron scale =====

// Single-block sum-of-squares reduction then per-element scale by rsqrt(ssq+eps): X = M / ||M||_F.
// Bounds the spectral norm so the cubic Newton-Schulz that follows always converges. Launch with ONE
// block of next_pow2(size)<=256 threads (matches metal's single-threadgroup dispatch).
extern "C" __global__ void muon_frob_normalize(const float* m, float* x, unsigned int size) {
    __shared__ float sdata[256];
    unsigned int tid = threadIdx.x;
    unsigned int tpg = blockDim.x;
    float local = 0.0f;
    for (unsigned int i = tid; i < size; i += tpg) { float v = m[i]; local += v * v; }
    sdata[tid] = local;
    __syncthreads();
    for (unsigned int stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    float inv = rsqrtf(sdata[0] + 1e-14f);
    for (unsigned int i = tid; i < size; i += tpg) { x[i] = m[i] * inv; }
}

// NorMuon per-neuron scale: out[i] = 1/(sqrt(v[i]*bias_correction) + eps), elementwise over [size].
extern "C" __global__ void inv_sqrt_bc(const float* v, float* out, unsigned int size, float bias_correction, float eps) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < size) {
        float vhat = v[gid] * bias_correction;
        out[gid] = 1.0f / (sqrtf(vhat) + eps);
    }
}

// ===== Precise fp32-tile matmul (no fp16 downcast/clamp) — C = A @ B, A[M,K] B[K,N] C[M,N] =====
// Identical tiling to matmul_tiled but float shared tiles: full fp32 range + precision (matmul_precise).
extern "C" __global__ void matmul_tiled_fp32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    int local_row = threadIdx.x / 8;
    int local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * TILE;
    int tile_col = blockIdx.x * TILE;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    for (int k_block = 0; k_block < K; k_block += TILE) {
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gr = tile_row + r, gc = k_block + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gr = k_block + r, gc = tile_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            float a_vals[THREAD_TILE], b_vals[THREAD_TILE];
            for (int i = 0; i < THREAD_TILE; i++)
                a_vals[i] = As[local_row * THREAD_TILE + i][k];
            for (int j = 0; j < THREAD_TILE; j++)
                b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            for (int i = 0; i < THREAD_TILE; i++)
                for (int j = 0; j < THREAD_TILE; j++)
                    acc[i][j] += a_vals[i] * b_vals[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < THREAD_TILE; i++)
        for (int j = 0; j < THREAD_TILE; j++) {
            int gr = tile_row + local_row * THREAD_TILE + i;
            int gc = tile_col + local_col * THREAD_TILE + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

// ===== MoE token routing: gather tokens for one expert + weighted scatter-add back =====

// gather: gathered[slot,d] = input[token_indices[slot], d]. 2D grid (slot, d).
extern "C" __global__ void moe_gather(const float* input, const unsigned int* token_indices, float* gathered,
    unsigned int n_routed, unsigned int dim) {
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int d = blockIdx.y * blockDim.y + threadIdx.y;
    if (slot >= n_routed || d >= dim) return;
    unsigned int token_idx = token_indices[slot];
    gathered[slot * dim + d] = input[token_idx * dim + d];
}

// scatter-add: combined[token,d] += weight[slot] * expert_output[slot,d]. Multiple slots may map to
// one token (top-k routing), so accumulate atomically (metal's plain read-add-write is a latent race).
extern "C" __global__ void moe_scatter_add(const float* expert_output, const unsigned int* token_indices,
    const float* weights, float* combined_output, unsigned int n_routed, unsigned int dim) {
    unsigned int slot = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int d = blockIdx.y * blockDim.y + threadIdx.y;
    if (slot >= n_routed || d >= dim) return;
    unsigned int token_idx = token_indices[slot];
    float val = expert_output[slot * dim + d] * weights[slot];
    atomicAdd(&combined_output[token_idx * dim + d], val);
}

// ===== bf16-tile matmul — C = A @ B with __nv_bfloat16 shared tiles, fp32 accumulate =====
// fp32 exponent range (NO ±65504 clamp, unlike the fp16 path) at bf16 mantissa precision (~7 bits).
extern "C" __global__ void matmul_tiled_bf16(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    int local_row = threadIdx.x / 8;
    int local_col = threadIdx.x % 8;
    int tile_row = blockIdx.y * TILE;
    int tile_col = blockIdx.x * TILE;

    __shared__ __nv_bfloat16 As[TILE][TILE];
    __shared__ __nv_bfloat16 Bs[TILE][TILE];
    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    for (int k_block = 0; k_block < K; k_block += TILE) {
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gr = tile_row + r, gc = k_block + c;
            As[r][c] = __float2bfloat16((gr < M && gc < K) ? A[gr * K + gc] : 0.0f);
        }
        for (int i = 0; i < 16; i++) {
            int flat = threadIdx.x * 16 + i;
            int r = flat / TILE, c = flat % TILE;
            int gr = k_block + r, gc = tile_col + c;
            Bs[r][c] = __float2bfloat16((gr < K && gc < N) ? B[gr * N + gc] : 0.0f);
        }
        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            __nv_bfloat16 a_vals[THREAD_TILE], b_vals[THREAD_TILE];
            for (int i = 0; i < THREAD_TILE; i++)
                a_vals[i] = As[local_row * THREAD_TILE + i][k];
            for (int j = 0; j < THREAD_TILE; j++)
                b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            for (int i = 0; i < THREAD_TILE; i++)
                for (int j = 0; j < THREAD_TILE; j++)
                    acc[i][j] += __bfloat162float(a_vals[i]) * __bfloat162float(b_vals[j]);
        }
        __syncthreads();
    }

    for (int i = 0; i < THREAD_TILE; i++)
        for (int j = 0; j < THREAD_TILE; j++) {
            int gr = tile_row + local_row * THREAD_TILE + i;
            int gc = tile_col + local_col * THREAD_TILE + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}

// ===== BitNet b1.58 ternary {-1,0,+1} matmul + quantize (16 weights packed per u32) =====
// C = A @ W, A[M,K] float, W[K,N] packed ternary. No fp multiply — conditional add/sub. 2D grid (col,row).
extern "C" __global__ void ternary_matmul(const float* A, const unsigned int* W_packed, float* C,
    unsigned int M, unsigned int N, unsigned int K) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    const float* a_row = A + row * K;
    float acc = 0.0f;
    unsigned int k = 0;
    for (; k + 16 <= K; k += 16) {
        unsigned int packed = W_packed[(k / 16) * N + col];
        for (unsigned int i = 0; i < 16; i++) {
            unsigned int bits = (packed >> (i * 2)) & 0x3u;
            if (bits == 1) acc += a_row[k + i];
            else if (bits == 2) acc -= a_row[k + i];
        }
    }
    if (k < K) {                                   // tail (K not a multiple of 16)
        unsigned int packed = W_packed[(k / 16) * N + col];
        for (unsigned int i = 0; i < K - k; i++) {
            unsigned int bits = (packed >> (i * 2)) & 0x3u;
            if (bits == 1) acc += a_row[k + i];
            else if (bits == 2) acc -= a_row[k + i];
        }
    }
    C[row * N + col] = acc;
}

// absmean per column (quantization threshold). 1D grid over cols.
extern "C" __global__ void ternary_absmean(const float* weights, float* absmean,
    unsigned int rows, unsigned int cols) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= cols) return;
    float sum = 0.0f;
    for (unsigned int r = 0; r < rows; r++) sum += fabsf(weights[r * cols + gid]);
    absmean[gid] = sum / (float)rows;
}

// quantize to ternary via absmean threshold and pack 16/u32. 2D grid (col, pack_row=K/16).
extern "C" __global__ void ternary_pack(const float* weights, const float* absmean, unsigned int* packed,
    unsigned int rows, unsigned int cols) {
    unsigned int pack_row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int K = rows, N = cols;
    if (col >= N) return;
    unsigned int k_start = pack_row * 16;
    if (k_start >= K) return;
    float threshold = absmean[col];
    float inv_thresh = (threshold > 1e-8f) ? (1.0f / threshold) : 0.0f;
    unsigned int packed_val = 0;
    unsigned int k_end = k_start + 16; if (k_end > K) k_end = K;
    for (unsigned int i = 0; i < k_end - k_start; i++) {
        float w = weights[(k_start + i) * N + col];
        float scaled = w * inv_thresh;
        unsigned int ternary;
        if (scaled > 0.5f) ternary = 1;            // +1
        else if (scaled < -0.5f) ternary = 2;      // -1 (0b10)
        else ternary = 0;
        packed_val |= (ternary << (i * 2));
    }
    packed[pack_row * N + col] = packed_val;
}

// ===== 8-bit (block-wise int8) AdamW — bitsandbytes-style, 1 block per 256-elem param-block =====
// m is signed-linear int8; v is int8 of sqrt(v) (range-compressed). Per-block fp32 absmax scales.
// Params packed in one POD struct (>12 launch args otherwise). signed char for the signed m quant.
struct Adam8Params {
    unsigned int size;
    float lr; float beta1; float beta2; float eps; float weight_decay;
    float bias_correction1; float bias_correction2; float update_clip;
};
extern "C" __global__ void adamw_8bit_update(
    float* param, const float* grad,
    signed char* m_q, signed char* v_q, float* m_scale, float* v_scale,
    Adam8Params p
) {
    __shared__ float sm[256];
    __shared__ float sv[256];
    unsigned int lid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int gid = bid * blockDim.x + lid;

    bool active = gid < p.size;
    float ms = m_scale[bid];
    float vs = v_scale[bid];                    // scale for sqrt(v), not v
    float g = active ? grad[gid] : 0.0f;
    float m_old = active ? (float)((int)m_q[gid]) * ms : 0.0f;
    float sqrt_v_old = active ? (float)((int)v_q[gid]) * vs : 0.0f;
    float v_old = sqrt_v_old * sqrt_v_old;
    float m_new = p.beta1 * m_old + (1.0f - p.beta1) * g;
    float v_new = p.beta2 * v_old + (1.0f - p.beta2) * g * g;
    float sqrt_v_new = sqrtf(v_new);

    if (active) {
        float m_hat = m_new / p.bias_correction1;
        float v_hat = v_new / p.bias_correction2;
        float upd = m_hat / (sqrtf(v_hat) + p.eps);
        if (p.update_clip > 0.0f) upd = fmaxf(-p.update_clip, fminf(p.update_clip, upd));
        param[gid] = param[gid] * (1.0f - p.lr * p.weight_decay) - p.lr * upd;
    }

    // Block absmax reduction (|m| and sqrt(v)) for requantization.
    sm[lid] = active ? fabsf(m_new) : 0.0f;
    sv[lid] = active ? sqrt_v_new : 0.0f;
    __syncthreads();
    for (unsigned int s = 128; s > 0; s >>= 1) {
        if (lid < s) { sm[lid] = fmaxf(sm[lid], sm[lid + s]); sv[lid] = fmaxf(sv[lid], sv[lid + s]); }
        __syncthreads();
    }
    float new_ms = sm[0] > 0.0f ? sm[0] / 127.0f : 0.0f;
    float new_vs = sv[0] > 0.0f ? sv[0] / 127.0f : 0.0f;
    if (lid == 0) { m_scale[bid] = new_ms; v_scale[bid] = new_vs; }

    if (active) {
        m_q[gid] = new_ms > 0.0f ? (signed char)(int)roundf(fmaxf(-127.0f, fminf(127.0f, m_new / new_ms))) : (signed char)0;
        v_q[gid] = new_vs > 0.0f ? (signed char)(int)roundf(fmaxf(0.0f, fminf(127.0f, sqrt_v_new / new_vs))) : (signed char)0;
    }
}

// ===== Flash Attention (Dao et al.) — fused tiled attention, online softmax, never stores N*N =====
#define FA_BR 32
#define FA_BC 32
#define NEG_INF __int_as_float(0xff800000)

// One block per (batch_head, query-block); 32 threads, one query row each. K/V tiled in shared (half).
extern "C" __global__ void flash_attention_forward(
    const float* Q, const float* K, const float* V, float* O,
    unsigned int batch_heads, unsigned int seq_q, unsigned int seq_k, unsigned int head_dim,
    float scale, unsigned int kv_offset
) {
    unsigned int bh = blockIdx.x;
    unsigned int q_block = blockIdx.y;
    unsigned int q_start = q_block * FA_BR;
    if (bh >= batch_heads) return;
    unsigned int d = head_dim;
    const float* Q_bh = Q + bh * seq_q * d;
    const float* K_bh = K + bh * seq_k * d;
    const float* V_bh = V + bh * seq_k * d;
    float* O_bh = O + bh * seq_q * d;

    __shared__ __half K_shared[FA_BC][128];
    __shared__ __half V_shared[FA_BC][128];

    unsigned int local_q = threadIdx.x;
    unsigned int global_q = q_start + local_q;
    bool active = (global_q < seq_q);

    float row_max = NEG_INF, row_sum = 0.0f;
    float o_acc[128];
    for (unsigned int i = 0; i < d; i++) o_acc[i] = 0.0f;
    float q_row[128];
    if (active) for (unsigned int i = 0; i < d; i++) q_row[i] = Q_bh[global_q * d + i];

    for (unsigned int k_start = 0; k_start < seq_k; k_start += FA_BC) {
        unsigned int k_end = k_start + FA_BC; if (k_end > seq_k) k_end = seq_k;
        unsigned int tile_len = k_end - k_start;
        // cooperative load (ALL threads, incl. inactive) — must hit every barrier in uniform control flow
        for (unsigned int j = threadIdx.x; j < tile_len; j += FA_BR) {
            unsigned int gk = k_start + j;
            for (unsigned int i = 0; i < d; i++) {
                K_shared[j][i] = __float2half(K_bh[gk * d + i]);
                V_shared[j][i] = __float2half(V_bh[gk * d + i]);
            }
        }
        for (unsigned int j = tile_len + threadIdx.x; j < FA_BC; j += FA_BR)
            for (unsigned int i = 0; i < d; i++) { K_shared[j][i] = __float2half(0.0f); V_shared[j][i] = __float2half(0.0f); }
        __syncthreads();

        if (active) {
            float s_vals[FA_BC];
            float block_max = NEG_INF;
            for (unsigned int j = 0; j < tile_len; j++) {
                unsigned int gk = k_start + j;
                if (gk > global_q + kv_offset) { s_vals[j] = NEG_INF; continue; }
                float dot = 0.0f;
                for (unsigned int i = 0; i < d; i++) dot += q_row[i] * __half2float(K_shared[j][i]);
                s_vals[j] = dot * scale;
                block_max = fmaxf(block_max, s_vals[j]);
            }
            float new_max = fmaxf(row_max, block_max);
            float old_correction = expf(row_max - new_max);
            float new_sum = old_correction * row_sum;
            for (unsigned int i = 0; i < d; i++) o_acc[i] *= old_correction;
            for (unsigned int j = 0; j < tile_len; j++) {
                float p = expf(s_vals[j] - new_max);
                new_sum += p;
                for (unsigned int i = 0; i < d; i++) o_acc[i] += p * __half2float(V_shared[j][i]);
            }
            row_max = new_max;
            row_sum = new_sum;
        }
        __syncthreads();
    }

    if (active) {
        float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
        for (unsigned int i = 0; i < d; i++) O_bh[global_q * d + i] = o_acc[i] * inv_sum;
    }
}

extern "C" __global__ void flash_attn_precompute_d(
    const float* dO, const float* O, float* D, unsigned int total_rows, unsigned int head_dim
) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_rows) return;
    float sum = 0.0f;
    for (unsigned int i = 0; i < head_dim; i++) sum += dO[gid * head_dim + i] * O[gid * head_dim + i];
    D[gid] = sum;
}

struct FlashBwdParams { unsigned int seq_q; unsigned int seq_k; unsigned int head_dim; unsigned int batch_heads; float scale; unsigned int kv_offset; };

// Recomputes attention tile-by-tile; dQ thread-local, dK/dV atomic scatter (caller pre-zeros them).
extern "C" __global__ void flash_attention_backward(
    const float* Q, const float* K, const float* V, const float* O, const float* dO, const float* D,
    float* dQ, float* dK, float* dV, FlashBwdParams p
) {
    unsigned int bh = blockIdx.x;
    unsigned int q_block = blockIdx.y;
    unsigned int q_start = q_block * 32;
    unsigned int local_q = threadIdx.x;
    unsigned int global_q = q_start + local_q;
    if (bh >= p.batch_heads) return;
    unsigned int seq_q = p.seq_q, seq_k = p.seq_k, d = p.head_dim, kv_offset = p.kv_offset;
    float scale = p.scale;
    bool active = (global_q < seq_q);

    const float* Q_bh = Q + bh * seq_q * d;
    const float* K_bh = K + bh * seq_k * d;
    const float* V_bh = V + bh * seq_k * d;
    const float* dO_bh = dO + bh * seq_q * d;
    const float* D_bh = D + bh * seq_q;
    float* dQ_bh = dQ + bh * seq_q * d;
    float* dK_bh = dK + bh * seq_k * d;
    float* dV_bh = dV + bh * seq_k * d;

    __shared__ __half K_shared[32][128];
    __shared__ __half V_shared[32][128];

    float q_row[128], do_row[128];
    float d_val = 0.0f;
    if (active) {
        for (unsigned int i = 0; i < d; i++) { q_row[i] = Q_bh[global_q * d + i]; do_row[i] = dO_bh[global_q * d + i]; }
        d_val = D_bh[global_q];
    }
    float dq_acc[128];
    for (unsigned int i = 0; i < d; i++) dq_acc[i] = 0.0f;

    // Pass 1: recompute row_max / row_sum
    float row_max = NEG_INF, row_sum = 0.0f;
    for (unsigned int k_start = 0; k_start < seq_k; k_start += 32) {
        unsigned int k_end = k_start + 32; if (k_end > seq_k) k_end = seq_k;
        unsigned int tile_len = k_end - k_start;
        for (unsigned int j = threadIdx.x; j < tile_len; j += 32)
            for (unsigned int i = 0; i < d; i++) K_shared[j][i] = __float2half(K_bh[(k_start + j) * d + i]);
        __syncthreads();
        if (active) {
            for (unsigned int j = 0; j < tile_len; j++) {
                unsigned int gk = k_start + j;
                if (gk > global_q + kv_offset) continue;
                float dot = 0.0f;
                for (unsigned int i = 0; i < d; i++) dot += q_row[i] * __half2float(K_shared[j][i]);
                float s = dot * scale;
                float new_max = fmaxf(row_max, s);
                row_sum = row_sum * expf(row_max - new_max) + expf(s - new_max);
                row_max = new_max;
            }
        }
        __syncthreads();
    }
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    // Pass 2: gradients
    for (unsigned int k_start = 0; k_start < seq_k; k_start += 32) {
        unsigned int k_end = k_start + 32; if (k_end > seq_k) k_end = seq_k;
        unsigned int tile_len = k_end - k_start;
        for (unsigned int j = threadIdx.x; j < tile_len; j += 32)
            for (unsigned int i = 0; i < d; i++) {
                K_shared[j][i] = __float2half(K_bh[(k_start + j) * d + i]);
                V_shared[j][i] = __float2half(V_bh[(k_start + j) * d + i]);
            }
        __syncthreads();
        if (active) {
            for (unsigned int j = 0; j < tile_len; j++) {
                unsigned int gk = k_start + j;
                if (gk > global_q + kv_offset) continue;
                float dot = 0.0f;
                for (unsigned int i = 0; i < d; i++) dot += q_row[i] * __half2float(K_shared[j][i]);
                float s = dot * scale;
                float pr = expf(s - row_max) * inv_sum;
                float dov = 0.0f;
                for (unsigned int i = 0; i < d; i++) dov += do_row[i] * __half2float(V_shared[j][i]);
                float ds = pr * (dov - d_val) * scale;
                for (unsigned int i = 0; i < d; i++) dq_acc[i] += ds * __half2float(K_shared[j][i]);
                for (unsigned int i = 0; i < d; i++) {
                    atomicAdd(&dK_bh[gk * d + i], ds * q_row[i]);
                    atomicAdd(&dV_bh[gk * d + i], pr * do_row[i]);
                }
            }
        }
        __syncthreads();
    }
    if (active) for (unsigned int i = 0; i < d; i++) dQ_bh[global_q * d + i] = dq_acc[i];
}

// ===== Optimizers: Lion (sign-momentum) + Sophia (clipped diagonal-Hessian) =====
extern "C" __global__ void cautious_mask(float* u, const float* g, float* keep, unsigned int size) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    float k = (u[gid] * g[gid] > 0.0f) ? 1.0f : 0.0f;
    keep[gid] = k;
    u[gid] *= k;
}

extern "C" __global__ void cautious_scale(float* x, const float* kept_sum, unsigned int size) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    float scale = (float)size / (kept_sum[0] + 1.0f);
    x[gid] *= scale;
}

extern "C" __global__ void lion_update(float* param, const float* grad, float* m,
    unsigned int size, float lr, float beta1, float beta2, float weight_decay) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    float g = grad[gid], m_val = m[gid];
    float update = m_val * beta1 + g * (1.0f - beta1);
    float s = (update > 0.0f) ? 1.0f : ((update < 0.0f) ? -1.0f : 0.0f);
    param[gid] = param[gid] * (1.0f - lr * weight_decay) - lr * s;
    m[gid] = m_val * beta2 + g * (1.0f - beta2);
}
extern "C" __global__ void sophia_update(float* param, const float* grad, float* m, float* h,
    unsigned int size, float lr, float beta1, float beta2, float eps, float rho, float weight_decay) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= size) return;
    float g = grad[gid];
    float m_val = beta1 * m[gid] + (1.0f - beta1) * g; m[gid] = m_val;
    float h_val = beta2 * h[gid] + (1.0f - beta2) * g * g; h[gid] = h_val;
    float update = m_val / fmaxf(h_val, eps);
    update = fmaxf(-rho, fminf(rho, update));
    param[gid] = param[gid] * (1.0f - lr * weight_decay) - lr * update;
}

// ===== Sampling / generation utilities =====
// argmax over a flat array → index, single-block reduction (grid-stride for size > blockDim).
extern "C" __global__ void argmax(const float* data, unsigned int* result, unsigned int size) {
    __shared__ float sv[256];
    __shared__ unsigned int si[256];
    unsigned int tid = threadIdx.x, tpg = blockDim.x;
    float local_max = __int_as_float(0xff800000); unsigned int local_idx = 0;
    for (unsigned int i = tid; i < size; i += tpg)
        if (data[i] > local_max) { local_max = data[i]; local_idx = i; }
    sv[tid] = local_max; si[tid] = local_idx;
    __syncthreads();
    for (unsigned int s = tpg / 2; s > 0; s >>= 1) {
        if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) result[0] = si[0];
}
extern "C" __global__ void temperature_scale(float* data, unsigned int offset, unsigned int count, float inv_temperature) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= count) return;
    data[offset + gid] *= inv_temperature;
}

// ===== Training utilities: zero rows by token index, mask gradients by position =====
extern "C" __global__ void zero_rows(const unsigned int* tokens, float* matrix, unsigned int n_tokens, unsigned int dim) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int token_idx = gid / dim, dim_idx = gid % dim;
    if (token_idx >= n_tokens) return;
    matrix[tokens[token_idx] * dim + dim_idx] = 0.0f;
}
extern "C" __global__ void gradient_mask(float* grad, const unsigned int* mask, unsigned int total, unsigned int vocab_size) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total) return;
    if (mask[gid / vocab_size] == 0u) grad[gid] = 0.0f;
}

// ===== Per-row reductions: logsumexp, inverse-RMS (one block per row, blockDim = next_pow2(cols)) =====
extern "C" __global__ void logsumexp(const float* input, float* output, unsigned int rows, unsigned int cols) {
    __shared__ float sh[256];
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float* row_in = input + row * cols;
    unsigned int tid = threadIdx.x, tpg = blockDim.x;
    float local_max = __int_as_float(0xff800000);
    for (unsigned int c = tid; c < cols; c += tpg) local_max = fmaxf(local_max, row_in[c]);
    sh[tid] = local_max; __syncthreads();
    for (unsigned int s = tpg / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] = fmaxf(sh[tid], sh[tid + s]); __syncthreads(); }
    float row_max = sh[0]; __syncthreads();
    float local_sum = 0.0f;
    for (unsigned int c = tid; c < cols; c += tpg) local_sum += expf(row_in[c] - row_max);
    sh[tid] = local_sum; __syncthreads();
    for (unsigned int s = tpg / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    if (tid == 0) output[row] = row_max + logf(sh[0]);
}
extern "C" __global__ void compute_inv_rms(const float* input, float* inv_rms, unsigned int rows, unsigned int cols, float eps) {
    __shared__ float sh[256];
    unsigned int row = blockIdx.x;
    if (row >= rows) return;
    const float* row_in = input + row * cols;
    unsigned int tid = threadIdx.x, tpg = blockDim.x;
    float local = 0.0f;
    for (unsigned int c = tid; c < cols; c += tpg) { float v = row_in[c]; local += v * v; }
    sh[tid] = local; __syncthreads();
    for (unsigned int s = tpg / 2; s > 0; s >>= 1) { if (tid < s) sh[tid] += sh[tid + s]; __syncthreads(); }
    if (tid == 0) inv_rms[row] = rsqrtf(sh[0] / (float)cols + eps);
}

// ===== Sliding-window causal mask: a query attends to exactly `window` keys [q_pos-window+1, q_pos];
// -inf for future (k>q_pos) OR too-far-back (k+window<=q_pos). Additive form is underflow-safe. =====
extern "C" __global__ void causal_mask_window(float* scores, unsigned int batch_heads, unsigned int seq_q,
    unsigned int seq_k, unsigned int offset, unsigned int window) {
    unsigned int bh = blockIdx.x, q = blockIdx.y, k = blockIdx.z * blockDim.x + threadIdx.x;
    if (bh >= batch_heads || q >= seq_q || k >= seq_k) return;
    unsigned int q_pos = q + offset;
    bool future = k > q_pos;
    bool too_far = (window > 0) && (k + window <= q_pos);
    if (future || too_far) scores[bh * seq_q * seq_k + q * seq_k + k] = __int_as_float(0xff800000);
}
"#;
