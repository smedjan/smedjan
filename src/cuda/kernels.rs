//! CUDA kernel source code — equivalent to metal/shaders.rs.
//! Compiled to PTX at runtime via cudarc::nvrtc.

/// All kernel names that get loaded into the CUDA module.
pub const KERNEL_NAMES: &[&str] = &[
    "matmul_tiled",
    "matmul_tiled_trans_b",
    "matmul_trans_a_tiled",
    "softmax",
    "rms_norm",
    "rope",
    "rope_backward",
    "add_kernel",
    "mul_kernel",
    "silu_gate",
    "cross_entropy",
    "reduce_sum",
    "adamw_update",
    "embedding_lookup",
    "scale_kernel",
    "fill_kernel",
    "cast_f32_to_f16",
    "cast_f16_to_f32",
    "add_inplace",
    "copy_kernel",
    "silu_backward",
    "silu_gate_backward",
    "rms_norm_backward",
    "softmax_backward",
    "embedding_backward",
    "causal_mask",
    "l2_norm_check",
    "buffer_copy",
];

/// All CUDA kernels in a single compilation unit.
pub const ALL_KERNELS: &str = r#"
#include <cuda_fp16.h>

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
    float eps, float weight_decay, int step
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

    param[i] = param[i] * (1.0f - lr * weight_decay) - lr * m_hat / (sqrtf(v_hat) + eps);
}

// ============================================================
// Embedding lookup
// ============================================================
extern "C" __global__ void embedding_lookup(
    const float* table, const unsigned int* tokens, float* output,
    unsigned int seq_len, unsigned int dim
) {
    int pos = blockIdx.x;
    int d = threadIdx.x;
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
    int k = threadIdx.x;
    if (bh >= batch_heads || q >= seq_q || k >= seq_k) return;
    if (k > q + offset)
        scores[bh * seq_q * seq_k + q * seq_k + k] = -1e9f;
}

// ============================================================
// FP16 cast kernels
// ============================================================
extern "C" __global__ void cast_f32_to_f16(const float* input, __half* output, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        output[i] = __float2half(fminf(fmaxf(input[i], -65504.0f), 65504.0f));
}

extern "C" __global__ void cast_f16_to_f32(const __half* input, float* output, unsigned int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size)
        output[i] = __half2float(input[i]);
}

// ============================================================
// L2 norm check (for gradient clipping)
// ============================================================
extern "C" __global__ void l2_norm_check(const float* data, float* output, unsigned int size) {
    __shared__ float shared_ss[256];
    __shared__ float shared_nan[256];
    int tid = threadIdx.x;
    float local_ss = 0.0f;
    float local_nan = 0.0f;
    for (int i = tid; i < size; i += blockDim.x) {
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
        output[0] = shared_ss[0];
        output[1] = shared_nan[0];
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
// Backward kernels (stubs — to be completed)
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
    unsigned int rows, unsigned int cols, float eps
) {
    // Simplified — full implementation needs shared memory reduction
    // This is a placeholder that handles the per-element gradient
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= rows || col >= cols) return;

    // Compute RMS for this row
    float ss = 0.0f;
    for (int c = 0; c < cols; c++) {
        float v = input[row * cols + c];
        ss += v * v;
    }
    float rms = sqrtf(ss / (float)cols + eps);
    float inv_rms = 1.0f / rms;

    float x = input[row * cols + col];
    float w = weight[col];
    float go = grad_out[row * cols + col];

    // grad_input (simplified — ignores cross-term)
    grad_input[row * cols + col] = go * w * inv_rms;

    // grad_weight (needs atomicAdd across rows)
    atomicAdd(&grad_weight[col], go * x * inv_rms);
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
    int d = threadIdx.x;
    if (pos >= seq_len || d >= dim) return;
    atomicAdd(&grad_table[tokens[pos] * dim + d], grad_out[pos * dim + d]);
}
"#;
