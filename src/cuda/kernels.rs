//! CUDA kernel source code — equivalent to metal/shaders.rs.
//! Compiled to PTX at runtime via cudarc::nvrtc.

/// All kernel names that get loaded into the CUDA module.
pub const KERNEL_NAMES: &[&str] = &[
    "matmul_tiled", "matmul_tiled_trans_b", "matmul_trans_a_tiled",
    "batched_matmul_tiled", "batched_matmul_tiled_trans_b", "batched_matmul_tiled_trans_a",
    "matmul_tiled_f16", "matmul_tiled_trans_b_f16", "matmul_trans_a_tiled_f16",
    "softmax", "rms_norm", "rms_norm_residual",
    "rope", "rope_backward",
    "add_kernel", "add_inplace", "mul_kernel", "scale_kernel", "fill_kernel", "copy_kernel",
    "silu", "silu_gate",
    "cross_entropy", "reduce_sum", "kl_divergence",
    "adamw_update", "embedding_lookup",
    "cast_f32_to_f16", "cast_f16_to_f32",
    "transpose_perm_forward", "transpose_perm_backward", "transpose_2d", "causal_mask",
    "compact_strided_copy", "strided_batch_copy", "buffer_copy",
    "silu_backward", "silu_gate_backward", "rms_norm_backward", "softmax_backward", "embedding_backward",
    "l2_norm_check", "argmax", "temperature_scale", "gradient_mask", "zero_rows",
    "ema_update", "repeat_kv", "repeat_kv_backward", "causal_doc_mask",
    "relu", "relu_backward", "exp_kernel", "axpy", "scale_rows", "row_dot_reduce",
    "broadcast_rows", "slice_cols", "concat_cols",
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
    int d = threadIdx.x;
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
extern "C" __global__ void argmax(
    const float* input, unsigned int* output, unsigned int size
) {
    __shared__ float shared_vals[256];
    __shared__ unsigned int shared_idxs[256];
    int tid = threadIdx.x, nt = blockDim.x;
    float best_val = -1e30f;
    unsigned int best_idx = 0;
    for (int i = tid; i < size; i += nt) {
        if (input[i] > best_val) { best_val = input[i]; best_idx = i; }
    }
    shared_vals[tid] = best_val;
    shared_idxs[tid] = best_idx;
    __syncthreads();
    for (int s=nt/2;s>0;s>>=1) {
        if (tid<s && shared_vals[tid+s]>shared_vals[tid]) {
            shared_vals[tid]=shared_vals[tid+s]; shared_idxs[tid]=shared_idxs[tid+s];
        }
        __syncthreads();
    }
    if (tid==0) output[0] = shared_idxs[0];
}

extern "C" __global__ void temperature_scale(
    float* logits, unsigned int offset, unsigned int vocab, float temperature
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vocab) logits[offset + i] /= temperature;
}

extern "C" __global__ void gradient_mask(
    float* grad, const unsigned int* mask, unsigned int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size && mask[i] == 0) grad[i] = 0.0f;
}

extern "C" __global__ void zero_rows(
    float* data, const unsigned int* row_indices,
    unsigned int n_rows, unsigned int cols
) {
    int idx = blockIdx.x;
    int col = threadIdx.x;
    if (idx >= n_rows || col >= cols) return;
    data[row_indices[idx] * cols + col] = 0.0f;
}

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
    unsigned int k = threadIdx.x;
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
"#;
