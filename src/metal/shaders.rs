/// Metal Shader Language (MSL) kernel sources for AndreAI.
/// All GPU compute happens through these shaders.
/// Optimized for Apple M1: 8-core GPU, 32KB threadgroup memory per core, 128 ALUs/core.
/// Tiled matrix multiplication: C = A @ B
/// A: [M, K], B: [K, N], C: [M, N]
/// Tile size: 32x32 output, K-blocking in chunks of 32.
/// Each threadgroup computes one 32x32 tile of C.
/// Each thread computes a 4x4 sub-tile (8x8 = 64 threads per group).
pub const MATMUL_TILED: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
    uint M;
    uint N;
    uint K;
};

#define TILE 32
#define THREAD_TILE 4
#define THREADS_PER_GROUP 64

kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    // Each thread has a position in the 8x8 grid within the threadgroup
    uint local_row = thread_index / 8;  // 0..7
    uint local_col = thread_index % 8;  // 0..7

    // Global starting position for this threadgroup's tile
    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;

    // Shared memory for A and B tiles
    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    // Accumulator for this thread's 4x4 sub-tile
    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    uint M = params.M;
    uint N = params.N;
    uint K = params.K;

    // Loop over K dimension in TILE-sized chunks
    for (uint k_block = 0; k_block < K; k_block += TILE) {
        // Cooperatively load A tile into shared memory
        // 64 threads need to load 32x32 = 1024 elements = 16 per thread
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE;
            uint c = flat % TILE;
            uint global_r = tile_row + r;
            uint global_c = k_block + c;
            As[r][c] = (global_r < M && global_c < K) ? A[global_r * K + global_c] : 0.0f;
        }

        // Cooperatively load B tile into shared memory
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE;
            uint c = flat % TILE;
            uint global_r = k_block + r;
            uint global_c = tile_col + c;
            Bs[r][c] = (global_r < K && global_c < N) ? B[global_r * N + global_c] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each thread computes its 4x4 sub-tile
        for (uint k = 0; k < TILE; k++) {
            float a_vals[THREAD_TILE];
            float b_vals[THREAD_TILE];

            for (uint i = 0; i < THREAD_TILE; i++) {
                a_vals[i] = As[local_row * THREAD_TILE + i][k];
            }
            for (uint j = 0; j < THREAD_TILE; j++) {
                b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            }

            for (uint i = 0; i < THREAD_TILE; i++) {
                for (uint j = 0; j < THREAD_TILE; j++) {
                    acc[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results to global memory
    for (uint i = 0; i < THREAD_TILE; i++) {
        for (uint j = 0; j < THREAD_TILE; j++) {
            uint global_r = tile_row + local_row * THREAD_TILE + i;
            uint global_c = tile_col + local_col * THREAD_TILE + j;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = acc[i][j];
            }
        }
    }
}
"#;

/// Matrix multiply with B transposed: C = A @ B^T
/// A: [M, K], B: [N, K] (stored row-major, but we treat it as transposed), C: [M, N]
/// Used for attention: scores = Q @ K^T
pub const MATMUL_TILED_TRANS_B: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
    uint M;
    uint N;
    uint K;
};

#define TILE 32
#define THREAD_TILE 4
#define THREADS_PER_GROUP 64

kernel void matmul_tiled_trans_b(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;

    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;

    threadgroup float As[TILE][TILE];
    threadgroup float Bs[TILE][TILE];

    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    uint M = params.M;
    uint N = params.N;
    uint K = params.K;

    for (uint k_block = 0; k_block < K; k_block += TILE) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE;
            uint c = flat % TILE;
            uint global_r = tile_row + r;
            uint global_c = k_block + c;
            As[r][c] = (global_r < M && global_c < K) ? A[global_r * K + global_c] : 0.0f;
        }

        // B is [N, K] and we want B^T, so B^T[k, n] = B[n, k]
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE;  // k index
            uint c = flat % TILE;  // n index
            uint global_k = k_block + r;
            uint global_n = tile_col + c;
            // B^T[k, n] = B[n, k]
            Bs[r][c] = (global_k < K && global_n < N) ? B[global_n * K + global_k] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            float a_vals[THREAD_TILE];
            float b_vals[THREAD_TILE];

            for (uint i = 0; i < THREAD_TILE; i++) {
                a_vals[i] = As[local_row * THREAD_TILE + i][k];
            }
            for (uint j = 0; j < THREAD_TILE; j++) {
                b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            }

            for (uint i = 0; i < THREAD_TILE; i++) {
                for (uint j = 0; j < THREAD_TILE; j++) {
                    acc[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < THREAD_TILE; i++) {
        for (uint j = 0; j < THREAD_TILE; j++) {
            uint global_r = tile_row + local_row * THREAD_TILE + i;
            uint global_c = tile_col + local_col * THREAD_TILE + j;
            if (global_r < M && global_c < N) {
                C[global_r * N + global_c] = acc[i][j];
            }
        }
    }
}
"#;

/// Row-wise softmax with numerical stability.
/// Input/output: [rows, cols]. Each threadgroup handles one row.
/// Uses two-pass: first compute max, then exp(x - max) / sum.
pub const SOFTMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SoftmaxParams {
    uint rows;
    uint cols;
};

kernel void softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SoftmaxParams& params [[buffer(2)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;

    uint cols = params.cols;
    device const float* row_in = input + row * cols;
    device float* row_out = output + row * cols;

    // Pass 1: find max for numerical stability
    threadgroup float shared_max[256];
    float local_max = -INFINITY;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        local_max = max(local_max, row_in[c]);
    }
    shared_max[thread_index] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce max
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_max[thread_index] = max(shared_max[thread_index], shared_max[thread_index + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];

    // Pass 2: compute exp(x - max) and sum
    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        float val = exp(row_in[c] - row_max);
        row_out[c] = val;
        local_sum += val;
    }
    shared_sum[thread_index] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce sum
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_sum[thread_index] += shared_sum[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = shared_sum[0];

    // Normalize
    float inv_sum = 1.0f / total;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        row_out[c] *= inv_sum;
    }
}
"#;

/// RMS Layer Normalization: output = (x / rms(x)) * weight
/// where rms(x) = sqrt(mean(x^2) + eps)
/// Input: [rows, cols], weight: [cols], output: [rows, cols]
pub const RMS_NORM: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct NormParams {
    uint rows;
    uint cols;
    float eps;
};

kernel void rms_norm(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant NormParams& params [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;

    uint cols = params.cols;
    device const float* row_in = input + row * cols;
    device float* row_out = output + row * cols;

    // Compute sum of squares
    threadgroup float shared_ss[256];
    float local_ss = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        float val = row_in[c];
        local_ss += val * val;
    }
    shared_ss[thread_index] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_ss[thread_index] += shared_ss[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_ss[0] / float(cols) + params.eps);

    // Normalize and scale
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        row_out[c] = row_in[c] * rms * weight[c];
    }
}
"#;

/// Rotary Positional Embedding (RoPE) applied in-place.
/// Input: [batch * n_heads, seq_len, head_dim]
/// Applies rotation to pairs of dimensions based on position.
pub const ROPE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RopeParams {
    uint seq_len;
    uint head_dim;
    uint total_rows; // batch * n_heads
    uint offset;     // for KV cache: start position
    float theta;     // base frequency (default 10000.0)
};

kernel void rope(
    device float* data [[buffer(0)]],
    constant RopeParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;       // batch * n_heads index
    uint pos = gid.x;       // sequence position
    uint pair = gid.z;      // which dimension pair (0..head_dim/2)

    if (row >= params.total_rows || pos >= params.seq_len || pair >= params.head_dim / 2) return;

    float freq = 1.0f / pow(params.theta, float(2 * pair) / float(params.head_dim));
    float angle = float(pos + params.offset) * freq;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

    uint base = row * params.seq_len * params.head_dim + pos * params.head_dim;
    uint i0 = base + pair * 2;
    uint i1 = base + pair * 2 + 1;

    float x0 = data[i0];
    float x1 = data[i1];

    data[i0] = x0 * cos_val - x1 * sin_val;
    data[i1] = x0 * sin_val + x1 * cos_val;
}
"#;

/// Elementwise addition: C = A + B (broadcast-compatible)
pub const ADD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct AddParams {
    uint size;
};

kernel void add(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant AddParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < params.size) {
        c[gid] = a[gid] + b[gid];
    }
}
"#;

/// Elementwise multiply: C = A * B
pub const MUL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MulParams {
    uint size;
};

kernel void mul(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* c [[buffer(2)]],
    constant MulParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < params.size) {
        c[gid] = a[gid] * b[gid];
    }
}
"#;

/// SiLU (Swish) activation: output = x * sigmoid(x)
pub const SILU: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SiluParams {
    uint size;
};

kernel void silu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant SiluParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < params.size) {
        float x = input[gid];
        output[gid] = x / (1.0f + exp(-x));
    }
}
"#;

/// Fused SiLU-gate: output[i] = silu(gate[i]) * up[i]
/// Saves one kernel dispatch and one temporary buffer vs separate silu + mul.
pub const SILU_GATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SiluGateParams {
    uint size;
};

kernel void silu_gate(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant SiluGateParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < params.size) {
        float x = gate[gid];
        float silu_x = x / (1.0f + exp(-x));
        output[gid] = silu_x * up[gid];
    }
}
"#;

/// Fused cross-entropy loss: log-softmax + NLL
/// logits: [batch, vocab], targets: [batch] (as uint)
/// Output: scalar loss (single float), plus grad_output: [batch, vocab] = softmax - one_hot
pub const CROSS_ENTROPY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct CEParams {
    uint batch_size;
    uint vocab_size;
};

// Per-row: compute log-sum-exp, then loss = -logit[target] + logsumexp
// Also compute gradient: softmax(logits) - one_hot(target)
kernel void cross_entropy(
    device const float* logits [[buffer(0)]],
    device const uint* targets [[buffer(1)]],
    device float* losses [[buffer(2)]],
    device float* grad_logits [[buffer(3)]],
    constant CEParams& params [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.batch_size) return;

    uint V = params.vocab_size;
    device const float* row_logits = logits + row * V;
    device float* row_grad = grad_logits + row * V;
    uint target = targets[row];

    // Find max
    threadgroup float shared_max[256];
    float local_max = -INFINITY;
    for (uint c = thread_index; c < V; c += threads_per_group) {
        local_max = max(local_max, row_logits[c]);
    }
    shared_max[thread_index] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_max[thread_index] = max(shared_max[thread_index], shared_max[thread_index + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared_max[0];

    // Compute exp and sum
    threadgroup float shared_sum[256];
    float local_sum = 0.0f;
    for (uint c = thread_index; c < V; c += threads_per_group) {
        float e = exp(row_logits[c] - row_max);
        row_grad[c] = e;  // temporarily store exp values
        local_sum += e;
    }
    shared_sum[thread_index] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_sum[thread_index] += shared_sum[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = shared_sum[0];
    float inv_sum = 1.0f / total;

    // Compute softmax gradient = softmax - one_hot
    float inv_batch = 1.0f / float(params.batch_size);
    for (uint c = thread_index; c < V; c += threads_per_group) {
        float softmax_val = row_grad[c] * inv_sum;
        float one_hot = (c == target) ? 1.0f : 0.0f;
        row_grad[c] = (softmax_val - one_hot) * inv_batch;
    }

    // Compute loss for this row: -log(softmax[target]) = -(logit[target] - max - log(sum))
    if (thread_index == 0) {
        losses[row] = -(row_logits[target] - row_max - log(total));
    }
}
"#;

/// Reduce sum: compute sum of array elements into a single float
pub const REDUCE_SUM: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ReduceParams {
    uint size;
};

kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceParams& params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float shared[256];
    float local_sum = 0.0f;
    for (uint i = thread_index; i < params.size; i += threads_per_group) {
        local_sum += input[i];
    }
    shared[thread_index] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared[thread_index] += shared[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        output[0] = shared[0];
    }
}
"#;

/// Fused AdamW optimizer update — one kernel launch per parameter tensor.
pub const ADAMW_UPDATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct AdamWParams {
    uint size;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_correction1;  // 1 - beta1^t
    float bias_correction2;  // 1 - beta2^t
};

kernel void adamw_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],
    device float* v [[buffer(3)]],
    constant AdamWParams& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;

    float g = grad[gid];
    float m_val = params.beta1 * m[gid] + (1.0f - params.beta1) * g;
    float v_val = params.beta2 * v[gid] + (1.0f - params.beta2) * g * g;

    m[gid] = m_val;
    v[gid] = v_val;

    float m_hat = m_val / params.bias_correction1;
    float v_hat = v_val / params.bias_correction2;

    // Weight decay applied to param directly (decoupled), then Adam step
    param[gid] = param[gid] * (1.0f - params.lr * params.weight_decay)
                 - params.lr * m_hat / (sqrt(v_hat) + params.eps);
}
"#;

/// Embedding lookup: gather rows from embedding matrix
/// tokens: [batch * seq_len] as uint, embeddings: [vocab, dim], output: [batch * seq_len, dim]
pub const EMBEDDING_LOOKUP: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct EmbedParams {
    uint n_tokens;
    uint dim;
};

kernel void embedding_lookup(
    device const uint* tokens [[buffer(0)]],
    device const float* embeddings [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant EmbedParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint dim_idx = gid.x;

    if (token_idx >= params.n_tokens || dim_idx >= params.dim) return;

    uint token_id = tokens[token_idx];
    output[token_idx * params.dim + dim_idx] = embeddings[token_id * params.dim + dim_idx];
}
"#;

/// Causal mask fill: set positions where col > row + offset to -infinity
/// scores: [batch_heads, seq_q, seq_k]
pub const CAUSAL_MASK: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MaskParams {
    uint batch_heads;
    uint seq_q;
    uint seq_k;
    uint offset; // for KV cache: offset into key sequence
};

kernel void causal_mask(
    device float* scores [[buffer(0)]],
    constant MaskParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint bh = gid.z;
    uint q = gid.y;
    uint k = gid.x;

    if (bh >= params.batch_heads || q >= params.seq_q || k >= params.seq_k) return;

    // q_pos = offset + q (for KV cache, queries start at offset)
    // k_pos = k
    // Mask if k_pos > q_pos (future positions)
    if (k > q + params.offset) {
        scores[bh * params.seq_q * params.seq_k + q * params.seq_k + k] = -INFINITY;
    }
}
"#;

/// Gradient clipping: compute L2 norm of a flat buffer
pub const L2_NORM: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct NormCalcParams {
    uint size;
};

kernel void l2_norm(
    device const float* data [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant NormCalcParams& params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float shared[256];
    float local_sum = 0.0f;
    for (uint i = thread_index; i < params.size; i += threads_per_group) {
        float val = data[i];
        local_sum += val * val;
    }
    shared[thread_index] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared[thread_index] += shared[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        output[0] = sqrt(shared[0]);
    }
}
"#;

/// Scale buffer in-place: data[i] *= scale
pub const SCALE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ScaleParams {
    uint size;
    float scale;
};

kernel void scale(
    device float* data [[buffer(0)]],
    constant ScaleParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < params.size) {
        data[gid] *= params.scale;
    }
}
"#;

/// Fill buffer with a constant value
pub const FILL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct FillParams {
    uint size;
    float value;
};

kernel void fill(
    device float* data [[buffer(0)]],
    constant FillParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < params.size) {
        data[gid] = params.value;
    }
}
"#;

/// Copy buffer: dst = src
pub const COPY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct CopyParams {
    uint size;
};

kernel void copy_buffer(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant CopyParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < params.size) {
        dst[gid] = src[gid];
    }
}
"#;

/// SiLU backward: grad_input = grad_output * (sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x)))
///              = grad_output * (sigmoid(x) * (1 + x * (1 - sigmoid(x))))
pub const SILU_BACKWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SiluBwdParams {
    uint size;
};

kernel void silu_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant SiluBwdParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;

    float x = input[gid];
    float sig = 1.0f / (1.0f + exp(-x));
    float grad_out = grad_output[gid];
    grad_input[gid] = grad_out * sig * (1.0f + x * (1.0f - sig));
}
"#;

/// Fused SiLU-gate backward:
/// d_gate = d_out * up * silu'(gate)  where silu'(x) = sigmoid(x)*(1+x*(1-sigmoid(x)))
/// d_up = d_out * silu(gate)
pub const SILU_GATE_BACKWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SiluGateBwdParams {
    uint size;
};

kernel void silu_gate_backward(
    device const float* gate [[buffer(0)]],
    device const float* up [[buffer(1)]],
    device const float* grad_output [[buffer(2)]],
    device float* grad_gate [[buffer(3)]],
    device float* grad_up [[buffer(4)]],
    constant SiluGateBwdParams& params [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;

    float x = gate[gid];
    float sig = 1.0f / (1.0f + exp(-x));
    float silu_x = x * sig;
    float silu_prime = sig * (1.0f + x * (1.0f - sig));
    float d_out = grad_output[gid];

    grad_gate[gid] = d_out * up[gid] * silu_prime;
    grad_up[gid] = d_out * silu_x;
}
"#;

/// RMS norm backward
/// Given: input x, weight w, output = (x / rms) * w where rms = sqrt(mean(x^2) + eps)
/// Need: grad_input, grad_weight given grad_output
pub const RMS_NORM_BACKWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct NormBwdParams {
    uint rows;
    uint cols;
    float eps;
};

kernel void rms_norm_backward(
    device const float* input [[buffer(0)]],
    device const float* weight [[buffer(1)]],
    device const float* grad_output [[buffer(2)]],
    device float* grad_input [[buffer(3)]],
    device float* grad_weight [[buffer(4)]],
    constant NormBwdParams& params [[buffer(5)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;

    uint cols = params.cols;
    device const float* x = input + row * cols;
    device const float* go = grad_output + row * cols;
    device float* gi = grad_input + row * cols;

    // Compute rms
    threadgroup float shared[256];
    float local_ss = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        local_ss += x[c] * x[c];
    }
    shared[thread_index] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) shared[thread_index] += shared[thread_index + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean_sq = shared[0] / float(cols);
    float rms = sqrt(mean_sq + params.eps);
    float inv_rms = 1.0f / rms;

    // Compute sum(grad_output * weight * input) for the correction term
    float local_dot = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        local_dot += go[c] * weight[c] * x[c];
    }
    shared[thread_index] = local_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) shared[thread_index] += shared[thread_index + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot_sum = shared[0];

    // grad_input = (grad_output * weight * inv_rms) - (input * dot_sum * inv_rms^3 / cols)
    float correction = dot_sum * inv_rms * inv_rms * inv_rms / float(cols);
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        gi[c] = go[c] * weight[c] * inv_rms - x[c] * correction;
    }

    // grad_weight: atomic accumulate across rows (only for first pass)
    // We accumulate grad_weight = sum_over_rows(grad_output * x * inv_rms)
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        // Use atomic add since multiple rows write to the same weight gradient
        float gw = go[c] * x[c] * inv_rms;
        // Atomic float add — supported on Apple GPU family 2+
        atomic_fetch_add_explicit((device atomic_float*)&grad_weight[c], gw, memory_order_relaxed);
    }
}
"#;

/// Softmax backward
/// Given cached softmax output S and grad_output dS:
/// grad_input = S * (dS - sum(dS * S))
pub const SOFTMAX_BACKWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SoftmaxBwdParams {
    uint rows;
    uint cols;
};

kernel void softmax_backward(
    device const float* softmax_out [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant SoftmaxBwdParams& params [[buffer(3)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;

    uint cols = params.cols;
    device const float* s = softmax_out + row * cols;
    device const float* go = grad_output + row * cols;
    device float* gi = grad_input + row * cols;

    // Compute dot = sum(grad_output * softmax_out) for this row
    threadgroup float shared[256];
    float local_dot = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        local_dot += go[c] * s[c];
    }
    shared[thread_index] = local_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) shared[thread_index] += shared[thread_index + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot_sum = shared[0];

    // grad_input = softmax_out * (grad_output - dot_sum)
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        gi[c] = s[c] * (go[c] - dot_sum);
    }
}
"#;

/// Embedding backward: scatter-add gradients back to embedding matrix
pub const EMBEDDING_BACKWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct EmbedBwdParams {
    uint n_tokens;
    uint dim;
};

kernel void embedding_backward(
    device const uint* tokens [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_embeddings [[buffer(2)]],
    constant EmbedBwdParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint token_idx = gid.y;
    uint dim_idx = gid.x;

    if (token_idx >= params.n_tokens || dim_idx >= params.dim) return;

    uint token_id = tokens[token_idx];
    float grad_val = grad_output[token_idx * params.dim + dim_idx];
    atomic_fetch_add_explicit(
        (device atomic_float*)&grad_embeddings[token_id * params.dim + dim_idx],
        grad_val,
        memory_order_relaxed
    );
}
"#;

/// 2D matrix transpose: out[j, i] = in[i, j]
/// in: [rows, cols], out: [cols, rows]
pub const TRANSPOSE_2D: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TransposeParams {
    uint rows;
    uint cols;
};

kernel void transpose_2d(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant TransposeParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.rows * params.cols) return;
    uint r = gid / params.cols;
    uint c = gid % params.cols;
    output[c * params.rows + r] = input[r * params.cols + c];
}
"#;

/// C = A^T @ B where A:[M,K] stored row-major, B:[M,N], C:[K,N]
/// A^T is [K,M], so C[i,j] = sum_m A[m,i] * B[m,j]
pub const MATMUL_TRANS_A: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MatmulTransAParams {
    uint M;  // shared (inner after transpose)
    uint K;  // rows of output (cols of A)
    uint N;  // cols of output (cols of B)
};

kernel void matmul_trans_a(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulTransAParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // K dimension
    uint col = gid.x;  // N dimension
    if (row >= params.K || col >= params.N) return;

    float sum = 0.0;
    for (uint m = 0; m < params.M; m++) {
        sum += A[m * params.K + row] * B[m * params.N + col];
    }
    C[row * params.N + col] = sum;
}
"#;

/// Buffer-to-buffer copy with offset: dst[dst_offset..dst_offset+count] = src[src_offset..src_offset+count]
pub const BUFFER_COPY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct CopyParams {
    uint src_offset;  // in floats
    uint dst_offset;  // in floats
    uint count;       // number of floats to copy
};

kernel void buffer_copy(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant CopyParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    dst[params.dst_offset + gid] = src[params.src_offset + gid];
}
"#;

/// Attention transpose permutation for backward pass.
/// Forward mapped: flat[batch*seq, n_heads*head_dim] → out[batch*n_heads, seq, head_dim]
/// Backward: grad_in[batch*n_heads, seq, head_dim] → grad_out[batch*seq, n_heads*head_dim]
pub const TRANSPOSE_PERM_BACKWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct PermParams {
    uint batch;
    uint seq;
    uint n_heads;
    uint head_dim;
};

kernel void transpose_perm_backward(
    device const float* grad_in [[buffer(0)]],
    device float* grad_out [[buffer(1)]],
    constant PermParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.batch * params.seq * params.n_heads * params.head_dim;
    if (gid >= total) return;

    // Decompose gid into (batch, n_heads, seq, head_dim) indices — input layout
    uint head_dim = params.head_dim;
    uint seq = params.seq;
    uint n_heads = params.n_heads;

    uint rem = gid;
    uint b = rem / (n_heads * seq * head_dim);
    rem %= n_heads * seq * head_dim;
    uint h = rem / (seq * head_dim);
    rem %= seq * head_dim;
    uint s = rem / head_dim;
    uint d = rem % head_dim;

    // Output layout: [batch*seq, n_heads*head_dim]
    uint out_idx = (b * seq + s) * (n_heads * head_dim) + h * head_dim + d;
    grad_out[out_idx] = grad_in[gid];
}
"#;

/// Forward attention transpose: [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim]
pub const TRANSPOSE_PERM_FORWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct PermParams {
    uint batch;
    uint seq;
    uint n_heads;
    uint head_dim;
};

kernel void transpose_perm_forward(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant PermParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.batch * params.seq * params.n_heads * params.head_dim;
    if (gid >= total) return;

    uint head_dim = params.head_dim;
    uint seq = params.seq;
    uint n_heads = params.n_heads;

    // Decompose gid into output indices: (batch, n_heads, seq, head_dim)
    uint rem = gid;
    uint b = rem / (n_heads * seq * head_dim);
    rem %= n_heads * seq * head_dim;
    uint h = rem / (seq * head_dim);
    rem %= seq * head_dim;
    uint s = rem / head_dim;
    uint d = rem % head_dim;

    // Input layout: [batch*seq, n_heads*head_dim]
    uint in_idx = (b * seq + s) * (n_heads * head_dim) + h * head_dim + d;
    output[gid] = input[in_idx];
}
"#;
