/// Metal Shader Language (MSL) kernel sources for AndreAI.
/// All GPU compute happens through these shaders.
/// Optimized for Apple M1: 8-core GPU, 32KB threadgroup memory per core, 128 ALUs/core.
/// Tiled matrix multiplication using simdgroup_matrix intrinsics (Metal 2.4+, M1+).
/// 128 threads = 4 simdgroups per threadgroup, 32x32 output tile.
/// Each simdgroup handles a 16x16 quadrant via 2x2 grid of 8x8 simdgroup_matrix ops.
/// K dimension blocked in chunks of 8 to match simdgroup_matrix width.
/// Bank conflict fix: +1 padding on all shared memory second dimension.
///
/// matmul_tiled: C = A @ B, A:[M,K], B:[K,N], C:[M,N]
pub const MATMUL_TILED: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct MatmulParams {
    uint M;
    uint N;
    uint K;
};

#define TILE 32
#define K_TILE 8
#define THREADS_PER_GROUP 128

kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;

    uint M = params.M;
    uint N = params.N;
    uint K = params.K;

    // +1 padding breaks power-of-2 stride to avoid bank conflicts
    threadgroup float As[TILE][K_TILE + 1];
    threadgroup float Bs[K_TILE][TILE + 1];

    // 4 simdgroups in 2x2 grid, each covers 16x16 of output
    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 16;

    // 2x2 grid of 8x8 accumulators per simdgroup = 16x16
    simdgroup_matrix<float, 8, 8> acc00(0.0f);
    simdgroup_matrix<float, 8, 8> acc01(0.0f);
    simdgroup_matrix<float, 8, 8> acc10(0.0f);
    simdgroup_matrix<float, 8, 8> acc11(0.0f);

    for (uint k_block = 0; k_block < K; k_block += K_TILE) {
        // Load A tile [32 x 8]: 128 threads, 2 elements each
        {
            uint flat = thread_index;
            uint r0 = flat / K_TILE;
            uint c0 = flat % K_TILE;
            As[r0][c0] = (tile_row + r0 < M && k_block + c0 < K) ? A[(tile_row + r0) * K + k_block + c0] : 0.0f;
            uint flat2 = flat + THREADS_PER_GROUP;
            uint r1 = flat2 / K_TILE;
            uint c1 = flat2 % K_TILE;
            As[r1][c1] = (tile_row + r1 < M && k_block + c1 < K) ? A[(tile_row + r1) * K + k_block + c1] : 0.0f;
        }
        // Load B tile [8 x 32]: 128 threads, 2 elements each
        {
            uint flat = thread_index;
            uint r0 = flat / TILE;
            uint c0 = flat % TILE;
            Bs[r0][c0] = (k_block + r0 < K && tile_col + c0 < N) ? B[(k_block + r0) * N + tile_col + c0] : 0.0f;
            uint flat2 = flat + THREADS_PER_GROUP;
            uint r1 = flat2 / TILE;
            uint c1 = flat2 % TILE;
            Bs[r1][c1] = (k_block + r1 < K && tile_col + c1 < N) ? B[(k_block + r1) * N + tile_col + c1] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each simdgroup loads its 8x8 sub-tiles and multiply-accumulates
        simdgroup_matrix<float, 8, 8> a0, a1, b0, b1;
        simdgroup_load(a0, &As[sg_row][0], K_TILE + 1);
        simdgroup_load(a1, &As[sg_row + 8][0], K_TILE + 1);
        simdgroup_load(b0, &Bs[0][sg_col], TILE + 1);
        simdgroup_load(b1, &Bs[0][sg_col + 8], TILE + 1);

        simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
        simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
        simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
        simdgroup_multiply_accumulate(acc11, a1, b1, acc11);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store accumulators to shared memory, then cooperatively write to global
    threadgroup float Cs[TILE][TILE + 1];
    simdgroup_store(acc00, &Cs[sg_row][sg_col], TILE + 1);
    simdgroup_store(acc01, &Cs[sg_row][sg_col + 8], TILE + 1);
    simdgroup_store(acc10, &Cs[sg_row + 8][sg_col], TILE + 1);
    simdgroup_store(acc11, &Cs[sg_row + 8][sg_col + 8], TILE + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 8; i++) {
        uint flat = thread_index * 8 + i;
        uint r = flat / TILE;
        uint c = flat % TILE;
        if (tile_row + r < M && tile_col + c < N) {
            C[(tile_row + r) * N + tile_col + c] = Cs[r][c];
        }
    }
}
"#;

/// Matrix multiply with B transposed: C = A @ B^T
/// A: [M, K], B: [N, K] (stored row-major, but we treat it as transposed), C: [M, N]
/// Used for attention: scores = Q @ K^T
/// Simdgroup matrix intrinsics, 128 threads = 4 simdgroups, 32x32 output tile.
pub const MATMUL_TILED_TRANS_B: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct MatmulParams {
    uint M;
    uint N;
    uint K;
};

#define TILE 32
#define K_TILE 8
#define THREADS_PER_GROUP 128

kernel void matmul_tiled_trans_b(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;

    uint M = params.M;
    uint N = params.N;
    uint K = params.K;

    threadgroup float As[TILE][K_TILE + 1];
    threadgroup float Bs[K_TILE][TILE + 1];

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 16;

    simdgroup_matrix<float, 8, 8> acc00(0.0f), acc01(0.0f), acc10(0.0f), acc11(0.0f);

    for (uint k_block = 0; k_block < K; k_block += K_TILE) {
        // Load A tile [32 x 8]
        {
            uint flat = thread_index;
            uint r0 = flat / K_TILE, c0 = flat % K_TILE;
            As[r0][c0] = (tile_row + r0 < M && k_block + c0 < K) ? A[(tile_row + r0) * K + k_block + c0] : 0.0f;
            uint flat2 = flat + THREADS_PER_GROUP;
            uint r1 = flat2 / K_TILE, c1 = flat2 % K_TILE;
            As[r1][c1] = (tile_row + r1 < M && k_block + c1 < K) ? A[(tile_row + r1) * K + k_block + c1] : 0.0f;
        }
        // Load B^T tile [8 x 32]: B is [N,K], B^T[k,n] = B[n,k]
        {
            uint flat = thread_index;
            uint r0 = flat / TILE, c0 = flat % TILE;
            uint gk0 = k_block + r0, gn0 = tile_col + c0;
            Bs[r0][c0] = (gk0 < K && gn0 < N) ? B[gn0 * K + gk0] : 0.0f;
            uint flat2 = flat + THREADS_PER_GROUP;
            uint r1 = flat2 / TILE, c1 = flat2 % TILE;
            uint gk1 = k_block + r1, gn1 = tile_col + c1;
            Bs[r1][c1] = (gk1 < K && gn1 < N) ? B[gn1 * K + gk1] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> a0, a1, b0, b1;
        simdgroup_load(a0, &As[sg_row][0], K_TILE + 1);
        simdgroup_load(a1, &As[sg_row + 8][0], K_TILE + 1);
        simdgroup_load(b0, &Bs[0][sg_col], TILE + 1);
        simdgroup_load(b1, &Bs[0][sg_col + 8], TILE + 1);

        simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
        simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
        simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
        simdgroup_multiply_accumulate(acc11, a1, b1, acc11);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float Cs[TILE][TILE + 1];
    simdgroup_store(acc00, &Cs[sg_row][sg_col], TILE + 1);
    simdgroup_store(acc01, &Cs[sg_row][sg_col + 8], TILE + 1);
    simdgroup_store(acc10, &Cs[sg_row + 8][sg_col], TILE + 1);
    simdgroup_store(acc11, &Cs[sg_row + 8][sg_col + 8], TILE + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 8; i++) {
        uint flat = thread_index * 8 + i;
        uint r = flat / TILE, c = flat % TILE;
        if (tile_row + r < M && tile_col + c < N) {
            C[(tile_row + r) * N + tile_col + c] = Cs[r][c];
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

/// RoPE backward pass: inverse rotation (negate sin to undo forward rotation).
/// Given grad_output with RoPE applied, produces grad_input by rotating by -θ.
pub const ROPE_BACKWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RopeParams {
    uint seq_len;
    uint head_dim;
    uint total_rows; // batch * n_heads
    uint offset;     // for KV cache: start position
    float theta;     // base frequency (default 10000.0)
};

kernel void rope_backward(
    device float* data [[buffer(0)]],
    constant RopeParams& params [[buffer(1)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;       // batch * n_heads index
    uint pos = gid.x;       // sequence position
    uint pair = gid.z;      // which dimension pair (0..head_dim/2)

    if (row >= params.total_rows || pos >= params.seq_len || pair >= params.head_dim / 2) return;

    float freq = 1.0 / pow(params.theta, float(2 * pair) / float(params.head_dim));
    float angle = float(pos + params.offset) * freq;
    float cos_val = cos(angle);
    float sin_val = sin(angle);

    uint base = row * params.seq_len * params.head_dim + pos * params.head_dim;
    uint i0 = base + 2 * pair;
    uint i1 = i0 + 1;

    float x0 = data[i0];
    float x1 = data[i1];

    // Inverse rotation: rotate by -θ (negate sin)
    data[i0] = x0 * cos_val + x1 * sin_val;
    data[i1] = -x0 * sin_val + x1 * cos_val;
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

/// In-place elementwise add: a += b
pub const ADD_INPLACE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct AddInplaceParams {
    uint size;
};

kernel void add_inplace(
    device float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    constant AddInplaceParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < params.size) {
        a[gid] += b[gid];
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

/// Fused residual add + RMS normalization: output = rms_norm(input + residual)
/// Saves one kernel dispatch + one temporary buffer vs separate add + rms_norm.
pub const RMS_NORM_RESIDUAL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct NormResParams {
    uint rows;
    uint cols;
    float eps;
};

kernel void rms_norm_residual(
    device const float* input [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* weight [[buffer(2)]],
    device float* output [[buffer(3)]],
    device float* sum_out [[buffer(4)]],
    constant NormResParams& params [[buffer(5)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;

    uint cols = params.cols;
    device const float* row_in = input + row * cols;
    device const float* row_res = residual + row * cols;
    device float* row_out = output + row * cols;
    device float* row_sum = sum_out + row * cols;

    // Phase 1: compute input + residual and sum of squares
    threadgroup float shared_ss[256];
    float local_ss = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        float v = row_in[c] + row_res[c];
        row_sum[c] = v;  // store the sum for backward pass
        local_ss += v * v;
    }
    shared_ss[thread_index] = local_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride /= 2) {
        if (thread_index < stride) {
            shared_ss[thread_index] += shared_ss[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float rms = rsqrt(shared_ss[0] / float(cols) + params.eps);

    // Phase 2: normalize and scale
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        row_out[c] = row_sum[c] * rms * weight[c];
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

/// Gradient clipping: compute L2 norm (sum of squares) and check for NaN/Inf
/// Output buffer: [0] = sum_of_squares, [1] = has_nan_or_inf (1.0 or 0.0)
/// Unlike L2_NORM which returns sqrt(sum_sq), this returns raw sum_sq for
/// accumulation across multiple parameter buffers before a single sqrt.
pub const L2_NORM_CHECK: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct NormCheckParams {
    uint size;
};

kernel void l2_norm_check(
    device const float* data [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant NormCheckParams& params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float shared_sum[256];
    threadgroup float shared_nan[256];

    float local_sum = 0.0f;
    float local_nan = 0.0f;
    for (uint i = thread_index; i < params.size; i += threads_per_group) {
        float val = data[i];
        if (isnan(val) || isinf(val)) {
            local_nan = 1.0f;
        } else {
            local_sum += val * val;
        }
    }
    shared_sum[thread_index] = local_sum;
    shared_nan[thread_index] = local_nan;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_sum[thread_index] += shared_sum[thread_index + stride];
            shared_nan[thread_index] = max(shared_nan[thread_index], shared_nan[thread_index + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        output[0] = shared_sum[0];
        output[1] = shared_nan[0];
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

/// Embedding backward: scatter-add gradients back to embedding matrix.
/// Uses threadgroup-local accumulation to reduce atomic contention: each thread
/// iterates over a chunk of tokens for one dim position, accumulating contributions
/// for runs of the same token_id locally before emitting a single atomic per unique
/// token_id per thread. For common tokens (space, newline, 'the'), this reduces
/// atomics from thousands-per-location to ~(n_tokens / threads_per_group) per location.
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
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    // Each threadgroup handles one dim position.
    // Threads within the group split n_tokens among themselves.
    uint dim_idx = group_id;
    if (dim_idx >= params.dim) return;

    uint n_tokens = params.n_tokens;
    uint dim = params.dim;

    // Each thread processes tokens: thread_index, thread_index + threads_per_group, ...
    // Accumulate locally for runs of the same token_id to reduce atomic pressure.
    uint prev_token_id = 0xFFFFFFFF; // sentinel
    float accum = 0.0f;

    for (uint t = thread_index; t < n_tokens; t += threads_per_group) {
        uint token_id = tokens[t];
        float grad_val = grad_output[t * dim + dim_idx];

        if (token_id == prev_token_id) {
            // Same token as previous iteration — accumulate locally
            accum += grad_val;
        } else {
            // Different token — flush previous accumulation
            if (prev_token_id != 0xFFFFFFFF) {
                atomic_fetch_add_explicit(
                    (device atomic_float*)&grad_embeddings[prev_token_id * dim + dim_idx],
                    accum,
                    memory_order_relaxed
                );
            }
            prev_token_id = token_id;
            accum = grad_val;
        }
    }

    // Flush final accumulation
    if (prev_token_id != 0xFFFFFFFF) {
        atomic_fetch_add_explicit(
            (device atomic_float*)&grad_embeddings[prev_token_id * dim + dim_idx],
            accum,
            memory_order_relaxed
        );
    }
}
"#;

/// Zero only the rows of a matrix that correspond to given token IDs.
/// Avoids zeroing the entire vocab_size × dim matrix when only a small
/// fraction of rows are touched during embedding backward.
pub const ZERO_ROWS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ZeroRowsParams {
    uint n_tokens;
    uint dim;
};

kernel void zero_rows(
    device const uint* tokens [[buffer(0)]],
    device float* matrix [[buffer(1)]],
    constant ZeroRowsParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint token_idx = gid / params.dim;
    uint dim_idx = gid % params.dim;
    if (token_idx >= params.n_tokens) return;
    uint row = tokens[token_idx];
    matrix[row * params.dim + dim_idx] = 0.0f;
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
/// Simdgroup matrix intrinsics, 128 threads = 4 simdgroups, 32x32 output tile.
/// Inner dimension M blocked in chunks of 8.
pub const MATMUL_TRANS_A: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct MatmulTransAParams {
    uint M;  // shared (inner after transpose)
    uint K;  // rows of output (cols of A)
    uint N;  // cols of output (cols of B)
};

#define TILE 32
#define M_TILE 8
#define THREADS_PER_GROUP 128

kernel void matmul_trans_a_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulTransAParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint tile_row = group_id.y * TILE;  // K dimension
    uint tile_col = group_id.x * TILE;  // N dimension

    uint M = params.M;
    uint K = params.K;
    uint N = params.N;

    threadgroup float As[TILE][M_TILE + 1];   // As[k][m]
    threadgroup float Bs[M_TILE][TILE + 1];   // Bs[m][n]

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 16;

    simdgroup_matrix<float, 8, 8> acc00(0.0f), acc01(0.0f), acc10(0.0f), acc11(0.0f);

    for (uint m_block = 0; m_block < M; m_block += M_TILE) {
        // Load A^T tile: As[k][m] = A[(m_block+m)*K + (tile_row+k)]
        {
            uint flat = thread_index;
            uint r0 = flat / M_TILE, c0 = flat % M_TILE;
            uint gk0 = tile_row + r0, gm0 = m_block + c0;
            As[r0][c0] = (gk0 < K && gm0 < M) ? A[gm0 * K + gk0] : 0.0f;
            uint flat2 = flat + THREADS_PER_GROUP;
            uint r1 = flat2 / M_TILE, c1 = flat2 % M_TILE;
            uint gk1 = tile_row + r1, gm1 = m_block + c1;
            As[r1][c1] = (gk1 < K && gm1 < M) ? A[gm1 * K + gk1] : 0.0f;
        }
        // Load B tile: Bs[m][n] = B[(m_block+m)*N + (tile_col+n)]
        {
            uint flat = thread_index;
            uint r0 = flat / TILE, c0 = flat % TILE;
            uint gm0 = m_block + r0, gn0 = tile_col + c0;
            Bs[r0][c0] = (gm0 < M && gn0 < N) ? B[gm0 * N + gn0] : 0.0f;
            uint flat2 = flat + THREADS_PER_GROUP;
            uint r1 = flat2 / TILE, c1 = flat2 % TILE;
            uint gm1 = m_block + r1, gn1 = tile_col + c1;
            Bs[r1][c1] = (gm1 < M && gn1 < N) ? B[gm1 * N + gn1] : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> a0, a1, b0, b1;
        simdgroup_load(a0, &As[sg_row][0], M_TILE + 1);
        simdgroup_load(a1, &As[sg_row + 8][0], M_TILE + 1);
        simdgroup_load(b0, &Bs[0][sg_col], TILE + 1);
        simdgroup_load(b1, &Bs[0][sg_col + 8], TILE + 1);

        simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
        simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
        simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
        simdgroup_multiply_accumulate(acc11, a1, b1, acc11);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float Cs[TILE][TILE + 1];
    simdgroup_store(acc00, &Cs[sg_row][sg_col], TILE + 1);
    simdgroup_store(acc01, &Cs[sg_row][sg_col + 8], TILE + 1);
    simdgroup_store(acc10, &Cs[sg_row + 8][sg_col], TILE + 1);
    simdgroup_store(acc11, &Cs[sg_row + 8][sg_col + 8], TILE + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 8; i++) {
        uint flat = thread_index * 8 + i;
        uint r = flat / TILE, c = flat % TILE;
        if (tile_row + r < K && tile_col + c < N) {
            C[(tile_row + r) * N + tile_col + c] = Cs[r][c];
        }
    }
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

/// Gradient masking: zero out entire rows in a [positions, vocab] gradient matrix.
/// mask[pos] == 0 → zero out grad[pos * vocab .. (pos+1) * vocab].
/// Used in SFT to mask loss on prompt tokens (only response tokens get gradients).
pub const GRADIENT_MASK: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GradMaskParams {
    uint total;      // positions * vocab_size
    uint vocab_size;
};

kernel void gradient_mask(
    device float* grad [[buffer(0)]],
    device const uint* mask [[buffer(1)]],
    constant GradMaskParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.total) return;
    uint pos = gid / params.vocab_size;
    if (mask[pos] == 0u) {
        grad[gid] = 0.0f;
    }
}
"#;

/// Batched strided copy: source [bh, src_seq_len, dim] (contiguous) →
/// destination [bh, dst_stride, dim] at offset dst_offset per batch-head.
/// Single dispatch replaces O(bh) individual gpu_buffer_copy calls.
/// Thread grid: bh * src_seq_len * dim total threads.
pub const STRIDED_BATCH_COPY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct StridedBatchCopyParams {
    uint bh;
    uint src_seq_len;
    uint dst_stride;
    uint dst_offset;
    uint dim;
};

kernel void strided_batch_copy(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant StridedBatchCopyParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.bh * params.src_seq_len * params.dim;
    if (gid >= total) return;

    uint elements_per_head = params.src_seq_len * params.dim;
    uint head = gid / elements_per_head;
    uint within = gid % elements_per_head;
    uint seq_pos = within / params.dim;
    uint d = within % params.dim;

    uint src_idx = head * elements_per_head + seq_pos * params.dim + d;
    uint dst_idx = head * params.dst_stride * params.dim + (params.dst_offset + seq_pos) * params.dim + d;

    dst[dst_idx] = src[src_idx];
}
"#;

/// Compact strided copy: source [bh, stride, dim] (only first seq_len valid) →
/// destination [bh, seq_len, dim] (contiguous). Reverse of strided_batch_copy.
/// Thread grid: bh * seq_len * dim total threads.
pub const COMPACT_STRIDED_COPY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct CompactStridedCopyParams {
    uint bh;
    uint seq_len;
    uint src_stride;
    uint dim;
};

kernel void compact_strided_copy(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant CompactStridedCopyParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.bh * params.seq_len * params.dim;
    if (gid >= total) return;

    uint elements_per_head = params.seq_len * params.dim;
    uint head = gid / elements_per_head;
    uint within = gid % elements_per_head;
    uint seq_pos = within / params.dim;
    uint d = within % params.dim;

    uint src_idx = head * params.src_stride * params.dim + seq_pos * params.dim + d;
    uint dst_idx = head * elements_per_head + seq_pos * params.dim + d;

    dst[dst_idx] = src[src_idx];
}
"#;

/// Argmax reduction: find the index of the maximum value in a float buffer.
/// Uses a single threadgroup with parallel reduction (256 threads).
/// Reads back just 4 bytes (one u32) instead of the entire buffer.
pub const ARGMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ArgmaxParams {
    uint size;
};

kernel void argmax(
    device const float* data [[buffer(0)]],
    device uint* result [[buffer(1)]],
    constant ArgmaxParams& params [[buffer(2)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    threadgroup float shared_vals[256];
    threadgroup uint shared_idxs[256];

    float local_max = -INFINITY;
    uint local_idx = 0;
    for (uint i = thread_index; i < params.size; i += threads_per_group) {
        if (data[i] > local_max) {
            local_max = data[i];
            local_idx = i;
        }
    }
    shared_vals[thread_index] = local_max;
    shared_idxs[thread_index] = local_idx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            if (shared_vals[thread_index + stride] > shared_vals[thread_index]) {
                shared_vals[thread_index] = shared_vals[thread_index + stride];
                shared_idxs[thread_index] = shared_idxs[thread_index + stride];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        result[0] = shared_idxs[0];
    }
}
"#;

/// Temperature scaling: divide logits by temperature in-place.
/// data[i] = data[i] / temperature for i in [offset, offset + count).
/// Operates on a sub-range so we can scale only the last token's logits.
pub const TEMPERATURE_SCALE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TempScaleParams {
    uint offset;
    uint count;
    float inv_temperature;
};

kernel void temperature_scale(
    device float* data [[buffer(0)]],
    constant TempScaleParams& params [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.count) return;
    data[params.offset + gid] = data[params.offset + gid] * params.inv_temperature;
}
"#;

/// Batched tiled matrix multiplication: C[b] = A[b] @ B[b]
/// A: [B, M, K], B: [B, K, N], C: [B, M, N]
/// Simdgroup matrix intrinsics, 128 threads = 4 simdgroups, 32x32 output tile.
pub const BATCHED_MATMUL_TILED: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct BatchedMatmulParams {
    uint M;
    uint N;
    uint K;
    uint batch;
};

#define TILE 32
#define K_TILE 8
#define THREADS_PER_GROUP 128

kernel void batched_matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint batch_idx = group_id.z;
    if (batch_idx >= params.batch) return;

    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;
    uint M = params.M, N = params.N, K = params.K;

    device const float* Ab = A + batch_idx * M * K;
    device const float* Bb = B + batch_idx * K * N;
    device float* Cb = C + batch_idx * M * N;

    threadgroup float As[TILE][K_TILE + 1];
    threadgroup float Bs[K_TILE][TILE + 1];

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 16;

    simdgroup_matrix<float, 8, 8> acc00(0.0f), acc01(0.0f), acc10(0.0f), acc11(0.0f);

    for (uint k_block = 0; k_block < K; k_block += K_TILE) {
        {
            uint f = thread_index;
            uint r0 = f / K_TILE, c0 = f % K_TILE;
            As[r0][c0] = (tile_row + r0 < M && k_block + c0 < K) ? Ab[(tile_row + r0) * K + k_block + c0] : 0.0f;
            uint f2 = f + THREADS_PER_GROUP;
            uint r1 = f2 / K_TILE, c1 = f2 % K_TILE;
            As[r1][c1] = (tile_row + r1 < M && k_block + c1 < K) ? Ab[(tile_row + r1) * K + k_block + c1] : 0.0f;
        }
        {
            uint f = thread_index;
            uint r0 = f / TILE, c0 = f % TILE;
            Bs[r0][c0] = (k_block + r0 < K && tile_col + c0 < N) ? Bb[(k_block + r0) * N + tile_col + c0] : 0.0f;
            uint f2 = f + THREADS_PER_GROUP;
            uint r1 = f2 / TILE, c1 = f2 % TILE;
            Bs[r1][c1] = (k_block + r1 < K && tile_col + c1 < N) ? Bb[(k_block + r1) * N + tile_col + c1] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> a0, a1, b0, b1;
        simdgroup_load(a0, &As[sg_row][0], K_TILE + 1);
        simdgroup_load(a1, &As[sg_row + 8][0], K_TILE + 1);
        simdgroup_load(b0, &Bs[0][sg_col], TILE + 1);
        simdgroup_load(b1, &Bs[0][sg_col + 8], TILE + 1);
        simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
        simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
        simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
        simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float Cs[TILE][TILE + 1];
    simdgroup_store(acc00, &Cs[sg_row][sg_col], TILE + 1);
    simdgroup_store(acc01, &Cs[sg_row][sg_col + 8], TILE + 1);
    simdgroup_store(acc10, &Cs[sg_row + 8][sg_col], TILE + 1);
    simdgroup_store(acc11, &Cs[sg_row + 8][sg_col + 8], TILE + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 8; i++) {
        uint flat = thread_index * 8 + i;
        uint r = flat / TILE, c = flat % TILE;
        if (tile_row + r < M && tile_col + c < N) {
            Cb[(tile_row + r) * N + tile_col + c] = Cs[r][c];
        }
    }
}
"#;

/// Batched tiled matmul with B transposed: C[b] = A[b] @ B[b]^T
/// A: [B, M, K], B: [B, N, K], C: [B, M, N]
/// Simdgroup matrix intrinsics, 128 threads = 4 simdgroups, 32x32 output tile.
pub const BATCHED_MATMUL_TILED_TRANS_B: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct BatchedMatmulParams {
    uint M;
    uint N;
    uint K;
    uint batch;
};

#define TILE 32
#define K_TILE 8
#define THREADS_PER_GROUP 128

kernel void batched_matmul_tiled_trans_b(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint batch_idx = group_id.z;
    if (batch_idx >= params.batch) return;

    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;
    uint M = params.M, N = params.N, K = params.K;

    device const float* Ab = A + batch_idx * M * K;
    device const float* Bb = B + batch_idx * N * K;
    device float* Cb = C + batch_idx * M * N;

    threadgroup float As[TILE][K_TILE + 1];
    threadgroup float Bs[K_TILE][TILE + 1];

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 16;

    simdgroup_matrix<float, 8, 8> acc00(0.0f), acc01(0.0f), acc10(0.0f), acc11(0.0f);

    for (uint k_block = 0; k_block < K; k_block += K_TILE) {
        {
            uint f = thread_index;
            uint r0 = f / K_TILE, c0 = f % K_TILE;
            As[r0][c0] = (tile_row + r0 < M && k_block + c0 < K) ? Ab[(tile_row + r0) * K + k_block + c0] : 0.0f;
            uint f2 = f + THREADS_PER_GROUP;
            uint r1 = f2 / K_TILE, c1 = f2 % K_TILE;
            As[r1][c1] = (tile_row + r1 < M && k_block + c1 < K) ? Ab[(tile_row + r1) * K + k_block + c1] : 0.0f;
        }
        {
            uint f = thread_index;
            uint r0 = f / TILE, c0 = f % TILE;
            uint gk0 = k_block + r0, gn0 = tile_col + c0;
            Bs[r0][c0] = (gk0 < K && gn0 < N) ? Bb[gn0 * K + gk0] : 0.0f;
            uint f2 = f + THREADS_PER_GROUP;
            uint r1 = f2 / TILE, c1 = f2 % TILE;
            uint gk1 = k_block + r1, gn1 = tile_col + c1;
            Bs[r1][c1] = (gk1 < K && gn1 < N) ? Bb[gn1 * K + gk1] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> a0, a1, b0, b1;
        simdgroup_load(a0, &As[sg_row][0], K_TILE + 1);
        simdgroup_load(a1, &As[sg_row + 8][0], K_TILE + 1);
        simdgroup_load(b0, &Bs[0][sg_col], TILE + 1);
        simdgroup_load(b1, &Bs[0][sg_col + 8], TILE + 1);
        simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
        simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
        simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
        simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float Cs[TILE][TILE + 1];
    simdgroup_store(acc00, &Cs[sg_row][sg_col], TILE + 1);
    simdgroup_store(acc01, &Cs[sg_row][sg_col + 8], TILE + 1);
    simdgroup_store(acc10, &Cs[sg_row + 8][sg_col], TILE + 1);
    simdgroup_store(acc11, &Cs[sg_row + 8][sg_col + 8], TILE + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 8; i++) {
        uint flat = thread_index * 8 + i;
        uint r = flat / TILE, c = flat % TILE;
        if (tile_row + r < M && tile_col + c < N) {
            Cb[(tile_row + r) * N + tile_col + c] = Cs[r][c];
        }
    }
}
"#;

/// Batched tiled matmul with A transposed: C[b] = A[b]^T @ B[b]
/// A: [B, M, K] (row-major), B: [B, M, N], C: [B, K, N]
/// A^T is [K, M], so C[b][i,j] = sum_m A[b][m,i] * B[b][m,j]
/// Uses group_id.z as the batch index. Single dispatch for all batch elements.
/// Used in backward pass for computing dB = A^T @ dC.
pub const KL_DIVERGENCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct KLParams {
    uint batch_size;
    uint vocab_size;
    float temperature;
};

// KL divergence: KL(p || q) where p = softmax(teacher/T), q = softmax(student/T)
// Per row: compute log-sum-exp for both teacher and student logits (scaled by 1/T),
// then KL = sum(p * (log_p - log_q)).
// Also outputs raw gradient w.r.t. student logits: (1/T) * (q - p) / batch_size.
// The caller applies alpha * T^2 to produce d(alpha * T^2 * KL)/d_z = alpha * T * (q - p) / batch.
kernel void kl_divergence(
    device const float* teacher_logits [[buffer(0)]],
    device const float* student_logits [[buffer(1)]],
    device float* losses [[buffer(2)]],
    device float* grad_student [[buffer(3)]],
    constant KLParams& params [[buffer(4)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.batch_size) return;

    uint V = params.vocab_size;
    float inv_T = 1.0f / params.temperature;
    device const float* t_row = teacher_logits + row * V;
    device const float* s_row = student_logits + row * V;
    device float* g_row = grad_student + row * V;

    // Phase 1: Find max of teacher/T and student/T for numerical stability
    threadgroup float shared_t_max[256];
    threadgroup float shared_s_max[256];
    float local_t_max = -INFINITY;
    float local_s_max = -INFINITY;
    for (uint c = thread_index; c < V; c += threads_per_group) {
        local_t_max = max(local_t_max, t_row[c] * inv_T);
        local_s_max = max(local_s_max, s_row[c] * inv_T);
    }
    shared_t_max[thread_index] = local_t_max;
    shared_s_max[thread_index] = local_s_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_t_max[thread_index] = max(shared_t_max[thread_index], shared_t_max[thread_index + stride]);
            shared_s_max[thread_index] = max(shared_s_max[thread_index], shared_s_max[thread_index + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float t_max = shared_t_max[0];
    float s_max = shared_s_max[0];

    // Phase 2: Compute exp sums for both distributions
    threadgroup float shared_t_sum[256];
    threadgroup float shared_s_sum[256];
    float local_t_sum = 0.0f;
    float local_s_sum = 0.0f;
    for (uint c = thread_index; c < V; c += threads_per_group) {
        float t_exp = exp(t_row[c] * inv_T - t_max);
        float s_exp = exp(s_row[c] * inv_T - s_max);
        // Store exps temporarily in grad buffer (will overwrite with actual grad below)
        g_row[c] = s_exp;
        local_t_sum += t_exp;
        local_s_sum += s_exp;
    }
    shared_t_sum[thread_index] = local_t_sum;
    shared_s_sum[thread_index] = local_s_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_t_sum[thread_index] += shared_t_sum[thread_index + stride];
            shared_s_sum[thread_index] += shared_s_sum[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float t_total = shared_t_sum[0];
    float s_total = shared_s_sum[0];
    float inv_t_total = 1.0f / t_total;
    float inv_s_total = 1.0f / s_total;
    float log_t_total = log(t_total);
    float log_s_total = log(s_total);

    // Phase 3: Compute KL divergence and gradient
    // KL = sum(p * (log_p - log_q))
    // log_p_c = t_row[c]*inv_T - t_max - log(t_total)
    // log_q_c = s_row[c]*inv_T - s_max - log(s_total)
    // Raw gradient: d_KL/d_z_c = (1/T) * (q_c - p_c)
    // The caller (loss.rs) applies alpha * T^2 scaling to get d(alpha * T^2 * KL)/d_z.
    threadgroup float shared_kl[256];
    float local_kl = 0.0f;
    float inv_batch = 1.0f / float(params.batch_size);
    for (uint c = thread_index; c < V; c += threads_per_group) {
        float t_scaled = t_row[c] * inv_T;
        float s_scaled = s_row[c] * inv_T;
        float p_c = exp(t_scaled - t_max) * inv_t_total;
        float q_c = g_row[c] * inv_s_total; // g_row[c] still holds s_exp from phase 2
        float log_p = t_scaled - t_max - log_t_total;
        float log_q = s_scaled - s_max - log_s_total;
        local_kl += p_c * (log_p - log_q);
        g_row[c] = inv_T * (q_c - p_c) * inv_batch;
    }
    shared_kl[thread_index] = local_kl;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_group / 2; stride > 0; stride >>= 1) {
        if (thread_index < stride) {
            shared_kl[thread_index] += shared_kl[thread_index + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (thread_index == 0) {
        losses[row] = shared_kl[0];
    }
}
"#;

pub const BATCHED_MATMUL_TILED_TRANS_A: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct BatchedMatmulTransAParams {
    uint M;
    uint K;
    uint N;
    uint batch;
};

#define TILE 32
#define M_TILE 8
#define THREADS_PER_GROUP 128

kernel void batched_matmul_tiled_trans_a(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulTransAParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]
) {
    uint batch_idx = group_id.z;
    if (batch_idx >= params.batch) return;

    uint tile_row = group_id.y * TILE;  // K dimension
    uint tile_col = group_id.x * TILE;  // N dimension
    uint M = params.M, K = params.K, N = params.N;

    device const float* Ab = A + batch_idx * M * K;
    device const float* Bb = B + batch_idx * M * N;
    device float* Cb = C + batch_idx * K * N;

    threadgroup float As[TILE][M_TILE + 1];   // As[k][m]
    threadgroup float Bs[M_TILE][TILE + 1];   // Bs[m][n]

    uint sg_row = (simd_group_id / 2) * 16;
    uint sg_col = (simd_group_id % 2) * 16;

    simdgroup_matrix<float, 8, 8> acc00(0.0f), acc01(0.0f), acc10(0.0f), acc11(0.0f);

    for (uint m_block = 0; m_block < M; m_block += M_TILE) {
        {
            uint f = thread_index;
            uint r0 = f / M_TILE, c0 = f % M_TILE;
            uint gk0 = tile_row + r0, gm0 = m_block + c0;
            As[r0][c0] = (gk0 < K && gm0 < M) ? Ab[gm0 * K + gk0] : 0.0f;
            uint f2 = f + THREADS_PER_GROUP;
            uint r1 = f2 / M_TILE, c1 = f2 % M_TILE;
            uint gk1 = tile_row + r1, gm1 = m_block + c1;
            As[r1][c1] = (gk1 < K && gm1 < M) ? Ab[gm1 * K + gk1] : 0.0f;
        }
        {
            uint f = thread_index;
            uint r0 = f / TILE, c0 = f % TILE;
            uint gm0 = m_block + r0, gn0 = tile_col + c0;
            Bs[r0][c0] = (gm0 < M && gn0 < N) ? Bb[gm0 * N + gn0] : 0.0f;
            uint f2 = f + THREADS_PER_GROUP;
            uint r1 = f2 / TILE, c1 = f2 % TILE;
            uint gm1 = m_block + r1, gn1 = tile_col + c1;
            Bs[r1][c1] = (gm1 < M && gn1 < N) ? Bb[gm1 * N + gn1] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<float, 8, 8> a0, a1, b0, b1;
        simdgroup_load(a0, &As[sg_row][0], M_TILE + 1);
        simdgroup_load(a1, &As[sg_row + 8][0], M_TILE + 1);
        simdgroup_load(b0, &Bs[0][sg_col], TILE + 1);
        simdgroup_load(b1, &Bs[0][sg_col + 8], TILE + 1);
        simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
        simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
        simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
        simdgroup_multiply_accumulate(acc11, a1, b1, acc11);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float Cs[TILE][TILE + 1];
    simdgroup_store(acc00, &Cs[sg_row][sg_col], TILE + 1);
    simdgroup_store(acc01, &Cs[sg_row][sg_col + 8], TILE + 1);
    simdgroup_store(acc10, &Cs[sg_row + 8][sg_col], TILE + 1);
    simdgroup_store(acc11, &Cs[sg_row + 8][sg_col + 8], TILE + 1);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = 0; i < 8; i++) {
        uint flat = thread_index * 8 + i;
        uint r = flat / TILE, c = flat % TILE;
        if (tile_row + r < K && tile_col + c < N) {
            Cb[(tile_row + r) * N + tile_col + c] = Cs[r][c];
        }
    }
}
"#;
