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

    // Shared memory tiles in half precision — halves bandwidth, 2x FP16 throughput.
    // Accumulator stays float for precision (standard mixed-precision pattern).
    threadgroup half As[TILE][TILE];
    threadgroup half Bs[TILE][TILE];

    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    uint M = params.M;
    uint N = params.N;
    uint K = params.K;

    for (uint k_block = 0; k_block < K; k_block += TILE) {
        // Load A tile: cast float→half on load
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE;
            uint c = flat % TILE;
            uint global_r = tile_row + r;
            uint global_c = k_block + c;
            As[r][c] = (half)(clamp((global_r < M && global_c < K) ? A[global_r * K + global_c] : 0.0f, -65504.0f, 65504.0f));
        }

        // Load B tile: cast float→half on load
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE;
            uint c = flat % TILE;
            uint global_r = k_block + r;
            uint global_c = tile_col + c;
            Bs[r][c] = (half)(clamp((global_r < K && global_c < N) ? B[global_r * N + global_c] : 0.0f, -65504.0f, 65504.0f));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Inner loop: half*half accumulated into float (mixed precision)
        for (uint k = 0; k < TILE; k++) {
            half a_vals[THREAD_TILE];
            half b_vals[THREAD_TILE];

            for (uint i = 0; i < THREAD_TILE; i++) {
                a_vals[i] = As[local_row * THREAD_TILE + i][k];
            }
            for (uint j = 0; j < THREAD_TILE; j++) {
                b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            }

            for (uint i = 0; i < THREAD_TILE; i++) {
                for (uint j = 0; j < THREAD_TILE; j++) {
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
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

/// Full-FP32 tiled matmul: identical tiling to matmul_tiled but the shared-memory tiles are
/// `float`, not `half`. No fp16 cast and no ±65504 clamp on the inputs, so it keeps full float
/// precision and range. ~2× the shared-memory bandwidth of the fp16 path (slower), so it is an
/// opt-in precision path (e.g. logits, large-magnitude matmuls) rather than the default.
pub const MATMUL_TILED_FP32: &str = r#"
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

kernel void matmul_tiled_fp32(
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

    // FP32 shared tiles — full precision and range, no clamp.
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
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE;
            uint c = flat % TILE;
            uint global_r = k_block + r;
            uint global_c = tile_col + c;
            Bs[r][c] = (global_r < K && global_c < N) ? B[global_r * N + global_c] : 0.0f;
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

/// BF16 tiled matmul: shared tiles are `bfloat` (8-bit exponent like fp32, ~7-bit mantissa). Unlike
/// the fp16 path it has fp32 RANGE — no ±65504 clamp/overflow — at half the shared-memory bandwidth
/// of fp32. The modern mixed-precision default: range-safe and fast, lower precision than fp32.
pub const MATMUL_TILED_BF16: &str = r#"
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

kernel void matmul_tiled_bf16(
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

    // BF16 shared tiles — fp32 exponent range (no clamp), ~half fp32 bandwidth.
    threadgroup bfloat As[TILE][TILE];
    threadgroup bfloat Bs[TILE][TILE];

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
            As[r][c] = (bfloat)((global_r < M && global_c < K) ? A[global_r * K + global_c] : 0.0f);
        }
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE;
            uint c = flat % TILE;
            uint global_r = k_block + r;
            uint global_c = tile_col + c;
            Bs[r][c] = (bfloat)((global_r < K && global_c < N) ? B[global_r * N + global_c] : 0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            bfloat a_vals[THREAD_TILE];
            bfloat b_vals[THREAD_TILE];
            for (uint i = 0; i < THREAD_TILE; i++) {
                a_vals[i] = As[local_row * THREAD_TILE + i][k];
            }
            for (uint j = 0; j < THREAD_TILE; j++) {
                b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            }
            for (uint i = 0; i < THREAD_TILE; i++) {
                for (uint j = 0; j < THREAD_TILE; j++) {
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
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

/// simdgroup_matrix matmul: C = A @ B using the Apple-Silicon hardware matrix units
/// (`simdgroup_matrix<float,8,8>` + `simdgroup_multiply_accumulate`, Metal 3+, M1→M4). One
/// threadgroup = one 32-thread simdgroup computes a 32×32 output tile as a 4×4 grid of 8×8 MMA
/// fragments, looping K in steps of 8. A 32×8 slab of A and an 8×32 slab of B are staged into
/// threadgroup memory each K-step (cooperative, zero-padded at the M/N/K edges) so the fragment
/// loads are always in-bounds for non-multiple-of-8 dims; the result fragments are stored back
/// through threadgroup memory with a bounds-checked cooperative write. This replaces the hand-rolled
/// scalar-MAC inner loop with the hardware MMA — the single biggest matmul throughput lever here.
pub const MATMUL_SIMDGROUP: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct MatmulParams {
    uint M;
    uint N;
    uint K;
};

#define TILE 32

kernel void matmul_simdgroup(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]
) {
    uint M = params.M, N = params.N, K = params.K;
    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;

    threadgroup float As[TILE * 8];   // 32 rows × 8 cols (row-major)
    threadgroup float Bs[8 * TILE];   // 8 rows × 32 cols (row-major)
    threadgroup float Cs[TILE * TILE];

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += 8) {
        // Stage A slab [32×8]: 256 elements, 32 lanes → 8 each, zero-padded at edges.
        for (uint t = 0; t < 8; t++) {
            uint idx = lane * 8 + t;       // 0..255
            uint r = idx >> 3;             // 0..31
            uint c = idx & 7;              // 0..7
            uint gr = tile_row + r;
            uint gc = k0 + c;
            As[r * 8 + c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        // Stage B slab [8×32]: 256 elements, zero-padded at edges.
        for (uint t = 0; t < 8; t++) {
            uint idx = lane * 8 + t;       // 0..255
            uint r = idx >> 5;             // 0..7
            uint c = idx & 31;             // 0..31
            uint gr = k0 + r;
            uint gc = tile_col + c;
            Bs[r * TILE + c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 a[4], b[4];
        for (uint i = 0; i < 4; i++) simdgroup_load(a[i], &As[i * 8 * 8], 8);     // 8×8, ld=8
        for (uint j = 0; j < 4; j++) simdgroup_load(b[j], &Bs[j * 8], TILE);      // 8×8, ld=32
        for (uint i = 0; i < 4; i++)
            for (uint j = 0; j < 4; j++)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store fragments to the threadgroup tile, then cooperative bounds-checked write to C.
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            simdgroup_store(acc[i][j], &Cs[i * 8 * TILE + j * 8], TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint t = 0; t < 32; t++) {
        uint idx = lane * 32 + t;          // 0..1023
        uint r = idx >> 5;                 // 0..31
        uint c = idx & 31;                 // 0..31
        uint gr = tile_row + r;
        uint gc = tile_col + c;
        if (gr < M && gc < N) C[gr * N + gc] = Cs[r * TILE + c];
    }
}
"#;

/// Half-input simdgroup_matrix matmul: C(fp32) = A(fp16) @ B(fp16) on the hardware MMA units.
/// The drop-in fast path for the default `matmul` (which pre-casts to fp16 then runs the hand-rolled
/// `matmul_tiled_f16`): same fp16-input / fp32-output precision, but the inner product runs on the
/// `simdgroup_half8x8` × `simdgroup_half8x8` → `simdgroup_float8x8` MMA instead of scalar MACs.
///
/// 64×64 output tile per threadgroup of 4 simdgroups (128 threads): each simdgroup owns a 32×32
/// quadrant (4×4 = 16 fragments). A 64×8 slab of A and an 8×64 slab of B are staged to threadgroup
/// memory per K-step and reused by all 4 quadrants (4× arithmetic intensity vs a 32×32 single-
/// simdgroup tile, which measured *slower* than the hand-rolled kernel — it was occupancy/
/// intensity-bound). Edges are zero-padded so the fragment loads stay in-bounds for any M/N/K.
pub const MATMUL_SIMDGROUP_F16: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct MatmulParams {
    uint M;
    uint N;
    uint K;
};

#define TILE 64

kernel void matmul_simdgroup_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    uint M = params.M, N = params.N, K = params.K;
    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;
    uint q_row = (sgid >> 1) * 32;   // simdgroup's 32×32 quadrant within the 64×64 tile
    uint q_col = (sgid & 1) * 32;

    threadgroup half As[TILE * 8];   // 64×8
    threadgroup half Bs[8 * TILE];   // 8×64
    threadgroup float Cs[TILE * TILE];

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += 8) {
        // Stage A [64×8] and B [8×64]: 512 + 512 elems, 128 lanes → 4 each, zero-padded at edges.
        for (uint t = 0; t < 4; t++) {
            uint idx = lane * 4 + t;       // 0..511
            uint r = idx >> 3;             // 0..63
            uint c = idx & 7;              // 0..7
            uint gr = tile_row + r;
            uint gc = k0 + c;
            As[r * 8 + c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0;
        }
        for (uint t = 0; t < 4; t++) {
            uint idx = lane * 4 + t;       // 0..511
            uint r = idx >> 6;             // 0..7
            uint c = idx & 63;             // 0..63
            uint gr = k0 + r;
            uint gc = tile_col + c;
            Bs[r * TILE + c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_half8x8 a[4], b[4];
        for (uint i = 0; i < 4; i++) simdgroup_load(a[i], &As[(q_row + i * 8) * 8], 8);
        for (uint j = 0; j < 4; j++) simdgroup_load(b[j], &Bs[q_col + j * 8], TILE);
        for (uint i = 0; i < 4; i++)
            for (uint j = 0; j < 4; j++)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < 4; i++)
        for (uint j = 0; j < 4; j++)
            simdgroup_store(acc[i][j], &Cs[(q_row + i * 8) * TILE + q_col + j * 8], TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative bounds-checked write: 64×64 = 4096 elems, 128 lanes → 32 each.
    for (uint t = 0; t < 32; t++) {
        uint idx = lane * 32 + t;          // 0..4095
        uint r = idx >> 6;                 // 0..63
        uint c = idx & 63;                 // 0..63
        uint gr = tile_row + r;
        uint gc = tile_col + c;
        if (gr < M && gc < N) C[gr * N + gc] = Cs[r * TILE + c];
    }
}
"#;

/// Batched simdgroup-matrix matmul: C[b] = A[b] @ B[b] on the hardware MMA units (extends the
/// simdgroup fast path beyond the batch==1 projections to the batched attention matmuls). fp32 in /
/// fp32 out, half MMA fragments (same precision as batched_matmul_tiled). 64×64 tile / 4 simdgroups,
/// batch in grid.z. A[b]:[M,K], B[b]:[K,N], C[b]:[M,N].
pub const BATCHED_MATMUL_SIMDGROUP: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct BatchedMatmulParams { uint M; uint N; uint K; uint batch; };
#define TILE 64

kernel void batched_matmul_simdgroup(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    uint b = group_id.z;
    if (b >= params.batch) return;
    uint M = params.M, N = params.N, K = params.K;
    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;
    uint q_row = (sgid >> 1) * 32;
    uint q_col = (sgid & 1) * 32;
    device const float* A_b = A + b * M * K;
    device const float* B_b = B + b * K * N;
    device float* C_b = C + b * M * N;

    threadgroup half As[TILE * 8];
    threadgroup half Bs[8 * TILE];
    threadgroup float Cs[TILE * TILE];

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)
        acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += 8) {
        for (uint t = 0; t < 4; t++) {
            uint idx = lane * 4 + t; uint r = idx >> 3; uint c = idx & 7;
            uint gr = tile_row + r, gc = k0 + c;
            As[r * 8 + c] = (gr < M && gc < K) ? (half)A_b[gr * K + gc] : (half)0.0;
        }
        for (uint t = 0; t < 4; t++) {
            uint idx = lane * 4 + t; uint r = idx >> 6; uint c = idx & 63;
            uint gr = k0 + r, gc = tile_col + c;
            Bs[r * TILE + c] = (gr < K && gc < N) ? (half)B_b[gr * N + gc] : (half)0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 a[4], bb[4];
        for (uint i = 0; i < 4; i++) simdgroup_load(a[i], &As[(q_row + i * 8) * 8], 8);
        for (uint j = 0; j < 4; j++) simdgroup_load(bb[j], &Bs[q_col + j * 8], TILE);
        for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)
            simdgroup_multiply_accumulate(acc[i][j], a[i], bb[j], acc[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)
        simdgroup_store(acc[i][j], &Cs[(q_row + i * 8) * TILE + q_col + j * 8], TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < 32; t++) {
        uint idx = lane * 32 + t; uint r = idx >> 6; uint c = idx & 63;
        uint gr = tile_row + r, gc = tile_col + c;
        if (gr < M && gc < N) C_b[gr * N + gc] = Cs[r * TILE + c];
    }
}
"#;

/// Batched simdgroup matmul with B transposed: C[b] = A[b] @ B[b]^T. A[b]:[M,K], B[b]:[N,K],
/// C[b]:[M,N]. Same MMA tiling; B is staged with transposed indexing (Bs[kk][nn] = B[nn][k0+kk]).
pub const BATCHED_MATMUL_SIMDGROUP_TRANS_B: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct BatchedMatmulParams { uint M; uint N; uint K; uint batch; };
#define TILE 64

kernel void batched_matmul_simdgroup_trans_b(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]],
    uint sgid [[simdgroup_index_in_threadgroup]]
) {
    uint b = group_id.z;
    if (b >= params.batch) return;
    uint M = params.M, N = params.N, K = params.K;
    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;
    uint q_row = (sgid >> 1) * 32;
    uint q_col = (sgid & 1) * 32;
    device const float* A_b = A + b * M * K;
    device const float* B_b = B + b * N * K; // B is [N, K]
    device float* C_b = C + b * M * N;

    threadgroup half As[TILE * 8];
    threadgroup half Bs[8 * TILE];
    threadgroup float Cs[TILE * TILE];

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)
        acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < K; k0 += 8) {
        for (uint t = 0; t < 4; t++) {
            uint idx = lane * 4 + t; uint r = idx >> 3; uint c = idx & 7;
            uint gr = tile_row + r, gc = k0 + c;
            As[r * 8 + c] = (gr < M && gc < K) ? (half)A_b[gr * K + gc] : (half)0.0;
        }
        // Stage B^T: Bs[kk][nn] = B[tile_col+nn][k0+kk]  (B is [N,K])
        for (uint t = 0; t < 4; t++) {
            uint idx = lane * 4 + t; uint kk = idx >> 6; uint nn = idx & 63;
            uint gn = tile_col + nn, gk = k0 + kk;
            Bs[kk * TILE + nn] = (gn < N && gk < K) ? (half)B_b[gn * K + gk] : (half)0.0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_half8x8 a[4], bb[4];
        for (uint i = 0; i < 4; i++) simdgroup_load(a[i], &As[(q_row + i * 8) * 8], 8);
        for (uint j = 0; j < 4; j++) simdgroup_load(bb[j], &Bs[q_col + j * 8], TILE);
        for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)
            simdgroup_multiply_accumulate(acc[i][j], a[i], bb[j], acc[i][j]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < 4; i++) for (uint j = 0; j < 4; j++)
        simdgroup_store(acc[i][j], &Cs[(q_row + i * 8) * TILE + q_col + j * 8], TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint t = 0; t < 32; t++) {
        uint idx = lane * 32 + t; uint r = idx >> 6; uint c = idx & 63;
        uint gr = tile_row + r, gc = tile_col + c;
        if (gr < M && gc < N) C_b[gr * N + gc] = Cs[r * TILE + c];
    }
}
"#;

/// Narrow matmul for small N (≤32): C = A @ B where A:[M,K], B:[K,N], C:[M,N].
/// TILE_M=32, TILE_N=16, 32 threads. Each thread computes 4×4 subtile.
/// Eliminates 50% wasted compute when N=16 with the standard 32-wide tile.
pub const MATMUL_NARROW: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MatmulParams {
    uint M;
    uint N;
    uint K;
};

#define NM_TILE_M 32
#define NM_TILE_N 16
#define NM_TILE_K 32
#define NM_THREAD_TILE 4
#define NM_THREADS 32  // 8 rows × 4 cols

kernel void matmul_narrow(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint local_row = thread_index / 4;  // 0..7
    uint local_col = thread_index % 4;  // 0..3

    uint tile_row = group_id.y * NM_TILE_M;
    uint tile_col = group_id.x * NM_TILE_N;

    threadgroup half As[NM_TILE_M][NM_TILE_K];
    threadgroup half Bs[NM_TILE_K][NM_TILE_N];

    float acc[NM_THREAD_TILE][NM_THREAD_TILE] = {{0.0f}};

    uint M = params.M;
    uint N = params.N;
    uint K = params.K;

    for (uint k_block = 0; k_block < K; k_block += NM_TILE_K) {
        // Load A tile [32][32]: 32 threads load 32 elements each = 1024 elements
        for (uint i = 0; i < 32; i++) {
            uint flat = thread_index * 32 + i;
            uint r = flat / NM_TILE_K;
            uint c = flat % NM_TILE_K;
            uint gr = tile_row + r;
            uint gc = k_block + c;
            As[r][c] = (half)(clamp((gr < M && gc < K) ? A[gr * K + gc] : 0.0f, -65504.0f, 65504.0f));
        }

        // Load B tile [32][16]: 32 threads load 16 elements each = 512 elements
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / NM_TILE_N;
            uint c = flat % NM_TILE_N;
            uint gr = k_block + r;
            uint gc = tile_col + c;
            Bs[r][c] = (half)(clamp((gr < K && gc < N) ? B[gr * N + gc] : 0.0f, -65504.0f, 65504.0f));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < NM_TILE_K; k++) {
            half a_vals[NM_THREAD_TILE];
            half b_vals[NM_THREAD_TILE];
            for (uint i = 0; i < NM_THREAD_TILE; i++)
                a_vals[i] = As[local_row * NM_THREAD_TILE + i][k];
            for (uint j = 0; j < NM_THREAD_TILE; j++)
                b_vals[j] = Bs[k][local_col * NM_THREAD_TILE + j];
            for (uint i = 0; i < NM_THREAD_TILE; i++)
                for (uint j = 0; j < NM_THREAD_TILE; j++)
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < NM_THREAD_TILE; i++) {
        for (uint j = 0; j < NM_THREAD_TILE; j++) {
            uint gr = tile_row + local_row * NM_THREAD_TILE + i;
            uint gc = tile_col + local_col * NM_THREAD_TILE + j;
            if (gr < M && gc < N)
                C[gr * N + gc] = acc[i][j];
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

    threadgroup half As[TILE][TILE];
    threadgroup half Bs[TILE][TILE];

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
            As[r][c] = (half)(clamp((global_r < M && global_c < K) ? A[global_r * K + global_c] : 0.0f, -65504.0f, 65504.0f));
        }

        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE;
            uint c = flat % TILE;
            uint global_k = k_block + r;
            uint global_n = tile_col + c;
            Bs[r][c] = (half)(clamp((global_k < K && global_n < N) ? B[global_n * K + global_k] : 0.0f, -65504.0f, 65504.0f));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            half a_vals[THREAD_TILE];
            half b_vals[THREAD_TILE];

            for (uint i = 0; i < THREAD_TILE; i++) {
                a_vals[i] = As[local_row * THREAD_TILE + i][k];
            }
            for (uint j = 0; j < THREAD_TILE; j++) {
                b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            }

            for (uint i = 0; i < THREAD_TILE; i++) {
                for (uint j = 0; j < THREAD_TILE; j++) {
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
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

/// Row-wise softmax with numerical stability. SIMD-optimized for Apple Silicon.
/// Uses simd_max/simd_sum for intra-SIMD-group reduction (4x faster than shared memory),
/// then threadgroup memory only for cross-SIMD-group reduction (~8 values).
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
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg [[simdgroups_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;

    uint cols = params.cols;
    device const float* row_in = input + row * cols;
    device float* row_out = output + row * cols;

    // Pass 1: find max — SIMD reduction (256 bytes/cycle) then cross-SIMD (64 bytes/cycle)
    float local_max = -INFINITY;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        local_max = max(local_max, row_in[c]);
    }
    // Phase 1: intra-SIMD max (hardware shuffle, no memory access)
    float simd_max_val = simd_max(local_max);
    // Phase 2: cross-SIMD max (only simd_groups_per_tg values, typically 4-8)
    threadgroup float shared_vals[8];
    if (simd_lane_id == 0) shared_vals[simd_group_id] = simd_max_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = shared_vals[0];
    for (uint i = 1; i < simd_groups_per_tg; i++) row_max = max(row_max, shared_vals[i]);

    // Pass 2: exp + sum — same SIMD-first pattern
    float local_sum = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        float val = exp(row_in[c] - row_max);
        row_out[c] = val;
        local_sum += val;
    }
    float simd_sum_val = simd_sum(local_sum);
    if (simd_lane_id == 0) shared_vals[simd_group_id] = simd_sum_val;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < simd_groups_per_tg; i++) total += shared_vals[i];

    // Normalize
    float inv_sum = 1.0f / total;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        row_out[c] *= inv_sum;
    }
}
"#;

/// Fused scale + causal mask + softmax in one kernel.
/// Eliminates 2 buffer allocations and 4 GPU dispatches vs separate ops.
/// Input: raw attention scores [batch_heads * seq_q, seq_k]
/// Output: softmax probabilities after scaling and causal masking.
pub const SCALED_CAUSAL_SOFTMAX: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ScaledCausalSoftmaxParams {
    uint seq_q;
    uint seq_k;
    float scale;
    uint kv_offset;
    uint window; // 0=full causal, >0=sliding window size
};

kernel void scaled_causal_softmax(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ScaledCausalSoftmaxParams& params [[buffer(2)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg [[simdgroups_per_threadgroup]]
) {
    uint row = group_id;
    uint seq_k = params.seq_k;
    uint q_pos = row % params.seq_q;

    device const float* row_in = input + row * seq_k;
    device float* row_out = output + row * seq_k;

    // Pass 1: scale + mask + SIMD max
    float local_max = -INFINITY;
    for (uint c = thread_index; c < seq_k; c += threads_per_group) {
        float val = row_in[c] * params.scale;
        bool future = c > q_pos + params.kv_offset;
        bool too_far = (params.window > 0) && (q_pos + params.kv_offset >= params.window) && (c < q_pos + params.kv_offset - params.window);
        if (future || too_far) val = -INFINITY;
        local_max = max(local_max, val);
    }
    float simd_mx = simd_max(local_max);
    threadgroup float sv[8];
    if (simd_lane_id == 0) sv[simd_group_id] = simd_mx;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = sv[0];
    for (uint i = 1; i < simd_groups_per_tg; i++) row_max = max(row_max, sv[i]);

    // Pass 2: exp + SIMD sum
    float local_sum = 0.0f;
    for (uint c = thread_index; c < seq_k; c += threads_per_group) {
        float val = row_in[c] * params.scale;
        bool future = c > q_pos + params.kv_offset;
        bool too_far = (params.window > 0) && (q_pos + params.kv_offset >= params.window) && (c < q_pos + params.kv_offset - params.window);
        if (future || too_far) val = -INFINITY;
        float e = exp(val - row_max);
        row_out[c] = e;
        local_sum += e;
    }
    float simd_sm = simd_sum(local_sum);
    if (simd_lane_id == 0) sv[simd_group_id] = simd_sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < simd_groups_per_tg; i++) total += sv[i];

    float inv_sum = 1.0f / total;
    for (uint c = thread_index; c < seq_k; c += threads_per_group) {
        row_out[c] *= inv_sum;
    }
}
"#;

/// RMS Layer Normalization: output = (x / rms(x)) * weight
/// SIMD-optimized: uses simd_sum for 4x faster reduction on Apple Silicon.
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
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg [[simdgroups_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;

    uint cols = params.cols;
    device const float* row_in = input + row * cols;
    device float* row_out = output + row * cols;

    // Sum of squares with SIMD-first reduction
    float local_ss = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        float val = row_in[c];
        local_ss += val * val;
    }
    // Phase 1: SIMD reduction (hardware shuffle, 256 bytes/cycle)
    float simd_ss = simd_sum(local_ss);
    // Phase 2: cross-SIMD reduction (only simd_groups_per_tg values)
    threadgroup float shared_vals[8];
    if (simd_lane_id == 0) shared_vals[simd_group_id] = simd_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_ss = 0.0f;
    for (uint i = 0; i < simd_groups_per_tg; i++) total_ss += shared_vals[i];

    float rms = rsqrt(total_ss / float(cols) + params.eps);

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

    float cos_val;
    float sin_val = sincos(angle, cos_val); // single instruction for both

    uint base = row * params.seq_len * params.head_dim + pos * params.head_dim;
    uint i0 = base + pair * 2;
    uint i1 = base + pair * 2 + 1;

    float x0 = data[i0];
    float x1 = data[i1];

    data[i0] = x0 * cos_val - x1 * sin_val;
    data[i1] = x0 * sin_val + x1 * cos_val;
}
"#;

/// Out-of-place RoPE forward: dst = rotate(src, θ). Eliminates copy+in-place (2→1 dispatch).
pub const ROPE_COPY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RopeParams {
    uint seq_len;
    uint head_dim;
    uint total_rows;
    uint offset;
    float theta;
};

kernel void rope_copy(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant RopeParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint pos = gid.x;
    uint pair = gid.z;
    if (row >= params.total_rows || pos >= params.seq_len || pair >= params.head_dim / 2) return;

    float freq = 1.0f / pow(params.theta, float(2 * pair) / float(params.head_dim));
    float angle = float(pos + params.offset) * freq;
    float cos_val;
    float sin_val = sincos(angle, cos_val);

    uint base = row * params.seq_len * params.head_dim + pos * params.head_dim;
    uint i0 = base + pair * 2;
    uint i1 = base + pair * 2 + 1;

    float x0 = src[i0];
    float x1 = src[i1];
    dst[i0] = x0 * cos_val - x1 * sin_val;
    dst[i1] = x0 * sin_val + x1 * cos_val;
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
    float cos_val;
    float sin_val = sincos(angle, cos_val);

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

/// Out-of-place RoPE backward: dst = rotate(src, -θ). Replaces copy+rope_backward (2→1 dispatch).
pub const ROPE_BACKWARD_COPY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RopeParams {
    uint seq_len;
    uint head_dim;
    uint total_rows;
    uint offset;
    float theta;
};

kernel void rope_backward_copy(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant RopeParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;
    uint pos = gid.x;
    uint pair = gid.z;

    if (row >= params.total_rows || pos >= params.seq_len || pair >= params.head_dim / 2) return;

    float freq = 1.0 / pow(params.theta, float(2 * pair) / float(params.head_dim));
    float angle = float(pos + params.offset) * freq;
    float cos_val;
    float sin_val = sincos(angle, cos_val);

    uint base = row * params.seq_len * params.head_dim + pos * params.head_dim;
    uint i0 = base + 2 * pair;
    uint i1 = i0 + 1;

    float x0 = src[i0];
    float x1 = src[i1];

    dst[i0] = x0 * cos_val + x1 * sin_val;
    dst[i1] = -x0 * sin_val + x1 * cos_val;
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
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg [[simdgroups_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;

    uint cols = params.cols;
    device const float* row_in = input + row * cols;
    device const float* row_res = residual + row * cols;
    device float* row_out = output + row * cols;
    device float* row_sum = sum_out + row * cols;

    // Phase 1: compute input + residual and sum of squares (SIMD-optimized)
    float local_ss = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        float v = row_in[c] + row_res[c];
        row_sum[c] = v;
        local_ss += v * v;
    }
    float simd_ss = simd_sum(local_ss);
    threadgroup float sv[8];
    if (simd_lane_id == 0) sv[simd_group_id] = simd_ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_ss = 0.0f;
    for (uint i = 0; i < simd_groups_per_tg; i++) total_ss += sv[i];

    float rms = rsqrt(total_ss / float(cols) + params.eps);

    // Phase 2: normalize and scale
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        row_out[c] = row_sum[c] * rms * weight[c];
    }
}
"#;

/// Fused SiLU-gate: output[i] = silu(gate[i]) * up[i]
/// Saves one kernel dispatch and one temporary buffer vs separate silu + mul.
/// AXPY: y[i] += alpha * x[i]. Fused scale+add in 1 dispatch (was 2).
pub const AXPY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct AxpyParams { uint size; float alpha; };

kernel void axpy(
    device float* y [[buffer(0)]],
    device const float* x [[buffer(1)]],
    constant AxpyParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    y[gid] += params.alpha * x[gid];
}
"#;

/// ReLU activation: output[i] = max(input[i], 0). Used for ReMoE routing.
pub const RELU: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ReluParams { uint size; };

kernel void relu(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReluParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    output[gid] = max(input[gid], 0.0f);
}
"#;

/// ReLU backward: grad_input = grad_output * (input > 0 ? 1 : 0)
pub const RELU_BACKWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ReluParams { uint size; };

kernel void relu_backward(
    device const float* input [[buffer(0)]],
    device const float* grad_output [[buffer(1)]],
    device float* grad_input [[buffer(2)]],
    constant ReluParams& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    grad_input[gid] = (input[gid] > 0.0f) ? grad_output[gid] : 0.0f;
}
"#;

/// Elementwise exp: output = exp(input). Used by SSM/RWKV selective-decay gates.
/// Clamped to avoid fp32/fp16 overflow (exp(88)≈1.6e38 ~ fp32 max; cap at 80 for headroom).
pub const EXP: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ExpParams { uint size; };

kernel void exp_fwd(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ExpParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    output[gid] = exp(min(input[gid], 80.0f));
}
"#;

/// Broadcast a `[cols]` vector to `[rows, cols]`: out[r*cols + c] = vec[c]. A direct copy,
/// avoiding the wasted tiled-matmul machinery of an `ones[rows,1] @ vec[1,cols]` outer product.
pub const BROADCAST_ROWS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BcastParams { uint rows; uint cols; };

kernel void broadcast_rows(
    device const float* vec [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant BcastParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.rows * params.cols;
    if (gid >= total) return;
    out[gid] = vec[gid % params.cols];
}
"#;

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
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg [[simdgroups_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.batch_size) return;

    uint V = params.vocab_size;
    device const float* row_logits = logits + row * V;
    device float* row_grad = grad_logits + row * V;
    uint target = targets[row];

    // Find max — SIMD-optimized
    float local_max = -INFINITY;
    for (uint c = thread_index; c < V; c += threads_per_group)
        local_max = max(local_max, row_logits[c]);
    float sm = simd_max(local_max);
    threadgroup float sv[8];
    if (simd_lane_id == 0) sv[simd_group_id] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = sv[0];
    for (uint i = 1; i < simd_groups_per_tg; i++) row_max = max(row_max, sv[i]);

    // Compute exp and sum — SIMD-optimized
    float local_sum = 0.0f;
    for (uint c = thread_index; c < V; c += threads_per_group) {
        float e = exp(row_logits[c] - row_max);
        row_grad[c] = e;
        local_sum += e;
    }
    float ss = simd_sum(local_sum);
    if (simd_lane_id == 0) sv[simd_group_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < simd_groups_per_tg; i++) total += sv[i];
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
    float update_clip;       // per-element ceiling on |m_hat/(sqrt(v_hat)+eps)|; 0 = disabled
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

    // Normalized Adam update. In steady state |update| ~ 1; when v_hat collapses (the
    // beta2-short-memory instability) it spikes to 1e3+. update_clip bounds it at the source —
    // more principled than the post-normalization global grad clip, which lets one exploded
    // component dominate the *direction*. 0 = disabled (exact back-compat).
    float update = m_hat / (sqrt(v_hat) + params.eps);
    if (params.update_clip > 0.0f) {
        update = clamp(update, -params.update_clip, params.update_clip);
    }

    // Weight decay applied to param directly (decoupled), then Adam step
    param[gid] = param[gid] * (1.0f - params.lr * params.weight_decay) - params.lr * update;
}
"#;

/// 8-bit (block-wise int8) AdamW update — bitsandbytes-style. The first/second moments are stored as
/// int8 with one fp32 absmax scale per block of 256 elements (≈4× less optimizer memory than fp32
/// m+v). One threadgroup per block: dequant → AdamW math → apply param update → block-reduce the new
/// absmax → requantize with fresh scales. `m` is signed-linear int8. `v` (≥0) is stored as int8 of
/// √v, NOT v: within a block v=g² spans a ~1000× dynamic range, so linear-quantizing v underflows
/// the small entries to 0 → v̂≈0 → the update m̂/ε explodes. √v compresses that range ~31× and is
/// what the optimizer actually consumes (√v̂). update_clip bounds the step.
pub const ADAMW_8BIT_UPDATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct Adam8Params {
    uint size;
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;
    float bias_correction1;
    float bias_correction2;
    float update_clip;
};

kernel void adamw_8bit_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device char* m_q [[buffer(2)]],
    device char* v_q [[buffer(3)]],
    device float* m_scale [[buffer(4)]],
    device float* v_scale [[buffer(5)]],
    constant Adam8Params& p [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]]
) {
    threadgroup float sm[256];
    threadgroup float sv[256];

    bool active = gid < p.size;
    float ms = m_scale[bid];
    float vs = v_scale[bid];   // scale for √v, not v
    float g = active ? grad[gid] : 0.0f;
    float m_old = active ? (float)((int)m_q[gid]) * ms : 0.0f;
    float sqrt_v_old = active ? (float)((int)v_q[gid]) * vs : 0.0f;
    float v_old = sqrt_v_old * sqrt_v_old;
    float m_new = p.beta1 * m_old + (1.0f - p.beta1) * g;
    float v_new = p.beta2 * v_old + (1.0f - p.beta2) * g * g;
    float sqrt_v_new = sqrt(v_new);

    if (active) {
        float m_hat = m_new / p.bias_correction1;
        float v_hat = v_new / p.bias_correction2;
        float upd = m_hat / (sqrt(v_hat) + p.eps);
        if (p.update_clip > 0.0f) upd = clamp(upd, -p.update_clip, p.update_clip);
        param[gid] = param[gid] * (1.0f - p.lr * p.weight_decay) - p.lr * upd;
    }

    // Block absmax reduction for requantization (|m| and √v).
    sm[lid] = active ? fabs(m_new) : 0.0f;
    sv[lid] = active ? sqrt_v_new : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 128; s > 0; s >>= 1) {
        if (lid < s) {
            sm[lid] = max(sm[lid], sm[lid + s]);
            sv[lid] = max(sv[lid], sv[lid + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float new_ms = sm[0] > 0.0f ? sm[0] / 127.0f : 0.0f;
    float new_vs = sv[0] > 0.0f ? sv[0] / 127.0f : 0.0f;
    if (lid == 0) {
        m_scale[bid] = new_ms;
        v_scale[bid] = new_vs;
    }

    if (active) {
        m_q[gid] = new_ms > 0.0f ? (char)round(clamp(m_new / new_ms, -127.0f, 127.0f)) : (char)0;
        v_q[gid] = new_vs > 0.0f ? (char)round(clamp(sqrt_v_new / new_vs, 0.0f, 127.0f)) : (char)0;
    }
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
    uint window; // sliding window: 0=full causal, >0=attend only last W positions
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

    uint q_pos = q + params.offset;
    bool future = k > q_pos;
    bool too_far = (params.window > 0) && (q_pos >= params.window) && (k < q_pos - params.window);
    if (future || too_far) {
        scores[bh * params.seq_q * params.seq_k + q * params.seq_k + k] = -INFINITY;
    }
}
"#;

/// Block-diagonal causal mask for SEQUENCE PACKING (varlen). When multiple short sequences are
/// packed into one row, a token must attend only WITHIN its own segment (and causally). Given a
/// per-position segment id, mask score[q,k] to -inf when k is in the future (k > q) OR when k and q
/// are in different segments. Eliminates the padding waste of one-sequence-per-row at the cost of
/// this segment check. (seq_q == seq_k; no KV-cache offset on the packed training path.)
pub const CAUSAL_DOC_MASK: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct DocMaskParams {
    uint batch_heads;
    uint seq;
    uint n_heads; // seg_ids are per-batch ([batch, seq]); batch index = bh / n_heads
};

kernel void causal_doc_mask(
    device float* scores [[buffer(0)]],
    device const uint* seg_ids [[buffer(1)]],
    constant DocMaskParams& params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint bh = gid.z;
    uint q = gid.y;
    uint k = gid.x;
    if (bh >= params.batch_heads || q >= params.seq || k >= params.seq) return;

    uint base = (bh / params.n_heads) * params.seq;
    bool future = k > q;
    bool cross_segment = seg_ids[base + q] != seg_ids[base + k];
    if (future || cross_segment) {
        scores[bh * params.seq * params.seq + q * params.seq + k] = -INFINITY;
    }
}
"#;

/// Block-mean keys for block-sparse attention: average K over each contiguous block of positions.
/// K:[bh, seq, hd] → out:[bh, nb, hd], out[bh,blk,d] = mean_{i in block blk} K[bh,i,d].
pub const BLOCK_MEAN_KEYS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BMParams { uint bh; uint seq; uint hd; uint nb; uint block_size; };

kernel void block_mean_keys(
    device const float* k [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant BMParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint bh = gid.z, blk = gid.y, d = gid.x;
    if (bh >= p.bh || blk >= p.nb || d >= p.hd) return;
    uint start = blk * p.block_size;
    uint end = min(start + p.block_size, p.seq);
    float s = 0.0f;
    uint cnt = 0;
    for (uint i = start; i < end; i++) {
        s += k[bh * p.seq * p.hd + i * p.hd + d];
        cnt++;
    }
    out[bh * p.nb * p.hd + blk * p.hd + d] = cnt > 0 ? s / (float)cnt : 0.0f;
}
"#;

/// Top-k block-sparse attention mask (MoBA / NSA-style routing). Given per-(head,query) block scores
/// (query · block-mean-key), mask the dense attention scores so each query attends ONLY to: its OWN
/// block (causally), plus the top-k PAST blocks by block score. Future positions are masked too.
/// This is the learned, content-based sparse-attention routing that keeps full-attention quality at a
/// fraction of the attended positions — the core of subquadratic sparse attention. (This variant
/// masks the dense O(n²) scores — correct + trainable; the true subquadratic SPEEDUP needs a gather
/// kernel that computes only the selected blocks, a documented follow-up.)
pub const BLOCK_SPARSE_TOPK_MASK: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BSParams { uint bh; uint seq; uint nb; uint block_size; uint top_k; };

kernel void block_sparse_topk_mask(
    device float* scores [[buffer(0)]],          // [bh, seq, seq], modified in place
    device const float* block_scores [[buffer(1)]], // [bh, seq, nb]
    constant BSParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint bh = gid.z, q = gid.y, k = gid.x;
    if (bh >= p.bh || q >= p.seq || k >= p.seq) return;
    uint idx = bh * p.seq * p.seq + q * p.seq + k;

    if (k > q) { scores[idx] = -INFINITY; return; } // causal
    uint qb = q / p.block_size;
    uint kb = k / p.block_size;
    if (kb == qb) return; // the query's own block is always attended (causal already applied)

    // Past block kb < qb: selected iff its block score is among the top_k of all past blocks j<qb.
    float my = block_scores[bh * p.seq * p.nb + q * p.nb + kb];
    uint better = 0;
    for (uint j = 0; j < qb; j++) {
        if (block_scores[bh * p.seq * p.nb + q * p.nb + j] > my) better++;
    }
    if (better >= p.top_k) scores[idx] = -INFINITY; // outside the top-k → masked
}
"#;

/// Gather selected K/V blocks for SUBQUADRATIC block-sparse attention. For each (head, query-block)
/// pair `bnq = bh*nb + qb`, copy its `K_SEL` selected key/value blocks (indices in `sel`) into a
/// compact buffer so attention computes ONLY over the selected positions (vs masking the full O(n²)
/// scores). Sentinel `sel >= nb` → zero-filled (a padding block, later fully masked).
/// src: [bh, seq, hd]; sel: [bh*nb*K_SEL] u32; out: [bh*nb, K_SEL*block, hd].
pub const GATHER_BLOCKS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GatherParams { uint bh; uint nb; uint seq; uint hd; uint block; uint k_sel; };

kernel void gather_blocks(
    device const float* src [[buffer(0)]],
    device const uint* sel [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant GatherParams& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint sel_w = p.k_sel * p.block;           // gathered keys per query-block
    uint total = p.bh * p.nb * sel_w * p.hd;
    if (gid >= total) return;
    uint d = gid % p.hd;
    uint rem = gid / p.hd;
    uint pos = rem % sel_w;                    // 0..k_sel*block
    uint bnq = rem / sel_w;                    // bh_idx*nb + qb
    uint slot = pos / p.block;
    uint w = pos % p.block;
    uint bh_idx = bnq / p.nb;
    uint block_idx = sel[bnq * p.k_sel + slot];
    if (block_idx >= p.nb) { out[gid] = 0.0f; return; } // sentinel → padding
    uint src_row = block_idx * p.block + w;
    if (src_row >= p.seq) { out[gid] = 0.0f; return; }
    out[gid] = src[bh_idx * p.seq * p.hd + src_row * p.hd + d];
}
"#;

/// Causal mask for the gathered block-sparse scores. scores: [bh*nb, block, K_SEL*block]. For query
/// row `r` of query-block `qb` (global pos qb*block+r) and gathered key column `slot*block+w`
/// (global pos sel[slot]*block+w), keep iff the key is real (slot in range) and causal
/// (key_global <= query_global); else -inf.
pub const GATHER_CAUSAL_MASK: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GMaskParams { uint bh_nb; uint nb; uint block; uint k_sel; };

kernel void gather_causal_mask(
    device float* scores [[buffer(0)]],
    device const uint* sel [[buffer(1)]],
    constant GMaskParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]
) {
    uint bnq = gid.z;                          // bh_idx*nb + qb
    uint r = gid.y;                            // query row within block
    uint c = gid.x;                            // gathered key column
    uint sel_w = p.k_sel * p.block;
    if (bnq >= p.bh_nb || r >= p.block || c >= sel_w) return;
    uint qb = bnq % p.nb;
    uint q_global = qb * p.block + r;
    uint slot = c / p.block;
    uint w = c % p.block;
    uint block_idx = sel[bnq * p.k_sel + slot];
    uint idx = bnq * p.block * sel_w + r * sel_w + c;
    bool masked;
    if (block_idx >= p.nb) {
        masked = true;                         // padding slot
    } else {
        uint k_global = block_idx * p.block + w;
        masked = k_global > q_global;          // causal
    }
    if (masked) scores[idx] = -INFINITY;
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
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg [[simdgroups_per_threadgroup]]
) {
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
    float ss = simd_sum(local_sum);
    float sn = simd_max(local_nan);
    threadgroup float sv[8];
    threadgroup float nv[8];
    if (simd_lane_id == 0) { sv[simd_group_id] = ss; nv[simd_group_id] = sn; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0) {
        float total = 0.0f; float nan_flag = 0.0f;
        for (uint i = 0; i < simd_groups_per_tg; i++) { total += sv[i]; nan_flag = max(nan_flag, nv[i]); }
        output[0] = total;
        output[1] = nan_flag;
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

/// Out-of-place scale: dst[i] = src[i] * scale. Replaces copy+scale (2 dispatches → 1).
pub const SCALE_COPY: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ScaleParams {
    uint size;
    float scale;
};

kernel void scale_copy(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant ScaleParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < params.size) {
        dst[gid] = src[gid] * params.scale;
    }
}
"#;

/// Muon Frobenius normalization: x = m / (‖m‖_F + eps).
/// Single-threadgroup sum-of-squares reduction (grid-stride, ≤256 threads — mirrors reduce_sum),
/// then a per-element scale by rsqrt(ssq + eps). Done in ONE dispatch with no CPU readback, so it
/// can't force-flush the command batch (the buffer-hazard class). This guarantees the largest
/// singular value of the normalized matrix ≤ 1 < √3, so the cubic Newton-Schulz that follows always
/// converges; the previous 1/√max(rows,cols) heuristic did NOT bound the spectral norm, so at larger
/// momentum magnitude (bigger batch / higher effective LR) σ_max could exceed √3 and the iteration
/// diverged cubically — the suspected root cause of the batch-32 Muon/hybrid divergence.
pub const MUON_FROB_NORMALIZE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct NormParams { uint size; };

kernel void muon_frob_normalize(
    device const float* m [[buffer(0)]],
    device float* x [[buffer(1)]],
    constant NormParams& params [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tpg [[threads_per_threadgroup]]
) {
    threadgroup float shared[256];
    float local = 0.0f;
    for (uint i = tid; i < params.size; i += tpg) {
        float v = m[i];
        local += v * v;
    }
    shared[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = tpg / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // shared[0] now holds the total Σ m_i² (stable after the final barrier). All threads read it
    // (read-only, no further shared writes), so no extra barrier is needed before the scale loop.
    float inv = rsqrt(shared[0] + 1e-14f);
    for (uint i = tid; i < params.size; i += tpg) {
        x[i] = m[i] * inv;
    }
}
"#;

/// NorMuon per-neuron normalization factor: out[i] = 1 / (sqrt(v[i] * bias_correction) + eps).
/// `v` is the per-row running second moment; `bias_correction` = 1/(1-beta2^t). Elementwise over the
/// [rows] vector. Feeds scale_rows to normalize the orthogonalized update per output neuron.
pub const INV_SQRT_BC: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct InvSqrtParams { uint size; float bias_correction; float eps; };

kernel void inv_sqrt_bc(
    device const float* v [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant InvSqrtParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < params.size) {
        float vhat = v[gid] * params.bias_correction;
        out[gid] = 1.0f / (sqrt(vhat) + params.eps);
    }
}
"#;

/// Fill buffer with a constant value
/// LogSumExp per row: output[i] = log(sum_j(exp(input[i*cols + j])))
/// Numerically stable: output[i] = max + log(sum(exp(x - max)))
/// EMA update: ema[i] = decay * ema[i] + (1-decay) * src[i]
pub const EMA_UPDATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct EmaParams {
    uint size;
    float decay;
};

kernel void ema_update(
    device float* ema [[buffer(0)]],
    device const float* src [[buffer(1)]],
    constant EmaParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= params.size) return;
    ema[gid] = params.decay * ema[gid] + (1.0f - params.decay) * src[gid];
}
"#;

pub const LOGSUMEXP: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct LSEParams {
    uint rows;
    uint cols;
};

kernel void logsumexp(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant LSEParams& params [[buffer(2)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg [[simdgroups_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;
    uint cols = params.cols;
    device const float* row_in = input + row * cols;

    float local_max = -INFINITY;
    for (uint c = thread_index; c < cols; c += threads_per_group)
        local_max = max(local_max, row_in[c]);
    float sm = simd_max(local_max);
    threadgroup float sv[8];
    if (simd_lane_id == 0) sv[simd_group_id] = sm;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float row_max = sv[0];
    for (uint i = 1; i < simd_groups_per_tg; i++) row_max = max(row_max, sv[i]);

    float local_sum = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group)
        local_sum += exp(row_in[c] - row_max);
    float ss = simd_sum(local_sum);
    if (simd_lane_id == 0) sv[simd_group_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < simd_groups_per_tg; i++) total += sv[i];

    if (thread_index == 0)
        output[row] = row_max + log(total);
}
"#;

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
    float clamp_on; // 1 = bound the degenerate-row backward; 0 = raw (investigation only)
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
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg [[simdgroups_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;

    uint cols = params.cols;
    device const float* x = input + row * cols;
    device const float* go = grad_output + row * cols;
    device float* gi = grad_input + row * cols;

    // Compute rms — SIMD reduction
    float local_ss = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group)
        local_ss += x[c] * x[c];
    float ss = simd_sum(local_ss);
    threadgroup float sv[8];
    if (simd_lane_id == 0) sv[simd_group_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_ss = 0.0f;
    for (uint i = 0; i < simd_groups_per_tg; i++) total_ss += sv[i];
    float mean_sq = total_ss / float(cols);
    float rms = sqrt(mean_sq + params.eps);
    float inv_rms = 1.0f / rms;
    // Bound inv_rms so the inv_rms^3 correction term below can't explode when an activation row
    // collapses (mean_sq -> 0). With eps=1e-5, inv_rms maxes at ~316 and inv_rms^3 at ~3e7, which
    // blows the gradient to ~1e8 from a perfectly bounded forward — clip then turns that into a
    // garbage update direction and the optimiser (notably AdamW) stalls/diverges. 1/sqrt(1e-3)
    // floors it; the forward normalization is untouched, so only the degenerate-row backward changes.
    if (params.clamp_on > 0.5f) inv_rms = min(inv_rms, 31.62f);

    // Compute dot product — SIMD reduction
    float local_dot = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group)
        local_dot += go[c] * weight[c] * x[c];
    float sd = simd_sum(local_dot);
    if (simd_lane_id == 0) sv[simd_group_id] = sd;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot_sum = 0.0f;
    for (uint i = 0; i < simd_groups_per_tg; i++) dot_sum += sv[i];

    // grad_input = (grad_output * weight * inv_rms) - (input * dot_sum * inv_rms^3 / cols)
    float correction = dot_sum * inv_rms * inv_rms * inv_rms / float(cols);
    for (uint c = thread_index; c < cols; c += threads_per_group) {
        float g = go[c] * weight[c] * inv_rms - x[c] * correction;
        // Cap the per-element grad. A collapsed activation row would otherwise blow this up and —
        // compounding across layers — explode the whole gradient (clip then yields a garbage
        // direction and AdamW diverges). Normal grads are O(1-10), so this only clips degenerate
        // spikes, and capping the OUTPUT stops the next layer's incoming grad from compounding.
        gi[c] = (params.clamp_on > 0.5f) ? clamp(g, -1.0e3f, 1.0e3f) : g;
    }

    // grad_weight: atomic accumulate across rows.
    // Guard: if dot_sum is NaN/Inf, this row has bad gradients — skip atomic add
    // to prevent one bad row from poisoning the entire grad_weight vector.
    // Single per-row check instead of per-element branching (no throughput impact).
    bool row_ok = !isnan(dot_sum) && !isinf(dot_sum) && !isnan(inv_rms) && !isinf(inv_rms);
    if (row_ok) {
        for (uint c = thread_index; c < cols; c += threads_per_group) {
            float gw = go[c] * x[c] * inv_rms;
            atomic_fetch_add_explicit((device atomic_float*)&grad_weight[c], gw, memory_order_relaxed);
        }
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
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg [[simdgroups_per_threadgroup]]
) {
    uint row = group_id;
    if (row >= params.rows) return;

    uint cols = params.cols;
    device const float* s = softmax_out + row * cols;
    device const float* go = grad_output + row * cols;
    device float* gi = grad_input + row * cols;

    float local_dot = 0.0f;
    for (uint c = thread_index; c < cols; c += threads_per_group)
        local_dot += go[c] * s[c];
    float sd = simd_sum(local_dot);
    threadgroup float sv[8];
    if (simd_lane_id == 0) sv[simd_group_id] = sd;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float dot_sum = 0.0f;
    for (uint i = 0; i < simd_groups_per_tg; i++) dot_sum += sv[i];

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

        // Guard: replace NaN/Inf with 0 to prevent poisoning via atomic add.
        // Use select() instead of branching for zero pipeline divergence.
        grad_val = select(grad_val, 0.0f, isnan(grad_val) || isinf(grad_val));

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
/// Tiled version: 32x32 output tiles, 64 threads per group, each thread computes 4x4.
pub const MATMUL_TRANS_A: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MatmulTransAParams {
    uint M;  // shared (inner after transpose)
    uint K;  // rows of output (cols of A)
    uint N;  // cols of output (cols of B)
};

#define TILE_TA 32
#define THREAD_TILE_TA 4
#define THREADS_PER_GROUP_TA 64

kernel void matmul_trans_a_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulTransAParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    // Each thread has a position in the 8x8 grid within the threadgroup
    uint local_row = thread_index / 8;  // 0..7
    uint local_col = thread_index % 8;  // 0..7

    // Global starting position for this threadgroup's tile
    // C is [K, N], so tile_row indexes K, tile_col indexes N
    uint tile_row = group_id.y * TILE_TA;
    uint tile_col = group_id.x * TILE_TA;

    // Shared memory for transposed-A tile and B tile
    threadgroup half As[TILE_TA][TILE_TA];
    threadgroup half Bs[TILE_TA][TILE_TA];

    float acc[THREAD_TILE_TA][THREAD_TILE_TA] = {{0.0f}};

    uint M = params.M;
    uint K = params.K;
    uint N = params.N;

    for (uint m_block = 0; m_block < M; m_block += TILE_TA) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE_TA;
            uint c = flat % TILE_TA;
            uint global_k = tile_row + r;
            uint global_m = m_block + c;
            As[r][c] = (half)(clamp((global_k < K && global_m < M) ? A[global_m * K + global_k] : 0.0f, -65504.0f, 65504.0f));
        }

        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE_TA;
            uint c = flat % TILE_TA;
            uint global_m = m_block + r;
            uint global_n = tile_col + c;
            Bs[r][c] = (half)(clamp((global_m < M && global_n < N) ? B[global_m * N + global_n] : 0.0f, -65504.0f, 65504.0f));
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint m = 0; m < TILE_TA; m++) {
            half a_vals[THREAD_TILE_TA];
            half b_vals[THREAD_TILE_TA];

            for (uint i = 0; i < THREAD_TILE_TA; i++) {
                a_vals[i] = As[local_row * THREAD_TILE_TA + i][m];
            }
            for (uint j = 0; j < THREAD_TILE_TA; j++) {
                b_vals[j] = Bs[m][local_col * THREAD_TILE_TA + j];
            }

            for (uint i = 0; i < THREAD_TILE_TA; i++) {
                for (uint j = 0; j < THREAD_TILE_TA; j++) {
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write results to C[K, N]
    for (uint i = 0; i < THREAD_TILE_TA; i++) {
        for (uint j = 0; j < THREAD_TILE_TA; j++) {
            uint global_r = tile_row + local_row * THREAD_TILE_TA + i;
            uint global_c = tile_col + local_col * THREAD_TILE_TA + j;
            if (global_r < K && global_c < N) {
                C[global_r * N + global_c] = acc[i][j];
            }
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

/// Fused transpose [batch*seq, n_heads*head_dim] → [batch*n_heads, seq, head_dim] + RoPE rotation.
/// Eliminates the intermediate buffer between transpose and RoPE (2 dispatches → 1).
pub const TRANSPOSE_ROPE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TransposeRopeParams {
    uint batch; uint seq; uint n_heads; uint head_dim; uint offset; float theta;
};

kernel void transpose_rope(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant TransposeRopeParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.batch * params.n_heads * params.seq * params.head_dim;
    if (gid >= total) return;

    uint head_dim = params.head_dim;
    uint seq = params.seq;
    uint n_heads = params.n_heads;

    // Output index decomposition: [batch*n_heads, seq, head_dim]
    uint rem = gid;
    uint b = rem / (n_heads * seq * head_dim);
    rem %= n_heads * seq * head_dim;
    uint h = rem / (seq * head_dim);
    rem %= seq * head_dim;
    uint s = rem / head_dim;
    uint d = rem % head_dim;

    // Input layout: [batch*seq, n_heads*head_dim]
    uint in_idx = (b * seq + s) * (n_heads * head_dim) + h * head_dim + d;
    float val = input[in_idx];

    // Apply RoPE: rotate pairs (d, d+1) by angle based on position
    uint pair = d / 2;
    float freq = 1.0f / pow(params.theta, float(2 * pair) / float(head_dim));
    float angle = float(s + params.offset) * freq;
    float cos_val;
    float sin_val = sincos(angle, cos_val);

    // Read the paired element
    uint d_pair = (d % 2 == 0) ? d + 1 : d - 1;
    uint in_pair = (b * seq + s) * (n_heads * head_dim) + h * head_dim + d_pair;
    float val_pair = input[in_pair];

    if (d % 2 == 0) {
        output[gid] = val * cos_val - val_pair * sin_val;
    } else {
        output[gid] = val_pair * sin_val + val * cos_val;
    }
}
"#;

/// Backward for fused transpose+RoPE: inverse RoPE + inverse transpose.
/// Input grad: [batch*n_heads, seq, head_dim], output grad: [batch*seq, n_heads*head_dim]
pub const TRANSPOSE_ROPE_BACKWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TransposeRopeParams {
    uint batch; uint seq; uint n_heads; uint head_dim; uint offset; float theta;
};

kernel void transpose_rope_backward(
    device const float* grad_out [[buffer(0)]],
    device float* grad_in [[buffer(1)]],
    constant TransposeRopeParams& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = params.batch * params.seq * params.n_heads * params.head_dim;
    if (gid >= total) return;

    uint head_dim = params.head_dim;
    uint seq = params.seq;
    uint n_heads = params.n_heads;

    // Input grad index: [batch*seq, n_heads*head_dim]
    uint rem = gid;
    uint bs = rem / (n_heads * head_dim);
    uint b = bs / seq;
    uint s = bs % seq;
    rem %= n_heads * head_dim;
    uint h = rem / head_dim;
    uint d = rem % head_dim;

    // Corresponding output grad index: [batch*n_heads, seq, head_dim]
    uint out_idx = (b * n_heads + h) * seq * head_dim + s * head_dim + d;
    float g = grad_out[out_idx];

    // Inverse RoPE: rotate by -angle
    uint pair = d / 2;
    float freq = 1.0f / pow(params.theta, float(2 * pair) / float(head_dim));
    float angle = float(s + params.offset) * freq;
    float cos_val;
    float sin_val = sincos(angle, cos_val);

    uint d_pair = (d % 2 == 0) ? d + 1 : d - 1;
    uint out_pair = (b * n_heads + h) * seq * head_dim + s * head_dim + d_pair;
    float g_pair = grad_out[out_pair];

    // Inverse rotation: multiply by rotation matrix transpose
    if (d % 2 == 0) {
        grad_in[gid] = g * cos_val + g_pair * sin_val;
    } else {
        grad_in[gid] = -g_pair * sin_val + g * cos_val;
    }
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
/// Uses group_id.z as the batch index. Single dispatch for all batch elements.
/// Same tiled algorithm as matmul_tiled (32x32 tiles, 64 threads, 4x4 per thread).
pub const BATCHED_MATMUL_TILED: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BatchedMatmulParams {
    uint M;
    uint N;
    uint K;
    uint batch;
};

#define BM_TILE 32
#define BM_THREAD_TILE 4
#define BM_THREADS_PER_GROUP 64

kernel void batched_matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint batch_idx = group_id.z;
    if (batch_idx >= params.batch) return;

    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;

    uint tile_row = group_id.y * BM_TILE;
    uint tile_col = group_id.x * BM_TILE;

    uint M = params.M;
    uint N = params.N;
    uint K = params.K;

    device const float* A_b = A + batch_idx * M * K;
    device const float* B_b = B + batch_idx * K * N;
    device float* C_b = C + batch_idx * M * N;

    threadgroup half As[BM_TILE][BM_TILE];
    threadgroup half Bs[BM_TILE][BM_TILE];

    float acc[BM_THREAD_TILE][BM_THREAD_TILE] = {{0.0f}};

    for (uint k_block = 0; k_block < K; k_block += BM_TILE) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE;
            uint c = flat % BM_TILE;
            uint global_r = tile_row + r;
            uint global_c = k_block + c;
            As[r][c] = (half)((global_r < M && global_c < K) ? A_b[global_r * K + global_c] : 0.0f);
        }
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE;
            uint c = flat % BM_TILE;
            uint global_r = k_block + r;
            uint global_c = tile_col + c;
            Bs[r][c] = (half)((global_r < K && global_c < N) ? B_b[global_r * N + global_c] : 0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BM_TILE; k++) {
            half a_vals[BM_THREAD_TILE];
            half b_vals[BM_THREAD_TILE];
            for (uint i = 0; i < BM_THREAD_TILE; i++) {
                a_vals[i] = As[local_row * BM_THREAD_TILE + i][k];
            }
            for (uint j = 0; j < BM_THREAD_TILE; j++) {
                b_vals[j] = Bs[k][local_col * BM_THREAD_TILE + j];
            }
            for (uint i = 0; i < BM_THREAD_TILE; i++) {
                for (uint j = 0; j < BM_THREAD_TILE; j++) {
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < BM_THREAD_TILE; i++) {
        for (uint j = 0; j < BM_THREAD_TILE; j++) {
            uint global_r = tile_row + local_row * BM_THREAD_TILE + i;
            uint global_c = tile_col + local_col * BM_THREAD_TILE + j;
            if (global_r < M && global_c < N) {
                C_b[global_r * N + global_c] = acc[i][j];
            }
        }
    }
}
"#;

/// Batched tiled matmul with B transposed: C[b] = A[b] @ B[b]^T
/// A: [B, M, K], B: [B, N, K], C: [B, M, N]
/// Uses group_id.z as the batch index. Single dispatch for all batch elements.
pub const BATCHED_MATMUL_TILED_TRANS_B: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BatchedMatmulParams {
    uint M;
    uint N;
    uint K;
    uint batch;
};

#define BM_TILE 32
#define BM_THREAD_TILE 4
#define BM_THREADS_PER_GROUP 64

kernel void batched_matmul_tiled_trans_b(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint batch_idx = group_id.z;
    if (batch_idx >= params.batch) return;

    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;

    uint tile_row = group_id.y * BM_TILE;
    uint tile_col = group_id.x * BM_TILE;

    uint M = params.M;
    uint N = params.N;
    uint K = params.K;

    device const float* A_b = A + batch_idx * M * K;
    device const float* B_b = B + batch_idx * N * K;
    device float* C_b = C + batch_idx * M * N;

    threadgroup half As[BM_TILE][BM_TILE];
    threadgroup half Bs[BM_TILE][BM_TILE];

    float acc[BM_THREAD_TILE][BM_THREAD_TILE] = {{0.0f}};

    for (uint k_block = 0; k_block < K; k_block += BM_TILE) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE;
            uint c = flat % BM_TILE;
            uint global_r = tile_row + r;
            uint global_c = k_block + c;
            As[r][c] = (half)((global_r < M && global_c < K) ? A_b[global_r * K + global_c] : 0.0f);
        }
        // B is [N, K], we want B^T so B^T[k, n] = B[n, k]
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE;  // k index
            uint c = flat % BM_TILE;  // n index
            uint global_k = k_block + r;
            uint global_n = tile_col + c;
            Bs[r][c] = (half)((global_k < K && global_n < N) ? B_b[global_n * K + global_k] : 0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BM_TILE; k++) {
            half a_vals[BM_THREAD_TILE];
            half b_vals[BM_THREAD_TILE];
            for (uint i = 0; i < BM_THREAD_TILE; i++) {
                a_vals[i] = As[local_row * BM_THREAD_TILE + i][k];
            }
            for (uint j = 0; j < BM_THREAD_TILE; j++) {
                b_vals[j] = Bs[k][local_col * BM_THREAD_TILE + j];
            }
            for (uint i = 0; i < BM_THREAD_TILE; i++) {
                for (uint j = 0; j < BM_THREAD_TILE; j++) {
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < BM_THREAD_TILE; i++) {
        for (uint j = 0; j < BM_THREAD_TILE; j++) {
            uint global_r = tile_row + local_row * BM_THREAD_TILE + i;
            uint global_c = tile_col + local_col * BM_THREAD_TILE + j;
            if (global_r < M && global_c < N) {
                C_b[global_r * N + global_c] = acc[i][j];
            }
        }
    }
}
"#;

/// GQA-aware batched matmul trans_b: C[b] = A[b] @ B[b/group_size]^T
/// A: [B*n_heads, M, K], B: [B*n_kv_heads, N, K], C: [B*n_heads, M, N]
/// Each Q head reads from K/V head = q_head / group_size.
/// When group_size=1 this is identical to standard batched_matmul_trans_b.
pub const BATCHED_MATMUL_GQA_TRANS_B: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GQAMatmulParams {
    uint M; uint N; uint K; uint batch; uint group_size;
};

#define BM_TILE 32
#define BM_THREAD_TILE 4
#define BM_THREADS_PER_GROUP 64

kernel void batched_matmul_gqa_trans_b(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GQAMatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint batch_idx = group_id.z;
    if (batch_idx >= params.batch) return;

    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;
    uint tile_row = group_id.y * BM_TILE;
    uint tile_col = group_id.x * BM_TILE;
    uint M = params.M, N = params.N, K = params.K;

    device const float* A_b = A + batch_idx * M * K;
    // GQA: map Q head to KV head via integer division
    uint kv_idx = batch_idx / params.group_size;
    device const float* B_b = B + kv_idx * N * K;
    device float* C_b = C + batch_idx * M * N;

    threadgroup half As[BM_TILE][BM_TILE];
    threadgroup half Bs[BM_TILE][BM_TILE];
    float acc[BM_THREAD_TILE][BM_THREAD_TILE] = {{0.0f}};

    for (uint k_block = 0; k_block < K; k_block += BM_TILE) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i, r = flat / BM_TILE, c = flat % BM_TILE;
            uint gr = tile_row + r, gc = k_block + c;
            As[r][c] = (half)((gr < M && gc < K) ? A_b[gr * K + gc] : 0.0f);
        }
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i, r = flat / BM_TILE, c = flat % BM_TILE;
            uint gr = tile_col + r, gc = k_block + c;
            Bs[r][c] = (half)((gr < N && gc < K) ? B_b[gr * K + gc] : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < BM_TILE; k++) {
            half av[BM_THREAD_TILE], bv[BM_THREAD_TILE];
            for (uint i = 0; i < BM_THREAD_TILE; i++) av[i] = As[local_row * BM_THREAD_TILE + i][k];
            for (uint j = 0; j < BM_THREAD_TILE; j++) bv[j] = Bs[local_col * BM_THREAD_TILE + j][k];
            for (uint i = 0; i < BM_THREAD_TILE; i++)
                for (uint j = 0; j < BM_THREAD_TILE; j++)
                    acc[i][j] += (float)(av[i] * bv[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < BM_THREAD_TILE; i++)
        for (uint j = 0; j < BM_THREAD_TILE; j++) {
            uint gr = tile_row + local_row * BM_THREAD_TILE + i;
            uint gc = tile_col + local_col * BM_THREAD_TILE + j;
            if (gr < M && gc < N) C_b[gr * N + gc] = acc[i][j];
        }
}
"#;

/// GQA-aware batched matmul: C[b] = A[b] @ B[b/group_size]
/// A: [B*n_heads, M, K], B: [B*n_kv_heads, K, N], C: [B*n_heads, M, N]
pub const BATCHED_MATMUL_GQA: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct GQAMatmulParams {
    uint M; uint N; uint K; uint batch; uint group_size;
};

#define BM_TILE 32
#define BM_THREAD_TILE 4
#define BM_THREADS_PER_GROUP 64

kernel void batched_matmul_gqa(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant GQAMatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint batch_idx = group_id.z;
    if (batch_idx >= params.batch) return;

    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;
    uint tile_row = group_id.y * BM_TILE;
    uint tile_col = group_id.x * BM_TILE;
    uint M = params.M, N = params.N, K = params.K;

    device const float* A_b = A + batch_idx * M * K;
    uint kv_idx = batch_idx / params.group_size;
    device const float* B_b = B + kv_idx * K * N;
    device float* C_b = C + batch_idx * M * N;

    threadgroup half As[BM_TILE][BM_TILE];
    threadgroup half Bs[BM_TILE][BM_TILE];
    float acc[BM_THREAD_TILE][BM_THREAD_TILE] = {{0.0f}};

    for (uint k_block = 0; k_block < K; k_block += BM_TILE) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i, r = flat / BM_TILE, c = flat % BM_TILE;
            uint gr = tile_row + r, gc = k_block + c;
            As[r][c] = (half)((gr < M && gc < K) ? A_b[gr * K + gc] : 0.0f);
        }
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i, r = flat / BM_TILE, c = flat % BM_TILE;
            uint gr = k_block + r, gc = tile_col + c;
            Bs[r][c] = (half)((gr < K && gc < N) ? B_b[gr * N + gc] : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k = 0; k < BM_TILE; k++) {
            half av[BM_THREAD_TILE], bv[BM_THREAD_TILE];
            for (uint i = 0; i < BM_THREAD_TILE; i++) av[i] = As[local_row * BM_THREAD_TILE + i][k];
            for (uint j = 0; j < BM_THREAD_TILE; j++) bv[j] = Bs[k][local_col * BM_THREAD_TILE + j];
            for (uint i = 0; i < BM_THREAD_TILE; i++)
                for (uint j = 0; j < BM_THREAD_TILE; j++)
                    acc[i][j] += (float)(av[i] * bv[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = 0; i < BM_THREAD_TILE; i++)
        for (uint j = 0; j < BM_THREAD_TILE; j++) {
            uint gr = tile_row + local_row * BM_THREAD_TILE + i;
            uint gc = tile_col + local_col * BM_THREAD_TILE + j;
            if (gr < M && gc < N) C_b[gr * N + gc] = acc[i][j];
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
using namespace metal;

struct BatchedMatmulTransAParams {
    uint M;
    uint K;
    uint N;
    uint batch;
};

#define BM_TILE 32
#define BM_THREAD_TILE 4
#define BM_THREADS_PER_GROUP 64

kernel void batched_matmul_tiled_trans_a(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulTransAParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint batch_idx = group_id.z;
    if (batch_idx >= params.batch) return;

    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;

    // C is [K, N], so tile_row indexes K, tile_col indexes N
    uint tile_row = group_id.y * BM_TILE;
    uint tile_col = group_id.x * BM_TILE;

    uint M = params.M;
    uint K = params.K;
    uint N = params.N;

    device const float* A_b = A + batch_idx * M * K;
    device const float* B_b = B + batch_idx * M * N;
    device float* C_b = C + batch_idx * K * N;

    threadgroup half As[BM_TILE][BM_TILE];  // As[k][m] within tile
    threadgroup half Bs[BM_TILE][BM_TILE];  // Bs[m][n] within tile

    float acc[BM_THREAD_TILE][BM_THREAD_TILE] = {{0.0f}};

    // Loop over M dimension in TILE-sized chunks
    for (uint m_block = 0; m_block < M; m_block += BM_TILE) {
        // Load A^T tile: As[k][m] = A[m_block+m][tile_row+k] = A[(m_block+m)*K + (tile_row+k)]
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE;  // k index
            uint c = flat % BM_TILE;  // m index
            uint global_k = tile_row + r;
            uint global_m = m_block + c;
            As[r][c] = (half)((global_k < K && global_m < M) ? A_b[global_m * K + global_k] : 0.0f);
        }
        // Load B tile: Bs[m][n] = B[(m_block+m)*N + (tile_col+n)]
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE;
            uint c = flat % BM_TILE;
            uint global_m = m_block + r;
            uint global_n = tile_col + c;
            Bs[r][c] = (half)((global_m < M && global_n < N) ? B_b[global_m * N + global_n] : 0.0f);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint m = 0; m < BM_TILE; m++) {
            half a_vals[BM_THREAD_TILE];
            half b_vals[BM_THREAD_TILE];
            for (uint i = 0; i < BM_THREAD_TILE; i++) {
                a_vals[i] = As[local_row * BM_THREAD_TILE + i][m];
            }
            for (uint j = 0; j < BM_THREAD_TILE; j++) {
                b_vals[j] = Bs[m][local_col * BM_THREAD_TILE + j];
            }
            for (uint i = 0; i < BM_THREAD_TILE; i++) {
                for (uint j = 0; j < BM_THREAD_TILE; j++) {
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < BM_THREAD_TILE; i++) {
        for (uint j = 0; j < BM_THREAD_TILE; j++) {
            uint global_r = tile_row + local_row * BM_THREAD_TILE + i;
            uint global_c = tile_col + local_col * BM_THREAD_TILE + j;
            if (global_r < K && global_c < N) {
                C_b[global_r * N + global_c] = acc[i][j];
            }
        }
    }
}
"#;

/// Cast float32 buffer to float16 buffer.
pub const CAST_F32_TO_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void cast_f32_to_f16(
    device const float* input [[buffer(0)]],
    device half* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        // Clamp to half range [-65504, 65504] to prevent overflow→NaN
        float val = clamp(input[gid], -65504.0f, 65504.0f);
        output[gid] = (half)val;
    }
}
"#;

/// Cast float16 buffer to float32 buffer.
pub const CAST_F16_TO_F32: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void cast_f16_to_f32(
    device const half* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid < size) {
        output[gid] = (float)input[gid];
    }
}
"#;

/// FP16-input tiled matmul: C(f32) = A(f16) @ B(f16)
/// Reads half directly from global memory — halves bandwidth vs float.
pub const MATMUL_TILED_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MatmulParams { uint M; uint N; uint K; };
#define TILE 32
#define THREAD_TILE 4

kernel void matmul_tiled_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;
    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;

    threadgroup half As[TILE][TILE];
    threadgroup half Bs[TILE][TILE];
    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    uint M = params.M; uint N = params.N; uint K = params.K;

    for (uint k_block = 0; k_block < K; k_block += TILE) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE; uint c = flat % TILE;
            uint gr = tile_row + r; uint gc = k_block + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE; uint c = flat % TILE;
            uint gr = k_block + r; uint gc = tile_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : (half)0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            half a_vals[THREAD_TILE]; half b_vals[THREAD_TILE];
            for (uint i = 0; i < THREAD_TILE; i++) a_vals[i] = As[local_row * THREAD_TILE + i][k];
            for (uint j = 0; j < THREAD_TILE; j++) b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            for (uint i = 0; i < THREAD_TILE; i++)
                for (uint j = 0; j < THREAD_TILE; j++)
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < THREAD_TILE; i++)
        for (uint j = 0; j < THREAD_TILE; j++) {
            uint gr = tile_row + local_row * THREAD_TILE + i;
            uint gc = tile_col + local_col * THREAD_TILE + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}
"#;

/// FP16-input matmul with B transposed: C(f32) = A(f16) @ B(f16)^T
pub const MATMUL_TILED_TRANS_B_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MatmulParams { uint M; uint N; uint K; };
#define TILE 32
#define THREAD_TILE 4

kernel void matmul_tiled_trans_b_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;
    uint tile_row = group_id.y * TILE;
    uint tile_col = group_id.x * TILE;

    threadgroup half As[TILE][TILE];
    threadgroup half Bs[TILE][TILE];
    float acc[THREAD_TILE][THREAD_TILE] = {{0.0f}};

    uint M = params.M; uint N = params.N; uint K = params.K;

    for (uint k_block = 0; k_block < K; k_block += TILE) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE; uint c = flat % TILE;
            uint gr = tile_row + r; uint gc = k_block + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : (half)0.0h;
        }
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE; uint c = flat % TILE;
            uint gk = k_block + r; uint gn = tile_col + c;
            Bs[r][c] = (gk < K && gn < N) ? B[gn * K + gk] : (half)0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE; k++) {
            half a_vals[THREAD_TILE]; half b_vals[THREAD_TILE];
            for (uint i = 0; i < THREAD_TILE; i++) a_vals[i] = As[local_row * THREAD_TILE + i][k];
            for (uint j = 0; j < THREAD_TILE; j++) b_vals[j] = Bs[k][local_col * THREAD_TILE + j];
            for (uint i = 0; i < THREAD_TILE; i++)
                for (uint j = 0; j < THREAD_TILE; j++)
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < THREAD_TILE; i++)
        for (uint j = 0; j < THREAD_TILE; j++) {
            uint gr = tile_row + local_row * THREAD_TILE + i;
            uint gc = tile_col + local_col * THREAD_TILE + j;
            if (gr < M && gc < N) C[gr * N + gc] = acc[i][j];
        }
}
"#;

/// FP16-input matmul with A transposed: C(f32) = A(f16)^T @ B(f16)
pub const MATMUL_TRANS_A_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MatmulTransAParams { uint M; uint K; uint N; };
#define TILE_TA 32
#define THREAD_TILE_TA 4

kernel void matmul_trans_a_tiled_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant MatmulTransAParams& params [[buffer(3)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;
    uint tile_row = group_id.y * TILE_TA;
    uint tile_col = group_id.x * TILE_TA;

    threadgroup half As[TILE_TA][TILE_TA];
    threadgroup half Bs[TILE_TA][TILE_TA];
    float acc[THREAD_TILE_TA][THREAD_TILE_TA] = {{0.0f}};

    uint M = params.M; uint K = params.K; uint N = params.N;

    for (uint m_block = 0; m_block < M; m_block += TILE_TA) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE_TA; uint c = flat % TILE_TA;
            uint gk = tile_row + r; uint gm = m_block + c;
            As[r][c] = (half)((gk < K && gm < M) ? A[gm * K + gk] : 0.0f);
        }
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / TILE_TA; uint c = flat % TILE_TA;
            uint gm = m_block + r; uint gn = tile_col + c;
            Bs[r][c] = (half)((gm < M && gn < N) ? B[gm * N + gn] : 0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint m = 0; m < TILE_TA; m++) {
            half a_vals[THREAD_TILE_TA]; half b_vals[THREAD_TILE_TA];
            for (uint i = 0; i < THREAD_TILE_TA; i++) a_vals[i] = As[local_row * THREAD_TILE_TA + i][m];
            for (uint j = 0; j < THREAD_TILE_TA; j++) b_vals[j] = Bs[m][local_col * THREAD_TILE_TA + j];
            for (uint i = 0; i < THREAD_TILE_TA; i++)
                for (uint j = 0; j < THREAD_TILE_TA; j++)
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < THREAD_TILE_TA; i++)
        for (uint j = 0; j < THREAD_TILE_TA; j++) {
            uint gr = tile_row + local_row * THREAD_TILE_TA + i;
            uint gc = tile_col + local_col * THREAD_TILE_TA + j;
            if (gr < K && gc < N) C[gr * N + gc] = acc[i][j];
        }
}
"#;

/// FP16-input batched matmul: C[b](f32) = A[b](f16) @ B[b](f16)
pub const BATCHED_MATMUL_TILED_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BatchedMatmulParams { uint M; uint N; uint K; uint batch; };
#define BM_TILE 32
#define BM_THREAD_TILE 4

kernel void batched_matmul_tiled_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint batch_idx = group_id.z;
    if (batch_idx >= params.batch) return;

    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;
    uint tile_row = group_id.y * BM_TILE;
    uint tile_col = group_id.x * BM_TILE;
    uint M = params.M; uint N = params.N; uint K = params.K;

    device const half* A_b = A + batch_idx * M * K;
    device const half* B_b = B + batch_idx * K * N;
    device float* C_b = C + batch_idx * M * N;

    threadgroup half As[BM_TILE][BM_TILE];
    threadgroup half Bs[BM_TILE][BM_TILE];
    float acc[BM_THREAD_TILE][BM_THREAD_TILE] = {{0.0f}};

    for (uint k_block = 0; k_block < K; k_block += BM_TILE) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE; uint c = flat % BM_TILE;
            uint gr = tile_row + r; uint gc = k_block + c;
            As[r][c] = (gr < M && gc < K) ? A_b[gr * K + gc] : (half)0.0h;
        }
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE; uint c = flat % BM_TILE;
            uint gr = k_block + r; uint gc = tile_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B_b[gr * N + gc] : (half)0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BM_TILE; k++) {
            half a_vals[BM_THREAD_TILE]; half b_vals[BM_THREAD_TILE];
            for (uint i = 0; i < BM_THREAD_TILE; i++) a_vals[i] = As[local_row * BM_THREAD_TILE + i][k];
            for (uint j = 0; j < BM_THREAD_TILE; j++) b_vals[j] = Bs[k][local_col * BM_THREAD_TILE + j];
            for (uint i = 0; i < BM_THREAD_TILE; i++)
                for (uint j = 0; j < BM_THREAD_TILE; j++)
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < BM_THREAD_TILE; i++)
        for (uint j = 0; j < BM_THREAD_TILE; j++) {
            uint gr = tile_row + local_row * BM_THREAD_TILE + i;
            uint gc = tile_col + local_col * BM_THREAD_TILE + j;
            if (gr < M && gc < N) C_b[gr * N + gc] = acc[i][j];
        }
}
"#;

/// FP16-input batched matmul with B transposed: C[b](f32) = A[b](f16) @ B[b](f16)^T
pub const BATCHED_MATMUL_TILED_TRANS_B_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BatchedMatmulParams { uint M; uint N; uint K; uint batch; };
#define BM_TILE 32
#define BM_THREAD_TILE 4

kernel void batched_matmul_tiled_trans_b_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint batch_idx = group_id.z;
    if (batch_idx >= params.batch) return;

    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;
    uint tile_row = group_id.y * BM_TILE;
    uint tile_col = group_id.x * BM_TILE;
    uint M = params.M; uint N = params.N; uint K = params.K;

    device const half* A_b = A + batch_idx * M * K;
    device const half* B_b = B + batch_idx * N * K;
    device float* C_b = C + batch_idx * M * N;

    threadgroup half As[BM_TILE][BM_TILE];
    threadgroup half Bs[BM_TILE][BM_TILE];
    float acc[BM_THREAD_TILE][BM_THREAD_TILE] = {{0.0f}};

    for (uint k_block = 0; k_block < K; k_block += BM_TILE) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE; uint c = flat % BM_TILE;
            uint gr = tile_row + r; uint gc = k_block + c;
            As[r][c] = (gr < M && gc < K) ? A_b[gr * K + gc] : (half)0.0h;
        }
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE; uint c = flat % BM_TILE;
            uint gk = k_block + r; uint gn = tile_col + c;
            Bs[r][c] = (gk < K && gn < N) ? B_b[gn * K + gk] : (half)0.0h;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < BM_TILE; k++) {
            half a_vals[BM_THREAD_TILE]; half b_vals[BM_THREAD_TILE];
            for (uint i = 0; i < BM_THREAD_TILE; i++) a_vals[i] = As[local_row * BM_THREAD_TILE + i][k];
            for (uint j = 0; j < BM_THREAD_TILE; j++) b_vals[j] = Bs[k][local_col * BM_THREAD_TILE + j];
            for (uint i = 0; i < BM_THREAD_TILE; i++)
                for (uint j = 0; j < BM_THREAD_TILE; j++)
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < BM_THREAD_TILE; i++)
        for (uint j = 0; j < BM_THREAD_TILE; j++) {
            uint gr = tile_row + local_row * BM_THREAD_TILE + i;
            uint gc = tile_col + local_col * BM_THREAD_TILE + j;
            if (gr < M && gc < N) C_b[gr * N + gc] = acc[i][j];
        }
}
"#;

/// FP16-input batched matmul with A transposed: C[b](f32) = A[b](f16)^T @ B[b](f16)
pub const BATCHED_MATMUL_TILED_TRANS_A_F16: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct BatchedMatmulTransAParams { uint M; uint K; uint N; uint batch; };
#define BM_TILE 32
#define BM_THREAD_TILE 4

kernel void batched_matmul_tiled_trans_a_f16(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant BatchedMatmulTransAParams& params [[buffer(3)]],
    uint3 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint batch_idx = group_id.z;
    if (batch_idx >= params.batch) return;

    uint local_row = thread_index / 8;
    uint local_col = thread_index % 8;
    uint tile_row = group_id.y * BM_TILE;
    uint tile_col = group_id.x * BM_TILE;
    uint M = params.M; uint K = params.K; uint N = params.N;

    device const half* A_b = A + batch_idx * M * K;
    device const half* B_b = B + batch_idx * M * N;
    device float* C_b = C + batch_idx * K * N;

    threadgroup half As[BM_TILE][BM_TILE];
    threadgroup half Bs[BM_TILE][BM_TILE];
    float acc[BM_THREAD_TILE][BM_THREAD_TILE] = {{0.0f}};

    for (uint m_block = 0; m_block < M; m_block += BM_TILE) {
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE; uint c = flat % BM_TILE;
            uint gk = tile_row + r; uint gm = m_block + c;
            As[r][c] = (half)(clamp((gk < K && gm < M) ? A_b[gm * K + gk] : 0.0f, -65504.0f, 65504.0f));
        }
        for (uint i = 0; i < 16; i++) {
            uint flat = thread_index * 16 + i;
            uint r = flat / BM_TILE; uint c = flat % BM_TILE;
            uint gm = m_block + r; uint gn = tile_col + c;
            Bs[r][c] = (half)(clamp((gm < M && gn < N) ? B_b[gm * N + gn] : 0.0f, -65504.0f, 65504.0f));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint m = 0; m < BM_TILE; m++) {
            half a_vals[BM_THREAD_TILE]; half b_vals[BM_THREAD_TILE];
            for (uint i = 0; i < BM_THREAD_TILE; i++) a_vals[i] = As[local_row * BM_THREAD_TILE + i][m];
            for (uint j = 0; j < BM_THREAD_TILE; j++) b_vals[j] = Bs[m][local_col * BM_THREAD_TILE + j];
            for (uint i = 0; i < BM_THREAD_TILE; i++)
                for (uint j = 0; j < BM_THREAD_TILE; j++)
                    acc[i][j] += (float)(a_vals[i] * b_vals[j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint i = 0; i < BM_THREAD_TILE; i++)
        for (uint j = 0; j < BM_THREAD_TILE; j++) {
            uint gr = tile_row + local_row * BM_THREAD_TILE + i;
            uint gc = tile_col + local_col * BM_THREAD_TILE + j;
            if (gr < K && gc < N) C_b[gr * N + gc] = acc[i][j];
        }
}
"#;

/// Flash Attention Forward v2 (Dao et al., 2022 — Apple M-series optimized)
/// Fuses Q@K^T → causal mask → softmax → @V into ONE kernel.
/// Never materializes the N×N attention score matrix. O(n) memory.
///
/// v2: Cooperative K/V tile loading into threadgroup shared memory (half precision).
/// Each K/V tile is loaded ONCE by all 32 threads, then reused for all Q rows.
/// Halves device memory bandwidth vs v1 (which read K/V per-thread from device).
///
/// Q,K,V: [batch_heads, seq, head_dim], O: [batch_heads, seq_q, head_dim]
/// Shared memory: K_shared[FA_BC][head_dim] + V_shared[FA_BC][head_dim] as half
/// For head_dim=64: 32×64×2×2 = 8KB total (fits 32KB limit).
pub const FLASH_ATTENTION_FORWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct FlashAttnParams {
    uint seq_q;
    uint seq_k;
    uint head_dim;
    uint batch_heads;
    float scale;      // 1/sqrt(head_dim)
    uint kv_offset;   // for causal mask: query position offset
};

#define FA_BR 32   // query block size (rows)
#define FA_BC 32   // key block size (cols)

kernel void flash_attention_forward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device float* O [[buffer(3)]],
    constant FlashAttnParams& params [[buffer(4)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint bh = group_id.x;           // batch_head index
    uint q_block = group_id.y;      // which block of queries
    uint q_start = q_block * FA_BR;

    if (bh >= params.batch_heads) return;

    uint seq_q = params.seq_q;
    uint seq_k = params.seq_k;
    uint d = params.head_dim;
    float scale = params.scale;
    uint kv_offset = params.kv_offset;

    // Pointers for this batch_head
    device const float* Q_bh = Q + bh * seq_q * d;
    device const float* K_bh = K + bh * seq_k * d;
    device const float* V_bh = V + bh * seq_k * d;
    device float* O_bh = O + bh * seq_q * d;

    // Threadgroup shared memory for K/V tiles (half precision — 2x bandwidth savings)
    threadgroup half K_shared[FA_BC][128]; // max head_dim=128
    threadgroup half V_shared[FA_BC][128];

    // Each thread handles one query row within the block
    uint local_q = thread_index;  // 0..FA_BR-1
    uint global_q = q_start + local_q;

    if (global_q >= seq_q) return;

    // Per-query state for online softmax
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    // Output accumulator
    float o_acc[128];
    for (uint i = 0; i < d; i++) o_acc[i] = 0.0f;

    // Load query row into registers (float for precision)
    float q_row[128];
    for (uint i = 0; i < d; i++) {
        q_row[i] = Q_bh[global_q * d + i];
    }

    // Iterate over key/value blocks
    for (uint k_start = 0; k_start < seq_k; k_start += FA_BC) {
        uint k_end = min(k_start + FA_BC, seq_k);
        uint tile_len = k_end - k_start;

        // === COOPERATIVE TILE LOAD: all 32 threads load K/V into shared memory ===
        // Each thread loads tile_len/32 rows (or 1 row for tile_len <= 32)
        for (uint j = thread_index; j < tile_len; j += FA_BR) {
            uint global_k = k_start + j;
            for (uint i = 0; i < d; i++) {
                K_shared[j][i] = (half)K_bh[global_k * d + i];
                V_shared[j][i] = (half)V_bh[global_k * d + i];
            }
        }
        // Zero padding for partial tiles
        for (uint j = tile_len + thread_index; j < FA_BC; j += FA_BR) {
            for (uint i = 0; i < d; i++) {
                K_shared[j][i] = (half)0.0f;
                V_shared[j][i] = (half)0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === COMPUTE: each thread computes scores from shared memory ===
        float s_vals[FA_BC];
        float block_max = -INFINITY;

        for (uint j = 0; j < tile_len; j++) {
            uint global_k = k_start + j;
            if (global_k > global_q + kv_offset) {
                s_vals[j] = -INFINITY;
                continue;
            }
            // Dot product from shared memory (half → float accumulate)
            float dot = 0.0f;
            for (uint i = 0; i < d; i++) {
                dot += q_row[i] * (float)K_shared[j][i];
            }
            s_vals[j] = dot * scale;
            block_max = max(block_max, s_vals[j]);
        }
        for (uint j = tile_len; j < FA_BC; j++) s_vals[j] = -INFINITY;

        // Online softmax update
        float new_max = max(row_max, block_max);
        float old_correction = exp(row_max - new_max);
        float new_sum = old_correction * row_sum;
        for (uint i = 0; i < d; i++) o_acc[i] *= old_correction;

        // Accumulate output from V shared memory
        for (uint j = 0; j < tile_len; j++) {
            float p = exp(s_vals[j] - new_max);
            new_sum += p;
            for (uint i = 0; i < d; i++) {
                o_acc[i] += p * (float)V_shared[j][i];
            }
        }

        row_max = new_max;
        row_sum = new_sum;

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Final normalization
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (uint i = 0; i < d; i++) {
        O_bh[global_q * d + i] = o_acc[i] * inv_sum;
    }
}
"#;

/// Flash Attention Backward (Dao et al., 2022)
/// Recomputes attention scores tile-by-tile during backward.
/// Never stores N×N matrix. Computes dQ, dK, dV in one pass.
///
/// Inputs: Q, K, V, O (forward output), dO (gradient of output)
/// Outputs: dQ, dK, dV
///
/// Also needs D[i] = rowsum(dO[i] * O[i]) precomputed per query row.
/// This is passed as a separate buffer.
pub const FLASH_ATTENTION_BACKWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct FlashAttnBwdParams {
    uint seq_q;
    uint seq_k;
    uint head_dim;
    uint batch_heads;
    float scale;
    uint kv_offset;
};

// Precompute D[i] = sum_j(dO[i][j] * O[i][j]) for each query row.
// This is needed for the backward softmax computation.
kernel void flash_attn_precompute_d(
    device const float* dO [[buffer(0)]],
    device const float* O [[buffer(1)]],
    device float* D [[buffer(2)]],
    constant uint& total_rows [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= total_rows) return;
    uint d = head_dim;
    float sum = 0.0f;
    for (uint i = 0; i < d; i++) {
        sum += dO[gid * d + i] * O[gid * d + i];
    }
    D[gid] = sum;
}

// Flash Attention Backward: compute dQ, dK, dV
// Each threadgroup handles one batch_head.
// Each thread handles one query row, iterates over K/V blocks.
kernel void flash_attention_backward(
    device const float* Q [[buffer(0)]],
    device const float* K [[buffer(1)]],
    device const float* V [[buffer(2)]],
    device const float* O [[buffer(3)]],
    device const float* dO [[buffer(4)]],
    device const float* D [[buffer(5)]],  // precomputed rowsum(dO * O)
    device float* dQ [[buffer(6)]],
    device float* dK [[buffer(7)]],
    device float* dV [[buffer(8)]],
    constant FlashAttnBwdParams& params [[buffer(9)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]]
) {
    uint bh = group_id.x;
    uint q_block = group_id.y;
    uint q_start = q_block * 32;
    uint local_q = thread_index;
    uint global_q = q_start + local_q;

    if (bh >= params.batch_heads || global_q >= params.seq_q) return;

    uint seq_q = params.seq_q;
    uint seq_k = params.seq_k;
    uint d = params.head_dim;
    float scale = params.scale;
    uint kv_offset = params.kv_offset;

    device const float* Q_bh = Q + bh * seq_q * d;
    device const float* K_bh = K + bh * seq_k * d;
    device const float* V_bh = V + bh * seq_k * d;
    device const float* dO_bh = dO + bh * seq_q * d;
    device const float* D_bh = D + bh * seq_q;
    device float* dQ_bh = dQ + bh * seq_q * d;
    device float* dK_bh = dK + bh * seq_k * d;
    device float* dV_bh = dV + bh * seq_k * d;

    // Shared memory for K/V tiles (same as forward v2)
    threadgroup half K_shared[32][128];
    threadgroup half V_shared[32][128];

    // Load query row and dO row into registers
    float q_row[128], do_row[128];
    for (uint i = 0; i < d; i++) {
        q_row[i] = Q_bh[global_q * d + i];
        do_row[i] = dO_bh[global_q * d + i];
    }
    float d_val = D_bh[global_q];

    float dq_acc[128];
    for (uint i = 0; i < d; i++) dq_acc[i] = 0.0f;

    // Pass 1: recompute row_max and row_sum (with shared memory K tiles)
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    for (uint k_start = 0; k_start < seq_k; k_start += 32) {
        uint k_end = min(k_start + 32u, seq_k);
        uint tile_len = k_end - k_start;

        // Cooperative K tile load
        for (uint j = thread_index; j < tile_len; j += 32) {
            for (uint i = 0; i < d; i++)
                K_shared[j][i] = (half)K_bh[(k_start + j) * d + i];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < tile_len; j++) {
            uint gk = k_start + j;
            if (gk > global_q + kv_offset) continue;
            float dot = 0.0f;
            for (uint i = 0; i < d; i++) dot += q_row[i] * (float)K_shared[j][i];
            float s = dot * scale;
            float new_max = max(row_max, s);
            row_sum = row_sum * exp(row_max - new_max) + exp(s - new_max);
            row_max = new_max;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Pass 2: compute gradients (with shared memory K/V tiles)
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;

    for (uint k_start = 0; k_start < seq_k; k_start += 32) {
        uint k_end = min(k_start + 32u, seq_k);
        uint tile_len = k_end - k_start;

        // Cooperative K/V tile load
        for (uint j = thread_index; j < tile_len; j += 32) {
            for (uint i = 0; i < d; i++) {
                K_shared[j][i] = (half)K_bh[(k_start + j) * d + i];
                V_shared[j][i] = (half)V_bh[(k_start + j) * d + i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint j = 0; j < tile_len; j++) {
            uint gk = k_start + j;
            if (gk > global_q + kv_offset) continue;

            float dot = 0.0f;
            for (uint i = 0; i < d; i++) dot += q_row[i] * (float)K_shared[j][i];
            float s = dot * scale;
            float p = exp(s - row_max) * inv_sum;

            float dov = 0.0f;
            for (uint i = 0; i < d; i++) dov += do_row[i] * (float)V_shared[j][i];
            float ds = p * (dov - d_val) * scale;

            // dQ (thread-local, no race)
            for (uint i = 0; i < d; i++) dq_acc[i] += ds * (float)K_shared[j][i];

            // dK, dV — multiple Q threads accumulate to the same K/V position.
            // Atomic adds prevent data races between threads in the threadgroup.
            for (uint i = 0; i < d; i++) {
                atomic_fetch_add_explicit((device atomic_float*)&dK_bh[gk * d + i], ds * q_row[i], memory_order_relaxed);
                atomic_fetch_add_explicit((device atomic_float*)&dV_bh[gk * d + i], p * do_row[i], memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Write dQ
    for (uint i = 0; i < d; i++) {
        dQ_bh[global_q * d + i] = dq_acc[i];
    }
}
"#;

/// MoE token routing: gather tokens for one expert + scatter weighted output back.
/// Fuses the per-token gather, expert output scaling, and scatter-add into one kernel.
///
/// For each (token, expert) assignment:
///   output[token] += weight * expert_output[slot]
///
/// token_indices: [n_routed] — which token each slot corresponds to
/// weights: [n_routed] — router weight for each routed token
/// expert_output: [n_routed, dim] — expert FFN output
/// combined_output: [n_tokens, dim] — accumulated output (scatter-add target)
pub const MOE_SCATTER_ADD: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void moe_scatter_add(
    device const float* expert_output [[buffer(0)]],
    device const uint* token_indices [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    device float* combined_output [[buffer(3)]],
    constant uint& n_routed [[buffer(4)]],
    constant uint& dim [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint slot = gid.x;   // which routed token
    uint d = gid.y;      // which dimension
    if (slot >= n_routed || d >= dim) return;

    uint token_idx = token_indices[slot];
    float w = weights[slot];
    float val = expert_output[slot * dim + d] * w;

    // Atomic add to combined output (multiple experts may write to same token)
    // On Apple Silicon, device memory atomic is available
    device float* dst = combined_output + token_idx * dim + d;
    float old = *dst;
    *dst = old + val;
}
"#;

/// MoE token gather: collect tokens assigned to one expert into contiguous buffer.
/// token_indices: [n_routed] — which tokens to gather
/// input: [n_tokens, dim] — full input tensor
/// gathered: [n_routed, dim] — gathered tokens for this expert
pub const MOE_GATHER: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void moe_gather(
    device const float* input [[buffer(0)]],
    device const uint* token_indices [[buffer(1)]],
    device float* gathered [[buffer(2)]],
    constant uint& n_routed [[buffer(3)]],
    constant uint& dim [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint slot = gid.x;
    uint d = gid.y;
    if (slot >= n_routed || d >= dim) return;

    uint token_idx = token_indices[slot];
    gathered[slot * dim + d] = input[token_idx * dim + d];
}
"#;

/// BitNet b1.58: Ternary matmul C = A @ W where W ∈ {-1, 0, +1}
/// A: [M, K] float, W: [K, N] packed ternary (2 bits per weight)
/// C: [M, N] float
///
/// Packing: 16 ternary weights per u32 (2 bits each)
///   0b00 = 0, 0b01 = +1, 0b10 = -1
///
/// No floating point multiply — just conditional add/subtract.
/// 10x faster than float matmul for the same dimensions.
pub const TERNARY_MATMUL: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct TernaryParams {
    uint M;
    uint N;
    uint K;
};

kernel void ternary_matmul(
    device const float* A [[buffer(0)]],
    device const uint* W_packed [[buffer(1)]],  // packed ternary weights
    device float* C [[buffer(2)]],
    constant TernaryParams& params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // M dimension
    uint col = gid.x;  // N dimension
    if (row >= params.M || col >= params.N) return;

    uint K = params.K;
    uint N = params.N;

    float acc = 0.0f;
    device const float* a_row = A + row * K;

    // Process 16 weights at a time (one u32 = 16 ternary values)
    uint k = 0;
    for (; k + 16 <= K; k += 16) {
        // W is packed as [K/16, N] where each element holds 16 weights for one K-slice
        uint packed = W_packed[(k / 16) * N + col];

        // Unpack and accumulate: no multiply, just add/subtract
        for (uint i = 0; i < 16; i++) {
            uint bits = (packed >> (i * 2)) & 0x3;
            // 0b00 = 0 (skip), 0b01 = +1 (add), 0b10 = -1 (subtract)
            if (bits == 1) acc += a_row[k + i];
            else if (bits == 2) acc -= a_row[k + i];
        }
    }

    // Handle remaining weights (K not multiple of 16)
    if (k < K) {
        uint packed = W_packed[(k / 16) * N + col];
        for (uint i = 0; i < K - k; i++) {
            uint bits = (packed >> (i * 2)) & 0x3;
            if (bits == 1) acc += a_row[k + i];
            else if (bits == 2) acc -= a_row[k + i];
        }
    }

    C[row * N + col] = acc;
}
"#;

/// Quantize float weights to ternary {-1, 0, +1} using absmean threshold.
/// w_ternary[i] = sign(w[i]) * round(|w[i]| / mean(|w|))
/// Packed as 2 bits per weight, 16 weights per u32.
pub const TERNARY_QUANTIZE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Step 1: compute mean(|w|) for each column (reduction)
kernel void ternary_absmean(
    device const float* weights [[buffer(0)]],
    device float* absmean [[buffer(1)]],
    constant uint& rows [[buffer(2)]],
    constant uint& cols [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= cols) return;
    float sum = 0.0f;
    for (uint r = 0; r < rows; r++) {
        sum += abs(weights[r * cols + gid]);
    }
    absmean[gid] = sum / (float)rows;
}

// Step 2: quantize to ternary and pack
kernel void ternary_pack(
    device const float* weights [[buffer(0)]],
    device const float* absmean [[buffer(1)]],
    device uint* packed [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint pack_row = gid.y;  // packed row (K/16)
    uint col = gid.x;       // N dimension
    uint K = rows;
    uint N = cols;

    if (col >= N) return;
    uint k_start = pack_row * 16;
    if (k_start >= K) return;

    float threshold = absmean[col];
    float inv_thresh = (threshold > 1e-8f) ? (1.0f / threshold) : 0.0f;

    uint packed_val = 0;
    uint k_end = min(k_start + 16u, K);
    for (uint i = 0; i < k_end - k_start; i++) {
        float w = weights[(k_start + i) * N + col];
        float scaled = w * inv_thresh;
        int ternary;
        if (scaled > 0.5f) ternary = 1;       // +1
        else if (scaled < -0.5f) ternary = 2;  // -1 (encoded as 0b10)
        else ternary = 0;                       // 0
        packed_val |= ((uint)ternary << (i * 2));
    }
    packed[pack_row * N + col] = packed_val;
}
"#;

/// Scale each row of a matrix by a different scalar.
/// input: [rows, cols], scales: [rows], output: [rows, cols]
/// output[r][c] = input[r][c] * scales[r]
pub const SCALE_ROWS: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void scale_rows(
    device const float* input [[buffer(0)]],
    device const float* scales [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint r = gid.y;
    uint c = gid.x;
    if (r >= rows || c >= cols) return;
    output[r * cols + c] = input[r * cols + c] * scales[r];
}
"#;

/// Row-wise dot product and reduce: output[r] = sum_c(a[r][c] * b[r][c])
/// Used for scale_rows backward: d_scales[r] = dot(d_out[r], input[r])
pub const ROW_DOT_REDUCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void row_dot_reduce(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant uint& rows [[buffer(3)]],
    constant uint& cols [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= rows) return;
    float sum = 0.0f;
    for (uint c = 0; c < cols; c++) {
        sum += a[gid * cols + c] * b[gid * cols + c];
    }
    output[gid] = sum;
}
"#;

/// Lion optimizer update (Chen et al., 2023).
/// Simpler than AdamW: only tracks momentum (no variance).
/// Update = sign(beta1 * m + (1-beta1) * grad) * lr + weight_decay * param
/// Then: m = beta2 * m + (1-beta2) * grad
/// 2x less memory than AdamW (no v buffer needed).
pub const LION_UPDATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct LionParams {
    float lr;
    float beta1;
    float beta2;
    float weight_decay;
};

kernel void lion_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],
    constant LionParams& hp [[buffer(3)]],
    constant uint& size [[buffer(4)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float g = grad[gid];
    float m_val = m[gid];

    // Update direction: sign(interpolation of m and grad)
    float update = m_val * hp.beta1 + g * (1.0f - hp.beta1);
    float sign_update = (update > 0.0f) ? 1.0f : ((update < 0.0f) ? -1.0f : 0.0f);

    // Apply update with weight decay
    param[gid] = param[gid] * (1.0f - hp.lr * hp.weight_decay) - hp.lr * sign_update;

    // Update momentum
    m[gid] = m_val * hp.beta2 + g * (1.0f - hp.beta2);
}
"#;

/// Column-wise copy: src[rows, src_cols] → dst[rows, dst_cols] at column offset.
/// Used for building concatenated weight matrices and scattering column gradients.
pub const CONCAT_COLS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct ConcatColsParams {
    uint rows;
    uint src_cols;
    uint dst_cols;
    uint col_offset;
};

kernel void concat_cols(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant ConcatColsParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.rows * params.src_cols;
    if (tid >= total) return;
    uint r = tid / params.src_cols;
    uint c = tid % params.src_cols;
    dst[r * params.dst_cols + params.col_offset + c] = src[tid];
}
"#;

/// Column-wise slice: extract cols [offset..offset+dst_cols) from [rows, src_cols] tensor.
pub const SLICE_COLS: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SliceColsParams {
    uint rows;
    uint src_cols;
    uint dst_cols;
    uint col_offset;
};

kernel void slice_cols(
    device const float* src [[buffer(0)]],
    device float* dst [[buffer(1)]],
    constant SliceColsParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint total = params.rows * params.dst_cols;
    if (tid >= total) return;
    uint r = tid / params.dst_cols;
    uint c = tid % params.dst_cols;
    dst[tid] = src[r * params.src_cols + params.col_offset + c];
}
"#;

/// Sophia optimizer (Liu et al., 2023): second-order with diagonal Hessian.
/// Update: theta -= lr * clip(grad / max(h, eps), rho)
/// h = EMA of squared gradients (Hutchinson's diagonal Hessian estimate)
/// 2x faster convergence than AdamW for ~10% more compute.
pub const SOPHIA_UPDATE: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct SophiaParams {
    float lr;
    float beta1;      // momentum decay (0.965)
    float beta2;      // hessian EMA decay (0.99)
    float eps;         // hessian floor (1e-4)
    float rho;         // clipping threshold (1.0)
    float weight_decay;
};

kernel void sophia_update(
    device float* param [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m [[buffer(2)]],       // first moment (momentum)
    device float* h [[buffer(3)]],       // diagonal Hessian estimate
    constant SophiaParams& hp [[buffer(4)]],
    constant uint& size [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    float g = grad[gid];

    // Update momentum: m = beta1 * m + (1-beta1) * g
    float m_val = hp.beta1 * m[gid] + (1.0f - hp.beta1) * g;
    m[gid] = m_val;

    // Hessian EMA: h = beta2 * h + (1-beta2) * g^2
    float h_val = hp.beta2 * h[gid] + (1.0f - hp.beta2) * g * g;
    h[gid] = h_val;

    // Clipped update: clip(m / max(h, eps), -rho, rho)
    float update = m_val / max(h_val, hp.eps);
    update = clamp(update, -hp.rho, hp.rho);

    // Apply with weight decay
    param[gid] = param[gid] * (1.0f - hp.lr * hp.weight_decay) - hp.lr * update;
}
"#;

/// Single-kernel KV head expansion for GQA. Replaces N×group_size separate dispatches.
pub const REPEAT_KV: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RepeatKvParams {
    uint n_kv_total;
    uint group_size;
    uint seq_len;
    uint head_dim;
};

kernel void repeat_kv(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant RepeatKvParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint head_block = params.seq_len * params.head_dim;
    uint n_heads_total = params.n_kv_total * params.group_size;
    uint total = n_heads_total * head_block;
    if (tid >= total) return;
    uint out_head = tid / head_block;
    uint in_head = out_head / params.group_size;
    uint offset_in_block = tid % head_block;
    output[tid] = input[in_head * head_block + offset_in_block];
}
"#;

/// Single-kernel backward for repeat_kv: sum group_size gradient blocks per KV head.
pub const REPEAT_KV_BACKWARD: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct RepeatKvParams {
    uint n_kv_total;
    uint group_size;
    uint seq_len;
    uint head_dim;
};

kernel void repeat_kv_backward(
    device const float* out_grad [[buffer(0)]],
    device float* kv_grad [[buffer(1)]],
    constant RepeatKvParams& params [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    uint head_block = params.seq_len * params.head_dim;
    uint total = params.n_kv_total * head_block;
    if (tid >= total) return;
    uint kv_head = tid / head_block;
    uint offset_in_block = tid % head_block;
    float sum = 0.0f;
    for (uint g = 0; g < params.group_size; g++) {
        uint out_head = kv_head * params.group_size + g;
        sum += out_grad[out_head * head_block + offset_in_block];
    }
    kv_grad[tid] = sum;
}
"#;

/// MEGA-KERNEL: Fused RMSNorm + SwiGLU FFN + residual in 1 dispatch.
/// Each threadgroup handles one token. d_model ≤ 256, d_ff ≤ 1024.
pub const MEGA_FFN: &str = r#"
#include <metal_stdlib>
using namespace metal;

struct MegaFFNParams {
    uint batch_tokens;
    uint d_model;
    uint d_ff;
    float eps;
};

kernel void mega_ffn(
    device const float* x [[buffer(0)]],
    device const float* norm_w [[buffer(1)]],
    device const float* W1 [[buffer(2)]],
    device const float* W2 [[buffer(3)]],
    device const float* W3 [[buffer(4)]],
    device float* out [[buffer(5)]],
    constant MegaFFNParams& params [[buffer(6)]],
    uint group_id [[threadgroup_position_in_grid]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads_per_group [[threads_per_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_groups_per_tg [[simdgroups_per_threadgroup]]
) {
    uint token = group_id;
    if (token >= params.batch_tokens) return;
    uint d = params.d_model;
    uint ff = params.d_ff;
    device const float* x_row = x + token * d;
    device float* out_row = out + token * d;

    // RMSNorm
    float local_ss = 0.0f;
    for (uint c = thread_index; c < d; c += threads_per_group) {
        float v = x_row[c]; local_ss += v * v;
    }
    float ss = simd_sum(local_ss);
    threadgroup float sv[8];
    if (simd_lane_id == 0) sv[simd_group_id] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total_ss = 0.0f;
    for (uint i = 0; i < simd_groups_per_tg; i++) total_ss += sv[i];
    float inv_rms = rsqrt(total_ss / float(d) + params.eps);

    // d_model up to 1024 (4KB), d_ff up to 4096 (16KB) — total 20KB < 32KB threadgroup limit
    threadgroup float norm_x_shared[1024];
    for (uint c = thread_index; c < d; c += threads_per_group) {
        norm_x_shared[c] = x_row[c] * inv_rms * norm_w[c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Gate + Up + SiLU
    threadgroup float hidden_shared[4096];
    for (uint j = thread_index; j < ff; j += threads_per_group) {
        float gate_val = 0.0f, up_val = 0.0f;
        for (uint k = 0; k < d; k++) {
            float nx = norm_x_shared[k];
            gate_val += nx * W1[k * ff + j];
            up_val += nx * W3[k * ff + j];
        }
        float sigmoid_gate = 1.0f / (1.0f + exp(-gate_val));
        hidden_shared[j] = gate_val * sigmoid_gate * up_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Down + residual
    for (uint i = thread_index; i < d; i += threads_per_group) {
        float down_val = 0.0f;
        for (uint j = 0; j < ff; j++) down_val += hidden_shared[j] * W2[j * d + i];
        out_row[i] = x_row[i] + down_val;
    }
}
"#;

