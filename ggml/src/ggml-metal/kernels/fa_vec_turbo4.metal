#include "common.h"
#include "dequantize.h"
#include "turbo4.h"
#include "fa_vec_common.metal"

#define TURBO_SPARSE_V 1

kernel void kernel_flash_attn_ext_vec_pair_kq8_0_vturbo4_dk128_dv128(
        constant ggml_metal_kargs_flash_attn_ext_vec & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * sinks,
        device const char * pad,
        device       char * dst,
        threadgroup  half * shmem_f16 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    (void) sinks;

#define NWG (FC_flash_attn_ext_vec_nwg)
#define NSG (FC_flash_attn_ext_vec_nsg)

    const short iwg = tgpig[2];
    const uint head0 = 2*tgpig[1];
    const uint ikv2 = head0/(args.ne02/args.ne_12_2);
    const short tx = tiisg%8;
    const short ty = tiisg/8;

    // Two complete vec scratch regions, matching nhptg*FATTN_SMEM on the host.
    const int head_stride = 512*NSG + 256; // half elements
    threadgroup half * sh0 = shmem_f16;
    threadgroup half * sh1 = shmem_f16 + head_stride;
    threadgroup half4 * sq0 = (threadgroup half4 *) sh0;
    threadgroup half4 * sq1 = (threadgroup half4 *) sh1;
    threadgroup float * ss0 = (threadgroup float *) (sh0 + 128*NSG + 128*sgitg);
    threadgroup float * ss1 = (threadgroup float *) (sh1 + 128*NSG + 128*sgitg);
    threadgroup float4 * so0 = (threadgroup float4 *) (sh0 + 256*NSG + 256*sgitg);
    threadgroup float4 * so1 = (threadgroup float4 *) (sh1 + 256*NSG + 256*sgitg);

    if (sgitg == 0) {
        sq0[tiisg] = half4(((device const float4 *) (q + head0*args.nb02))[tiisg]);
        sq1[tiisg] = half4(((device const float4 *) (q + (head0 + 1)*args.nb02))[tiisg]);
    }
    so0[tiisg] = float4(0.0f);
    so1[tiisg] = float4(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S0 = 0.0f;
    float S1 = 0.0f;
    float M0 = -FLT_MAX/2;
    float M1 = -FLT_MAX/2;

    for (int ic0 = iwg*NSG + sgitg; ic0*32 < args.ne11; ic0 += NWG*NSG) {
        int ic = ic0*32;
        device const char * pk_base = k + ikv2*args.nb12;
        device const char * pv_base = v + ikv2*args.nb22;
        device const half * pm = (device const half *) mask;
        half mask_value = 0.0h;

        if (FC_flash_attn_ext_vec_has_kvpad && ic + 32 > args.ne11) {
            device const char * pad_v = pad + args.nb11*32*args.ne_12_2*args.ne_12_3;
            pk_base = pad + ikv2*args.nb11*32;
            pv_base = pad_v + ikv2*args.nb21*32;
            pm = (device const half *) (pad_v + args.nb21*32*args.ne_12_2*args.ne_12_3);
            if (!FC_flash_attn_ext_vec_has_mask && ic + tiisg >= args.ne11) {
                mask_value = -MAXHALF;
            }
            ic = 0;
        }
        if (FC_flash_attn_ext_vec_has_mask) {
            mask_value = pm[ic + tiisg];
        }
        if (simd_max(mask_value) <= -MAXHALF) {
            continue;
        }

        const float score_mask = simd_shuffle(float(mask_value), 4*tx + ty);
        _Pragma("clang loop unroll(disable)")
        for (short cc = 0; cc < 8; ++cc) {
            float score0 = 0.0f;
            float score1 = 0.0f;
            device const block_q8_0 * pk = (device const block_q8_0 *) (pk_base + (ic + 4*cc + ty)*args.nb11);
            FOR_UNROLL (short ii = 0; ii < 4; ++ii) {
                const short i = ii*8 + tx;
                half4 mk;
                dequantize_q8_0_t4(pk + i/8, i%8, mk);
                score0 += dot(float4(mk), float4(sq0[i]));
                score1 += dot(float4(mk), float4(sq1[i]));
            }
            score0 += simd_shuffle_down(score0, 4);
            score1 += simd_shuffle_down(score1, 4);
            score0 += simd_shuffle_down(score0, 2);
            score1 += simd_shuffle_down(score1, 2);
            score0 += simd_shuffle_down(score0, 1);
            score1 += simd_shuffle_down(score1, 1);
            score0 = simd_shuffle(score0, 8*ty);
            score1 = simd_shuffle(score1, 8*ty);

            if (tx == cc) {
                if (FC_flash_attn_ext_vec_has_mask) {
                    ss0[4*cc + ty] = fma(score0, args.scale, score_mask);
                    ss1[4*cc + ty] = fma(score1, args.scale, score_mask);
                } else {
                    ss0[4*cc + ty] = score0*args.scale + score_mask;
                    ss1[4*cc + ty] = score1*args.scale + score_mask;
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float previous0 = M0;
        const float previous1 = M1;
        const float s0 = ss0[tiisg];
        const float s1 = ss1[tiisg];
        M0 = simd_max(max(M0, s0));
        M1 = simd_max(max(M1, s1));
        const float ms0 = exp(previous0 - M0);
        const float ms1 = exp(previous1 - M1);
        const float vs0 = exp(s0 - M0);
        const float vs1 = exp(s1 - M1);
        S0 = S0*ms0 + simd_sum(vs0);
        S1 = S1*ms1 + simd_sum(vs1);
        ss0[tiisg] = vs0;
        ss1[tiisg] = vs1;
        so0[tiisg] *= ms0;
        so1[tiisg] *= ms1;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Each lane owns four output dimensions and visits all 32 cache rows.
        float4 lo0 = float4(0.0f);
        float4 lo1 = float4(0.0f);
        FOR_UNROLL (short cc = 0; cc < 32; ++cc) {
            const float weight0 = ss0[cc];
            const float weight1 = ss1[cc];
#if TURBO_SPARSE_V
            const bool active0 = !(weight0 < 1e-6f);
            const bool active1 = !(weight1 < 1e-6f);
            if (!active0 && !active1) {
                continue;
            }
#else
            const bool active0 = true;
            const bool active1 = true;
#endif
            device const block_turbo4_0 * pv = (device const block_turbo4_0 *) (pv_base + (ic + cc)*args.nb21);
            half4 mv;
            dequantize_turbo4_0_t4(pv, tiisg, mv);
            if (active0) {
                lo0 += float4(mv)*weight0;
            }
            if (active1) {
                lo1 += float4(mv)*weight1;
            }
        }
        so0[tiisg] += lo0;
        so1[tiisg] += lo1;
    }

    if (tiisg == 0) {
        ss0[0] = S0;
        ss0[1] = M0;
        ss1[0] = S1;
        ss1[1] = M1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float m0 = max(ss0[1], ss0[r*64 + 1]);
            const float m1 = max(ss1[1], ss1[r*64 + 1]);
            const float a0 = exp(ss0[1] - m0);
            const float b0 = exp(ss0[r*64 + 1] - m0);
            const float a1 = exp(ss1[1] - m1);
            const float b1 = exp(ss1[r*64 + 1] - m1);
            const float sum0 = ss0[0]*a0 + ss0[r*64]*b0;
            const float sum1 = ss1[0]*a1 + ss1[r*64]*b1;
            if (tiisg == 0) {
                ss0[0] = sum0;
                ss0[1] = m0;
                ss1[0] = sum1;
                ss1[1] = m1;
            }
            so0[tiisg] = so0[tiisg]*a0 + so0[tiisg + r*32]*b0;
            so1[tiisg] = so1[tiisg]*a1 + so1[tiisg + r*32]*b1;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;
        const int64_t rid0 = head0;
        const int64_t rid1 = head0 + 1;
        device float4 * dst4 = (device float4 *) dst;
        device float * dst1 = (device float *) dst + nrows*128*NWG;
        const float scale0 = NWG == 1 ? (ss0[0] == 0.0f ? 0.0f : 1.0f/ss0[0]) : 1.0f;
        const float scale1 = NWG == 1 ? (ss1[0] == 0.0f ? 0.0f : 1.0f/ss1[0]) : 1.0f;
        dst4[rid0*32*NWG + NWG*tiisg + iwg] = so0[tiisg]*scale0;
        dst4[rid1*32*NWG + NWG*tiisg + iwg] = so1[tiisg]*scale1;
        if (NWG > 1 && tiisg == 0) {
            dst1[rid0*(2*NWG) + 2*iwg + 0] = ss0[0];
            dst1[rid0*(2*NWG) + 2*iwg + 1] = ss0[1];
            dst1[rid1*(2*NWG) + 2*iwg + 0] = ss1[0];
            dst1[rid1*(2*NWG) + 2*iwg + 1] = ss1[1];
        }
    }

#undef NWG
#undef NSG
}

#define FA_TYPES half4, half4, half4, float, float, float4, float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 4>) flash_attn_turbo4_vec_t;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk128_dv128")]] kernel flash_attn_turbo4_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 4>;
