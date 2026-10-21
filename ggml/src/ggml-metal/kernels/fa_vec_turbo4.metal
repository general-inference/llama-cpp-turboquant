#include "common.h"
#include "dequantize.h"
#include "turbo4.h"
#include "fa_vec_common.metal"

template<typename kd4_t, short nl_k,
         void (*deq_k_t4)(device const kd4_t *, short, thread half4 &),
         bool sparse_v>
kernel void kernel_flash_attn_ext_vec_pair_turbo4(
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
            device const kd4_t * pk = (device const kd4_t *) (pk_base + (ic + 4*cc + ty)*args.nb11);
            FOR_UNROLL (short ii = 0; ii < 4; ++ii) {
                const short i = ii*8 + tx;
                half4 mk;
                deq_k_t4(pk + i/nl_k, i%nl_k, mk);
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
            const bool active0 = !sparse_v || !(weight0 < 1e-6f);
            const bool active1 = !sparse_v || !(weight1 < 1e-6f);
            if (!active0 && !active1) {
                continue;
            }
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

typedef decltype(kernel_flash_attn_ext_vec_pair_turbo4<block_q8_0, 8, dequantize_q8_0_t4, true>) flash_attn_turbo4_pair_t;
template [[host_name("kernel_flash_attn_ext_vec_pair_kq8_0_vturbo4_dk128_dv128")]] kernel flash_attn_turbo4_pair_t kernel_flash_attn_ext_vec_pair_turbo4<block_q8_0, 8, dequantize_q8_0_t4, true>;
template [[host_name("kernel_flash_attn_ext_vec_pair_kf16_vturbo4_dk128_dv128")]] kernel flash_attn_turbo4_pair_t kernel_flash_attn_ext_vec_pair_turbo4<half4, 1, dequantize_f16_t4, false>;

kernel void kernel_flash_attn_ext_vec_quartet_kf16_vturbo4_dk128_dv128(
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
    const uint head0 = 4*tgpig[1];
    const uint ikv2 = head0/4;
    const short tx = tiisg%8;
    const short ty = tiisg/8;

    // Four independent head regions; global partial-output slots remain per head.
    const int head_stride = 128 + 320*NSG; // half elements
    threadgroup half * sh0 = shmem_f16;
    threadgroup half * sh1 = shmem_f16 + head_stride;
    threadgroup half * sh2 = shmem_f16 + 2*head_stride;
    threadgroup half * sh3 = shmem_f16 + 3*head_stride;
    threadgroup half4 * sq0 = (threadgroup half4 *) sh0;
    threadgroup half4 * sq1 = (threadgroup half4 *) sh1;
    threadgroup half4 * sq2 = (threadgroup half4 *) sh2;
    threadgroup half4 * sq3 = (threadgroup half4 *) sh3;
    threadgroup float * ss0 = (threadgroup float *) (sh0 + 128 + 64*sgitg);
    threadgroup float * ss1 = (threadgroup float *) (sh1 + 128 + 64*sgitg);
    threadgroup float * ss2 = (threadgroup float *) (sh2 + 128 + 64*sgitg);
    threadgroup float * ss3 = (threadgroup float *) (sh3 + 128 + 64*sgitg);
    threadgroup float4 * so0 = (threadgroup float4 *) (sh0 + 128 + 64*NSG + 256*sgitg);
    threadgroup float4 * so1 = (threadgroup float4 *) (sh1 + 128 + 64*NSG + 256*sgitg);
    threadgroup float4 * so2 = (threadgroup float4 *) (sh2 + 128 + 64*NSG + 256*sgitg);
    threadgroup float4 * so3 = (threadgroup float4 *) (sh3 + 128 + 64*NSG + 256*sgitg);

    if (sgitg == 0) {
        sq0[tiisg] = half4(((device const float4 *) (q + head0*args.nb02))[tiisg]);
        sq1[tiisg] = half4(((device const float4 *) (q + (head0 + 1)*args.nb02))[tiisg]);
        sq2[tiisg] = half4(((device const float4 *) (q + (head0 + 2)*args.nb02))[tiisg]);
        sq3[tiisg] = half4(((device const float4 *) (q + (head0 + 3)*args.nb02))[tiisg]);
    }
    so0[tiisg] = float4(0.0f);
    so1[tiisg] = float4(0.0f);
    so2[tiisg] = float4(0.0f);
    so3[tiisg] = float4(0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S0 = 0.0f;
    float S1 = 0.0f;
    float S2 = 0.0f;
    float S3 = 0.0f;
    float M0 = -FLT_MAX/2;
    float M1 = -FLT_MAX/2;
    float M2 = -FLT_MAX/2;
    float M3 = -FLT_MAX/2;

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
            float score2 = 0.0f;
            float score3 = 0.0f;
            device const half4 * pk = (device const half4 *) (pk_base + (ic + 4*cc + ty)*args.nb11);
            FOR_UNROLL (short ii = 0; ii < 4; ++ii) {
                const short i = ii*8 + tx;
                half4 mk;
                dequantize_f16_t4(pk + i, 0, mk);
                score0 += dot(float4(mk), float4(sq0[i]));
                score1 += dot(float4(mk), float4(sq1[i]));
                score2 += dot(float4(mk), float4(sq2[i]));
                score3 += dot(float4(mk), float4(sq3[i]));
            }
            score0 += simd_shuffle_down(score0, 4);
            score1 += simd_shuffle_down(score1, 4);
            score2 += simd_shuffle_down(score2, 4);
            score3 += simd_shuffle_down(score3, 4);
            score0 += simd_shuffle_down(score0, 2);
            score1 += simd_shuffle_down(score1, 2);
            score2 += simd_shuffle_down(score2, 2);
            score3 += simd_shuffle_down(score3, 2);
            score0 += simd_shuffle_down(score0, 1);
            score1 += simd_shuffle_down(score1, 1);
            score2 += simd_shuffle_down(score2, 1);
            score3 += simd_shuffle_down(score3, 1);
            score0 = simd_shuffle(score0, 8*ty);
            score1 = simd_shuffle(score1, 8*ty);
            score2 = simd_shuffle(score2, 8*ty);
            score3 = simd_shuffle(score3, 8*ty);

            if (tx == cc) {
                if (FC_flash_attn_ext_vec_has_mask) {
                    ss0[4*cc + ty] = fma(score0, args.scale, score_mask);
                    ss1[4*cc + ty] = fma(score1, args.scale, score_mask);
                    ss2[4*cc + ty] = fma(score2, args.scale, score_mask);
                    ss3[4*cc + ty] = fma(score3, args.scale, score_mask);
                } else {
                    ss0[4*cc + ty] = score0*args.scale + score_mask;
                    ss1[4*cc + ty] = score1*args.scale + score_mask;
                    ss2[4*cc + ty] = score2*args.scale + score_mask;
                    ss3[4*cc + ty] = score3*args.scale + score_mask;
                }
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        const float previous0 = M0;
        const float previous1 = M1;
        const float previous2 = M2;
        const float previous3 = M3;
        const float s0 = ss0[tiisg];
        const float s1 = ss1[tiisg];
        const float s2 = ss2[tiisg];
        const float s3 = ss3[tiisg];
        M0 = simd_max(max(M0, s0));
        M1 = simd_max(max(M1, s1));
        M2 = simd_max(max(M2, s2));
        M3 = simd_max(max(M3, s3));
        const float ms0 = exp(previous0 - M0);
        const float ms1 = exp(previous1 - M1);
        const float ms2 = exp(previous2 - M2);
        const float ms3 = exp(previous3 - M3);
        const float vs0 = exp(s0 - M0);
        const float vs1 = exp(s1 - M1);
        const float vs2 = exp(s2 - M2);
        const float vs3 = exp(s3 - M3);
        S0 = S0*ms0 + simd_sum(vs0);
        S1 = S1*ms1 + simd_sum(vs1);
        S2 = S2*ms2 + simd_sum(vs2);
        S3 = S3*ms3 + simd_sum(vs3);
        ss0[tiisg] = vs0;
        ss1[tiisg] = vs1;
        ss2[tiisg] = vs2;
        ss3[tiisg] = vs3;
        so0[tiisg] *= ms0;
        so1[tiisg] *= ms1;
        so2[tiisg] *= ms2;
        so3[tiisg] *= ms3;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Each lane decodes one float4 per row and reuses it for all four heads.
        float4 lo0 = float4(0.0f);
        float4 lo1 = float4(0.0f);
        float4 lo2 = float4(0.0f);
        float4 lo3 = float4(0.0f);
        FOR_UNROLL (short cc = 0; cc < 32; ++cc) {
            const float weight0 = ss0[cc];
            const float weight1 = ss1[cc];
            const float weight2 = ss2[cc];
            const float weight3 = ss3[cc];
            device const block_turbo4_0 * pv = (device const block_turbo4_0 *) (pv_base + (ic + cc)*args.nb21);
            half4 mv;
            dequantize_turbo4_0_t4(pv, tiisg, mv);
            lo0 += float4(mv)*weight0;
            lo1 += float4(mv)*weight1;
            lo2 += float4(mv)*weight2;
            lo3 += float4(mv)*weight3;
        }
        so0[tiisg] += lo0;
        so1[tiisg] += lo1;
        so2[tiisg] += lo2;
        so3[tiisg] += lo3;
    }

    if (tiisg == 0) {
        ss0[0] = S0;
        ss0[1] = M0;
        ss1[0] = S1;
        ss2[0] = S2;
        ss3[0] = S3;
        ss1[1] = M1;
        ss2[1] = M2;
        ss3[1] = M3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short r = NSG/2; r > 0; r >>= 1) {
        if (sgitg < r) {
            const float m0 = max(ss0[1], ss0[r*32 + 1]);
            const float m1 = max(ss1[1], ss1[r*32 + 1]);
            const float m2 = max(ss2[1], ss2[r*32 + 1]);
            const float m3 = max(ss3[1], ss3[r*32 + 1]);
            const float a0 = exp(ss0[1] - m0);
            const float b0 = exp(ss0[r*32 + 1] - m0);
            const float a1 = exp(ss1[1] - m1);
            const float a2 = exp(ss2[1] - m2);
            const float a3 = exp(ss3[1] - m3);
            const float b1 = exp(ss1[r*32 + 1] - m1);
            const float b2 = exp(ss2[r*32 + 1] - m2);
            const float b3 = exp(ss3[r*32 + 1] - m3);
            const float sum0 = ss0[0]*a0 + ss0[r*32]*b0;
            const float sum1 = ss1[0]*a1 + ss1[r*32]*b1;
            const float sum2 = ss2[0]*a2 + ss2[r*32]*b2;
            const float sum3 = ss3[0]*a3 + ss3[r*32]*b3;
            if (tiisg == 0) {
                ss0[0] = sum0;
                ss0[1] = m0;
                ss1[0] = sum1;
                ss2[0] = sum2;
                ss3[0] = sum3;
                ss1[1] = m1;
                ss2[1] = m2;
                ss3[1] = m3;
            }
            so0[tiisg] = so0[tiisg]*a0 + so0[tiisg + r*32]*b0;
            so1[tiisg] = so1[tiisg]*a1 + so1[tiisg + r*32]*b1;
            so2[tiisg] = so2[tiisg]*a2 + so2[tiisg + r*32]*b2;
            so3[tiisg] = so3[tiisg]*a3 + so3[tiisg + r*32]*b3;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (sgitg == 0) {
        const int64_t nrows = args.ne3*args.ne2*args.ne1;
        const int64_t rid0 = head0;
        const int64_t rid1 = head0 + 1;
        const int64_t rid2 = head0 + 2;
        const int64_t rid3 = head0 + 3;
        device float4 * dst4 = (device float4 *) dst;
        device float * dst1 = (device float *) dst + nrows*128*NWG;
        const float scale0 = NWG == 1 ? (ss0[0] == 0.0f ? 0.0f : 1.0f/ss0[0]) : 1.0f;
        const float scale1 = NWG == 1 ? (ss1[0] == 0.0f ? 0.0f : 1.0f/ss1[0]) : 1.0f;
        const float scale2 = NWG == 1 ? (ss2[0] == 0.0f ? 0.0f : 1.0f/ss2[0]) : 1.0f;
        const float scale3 = NWG == 1 ? (ss3[0] == 0.0f ? 0.0f : 1.0f/ss3[0]) : 1.0f;
        dst4[rid0*32*NWG + NWG*tiisg + iwg] = so0[tiisg]*scale0;
        dst4[rid1*32*NWG + NWG*tiisg + iwg] = so1[tiisg]*scale1;
        dst4[rid2*32*NWG + NWG*tiisg + iwg] = so2[tiisg]*scale2;
        dst4[rid3*32*NWG + NWG*tiisg + iwg] = so3[tiisg]*scale3;
        if (NWG > 1 && tiisg == 0) {
            dst1[rid0*(2*NWG) + 2*iwg + 0] = ss0[0];
            dst1[rid0*(2*NWG) + 2*iwg + 1] = ss0[1];
            dst1[rid1*(2*NWG) + 2*iwg + 0] = ss1[0];
            dst1[rid2*(2*NWG) + 2*iwg + 0] = ss2[0];
            dst1[rid3*(2*NWG) + 2*iwg + 0] = ss3[0];
            dst1[rid1*(2*NWG) + 2*iwg + 1] = ss1[1];
            dst1[rid2*(2*NWG) + 2*iwg + 1] = ss2[1];
            dst1[rid3*(2*NWG) + 2*iwg + 1] = ss3[1];
        }
    }

#undef NWG
#undef NSG
}

#define FA_TYPES half4, half4, half4, float, float, float4, float4

typedef decltype(kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 4>) flash_attn_turbo4_vec_t;
template [[host_name("kernel_flash_attn_ext_vec_kq8_0_vturbo4_dk128_dv128")]] kernel flash_attn_turbo4_vec_t kernel_flash_attn_ext_vec<FA_TYPES, block_q8_0, 8, dequantize_q8_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 4>;
template [[host_name("kernel_flash_attn_ext_vec_kf16_vturbo4_dk128_dv128")]] kernel flash_attn_turbo4_vec_t kernel_flash_attn_ext_vec<FA_TYPES, half4, 1, dequantize_f16_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4, 128, 128, 4>;
