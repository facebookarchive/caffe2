// @generated
// Copyright 2004-present Facebook. All Rights Reserved.

static const char* MPSCNN_KERNELS = R"V0G0N(

//  Copyright 2004-present Facebook. All Rights Reserved.

#include <metal_stdlib>

using namespace metal;

constant ushort ushort_arg_0[[function_constant(0)]];
constant ushort ushort_arg_1[[function_constant(1)]];
constant ushort ushort_arg_2[[function_constant(2)]];
constant ushort ushort_arg_3[[function_constant(3)]];

kernel void affine(constant half4* scale[[buffer(0)]],
                   constant half4* shift[[buffer(1)]],
                   texture2d_array<half, access::read> in[[texture(0)]],
                   texture2d_array<half, access::write> out[[texture(1)]],
                   ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    // help out compiler with static bound on shale/shift access?
    if (gid.z * 4 >= ushort_arg_0) {
        return;
    }
    const half4 scale_c = scale[gid.z];
    const half4 shift_c = shift[gid.z];
    ushort2 gid_(gid.x, gid.y);
    const half4 x = in.read(gid_, gid.z);
    const half4 y = scale_c * x + shift_c;
    out.write(y, gid_, gid.z);
}

kernel void affine_nonarray(constant half4* scale[[buffer(0)]],
                            constant half4* shift[[buffer(1)]],
                            texture2d<half, access::read> in[[texture(0)]],
                            texture2d<half, access::write> out[[texture(1)]],
                            ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    const half4 scale_c = scale[0];
    const half4 shift_c = shift[0];
    half4 x = in.read(gid);
    const half4 y = scale_c * x + shift_c;
    out.write(y, gid);
}

kernel void prelu_nonshared(constant half4* weights[[buffer(0)]],
                            texture2d_array<half, access::read> in[[texture(0)]],
                            texture2d_array<half, access::write> out[[texture(1)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    const bool channel_shared = ushort_arg_0 == 0;
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    half4 w = channel_shared ? half4(weights[0][0], weights[0][0], weights[0][0], weights[0][0])
    : weights[gid.z];
    ushort2 gid_(gid.x, gid.y);
    half4 x = in.read(gid_, gid.z);
    half4 y = select(x * w, x, x > 0.0h);
    out.write(y, gid_, gid.z);
}

kernel void prelu_nonshared_nonarray(constant half4* weights[[buffer(0)]],
                                     texture2d<half, access::read> in[[texture(0)]],
                                     texture2d<half, access::write> out[[texture(1)]],
                                     ushort2 gid[[thread_position_in_grid]]) {
    const bool channel_shared = ushort_arg_0 == 0;
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    half4 w = channel_shared ? half4(weights[0][0], weights[0][0], weights[0][0], weights[0][0])
    : weights[0];
    half4 x = in.read(gid);
    half4 y = select(x * w, x, x > 0.0h);
    out.write(y, gid);
}

// One block per texture.
// 256 threads per block.
using AccT = float4;
kernel void instance_norm(constant half* weights[[buffer(0)]],
                          constant half* bias[[buffer(1)]],
                          texture2d_array<half, access::read> in[[texture(0)]],
                          texture2d_array<half, access::write> out[[texture(1)]],
                          ushort3 gid[[thread_position_in_grid]],
                          ushort tid[[thread_index_in_threadgroup]],
                          ushort3 tcount[[threads_per_threadgroup]]) {
    if (gid.z >= out.get_array_size()) {
        return;
    }
    
    constexpr ushort THREADGROUP_SIZE = 256;
    
    threadgroup AccT per_thread_state[THREADGROUP_SIZE];
    // Each block handles a single texture.
    per_thread_state[tid] = 0;
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            per_thread_state[tid] += static_cast<AccT>(in.read(ushort2(x, y), gid.z));
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 256 -> 32 reduction
    if (tid < 32) {
        per_thread_state[tid] += per_thread_state[tid + 32] + per_thread_state[tid + 64] +
        per_thread_state[tid + 96] + per_thread_state[tid + 128] +
        per_thread_state[tid + 160] + per_thread_state[tid + 192] +
        per_thread_state[tid + 224];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        AccT sum = 0.0;
        for (ushort i = 0; i < 32; ++i) {
            sum += per_thread_state[i];
        }
        sum /= (in.get_width() * in.get_height());
        per_thread_state[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Broadcast to all threads.
    const AccT mean = per_thread_state[0];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    per_thread_state[tid] = 0;
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            AccT delta = static_cast<AccT>(in.read(ushort2(x, y), gid.z)) - mean;
            per_thread_state[tid] += delta * delta;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 256 -> 32 reduction
    if (tid < 32) {
        per_thread_state[tid] += per_thread_state[tid + 32] + per_thread_state[tid + 64] +
        per_thread_state[tid + 96] + per_thread_state[tid + 128] +
        per_thread_state[tid + 160] + per_thread_state[tid + 192] +
        per_thread_state[tid + 224];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        AccT sum = 0.0;
        for (ushort i = 0; i < 32; ++i) {
            sum += per_thread_state[i];
        }
        sum /= (in.get_width() * in.get_height());
        per_thread_state[0] = 1.0 / sqrt(max(sum, AccT(1e-5, 1e-5, 1e-5, 1e-5)) + 1.0e-5);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Broadcast to all threads.
    const AccT inv_var = per_thread_state[0];
    
    const AccT c_weights(
                         weights[4 * gid.z], weights[4 * gid.z + 1], weights[4 * gid.z + 2], weights[4 * gid.z + 3]);
    const AccT c_bias(bias[4 * gid.z], bias[4 * gid.z + 1], bias[4 * gid.z + 2], bias[4 * gid.z + 3]);
    
    const AccT scale = inv_var * c_weights;
    const AccT shift = c_bias - mean * scale;
    
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            AccT scaled = static_cast<AccT>(in.read(ushort2(x, y), gid.z)) * scale + shift;
            out.write(static_cast<half4>(scaled), ushort2(x, y), gid.z);
        }
    }
}

// One block per texture.
// 256 threads per block.
kernel void instance_norm_nonarray(constant half* weights[[buffer(0)]],
                                   constant half* bias[[buffer(1)]],
                                   texture2d<half, access::read> in[[texture(0)]],
                                   texture2d<half, access::write> out[[texture(1)]],
                                   ushort3 gid[[thread_position_in_grid]],
                                   ushort tid[[thread_index_in_threadgroup]],
                                   ushort3 tcount[[threads_per_threadgroup]]) {
    constexpr ushort THREADGROUP_SIZE = 256;
    
    threadgroup AccT per_thread_state[THREADGROUP_SIZE];
    // Each block handles a single texture.
    per_thread_state[tid] = 0;
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            per_thread_state[tid] += static_cast<AccT>(in.read(ushort2(x, y)));
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 256 -> 32 reduction
    if (tid < 32) {
        per_thread_state[tid] += per_thread_state[tid + 32] + per_thread_state[tid + 64] +
        per_thread_state[tid + 96] + per_thread_state[tid + 128] +
        per_thread_state[tid + 160] + per_thread_state[tid + 192] +
        per_thread_state[tid + 224];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        AccT sum = 0.0;
        for (ushort i = 0; i < 32; ++i) {
            sum += per_thread_state[i];
        }
        sum /= (in.get_width() * in.get_height());
        per_thread_state[0] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Broadcast to all threads.
    const AccT mean = per_thread_state[0];
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    per_thread_state[tid] = 0;
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            AccT delta = static_cast<AccT>(in.read(ushort2(x, y))) - mean;
            per_thread_state[tid] += delta * delta;
        }
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 256 -> 32 reduction
    if (tid < 32) {
        per_thread_state[tid] += per_thread_state[tid + 32] + per_thread_state[tid + 64] +
        per_thread_state[tid + 96] + per_thread_state[tid + 128] +
        per_thread_state[tid + 160] + per_thread_state[tid + 192] +
        per_thread_state[tid + 224];
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (tid == 0) {
        AccT sum = 0.0;
        for (ushort i = 0; i < 32; ++i) {
            sum += per_thread_state[i];
        }
        sum /= (in.get_width() * in.get_height());
        per_thread_state[0] = 1.0 / sqrt(max(sum, AccT(1e-5, 1e-5, 1e-5, 1e-5)) + 1.0e-5);
    }
    
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Broadcast to all threads.
    const AccT inv_var = per_thread_state[0];
    
    const AccT c_weights(
                         weights[4 * gid.z], weights[4 * gid.z + 1], weights[4 * gid.z + 2], weights[4 * gid.z + 3]);
    const AccT c_bias(bias[4 * gid.z], bias[4 * gid.z + 1], bias[4 * gid.z + 2], bias[4 * gid.z + 3]);
    
    const AccT scale = inv_var * c_weights;
    const AccT shift = c_bias - mean * scale;
    
    for (ushort y = 0; y < in.get_height(); y += 1) {
        for (ushort x = gid.x; x < in.get_width(); x += tcount.x) {
            AccT scaled = static_cast<AccT>(in.read(ushort2(x, y))) * scale + shift;
            out.write(static_cast<half4>(scaled), ushort2(x, y));
        }
    }
}

kernel void copy_nchw_to_metal(constant float* in[[buffer(0)]],
                               texture2d_array<half, access::write> out[[texture(0)]],
                               ushort3 gid[[thread_position_in_grid]]) {
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    
    // TODO: are the `else` branches needed?
    // TODO: trick the optimizer for case where C == 4?
#define CHW_TO_CHWP4(idx, c, h, w)                        \
if ((c) < C) {                                          \
trns[idx] = in[int(c) * H * W + int(h) * W + int(w)]; \
} else {                                                \
trns[idx] = 0.0h;                                     \
}
    
    half4 trns;
    CHW_TO_CHWP4(0, 4 * gid.z + 0, gid.y, gid.x);
    CHW_TO_CHWP4(1, 4 * gid.z + 1, gid.y, gid.x);
    CHW_TO_CHWP4(2, 4 * gid.z + 2, gid.y, gid.x);
    CHW_TO_CHWP4(3, 4 * gid.z + 3, gid.y, gid.x);
#undef CHW_TO_CHWP4
    
    out.write(trns, ushort2(gid.x, gid.y), gid.z);
}

kernel void copy_nchw_to_metal_nonarray(constant float* in[[buffer(0)]],
                                        texture2d<half, access::write> out[[texture(0)]],
                                        ushort2 gid[[thread_position_in_grid]]) {
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    
    half4 trns;
    // TODO: are the `else` branches needed?
    // TODO: trick the optimizer for case where C % 4 == 0?
    
#define CHW_TO_CHWP4(idx, c, h, w)                        \
if ((c) < C) {                                          \
trns[idx] = in[int(c) * H * W + int(h) * W + int(w)]; \
} else {                                                \
trns[idx] = 0.0h;                                     \
}
    
    CHW_TO_CHWP4(0, 0, gid.y, gid.x);
    CHW_TO_CHWP4(1, 1, gid.y, gid.x);
    CHW_TO_CHWP4(2, 2, gid.y, gid.x);
    CHW_TO_CHWP4(3, 3, gid.y, gid.x);
#undef CHW_TO_CHWP4
    
    out.write(trns, ushort2(gid.x, gid.y));
}

kernel void copy_metal_to_nchw(texture2d_array<half, access::read> in[[texture(0)]],
                               device float* out[[buffer(0)]],
                               ushort3 gid[[thread_position_in_grid]]) {
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    
    half4 cs = in.read(ushort2(gid.x, gid.y), gid.z);
    
#define CHWP4_TO_CHW(idx, c, h, w)                       \
if ((c) < C) {                                         \
out[int(c) * H * W + int(h) * W + int(w)] = cs[idx]; \
}
    
    CHWP4_TO_CHW(0, gid.z * 4 + 0, gid.y, gid.x);
    CHWP4_TO_CHW(1, gid.z * 4 + 1, gid.y, gid.x);
    CHWP4_TO_CHW(2, gid.z * 4 + 2, gid.y, gid.x);
    CHWP4_TO_CHW(3, gid.z * 4 + 3, gid.y, gid.x);
#undef CHWP4_TO_CHW
}

kernel void copy_metal_to_nchw_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                        device float* out[[buffer(0)]],
                                        ushort2 gid[[thread_position_in_grid]]) {
    const ushort C = ushort_arg_0;
    const ushort H = ushort_arg_1;
    const ushort W = ushort_arg_2;
    
    if (gid.x >= W || gid.y >= H) {
        return;
    }
    
    half4 cs = in.read(ushort2(gid.x, gid.y));
    
#define CHWP4_TO_CHW(idx, c, h, w)                       \
if ((c) < C) {                                         \
out[int(c) * H * W + int(h) * W + int(w)] = cs[idx]; \
}
    
    CHWP4_TO_CHW(0, 0, gid.y, gid.x);
    CHWP4_TO_CHW(1, 1, gid.y, gid.x);
    CHWP4_TO_CHW(2, 2, gid.y, gid.x);
    CHWP4_TO_CHW(3, 3, gid.y, gid.x);
#undef CHWP4_TO_CHW
}

kernel void convtranspose_upscale(texture2d_array<half, access::read> in[[texture(0)]],
                                  texture2d_array<half, access::write> out[[texture(1)]],
                                  ushort3 gid[[thread_position_in_grid]]) {
    // All resolved at compile time.
    // Assume symmetric kernel/stride/pad for now.
    const ushort kernel_ = ushort_arg_0;
    const ushort stride = ushort_arg_1;
    const ushort pad = ushort_arg_2;
    
    half4 zero(0.0h, 0.0h, 0.0h, 0.0h);
    
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    const ushort2 gid_ = ushort2(gid.x, gid.y);
    if (gid.x < kernel_ - 1 - pad || gid.y < kernel_ - 1 - pad) {
        out.write(zero, gid_, gid.z);
        return;
    }
    
    if (((gid.x - (kernel_ - 1 - pad)) % stride == 0) &&
        ((gid.y - (kernel_ - 1 - pad)) % stride == 0)) {
        ushort2 in_pos((gid.x - (kernel_ - 1 - pad)) / stride, (gid.y - (kernel_ - 1 - pad)) / stride);
        
        if (in_pos.x < in.get_width() && in_pos.y < in.get_height()) {
            half4 input = in.read(in_pos, gid.z);
            out.write(input, gid_, gid.z);
        } else {
            out.write(zero, gid_, gid.z);
        }
    } else {
        out.write(zero, gid_, gid.z);
    }
}

kernel void convtranspose_upscale_4(texture2d<half, access::read> in[[texture(0)]],
                                    texture2d<half, access::write> out[[texture(1)]],
                                    ushort2 gid[[thread_position_in_grid]]) {
    // All resolved at compile time.
    // Assume symmetric kernel/stride/pad for now.
    const ushort kernel_ = ushort_arg_0;
    const ushort stride = ushort_arg_1;
    const ushort pad = ushort_arg_2;
    
    half4 zero(0.0h, 0.0h, 0.0h, 0.0h);
    
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    
    if (gid.x < kernel_ - 1 - pad || gid.y < kernel_ - 1 - pad) {
        out.write(zero, gid);
        return;
    }
    
    if (((gid.x - (kernel_ - 1 - pad)) % stride == 0) &&
        ((gid.y - (kernel_ - 1 - pad)) % stride == 0)) {
        ushort2 in_pos((gid.x - (kernel_ - 1 - pad)) / stride, (gid.y - (kernel_ - 1 - pad)) / stride);
        if (in_pos.x < in.get_width() && in_pos.y < in.get_height()) {
            half4 input = in.read(in_pos);
            out.write(input, gid);
        } else {
            out.write(zero, gid);
        }
    } else {
        out.write(zero, gid);
    }
}

kernel void preprocess_stylizer(device uchar4* in[[buffer(0)]],
                                constant half* mean[[buffer(1)]],
                                constant half4* noise[[buffer(2)]],
                                texture2d<half, access::write> out[[texture(0)]],
                                ushort2 gid[[thread_position_in_grid]]) {
    
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    const ushort noise_size = ushort_arg_0;
    
    half4 mean_half(mean[0], mean[1], mean[2], 0.0h);
    uint input_noise_idx = ((uint)out.get_width() * (uint)gid.y + (uint)gid.x) % (noise_size / 4);
    const half4 input_noise = noise[input_noise_idx];
    const uint W = out.get_width();
#define in_at(h, w) in[(uint)(h)*W + (uint)(w)]
    uchar4 input = in_at(gid.y, gid.x);
#undef in_at
    half4 input_half = static_cast<half4>(input);
    out.write(input_half - mean_half + input_noise, gid);
}

kernel void deprocess_stylizer(texture2d<half, access::read> in[[texture(0)]],
                               device uchar4* out[[buffer(0)]],
                               constant half* mean[[buffer(1)]],
                               ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= in.get_width() || gid.y >= in.get_height()) {
        return;
    }
    
    half4 value = in.read(gid);
    
    half4 mean_h(mean[0], mean[1], mean[2], 0.0h);
    half4 min_h(0.0h, 0.0h, 0.0h, 255.0h);
    half4 max_h(255.0h, 255.0h, 255.0h, 255.0h);
    half4 clamped = clamp(value + mean_h, min_h, max_h);
    const uint W = in.get_width();
#define out_at(h, w, v) out[(uint)(h)*W + (uint)(w)] = (v)
    out_at(gid.y, gid.x, static_cast<uchar4>(clamped));
#undef out_at
}

kernel void reflection_padding_nonarray(texture2d<half, access::read> in[[texture(0)]],
                                        texture2d<half, access::write> out[[texture(1)]],
                                        ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort H = in.get_height();
    ushort PH = out.get_height();
    
    // Note: we assume symmetric padding on H/W here, which is verified
    // in the calling code.
    ushort pad_h = (PH - H) / 2;
    ushort W = in.get_width();
    ushort PW = out.get_width();
    ushort pad_w = (PW - W) / 2;
    
    short h = short(gid.y) - short(pad_h);
    h = max(h, short(-h));
    h = min(h, short(2 * H - h - 2));
    
    short w = short(gid.x) - short(pad_w);
    w = max(w, short(-w));
    w = min(w, short(2 * W - w - 2));
    
    ushort2 inid(w, h);
    out.write(in.read(inid), gid);
}

kernel void reflection_padding(texture2d_array<half, access::read> in[[texture(0)]],
                               texture2d_array<half, access::write> out[[texture(1)]],
                               ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort H = in.get_height();
    ushort PH = out.get_height();
    
    // Note: we assume symmetric padding on H/W here, which is verified
    // in the calling code.
    ushort pad_h = (PH - H) / 2;
    ushort W = in.get_width();
    ushort PW = out.get_width();
    ushort pad_w = (PW - W) / 2;
    
    short h = short(gid.y) - short(pad_h);
    h = max(h, short(-h));
    h = min(h, short(2 * H - h - 2));
    
    short w = short(gid.x) - short(pad_w);
    w = max(w, short(-w));
    w = min(w, short(2 * W - w - 2));
    
    ushort2 inid(w, h);
    
    out.write(in.read(inid, gid.z), ushort2(gid.x, gid.y), gid.z);
}

kernel void bilinear_upsample(texture2d<half, access::sample> in[[texture(0)]],
                              texture2d<half, access::write> out[[texture(1)]],
                              ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 src = gid / 2;
    constexpr sampler sampler(address::clamp_to_edge, filter::linear, coord::pixel);
    half4 value = in.sample(sampler, static_cast<float2>(src));
    out.write(value, gid);
}

kernel void elementwise_add_nonarray(texture2d<half, access::read> in0[[texture(0)]],
                                     texture2d<half, access::read> in1[[texture(1)]],
                                     texture2d<half, access::write> out[[texture(2)]],
                                     ushort2 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    out.write(in0.read(gid) + in1.read(gid), gid);
}

kernel void elementwise_add(texture2d_array<half, access::read> in0[[texture(0)]],
                            texture2d_array<half, access::read> in1[[texture(1)]],
                            texture2d_array<half, access::write> out[[texture(2)]],
                            ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    ushort2 gid_ = ushort2(gid.x, gid.y);
    out.write(in0.read(gid_, gid.z) + in1.read(gid_, gid.z), gid_, gid.z);
}

constant bool has_in0_arg = (ushort_arg_0 > 0);
constant bool has_in1_arg = (ushort_arg_1 > 0);
constant bool has_in2_arg = (ushort_arg_2 > 0);
constant bool has_in3_arg = (ushort_arg_3 > 0);

constant bool has_in0_tex = (ushort_arg_0 > 0 && ushort_arg_0 <= 4);
constant bool has_in1_tex = (ushort_arg_1 > 0 && ushort_arg_1 <= 4);
constant bool has_in2_tex = (ushort_arg_2 > 0 && ushort_arg_2 <= 4);
constant bool has_in3_tex = (ushort_arg_3 > 0 && ushort_arg_3 <= 4);

constant bool has_in0_array = (ushort_arg_0 > 4);
constant bool has_in1_array = (ushort_arg_1 > 4);
constant bool has_in2_array = (ushort_arg_2 > 4);
constant bool has_in3_array = (ushort_arg_3 > 4);

inline ushort idx_3(ushort z, ushort C0, ushort C1, ushort C2, ushort C3) {
    if (z < C0 / 4) {
        return 0;
    }
    if (z < (C0 + C1) / 4) {
        return 1;
    }
    if (z < (C0 + C1 + C2) / 4) {
        return 2;
    }
    return 3;
}

inline ushort idx_2(ushort z, ushort C0, ushort C1, ushort C2) {
    if (z < C0 / 4) {
        return 0;
    }
    if (z < (C0 + C1) / 4) {
        return 1;
    }
    return 2;
}

inline ushort idx_1(ushort z, ushort C0, ushort C1) {
    if (z < C0 / 4) {
        return 0;
    } else {
        return 1;
    }
}

inline ushort idx_0(ushort z, ushort C0) { return 0; }

kernel void concat(
                   texture2d<half, access::read> in0[[ texture(0), function_constant(has_in0_tex) ]],
                   texture2d<half, access::read> in1[[ texture(1), function_constant(has_in1_tex) ]],
                   texture2d<half, access::read> in2[[ texture(2), function_constant(has_in2_tex) ]],
                   texture2d<half, access::read> in3[[ texture(3), function_constant(has_in3_tex) ]],
                   texture2d_array<half, access::read> ina0[[ texture(0), function_constant(has_in0_array) ]],
                   texture2d_array<half, access::read> ina1[[ texture(1), function_constant(has_in1_array) ]],
                   texture2d_array<half, access::read> ina2[[ texture(2), function_constant(has_in2_array) ]],
                   texture2d_array<half, access::read> ina3[[ texture(3), function_constant(has_in3_array) ]],
                   texture2d_array<half, access::write> out[[texture(5)]],
                   ushort3 gid[[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) {
        return;
    }
    
    const ushort C0 = ushort_arg_0;
    const ushort C1 = ushort_arg_1;
    const ushort C2 = ushort_arg_2;
    const ushort C3 = ushort_arg_3;
    
    ushort idx = 0;
    if (has_in3_arg) {
        idx = idx_3(gid.z, C0, C1, C2, C3);
    } else if (has_in2_arg) {
        idx = idx_2(gid.z, C0, C1, C2);
    } else if (has_in1_arg) {
        idx = idx_1(gid.z, C0, C1);
    } else if (has_in0_arg) {
        idx = idx_0(gid.z, C0);
    } else {
        // never reached.
        idx = 0;
    }
    
    ushort2 gid_ = ushort2(gid.x, gid.y);
    half4 value;
    switch (idx) {
        case 0: {
            if (has_in0_tex) {
                value = in0.read(gid_);
            }
            if (has_in0_array) {
                value = ina0.read(gid_, gid.z);
            }
            break;
        }
        case 1: {
            if (has_in1_tex) {
                value = in1.read(gid_);
            }
            if (has_in1_array) {
                value = ina1.read(gid_, gid.z - (C0) / 4);
            }
            break;
        }
        case 2: {
            if (has_in2_tex) {
                value = in2.read(gid_);
            }
            if (has_in2_array) {
                value = ina2.read(gid_, gid.z - (C0 + C1) / 4);
            }
            break;
        }
        case 3: {
            if (has_in3_tex) {
                value = in3.read(gid_);
            }
            if (has_in3_array) {
                value = ina3.read(gid_, gid.z - (C0 + C1 + C2) / 4);
            }
            break;
        }
    }
    out.write(value, gid_, gid.z);
}



)V0G0N";
