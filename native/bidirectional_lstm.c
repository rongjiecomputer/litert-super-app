// CPU kernel for the `litert_custom_op.bidirectional_lstm` custom op that the
// LiteRT-LM Kokoro acoustic graph uses (one node per bidirectional LSTM layer).
//
// The math is a C port of LiteRT's reference kernel,
//   litert/experimental/custom_ops/bidirectional_lstm/
//     bidirectional_lstm_impl.cc / bidirectional_lstm_custom_op.cc
// (Copyright 2026 Google LLC, Apache License 2.0), rewritten without Eigen
// and against the LiteRT C API so it can be registered from Dart FFI.
//
// Operands (the order the exporter marks them), all float32 unless noted:
//   0 x           [B, T, D]       5 bias_fwd  [4H]
//   1 seq_lengths [B]      int32  6 w_ih_bwd  [4H, D]
//   2 mask        [B, T]          7 w_hh_bwd  [4H, H]
//   3 w_ih_fwd    [4H, D]         8 bias_bwd  [4H]
//   4 w_hh_fwd    [4H, H]
// Output 0: y [B, T, 2H] (forward in [0, H), backward in [H, 2H)).
// Gates are packed i, f, g, o (PyTorch nn.LSTM). `mask` is 1.0 on padding.
//
// The kernel is stateless: every dimension comes from the operand shapes.
//
// LiteRT symbols are not linked. The host passes the few C API entry points
// it needs through `LitertCustomOpsApi`, so this library has no link-time
// dependency on libLiteRt.

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Minimal copies of the LiteRT C API types the kernel touches (see
// litert/c/litert_common.h, litert_layout.h, litert_model_types.h and
// litert_custom_op_kernel.h). They are ABI-stable, and defining them here
// keeps this library buildable without the generated LiteRT headers.
typedef int LiteRtStatus;
enum {
  kLiteRtStatusOk = 0,
  kLiteRtStatusErrorInvalidArgument = 1,
  kLiteRtStatusErrorRuntimeFailure = 3,
};
typedef void* LiteRtOptions;
typedef void* LiteRtTensorBuffer;
typedef int LiteRtTensorBufferLockMode;
enum {
  kLiteRtTensorBufferLockModeRead = 0,
  kLiteRtTensorBufferLockModeWrite = 1,
};

#define LITERT_TENSOR_MAX_RANK 8
typedef struct {
  unsigned int rank : 7;
  unsigned int has_strides : 1;
  int32_t dimensions[LITERT_TENSOR_MAX_RANK];
  uint32_t strides[LITERT_TENSOR_MAX_RANK];
} LiteRtLayout;

typedef struct {
  int element_type;  // LiteRtElementType
  LiteRtLayout layout;
} LiteRtRankedTensorType;

typedef struct {
  LiteRtStatus (*Init)(void* user_data, const void* init_data,
                       size_t init_data_size);
  LiteRtStatus (*GetOutputLayouts)(void* user_data, size_t num_inputs,
                                   const LiteRtLayout* input_layouts,
                                   size_t num_outputs,
                                   LiteRtLayout* output_layouts);
  LiteRtStatus (*Run)(void* user_data, size_t num_inputs,
                      const LiteRtTensorBuffer* inputs, size_t num_outputs,
                      LiteRtTensorBuffer* outputs);
  LiteRtStatus (*Destroy)(void* user_data);
} LiteRtCustomOpKernel;

// Compile-time ABI checks against the 64-bit sizes asserted in the headers.
#define LITERT_STATIC_ASSERT(name, cond) typedef char name[(cond) ? 1 : -1]
#if defined(_WIN64) || defined(__LP64__)
LITERT_STATIC_ASSERT(layout_size_is_68, sizeof(LiteRtLayout) == 68);
LITERT_STATIC_ASSERT(ranked_type_size_is_72, sizeof(LiteRtRankedTensorType) == 72);
#endif

#if defined(_WIN32)
#define LITERT_CUSTOM_OPS_EXPORT __declspec(dllexport)
#else
#define LITERT_CUSTOM_OPS_EXPORT __attribute__((visibility("default")))
#endif

enum {
  kNumInputs = 9,
  kNumOutputs = 1,
  kInputX = 0,
  kInputSeqLengths = 1,
  kInputMask = 2,
  kInputWeightIhFwd = 3,
  kInputWeightHhFwd = 4,
  kInputBiasFwd = 5,
  kInputWeightIhBwd = 6,
  kInputWeightHhBwd = 7,
  kInputBiasBwd = 8,
};

// Entry points resolved by the host from libLiteRt.
typedef struct {
  LiteRtStatus (*add_custom_op_kernel_option)(
      LiteRtOptions options, const char* custom_op_name, int custom_op_version,
      const LiteRtCustomOpKernel* custom_op_kernel,
      void* custom_op_kernel_user_data);
  LiteRtStatus (*get_tensor_buffer_tensor_type)(
      LiteRtTensorBuffer buffer, LiteRtRankedTensorType* tensor_type);
  LiteRtStatus (*lock_tensor_buffer)(LiteRtTensorBuffer buffer,
                                     void** host_mem_addr,
                                     LiteRtTensorBufferLockMode lock_mode);
  LiteRtStatus (*unlock_tensor_buffer)(LiteRtTensorBuffer buffer);
} LitertCustomOpsApi;

// Kernels outlive the options that reference them, so the table is static.
static LitertCustomOpsApi g_api;

// ---------------------------------------------------------------- math ----

static float dot(const float* a, const float* b, int n) {
  float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, s7 = 0;
  int i = 0;
  for (; i + 8 <= n; i += 8) {
    s0 += a[i] * b[i];
    s1 += a[i + 1] * b[i + 1];
    s2 += a[i + 2] * b[i + 2];
    s3 += a[i + 3] * b[i + 3];
    s4 += a[i + 4] * b[i + 4];
    s5 += a[i + 5] * b[i + 5];
    s6 += a[i + 6] * b[i + 6];
    s7 += a[i + 7] * b[i + 7];
  }
  float s = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

static float sigmoidf(float v) { return 1.0f / (1.0f + expf(-v)); }

static int reverse_index(int length, int t, int t_steps) {
  int idx = length - 1 - t;
  if (idx < 0) return 0;
  if (idx > t_steps - 1) return t_steps - 1;
  return idx;
}

// One unidirectional LSTM over a single sequence. x [T, D], out [T, H].
// `gates_all` must hold T * 4H floats and `state`/`cell` 2H floats.
static void unidirectional_lstm(const float* x, const float* w_ih,
                                const float* w_hh, const float* bias,
                                float* out, int t_steps, int d, int h,
                                float* gates_all, float* state, float* cell) {
  const int g4 = 4 * h;
  // The input projection does not depend on the recurrent state: one GEMM
  // for the whole sequence, weight row outermost so it is reused over T.
  for (int j = 0; j < g4; ++j) {
    const float* w = w_ih + (size_t)j * d;
    const float b = bias[j];
    for (int t = 0; t < t_steps; ++t) {
      gates_all[(size_t)t * g4 + j] = dot(x + (size_t)t * d, w, d) + b;
    }
  }
  memset(state, 0, (size_t)h * sizeof(float));
  memset(cell, 0, (size_t)h * sizeof(float));
  for (int t = 0; t < t_steps; ++t) {
    float* g = gates_all + (size_t)t * g4;
    for (int j = 0; j < g4; ++j) {
      g[j] += dot(w_hh + (size_t)j * h, state, h);
    }
    float* o = out + (size_t)t * h;
    for (int k = 0; k < h; ++k) {
      const float ig = sigmoidf(g[k]);
      const float fg = sigmoidf(g[h + k]);
      const float cg = tanhf(g[2 * h + k]);
      const float og = sigmoidf(g[3 * h + k]);
      cell[k] = fg * cell[k] + ig * cg;
      state[k] = og * tanhf(cell[k]);
      o[k] = state[k];
    }
  }
}

static int compute_bidirectional_lstm(
    const float* x, const int32_t* seq_lengths, const float* mask,
    const float* w_ih_fwd, const float* w_hh_fwd, const float* b_fwd,
    const float* w_ih_bwd, const float* w_hh_bwd, const float* b_bwd,
    float* out, int batch, int t_steps, int d, int h) {
  // Scratch: fwd_out [T,H], reversed_in [T,D], reversed_out [T,H],
  // gates_all [T,4H], state [H], cell [H].
  const size_t floats = (size_t)t_steps * (2 * h + d + 4 * h) + 2 * (size_t)h;
  float* scratch = (float*)malloc(floats * sizeof(float));
  if (scratch == NULL) return 0;
  float* fwd_out = scratch;
  float* reversed_in = fwd_out + (size_t)t_steps * h;
  float* reversed_out = reversed_in + (size_t)t_steps * d;
  float* gates_all = reversed_out + (size_t)t_steps * h;
  float* state = gates_all + (size_t)t_steps * 4 * h;
  float* cell = state + h;
  const int out_stride = 2 * h;

  for (int b = 0; b < batch; ++b) {
    const float* x_b = x + (size_t)b * t_steps * d;
    const float* mask_b = mask + (size_t)b * t_steps;
    float* out_b = out + (size_t)b * t_steps * out_stride;
    const int length = seq_lengths[b];
    // Only t < length is unmasked; skip the padded bucket steps entirely.
    int active = length < 1 ? 1 : length;
    if (active > t_steps) active = t_steps;

    unidirectional_lstm(x_b, w_ih_fwd, w_hh_fwd, b_fwd, fwd_out, active, d, h,
                        gates_all, state, cell);
    if (active < t_steps) {
      memset(fwd_out + (size_t)active * h, 0,
             (size_t)(t_steps - active) * h * sizeof(float));
    }

    // Gather the active input "last real token first", zeroing padding.
    for (int t = 0; t < active; ++t) {
      const float keep = 1.0f - mask_b[t];
      const float* src = x_b + (size_t)reverse_index(length, t, t_steps) * d;
      float* dst = reversed_in + (size_t)t * d;
      for (int i = 0; i < d; ++i) dst[i] = src[i] * keep;
    }
    unidirectional_lstm(reversed_in, w_ih_bwd, w_hh_bwd, b_bwd, reversed_out,
                        active, d, h, gates_all, state, cell);

    // Scatter back to forward time order, mask padding, concatenate.
    for (int t = 0; t < t_steps; ++t) {
      const float keep = 1.0f - mask_b[t];
      const float* fwd = fwd_out + (size_t)t * h;
      const float* bwd =
          reversed_out + (size_t)reverse_index(length, t, t_steps) * h;
      float* dst = out_b + (size_t)t * out_stride;
      for (int i = 0; i < h; ++i) {
        dst[i] = fwd[i] * keep;
        dst[h + i] = bwd[i] * keep;
      }
    }
  }
  free(scratch);
  return 1;
}

// -------------------------------------------------------------- kernel ----

static LiteRtStatus kernel_init(void* user_data, const void* init_data,
                                size_t init_data_size) {
  (void)user_data;
  // Only PyTorch's "ifgo" gate order is implemented. The attributes are a
  // flexbuffer map; its keys and string values are stored as plain C strings,
  // so a substring check is enough to reject a different packing.
  if (init_data == NULL || init_data_size == 0) return kLiteRtStatusOk;
  const char* p = (const char*)init_data;
  for (size_t i = 0; i + 10 <= init_data_size; ++i) {
    if (memcmp(p + i, "gate_order", 10) == 0) {
      for (size_t j = i; j + 4 <= init_data_size; ++j) {
        if (memcmp(p + j, "ifgo", 4) == 0) return kLiteRtStatusOk;
      }
      return kLiteRtStatusErrorInvalidArgument;
    }
  }
  return kLiteRtStatusOk;
}

static LiteRtStatus kernel_get_output_layouts(
    void* user_data, size_t num_inputs, const LiteRtLayout* input_layouts,
    size_t num_outputs, LiteRtLayout* output_layouts) {
  (void)user_data;
  if (num_inputs != kNumInputs || num_outputs != kNumOutputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const LiteRtLayout* x = &input_layouts[kInputX];
  const LiteRtLayout* w_hh = &input_layouts[kInputWeightHhFwd];
  if (x->rank != 3 || w_hh->rank != 2) return kLiteRtStatusErrorInvalidArgument;
  const int32_t hidden = w_hh->dimensions[1];
  if (w_hh->dimensions[0] != 4 * hidden) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtLayout* y = &output_layouts[0];
  memset(y, 0, sizeof(*y));
  y->rank = 3;
  y->has_strides = 0;
  y->dimensions[0] = x->dimensions[0];
  y->dimensions[1] = x->dimensions[1];
  y->dimensions[2] = 2 * hidden;
  return kLiteRtStatusOk;
}

static LiteRtStatus kernel_run(void* user_data, size_t num_inputs,
                               const LiteRtTensorBuffer* inputs,
                               size_t num_outputs, LiteRtTensorBuffer* outputs) {
  (void)user_data;
  if (num_inputs != kNumInputs || num_outputs != kNumOutputs) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  LiteRtRankedTensorType x_type, w_type;
  if (g_api.get_tensor_buffer_tensor_type(inputs[kInputX], &x_type) !=
          kLiteRtStatusOk ||
      g_api.get_tensor_buffer_tensor_type(inputs[kInputWeightHhFwd],
                                          &w_type) != kLiteRtStatusOk) {
    return kLiteRtStatusErrorRuntimeFailure;
  }
  if (x_type.layout.rank != 3 || w_type.layout.rank != 2) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  const int batch = x_type.layout.dimensions[0];
  const int t_steps = x_type.layout.dimensions[1];
  const int d = x_type.layout.dimensions[2];
  const int h = w_type.layout.dimensions[1];

  void* in[kNumInputs] = {0};
  void* out = NULL;
  int locked_in = 0, locked_out = 0;
  LiteRtStatus status = kLiteRtStatusOk;
  for (int i = 0; i < kNumInputs; ++i) {
    status = g_api.lock_tensor_buffer(inputs[i], &in[i],
                                      kLiteRtTensorBufferLockModeRead);
    if (status != kLiteRtStatusOk) goto done;
    ++locked_in;
  }
  status = g_api.lock_tensor_buffer(outputs[0], &out,
                                    kLiteRtTensorBufferLockModeWrite);
  if (status != kLiteRtStatusOk) goto done;
  locked_out = 1;

  if (!compute_bidirectional_lstm(
          (const float*)in[kInputX], (const int32_t*)in[kInputSeqLengths],
          (const float*)in[kInputMask], (const float*)in[kInputWeightIhFwd],
          (const float*)in[kInputWeightHhFwd], (const float*)in[kInputBiasFwd],
          (const float*)in[kInputWeightIhBwd],
          (const float*)in[kInputWeightHhBwd], (const float*)in[kInputBiasBwd],
          (float*)out, batch, t_steps, d, h)) {
    status = kLiteRtStatusErrorRuntimeFailure;
  }

done:
  if (locked_out) g_api.unlock_tensor_buffer(outputs[0]);
  for (int i = 0; i < locked_in; ++i) g_api.unlock_tensor_buffer(inputs[i]);
  return status;
}

static LiteRtStatus kernel_destroy(void* user_data) {
  (void)user_data;
  return kLiteRtStatusOk;
}

static const LiteRtCustomOpKernel kKernel = {
    kernel_init,
    kernel_get_output_layouts,
    kernel_run,
    kernel_destroy,
};

// The `custom_code` the exporter writes is the composite name verbatim,
// dotted prefix included.
LITERT_CUSTOM_OPS_EXPORT LiteRtStatus litert_custom_ops_register_bidirectional_lstm(
    LiteRtOptions options, const LitertCustomOpsApi* api) {
  if (options == NULL || api == NULL || api->add_custom_op_kernel_option == NULL ||
      api->get_tensor_buffer_tensor_type == NULL ||
      api->lock_tensor_buffer == NULL || api->unlock_tensor_buffer == NULL) {
    return kLiteRtStatusErrorInvalidArgument;
  }
  g_api = *api;
  return g_api.add_custom_op_kernel_option(
      options, "litert_custom_op.bidirectional_lstm", /*version=*/1, &kKernel,
      /*user_data=*/NULL);
}
