import 'dart:ffi';

import 'package:ffi/ffi.dart';

import 'custom_ops_native.dart';
import 'ffi.dart';

/// Registers the custom op kernels LiteRT does not ship, so graphs using them
/// can be compiled with the `CompiledModel` API.
///
/// Currently: `litert_custom_op.bidirectional_lstm`, which the LiteRT-LM
/// Kokoro acoustic graph uses (see `native/bidirectional_lstm.c`). It runs on
/// the CPU. Registering is harmless for graphs that don't use the op.
///
/// Must be called on the [options] before `LiteRtCreateCompiledModel`.
void registerCustomOps(LiteRtApi api, Handle options) {
  final lib = api.library;
  // Mirror of `LitertCustomOpsApi` in native/bidirectional_lstm.c; the C side
  // copies it, so it can be freed right after the call.
  final fns = calloc<Pointer<Void>>(4);
  try {
    fns[0] = lib.lookup('LiteRtAddCustomOpKernelOption').cast();
    fns[1] = lib.lookup('LiteRtGetTensorBufferTensorType').cast();
    fns[2] = lib.lookup('LiteRtLockTensorBuffer').cast();
    fns[3] = lib.lookup('LiteRtUnlockTensorBuffer').cast();
    api.check(
      litertCustomOpsRegisterBidirectionalLstm(options, fns.cast()),
      'RegisterCustomOps(bidirectional_lstm)',
    );
  } finally {
    calloc.free(fns);
  }
}
