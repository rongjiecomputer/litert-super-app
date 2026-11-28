import 'dart:ffi';

/// Entry point of the `litert_custom_ops` code asset, built from
/// `native/bidirectional_lstm.c` by `hook/build.dart`.
///
/// Adds the `litert_custom_op.bidirectional_lstm` kernel to a `LiteRtOptions`.
/// [api] points at a struct of the four LiteRT C API functions the kernel
/// calls (see `LitertCustomOpsApi` in the C source).
@Native<Int32 Function(Pointer<Void>, Pointer<Void>)>(
  symbol: 'litert_custom_ops_register_bidirectional_lstm',
)
external int litertCustomOpsRegisterBidirectionalLstm(
  Pointer<Void> options,
  Pointer<Void> api,
);
