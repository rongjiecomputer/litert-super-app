// Compiles native/bidirectional_lstm.c into the `litert_custom_ops` code
// asset (see lib/core/litert/custom_ops.dart).
import 'package:code_assets/code_assets.dart';
import 'package:hooks/hooks.dart';
import 'package:native_toolchain_c/native_toolchain_c.dart';

Future<void> main(List<String> args) async {
  await build(args, (input, output) async {
    if (!input.config.buildCodeAssets) return;
    final builder = CBuilder.library(
      name: 'litert_custom_ops',
      assetName: 'core/litert/custom_ops_native.dart',
      sources: ['native/bidirectional_lstm.c'],
    );
    await builder.run(input: input, output: output);
  });
}
