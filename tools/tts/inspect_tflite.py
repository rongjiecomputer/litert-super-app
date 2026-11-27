"""Dumps signatures and tensor details of .tflite graphs.

    py -I tools/tts/inspect_tflite.py <model.tflite> [...]

Needs `pip install ai-edge-litert numpy`. Used to confirm the I/O of the
legacy Qwen3-TTS / Kokoro graphs that the Dart engines drive.
"""
import sys

from ai_edge_litert.interpreter import Interpreter


def main(paths):
    for path in paths:
        print(f'=== {path}')
        interp = Interpreter(model_path=path)
        sigs = interp.get_signature_list()
        if not sigs:
            for d in interp.get_input_details():
                print('  in ', d['name'], d['shape'].tolist(), d['dtype'].__name__)
            for d in interp.get_output_details():
                print('  out', d['name'], d['shape'].tolist(), d['dtype'].__name__)
            continue
        for key in sigs:
            print(f'  [{key}]')
            runner = interp.get_signature_runner(key)
            for name, d in sorted(runner.get_input_details().items()):
                print('    in ', name, d['shape'].tolist(), d['dtype'].__name__)
            for name, d in sorted(runner.get_output_details().items()):
                print('    out', name, d['shape'].tolist(), d['dtype'].__name__)


if __name__ == '__main__':
    main(sys.argv[1:])
