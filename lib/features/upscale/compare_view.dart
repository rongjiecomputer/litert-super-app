import 'dart:ui' as ui;

import 'package:material_ui/material_ui.dart';

/// Before/after split view with zoom & pan. Both images are drawn at the
/// same on-screen size, so the input appears bilinear-upscaled.
class CompareView extends StatefulWidget {
  const CompareView({
    super.key,
    required this.before,
    required this.after,
    this.beforeLabel = 'Input (bilinear)',
    this.afterLabel = 'Real-ESRGAN ×4',
  });

  final ui.Image before;
  final ui.Image after;
  final String beforeLabel, afterLabel;

  @override
  State<CompareView> createState() => _CompareViewState();
}

class _CompareViewState extends State<CompareView> {
  double _split = 0.5;
  final _transform = TransformationController();

  @override
  void dispose() {
    _transform.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final cs = Theme.of(context).colorScheme;
    final aspect = widget.after.width / widget.after.height;
    return Column(
      children: [
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: ColoredBox(
              color: cs.surfaceContainerHighest,
              child: LayoutBuilder(
                builder: (context, c) {
                  var w = c.maxWidth, h = w / aspect;
                  if (h > c.maxHeight) {
                    h = c.maxHeight;
                    w = h * aspect;
                  }
                  return InteractiveViewer(
                    transformationController: _transform,
                    maxScale: 32,
                    child: Center(
                      child: SizedBox(
                        width: w,
                        height: h,
                        child: Stack(
                          fit: StackFit.expand,
                          children: [
                            RawImage(
                              image: widget.before,
                              fit: BoxFit.fill,
                              filterQuality: FilterQuality.low,
                            ),
                            ClipRect(
                              clipper: _RightOf(_split),
                              child: RawImage(
                                image: widget.after,
                                fit: BoxFit.fill,
                                filterQuality: FilterQuality.medium,
                              ),
                            ),
                            Align(
                              alignment: Alignment(_split * 2 - 1, 0),
                              child: Container(
                                width: 2,
                                color: Colors.white.withValues(alpha: 0.85),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
        ),
        const SizedBox(height: 4),
        Row(
          children: [
            _Tag(widget.beforeLabel),
            Expanded(
              child: Slider(
                value: _split,
                onChanged: (v) => setState(() => _split = v),
              ),
            ),
            _Tag(widget.afterLabel),
            IconButton(
              tooltip: 'Reset zoom',
              onPressed: () => _transform.value = Matrix4.identity(),
              icon: const Icon(Icons.zoom_out_map, size: 18),
            ),
          ],
        ),
      ],
    );
  }
}

class _RightOf extends CustomClipper<Rect> {
  _RightOf(this.split);
  final double split;
  @override
  Rect getClip(Size size) =>
      Rect.fromLTRB(size.width * split, 0, size.width, size.height);
  @override
  bool shouldReclip(_RightOf old) => old.split != split;
}

class _Tag extends StatelessWidget {
  const _Tag(this.text);
  final String text;
  @override
  Widget build(BuildContext context) =>
      Text(text, style: Theme.of(context).textTheme.labelMedium);
}
