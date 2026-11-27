import 'dart:math' as math;
import 'dart:typed_data';

/// Greedy argmax or top-k/temperature sampling (HF semantics), matching
/// `_pick` in the Qwen3-TTS LiteRT reference pipeline.
int pickToken(
  Float32List logits,
  math.Random rng, {
  required bool greedy,
  int topK = 50,
  double temperature = 0.9,
  int offset = 0,
  int? length,
}) {
  final n = length ?? logits.length - offset;
  if (greedy) {
    var best = 0;
    var bestV = logits[offset];
    for (var i = 1; i < n; i++) {
      final v = logits[offset + i];
      if (v > bestV) {
        bestV = v;
        best = i;
      }
    }
    return best;
  }
  final t = math.max(temperature, 1e-6);
  final k = topK > 0 && topK < n ? topK : n;
  // Indices of the k largest logits (partial selection by min-heap).
  final idx = Int32List(k);
  final val = Float64List(k);
  var size = 0;
  void siftDown(int i) {
    while (true) {
      final l = 2 * i + 1, r = l + 1;
      var m = i;
      if (l < size && val[l] < val[m]) m = l;
      if (r < size && val[r] < val[m]) m = r;
      if (m == i) return;
      final tv = val[i], ti = idx[i];
      val[i] = val[m];
      idx[i] = idx[m];
      val[m] = tv;
      idx[m] = ti;
      i = m;
    }
  }

  for (var i = 0; i < n; i++) {
    final v = logits[offset + i] / t;
    if (size < k) {
      var j = size++;
      val[j] = v;
      idx[j] = i;
      while (j > 0) {
        final p = (j - 1) >> 1;
        if (val[p] <= val[j]) break;
        final tv = val[p], ti = idx[p];
        val[p] = val[j];
        idx[p] = idx[j];
        val[j] = tv;
        idx[j] = ti;
        j = p;
      }
    } else if (v > val[0]) {
      val[0] = v;
      idx[0] = i;
      siftDown(0);
    }
  }
  var maxV = double.negativeInfinity;
  for (var i = 0; i < size; i++) {
    maxV = math.max(maxV, val[i]);
  }
  var sum = 0.0;
  for (var i = 0; i < size; i++) {
    val[i] = math.exp(val[i] - maxV);
    sum += val[i];
  }
  var r = rng.nextDouble() * sum;
  for (var i = 0; i < size; i++) {
    r -= val[i];
    if (r <= 0) return idx[i];
  }
  return idx[size - 1];
}

/// HF repetition penalty, in place: positive scores are divided, negative
/// ones multiplied, for every token in [history].
void applyRepetitionPenalty(
  Float32List scores,
  Iterable<int> history,
  double penalty,
) {
  if (penalty == 1.0) return;
  for (final t in history) {
    final s = scores[t];
    scores[t] = s > 0 ? s / penalty : s * penalty;
  }
}
