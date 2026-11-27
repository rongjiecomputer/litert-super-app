import 'dart:convert';

import 'package:material_ui/material_ui.dart';
import 'package:shared_preferences/shared_preferences.dart';

import 'inference/inference_options.dart';

/// App-wide preferences plus per-use-case inference options.
class AppSettings extends ChangeNotifier {
  AppSettings(this._prefs);

  final SharedPreferences _prefs;

  ThemeMode get themeMode => ThemeMode.values.firstWhere(
    (m) => m.name == _prefs.getString('themeMode'),
    orElse: () => ThemeMode.system,
  );
  set themeMode(ThemeMode mode) {
    _prefs.setString('themeMode', mode.name);
    notifyListeners();
  }

  /// Optional explicit ffmpeg executable (or its folder).
  String? get ffmpegPath {
    final v = _prefs.getString('ffmpegPath');
    return v == null || v.trim().isEmpty ? null : v.trim();
  }

  set ffmpegPath(String? value) {
    if (value == null || value.trim().isEmpty) {
      _prefs.remove('ffmpegPath');
    } else {
      _prefs.setString('ffmpegPath', value.trim());
    }
    notifyListeners();
  }

  InferenceOptions inferenceOptions(String useCaseId, PlatformCaps caps) {
    final raw = _prefs.getString('options.$useCaseId');
    if (raw == null) return const InferenceOptions().sanitized(caps);
    try {
      final json = jsonDecode(raw) as Map<String, Object?>;
      return InferenceOptions.fromJson(json).sanitized(caps);
    } catch (_) {
      return const InferenceOptions().sanitized(caps);
    }
  }

  void setInferenceOptions(String useCaseId, InferenceOptions options) {
    _prefs.setString('options.$useCaseId', jsonEncode(options.toJson()));
  }

  Map<String, Object?> useCaseParams(String useCaseId) {
    final raw = _prefs.getString('params.$useCaseId');
    if (raw == null) return {};
    try {
      return jsonDecode(raw) as Map<String, Object?>;
    } catch (_) {
      return {};
    }
  }

  void setUseCaseParams(String useCaseId, Map<String, Object?> params) {
    _prefs.setString('params.$useCaseId', jsonEncode(params));
  }
}
