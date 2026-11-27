import 'package:desktop_drop/desktop_drop.dart';
import 'package:file_selector/file_selector.dart';
import 'package:material_ui/material_ui.dart';
import 'package:path/path.dart' as p;

import '../../app/services.dart';
import '../../core/inference/inference_options.dart';
import '../../core/metrics/metrics_panel.dart';
import '../../core/metrics/run_metrics.dart';
import '../../core/models/model_catalog.dart';
import '../../core/widgets/inference_options_panel.dart';
import '../../core/widgets/model_banner.dart';
import '../../core/widgets/page_header.dart';
import '../../core/widgets/section_card.dart';
import 'engines/qwen3_tts_engine.dart';
import 'engines/tts_engine.dart';
import 'tts_controller.dart';

const _audioTypes = XTypeGroup(
  label: 'Audio',
  extensions: ['wav', 'mp3', 'm4a', 'flac', 'ogg', 'opus', 'aac', 'webm', 'mp4'],
);

class TtsPage extends StatefulWidget {
  const TtsPage({super.key});

  @override
  State<TtsPage> createState() => _TtsPageState();
}

class _TtsPageState extends State<TtsPage> {
  late final TtsController c;
  late final TextEditingController _text;
  final _voiceName = TextEditingController();

  @override
  void initState() {
    super.initState();
    final s = AppServices.instance;
    c = TtsController(
      settings: s.settings,
      models: s.models,
      caps: PlatformCaps.current(),
      // Cloned voices live next to the model but outside its catalog files,
      // so deleting/re-downloading the model keeps them.
      voicesDir: p.join(s.paths.modelDir(ModelCatalog.qwen3Tts06b.id), 'voices', 'cloned'),
      tempDir: p.join(s.paths.cache, 'tts'),
    );
    _text = TextEditingController(text: c.text);
  }

  @override
  void dispose() {
    c.dispose();
    _text.dispose();
    _voiceName.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: Listenable.merge([c, c.models, c.player, c.recorder]),
      builder: (context, _) {
        if (_text.text != c.text) _text.text = c.text;
        return LayoutBuilder(
          builder: (context, box) {
            final wide = box.maxWidth >= 1000;
            final header = Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                PageHeader(
                  icon: Icons.record_voice_over,
                  title: 'Text to speech',
                  subtitle:
                      '${c.engine.label} · ${c.engine.blurb}, fully on-device '
                      'with LiteRT',
                ),
                ModelBanner(manager: c.models, spec: c.spec),
              ],
            );
            final workspace = _workspace(context);
            final side = _sidePanel();
            if (wide) {
              return Padding(
                padding: const EdgeInsets.fromLTRB(24, 20, 24, 20),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    header,
                    Expanded(
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Expanded(child: SingleChildScrollView(child: workspace)),
                          const SizedBox(width: 20),
                          SizedBox(
                            width: 380,
                            child: SingleChildScrollView(child: side),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              );
            }
            return ListView(
              padding: const EdgeInsets.all(16),
              children: [header, workspace, const SizedBox(height: 16), side],
            );
          },
        );
      },
    );
  }

  // ---- workspace ------------------------------------------------------------

  Widget _workspace(BuildContext context) {
    final theme = Theme.of(context);
    final busy = c.isBusy;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SegmentedButton<TtsEngineKind>(
          segments: [
            for (final k in TtsEngineKind.values)
              ButtonSegment(
                value: k,
                label: Text(k.label),
                icon: Icon(
                  k == TtsEngineKind.kokoro ? Icons.bolt_outlined : Icons.translate,
                ),
              ),
          ],
          selected: {c.engine},
          onSelectionChanged: busy ? null : (s) => c.setEngine(s.first),
        ),
        const SizedBox(height: 12),
        SectionCard(
          title: 'Text',
          icon: Icons.notes,
          trailing: Text(
            '${c.text.characters.length} chars',
            style: theme.textTheme.labelSmall,
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextField(
                controller: _text,
                minLines: 4,
                maxLines: 10,
                enabled: !busy,
                decoration: InputDecoration(
                  border: const OutlineInputBorder(),
                  hintText: c.engine == TtsEngineKind.kokoro
                      ? 'English text to speak'
                      : 'Text in any of the 10 supported languages',
                ),
                onChanged: (v) => c.setParams(text: v),
              ),
              const SizedBox(height: 12),
              _voiceControls(),
              const SizedBox(height: 4),
              Row(
                children: [
                  if (c.busy == TtsBusy.synthesizing || c.busy == TtsBusy.loading)
                    FilledButton.tonalIcon(
                      onPressed: c.cancel,
                      icon: const Icon(Icons.stop),
                      label: const Text('Stop'),
                    )
                  else
                    FilledButton.icon(
                      onPressed: !busy && c.modelReady && c.text.trim().isNotEmpty
                          ? c.synthesize
                          : null,
                      icon: const Icon(Icons.graphic_eq),
                      label: const Text('Speak'),
                    ),
                  const SizedBox(width: 12),
                  Expanded(child: _statusLine(theme)),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),
        _playerCard(theme),
        if (c.engine == TtsEngineKind.qwen3) ...[
          const SizedBox(height: 12),
          _cloneCard(context, theme),
        ],
      ],
    );
  }

  Widget _statusLine(ThemeData theme) {
    if (c.error != null) {
      return Text(
        c.error!,
        style: TextStyle(color: theme.colorScheme.error),
        maxLines: 3,
        overflow: TextOverflow.ellipsis,
      );
    }
    return Row(
      children: [
        if (c.isBusy) ...[
          const SizedBox.square(dimension: 16, child: CircularProgressIndicator(strokeWidth: 2)),
          const SizedBox(width: 8),
        ],
        Expanded(
          child: Text(
            c.status ?? (c.modelReady ? 'Ready' : 'Download the model to start'),
            style: theme.textTheme.bodySmall,
            overflow: TextOverflow.ellipsis,
            maxLines: 2,
          ),
        ),
      ],
    );
  }

  Widget _voiceControls() {
    final voices = c.voices;
    final selected = voices.any((v) => v.id == c.voiceId) ? c.voiceId : voices.first.id;
    final enabled = !c.isBusy;
    return Wrap(
      spacing: 16,
      runSpacing: 8,
      crossAxisAlignment: WrapCrossAlignment.center,
      children: [
        SizedBox(
          width: 280,
          child: DropdownButtonFormField<String>(
            initialValue: selected,
            isExpanded: true,
            decoration: const InputDecoration(
              labelText: 'Voice',
              border: OutlineInputBorder(),
              isDense: true,
            ),
            items: [
              for (final v in voices)
                DropdownMenuItem(
                  value: v.id,
                  child: Text(
                    v.group.isEmpty ? v.label : '${v.label}  ·  ${v.group}',
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
            ],
            onChanged: enabled ? (v) => c.setParams(voice: v) : null,
          ),
        ),
        if (c.engine == TtsEngineKind.qwen3)
          SizedBox(
            width: 180,
            child: DropdownButtonFormField<String>(
              initialValue: c.language,
              decoration: const InputDecoration(
                labelText: 'Language',
                border: OutlineInputBorder(),
                isDense: true,
              ),
              items: [
                const DropdownMenuItem(value: 'auto', child: Text('Auto-detect')),
                for (final l in Qwen3TtsEngine.languages.keys)
                  DropdownMenuItem(
                    value: l,
                    child: Text('${l[0].toUpperCase()}${l.substring(1)}'),
                  ),
              ],
              onChanged: enabled ? (v) => c.setParams(language: v) : null,
            ),
          )
        else
          SizedBox(
            width: 260,
            child: Row(
              children: [
                Text('Speed ${c.speed.toStringAsFixed(2)}×'),
                Expanded(
                  child: Slider(
                    value: c.speed,
                    min: 0.5,
                    max: 2.0,
                    divisions: 30,
                    onChanged: enabled ? (v) => c.setParams(speed: v) : null,
                  ),
                ),
              ],
            ),
          ),
      ],
    );
  }

  Widget _playerCard(ThemeData theme) {
    final pl = c.player;
    final total = pl.seconds;
    final pos = pl.position.inMilliseconds / 1000;
    String fmt(double s) =>
        '${s ~/ 60}:${(s % 60).floor().toString().padLeft(2, '0')}';
    return SectionCard(
      title: 'Audio',
      icon: Icons.volume_up_outlined,
      trailing: pl.buffering && pl.playing
          ? Text('Buffering…', style: theme.textTheme.labelSmall)
          : null,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              IconButton.filledTonal(
                tooltip: pl.playing ? 'Pause' : 'Play from start',
                onPressed: !pl.hasAudio
                    ? null
                    : pl.playing
                    ? pl.togglePause
                    : pl.play,
                icon: Icon(pl.playing ? Icons.pause : Icons.play_arrow),
              ),
              IconButton(
                tooltip: 'Stop',
                onPressed: pl.playing ? pl.stop : null,
                icon: const Icon(Icons.stop),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: LinearProgressIndicator(
                  value: total == 0 ? 0 : (pos / total).clamp(0.0, 1.0),
                ),
              ),
              const SizedBox(width: 12),
              Text(
                '${fmt(pos)} / ${fmt(total)}${pl.streaming ? ' …' : ''}',
                style: theme.textTheme.bodySmall,
              ),
              const SizedBox(width: 8),
              TextButton.icon(
                onPressed: pl.hasAudio && !pl.streaming ? _saveWav : null,
                icon: const Icon(Icons.save_alt, size: 18),
                label: const Text('Save WAV'),
              ),
            ],
          ),
          if (pl.error != null)
            Text(pl.error!, style: TextStyle(color: theme.colorScheme.error)),
        ],
      ),
    );
  }

  Future<void> _saveWav() async {
    final loc = await getSaveLocation(
      suggestedName: 'speech-${DateTime.now().millisecondsSinceEpoch}.wav',
      acceptedTypeGroups: const [XTypeGroup(label: 'WAV', extensions: ['wav'])],
    );
    if (loc == null) return;
    final path = loc.path.toLowerCase().endsWith('.wav') ? loc.path : '${loc.path}.wav';
    await c.saveWav(path);
  }

  // ---- voice cloning ----------------------------------------------------------

  Widget _cloneCard(BuildContext context, ThemeData theme) {
    final busy = c.isBusy;
    final rec = c.recorder;
    final ref = c.reference;
    return SectionCard(
      title: 'Clone a voice',
      icon: Icons.content_copy_outlined,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Text(
            'Qwen3-TTS Base clones the timbre of a reference voice from its '
            'speaker embedding (x-vector). Use 5–15 s of clean speech; the '
            'reference transcript is not needed.',
            style: theme.textTheme.bodySmall,
          ),
          const SizedBox(height: 12),
          if (!c.hasSpeakerEncoder) _encoderMissing(theme),
          DropTarget(
            onDragDone: (d) {
              if (d.files.isNotEmpty && !busy) c.setReferenceFile(d.files.first.path);
            },
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                border: Border.all(color: theme.colorScheme.outlineVariant),
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                children: [
                  Icon(
                    ref == null ? Icons.upload_file : Icons.audio_file,
                    color: theme.colorScheme.primary,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      rec.recording
                          ? 'Recording… ${rec.elapsed.inSeconds} s'
                          : ref == null
                          ? 'Drop an audio file here, pick one, or record'
                          : '${c.referenceName} · ${ref.seconds.toStringAsFixed(1)} s',
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  if (rec.recording)
                    SizedBox(
                      width: 60,
                      child: LinearProgressIndicator(value: rec.level),
                    ),
                  IconButton(
                    tooltip: 'Pick audio file',
                    onPressed: busy || rec.recording ? null : _pickReference,
                    icon: const Icon(Icons.folder_open_outlined),
                  ),
                  IconButton(
                    tooltip: rec.recording ? 'Stop recording' : 'Record from microphone',
                    onPressed: busy ? null : c.toggleRecording,
                    icon: Icon(rec.recording ? Icons.stop_circle : Icons.mic_none),
                    color: rec.recording ? theme.colorScheme.error : null,
                  ),
                  IconButton(
                    tooltip: 'Listen to reference',
                    onPressed: ref == null || busy || rec.recording ? null : c.previewReference,
                    icon: const Icon(Icons.hearing),
                  ),
                ],
              ),
            ),
          ),
          if (rec.error != null)
            Padding(
              padding: const EdgeInsets.only(top: 6),
              child: Text(rec.error!, style: TextStyle(color: theme.colorScheme.error)),
            ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _voiceName,
                  enabled: !busy,
                  decoration: const InputDecoration(
                    labelText: 'Voice name',
                    border: OutlineInputBorder(),
                    isDense: true,
                  ),
                  onChanged: (_) => setState(() {}),
                ),
              ),
              const SizedBox(width: 12),
              FilledButton.icon(
                onPressed: !busy &&
                        ref != null &&
                        c.hasSpeakerEncoder &&
                        _voiceName.text.trim().isNotEmpty
                    ? () => c.cloneVoice(_voiceName.text)
                    : null,
                icon: const Icon(Icons.person_add_alt),
                label: const Text('Clone'),
              ),
            ],
          ),
          if (c.customVoices.isNotEmpty) ...[
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final v in c.customVoices)
                  InputChip(
                    label: Text(v.name),
                    selected: c.qwenVoice == v.id,
                    onPressed: busy ? null : () => c.setParams(voice: v.id),
                    onDeleted: busy ? null : () => _confirmDelete(context, v),
                  ),
              ],
            ),
          ],
        ],
      ),
    );
  }

  Widget _encoderMissing(ThemeData theme) {
    return Card(
      elevation: 0,
      color: theme.colorScheme.secondaryContainer,
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          children: [
            Icon(Icons.info_outline, color: theme.colorScheme.secondary),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                'Cloning needs speaker_encoder.tflite, which the LiteRT '
                'conversion does not ship. Build it once with '
                'tools/tts/export_qwen3_speaker_encoder.py (Linux/macOS/WSL), '
                'then import it here.',
                style: theme.textTheme.bodySmall,
              ),
            ),
            const SizedBox(width: 12),
            OutlinedButton(
              onPressed: c.isBusy ? null : _importEncoder,
              child: const Text('Import…'),
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _pickReference() async {
    final f = await openFile(acceptedTypeGroups: const [_audioTypes]);
    if (f != null) await c.setReferenceFile(f.path);
  }

  Future<void> _importEncoder() async {
    final f = await openFile(
      acceptedTypeGroups: const [
        XTypeGroup(label: 'LiteRT model', extensions: ['tflite']),
      ],
    );
    if (f != null) await c.importSpeakerEncoder(f.path);
  }

  Future<void> _confirmDelete(BuildContext context, CustomVoice v) async {
    final ok = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: Text('Delete voice "${v.name}"?'),
        content: Text('This removes ${v.path}.'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx, false), child: const Text('Cancel')),
          FilledButton(onPressed: () => Navigator.pop(ctx, true), child: const Text('Delete')),
        ],
      ),
    );
    if (ok == true) await c.deleteVoice(v);
  }

  // ---- side panel ---------------------------------------------------------------

  Widget _sidePanel() {
    final enabled = !c.isBusy;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        InferenceOptionsPanel(
          options: c.options,
          caps: c.caps,
          enabled: enabled,
          loadInfo: c.loaded ? c.loadInfo : null,
          needsReload: c.needsReload,
          onChanged: c.setOptions,
          onLoad: c.modelReady ? c.loadModel : null,
          onUnload: c.unloadModel,
          extra: [
            if (c.engine == TtsEngineKind.qwen3) ...[
              SwitchListTile(
                dense: true,
                contentPadding: EdgeInsets.zero,
                title: const Text('Greedy decoding'),
                subtitle: const Text('Deterministic; matches the PyTorch reference'),
                value: c.greedy,
                onChanged: enabled ? (v) => c.setParams(greedy: v) : null,
              ),
              if (!c.greedy) ...[
                LabeledField(
                  label: 'Temperature: ${c.temperature.toStringAsFixed(2)}',
                  child: Slider(
                    value: c.temperature,
                    min: 0.1,
                    max: 1.5,
                    divisions: 28,
                    onChanged: enabled ? (v) => c.setParams(temperature: v) : null,
                  ),
                ),
                LabeledField(
                  label: 'Top-k: ${c.topK}',
                  child: Slider(
                    value: c.topK.toDouble(),
                    min: 1,
                    max: 100,
                    divisions: 99,
                    onChanged: enabled ? (v) => c.setParams(topK: v.round()) : null,
                  ),
                ),
              ],
            ],
          ],
        ),
        const SizedBox(height: 12),
        MetricsPanel(
          loadInfo: c.loaded ? c.loadInfo : null,
          lastRun: null,
          history: const [],
          runSection: _TtsMetrics(c),
        ),
      ],
    );
  }
}

class _TtsMetrics extends StatelessWidget {
  const _TtsMetrics(this.c);
  final TtsController c;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final r = c.lastRun;
    Widget grid(List<(String, String)> items) => Wrap(
      spacing: 16,
      runSpacing: 8,
      children: [
        for (final (k, v) in items)
          SizedBox(
            width: 150,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(k, style: theme.textTheme.labelSmall?.copyWith(color: theme.colorScheme.onSurfaceVariant)),
                Text(v, style: theme.textTheme.titleSmall),
              ],
            ),
          ),
      ],
    );
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        SectionCard(
          title: 'Synthesis cost (last run)',
          icon: Icons.speed,
          child: r == null
              ? Text(
                  'Speak some text to see timings.',
                  style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                )
              : Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Text(
                      '${r.engine.label} · ${r.voice}${r.cancelled ? ' · stopped' : ''}',
                      style: theme.textTheme.bodyMedium,
                    ),
                    Text(
                      '${r.optionsSummary}  →  ${r.accelerator}',
                      style: theme.textTheme.bodySmall?.copyWith(color: theme.colorScheme.onSurfaceVariant),
                    ),
                    const SizedBox(height: 10),
                    grid([
                      ('Audio', '${r.stats.audioSeconds.toStringAsFixed(2)} s'),
                      ('Wall time', formatMs(r.stats.totalMs)),
                      ('Real-time factor', r.stats.rtf.toStringAsFixed(2)),
                      ('First audio', formatMs(r.stats.firstAudioMs)),
                      if (r.stats.units > 0) ('${r.stats.unitLabel}s', '${r.stats.units}'),
                      if (r.loadMs != null) ('Load + warm-up', formatMs(r.loadMs!)),
                      ('Peak RAM', formatBytes(r.rssPeak)),
                      ('RAM growth', formatBytes(r.rssDelta)),
                    ]),
                    if (r.stats.stagesMs.isNotEmpty) ...[
                      const SizedBox(height: 10),
                      Text('Stages', style: theme.textTheme.labelMedium),
                      const SizedBox(height: 4),
                      for (final e in r.stats.stagesMs.entries)
                        Row(
                          children: [
                            SizedBox(width: 90, child: Text(e.key, style: theme.textTheme.bodySmall)),
                            Expanded(
                              child: LinearProgressIndicator(
                                value: r.stats.totalMs == 0 ? 0 : (e.value / r.stats.totalMs).clamp(0.0, 1.0),
                              ),
                            ),
                            const SizedBox(width: 8),
                            SizedBox(
                              width: 70,
                              child: Text(formatMs(e.value), textAlign: TextAlign.end, style: theme.textTheme.bodySmall),
                            ),
                          ],
                        ),
                    ],
                  ],
                ),
        ),
        if (c.history.length > 1) ...[
          const SizedBox(height: 12),
          SectionCard(
            title: 'History',
            icon: Icons.history,
            child: Column(
              children: [
                for (final h in c.history.take(10))
                  ListTile(
                    dense: true,
                    contentPadding: EdgeInsets.zero,
                    title: Text(
                      h.text,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    subtitle: Text(
                      '${h.engine.label} · ${h.voice} · ${h.optionsSummary}',
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    trailing: Text(
                      '${h.stats.audioSeconds.toStringAsFixed(1)} s · RTF ${h.stats.rtf.toStringAsFixed(2)}',
                      style: theme.textTheme.bodySmall,
                    ),
                  ),
              ],
            ),
          ),
        ],
      ],
    );
  }
}
