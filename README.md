# creative-agent

音声・音楽を理解し、制作アプリで作成・修正・修復を完了するエージェントの学習基盤。

音声の自己教師あり学習スモークを実装済み。[実行手順](docs/audio-ssl-smoke.ja.md)に従って、自作の48 kHzステレオ音声で前処理、マスク付き復元、実更新、保存、再開を確認できる。

RTX 3070で20更新と別プロセスからの21更新目への再開を確認した。[実測結果](docs/audio-ssl-smoke-report.ja.md)に、検証内容と現在の学習品質の限界をまとめている。

[キャプション学習スモーク](docs/audio-caption-smoke.ja.md)では、音声エンコーダーと小型Qwen3を凍結し、キャプションの応答lossから接続部だけを更新する。SSL checkpointの読み込み、loss mask、生成、保存・再開を検証できる。

[問題検出の学習スモーク](docs/audio-diagnosis-smoke.ja.md)では、音切れ・クリッピング・左右レベルの偏りを多ラベルで学習する。正常・単一・複合問題のデータ生成、時間とchの教師、誤警報の採点、encoderと診断headの更新・保存・再開を扱う。

診断スモークの[実測報告](docs/audio-diagnosis-smoke-report.ja.md)に、正常・複合別の結果と、現在の検出品質の未達を記録している。

[発話の書き起こしスモーク](docs/audio-transcription-smoke.ja.md)では、音声の潜在表現をprojector経由で小型Qwen3へ渡す。書き起こし文の応答CE、教師なし生成、CER/WER、保存・再開を検証する。

書き起こしの[実測報告](docs/audio-transcription-smoke-report.ja.md)には、CPU/GPUでの実更新・再開と、現在の認識品質の未達を記録している。

[音源からMIDIへの採譜スモーク](docs/audio-midi-smoke.ja.md)では、音声潜在表現からLLMが音高・開始・終了・velocityの音符列を生成する。旋律・和音・同音連打・無音の合成教師、音符F1、MIDI往復、単体推論、学習の保存・再開を扱う。

[開発と学習の実行計画](docs/implementation-plan.ja.md)に、Aurisでの既存実験、2段階学習、評価条件、実装順序、計算予算をまとめている。

最初の実用成果は、時刻と対象を添えた音声講評。制作ツールでの編集と再観測を段階的に統合する。初回は音声・音楽を対象とし、動画学習は後続とする。

ソース、設定、依存lock、スキーマ、fixture生成コード、公開用の小さな実験集計をGitで管理する。学習データと完全なmanifestは `.local/` または `data/`、実行ログとチェックポイントは `runs/` に保存する。これらとモデル重みは [.gitignore](.gitignore) で除外する。

コードとドキュメントは作業単位ごとに検証し、commit・pushする。作業規約は [AGENTS.md](AGENTS.md) を参照。
