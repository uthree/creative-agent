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

採譜の[実測報告](docs/audio-midi-smoke-report.ja.md)に、CPU/GPUでの実更新・再開、MIDI変換の確認と、現在の生成品質の未達を記録している。

[LFM3の5タスク混合学習](docs/audio-multitask-lfm3.ja.md)は、事前学習済みLFM3-1.7Bへencoder・projector・LoRAを接続し、SSL・caption・正常/複合診断・発話書き起こし・MIDIを交互に学習する。TensorBoardで入力音声、正解、LLM応答と、MIDI楽譜の比較画像を確認できる。

合計5,000更新・各タスク1,000更新を完了した。[実測報告](docs/audio-multitask-lfm3-5000-report.ja.md)に、訓練例への適合、未学習例での汎化不足、保存済み重みとTensorBoardの照合を記録している。

[公開データセットの調査](docs/public-audio-datasets-survey.ja.md)に、音楽波形、発話と書き起こし、同期MIDI、記譜データの規模・入手方法・利用条件と、実データへ拡張する順序をまとめている。

[公開実データの学習経路](docs/audio-public-multitask.ja.md)では、Mini LibriSpeech・JSUT・BabySlakhを取得し、原音の分割、区間の遅延読み込み、明示的resample、5課題の学習前後の自由生成評価を行う。楽器説明の弱教師と、日本語・音楽の評価範囲も明記している。

公開データでの追加5,000更新と再現監査を完了した。[実測報告](docs/audio-public-multitask-lfm3-5000-report.ja.md)に、楽器短文の改善、発話・採譜・診断の汎化未達、無音・別音声による対照結果を記録している。

[課題ごとの単独学習](docs/audio-public-single-task.ja.md)では、同じ初期重みから各課題1,000更新を行い、混合学習と比較する。採譜には30秒・60秒の追加条件を用意する。

[単独学習の実測結果](docs/audio-public-single-task-report.ja.md)では、SSL・診断の部分改善、楽器短文・ASR・短区間採譜の比較、30秒・60秒採譜の精度未達を記録している。計6,000更新と保存重みからの再現監査を完了した。

[開発と学習の実行計画](docs/implementation-plan.ja.md)に、Aurisでの既存実験、2段階学習、評価条件、実装順序、計算予算をまとめている。

最初の実用成果は、時刻と対象を添えた音声講評。制作ツールでの編集と再観測を段階的に統合する。初回は音声・音楽を対象とし、動画学習は後続とする。

ソース、設定、依存lock、スキーマ、fixture生成コード、公開用の小さな実験集計をGitで管理する。学習データと完全なmanifestは `.local/` または `data/`、実行ログとチェックポイントは `runs/` に保存する。これらとモデル重みは [.gitignore](.gitignore) で除外する。

コードとドキュメントは作業単位ごとに検証し、commit・pushする。作業規約は [AGENTS.md](AGENTS.md) を参照。
