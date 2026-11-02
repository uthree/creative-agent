# creative-agent

音声・音楽を理解し、制作アプリで作成・修正・修復を完了するエージェントの学習基盤。

新規SSLの標準構成は、**50 ms波形パッチ → Linear 4,800→128 → 双方向Transformer 2層**の`patch20`とする。20 FPS・128次元を出力し、CNN前段と平均プーリングを使わない。[採用方針と起動方法](docs/audio-patch20.ja.md)、[比較結果](docs/audio-patch20-report.ja.md)を参照。新規plan・encoder・単独学習の既定値も`patch20`である。

新規SSLの学習目的は**EMA教師の潜在予測のみ（`latent`）**を採用する。[損失・教師更新・実際の設定値](docs/audio-latent-ssl.ja.md)を参照。`run_ssl_objectives.py train`の既定値も`latent`で、`suite`は過去の③→②比較を再現する専用経路として残す。旧クラスタSSL・LLM課題の学習経路は別であり、この選択だけでは置き換わらない。

[GPT-2式の残差初期化とVICReg型損失](docs/audio-gpt2-vicreg.ja.md)では、残差出力層を標準偏差`0.02 / sqrt(2L)`で初期化し、潜在予測のみを先に検証した後、同じ初期値で分散・共分散項を加える。新規latent用CLIはGPT-2式を既定値とし、旧configの再現用fallbackは保持する。

[初期化の対照実験](docs/audio-ssl-initialization-comparison.ja.md)では、残差出力重みの1e-4初期化と、メル尺度に並べたsin/cosによるパッチ投影初期化を個別に検証する。共通の凍結probeに加え、固定validationでゼロ予測との差、教師・生徒の分散、有効rank、入力依存、残差の相対量を追う。
[各10k更新の実測](docs/audio-ssl-initialization-report.ja.md)では、残差初期化がtest音響accuracy 34.13%・MIDIフレームF1 19.26%・AP 17.27%で対照を上回り、正弦波初期化は対照を下回った。1 seedの結果で、MIDI validationの優位は一貫していない。

[発話・音楽・環境音に共通するSSLの調査](docs/audio-universal-ssl-survey.ja.md)では、M2D、WavJEPA、SSLAM、USAD 2.0等を比較し、直接パッチモデルのEMA潜在予測と音響復元補助損失を候補として整理した。各方式の限界と汎化評価を記載している。

[SSL目的の③→②比較](docs/audio-ssl-objective-comparison.ja.md)では、音響復元補助付きのEMA潜在予測、潜在予測だけの順で学習し、共通の凍結線形probeによる音響クラスタ精度とMIDI押下フレームF1を比較できる。
[各1,000更新の実測](docs/audio-ssl-objective-comparison-report.ja.md)では、②が音響accuracy 31.14%・MIDIフレームF1 18.53%で③を上回った。1 seedの簡易比較で、採譜やASRの完成精度とは区別する。
[各10,000更新への継続結果](docs/audio-ssl-objectives-10k-report.ja.md)では、音響accuracyは③33.33%・②33.40%でほぼ同率となった。MIDIフレームF1は③17.23%・②15.38%、音高micro APは③11.74%・②11.97%。学習量の追加だけで採譜精度が大きく改善する結果ではなかった。

初期方式の自己教師あり学習スモークも実装済み。[実行手順](docs/audio-ssl-smoke.ja.md)に従って、自作の48 kHzステレオ音声で前処理、マスク付き復元、実更新、保存、再開を確認できる。

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

[LLMを変更した切り分け実験](docs/audio-llm-diagnostics-report.ja.md)では、LFM3とQwen3の音声入力・正解情報テキスト入力をASRと30秒・60秒採譜で比較した。計8,000更新で全条件の訓練例を完全再現できたが、未知音声では訓練答案のコピーが多く、LLM変更では解決しなかった。[条件と実行手順](docs/audio-llm-diagnostics.ja.md)も参照。

既存の100 FPS encoderを使うLLM学習では、5フレームを時刻順に連結してからLLM次元へ投影する。[接続方式と旧checkpointの扱い](docs/audio-multitask-lfm3.ja.md#フレーム連結への変更と旧成果物)に、末尾padding、旧平均cacheの識別、projectorの再学習を記載している。標準の`patch20`はencoder自体が20 FPSを返すため、今後のLLM接続ではこの表現を直接投影する契約へ移行する。

[20 FPS CNNと浅いTransformerのSSL比較](docs/audio-representation-comparison.ja.md)では、音楽・英語・日本語を使い、固定の離散教師に対するmask予測と混合音声のdenoisingを学習する。100 FPSで文脈処理する対照、先に連結する対照、Conformerを参考にしたCNNを同じ条件で比較する。 [実測報告](docs/audio-representation-comparison-report.ja.md)では9 run・9,000更新を監査し、先行連結20 FPSの近い集計CEと長音声の高速化、今回のdepthwise CNN案の精度未達を記録している。

[段階的な残差CNN → 20 FPS → Transformer](docs/audio-seanet20.ja.md)は、EnCodecのSEANetを参考にしたLSTMなしの前段を追加する。`seanet20`と`early_concat`を同じSSL経路で比較できる。
[ELU構成の比較結果](docs/audio-seanet20-report.ja.md)では各1,000更新・3 seedを実行し、新CNNの入力依存の弱さと対照に対する精度未達を記録した。

[50 ms波形パッチ → 線形投影 → Transformer](docs/audio-patch20.ja.md)の`patch20`では、CNN前段を使わない直接パッチ埋め込みを同じSSL経路で比較できる。
[6,000更新の実測結果](docs/audio-patch20-report.ja.md)では、対照より通常音声CEが全3 seedで低下し、混合CEも平均で低下した。パラメータ数は約1.94倍の比較である。

[開発と学習の実行計画](docs/implementation-plan.ja.md)に、Aurisでの既存実験、2段階学習、評価条件、実装順序、計算予算をまとめている。

最初の実用成果は、時刻と対象を添えた音声講評。制作ツールでの編集と再観測を段階的に統合する。初回は音声・音楽を対象とし、動画学習は後続とする。

ソース、設定、依存lock、スキーマ、fixture生成コード、公開用の小さな実験集計をGitで管理する。学習データと完全なmanifestは `.local/` または `data/`、実行ログとチェックポイントは `runs/` に保存する。これらとモデル重みは [.gitignore](.gitignore) で除外する。

コードとドキュメントは作業単位ごとに検証し、commit・pushする。作業規約は [AGENTS.md](AGENTS.md) を参照。
