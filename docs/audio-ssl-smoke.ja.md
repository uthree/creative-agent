# 音声の自己教師あり学習スモーク

最初の音声学習経路として、原音の一部を隠して、その区間の左右別スペクトルを予測する小型モデルを実装した。自作音声の生成、読み込み、可変長batch、loss、逆伝播、optimizer更新、TensorBoard、保存、別プロセスからの再開までを確認する。

このスモークが確認するのは第1段階の学習処理である。音楽理解、局所異常の診断精度、講評、LLMへの接続、長時間音源の性能は後続の評価対象となる。

## 環境

Python 3.12とuvを使う。依存は `uv.lock` に固定し、CPUとCUDAのPyTorchを排他的なgroupに分ける。LinuxとWindows向けのCPU検証workflowを用意している。

CPUでは次を実行する。

```bash
uv sync --locked --group dev
uv run --no-sync pytest
uv run --no-sync creative-agent smoke-audio-ssl \
  --config recipes/audio-ssl-smoke.json \
  --device cpu --steps 20 --output runs/audio-ssl-cpu-001
```

RTX 3070等のCUDA環境では、CPU groupを外してCUDA groupを指定する。

```bash
uv sync --locked --no-group cpu --group cuda
uv run --no-sync pytest
uv run --no-sync creative-agent smoke-audio-ssl \
  --config recipes/audio-ssl-smoke.json \
  --device cuda --steps 20 --output runs/audio-ssl-cuda-001
```

CUDA環境を保持するため、その後も `uv run --no-sync` を使う。単に `uv run` とすると既定のCPU groupへ同期される。

## 入力とモデル

既定では、独立した16作品を数式で合成する。外部音源、音色bank、MIDI、事前学習済み重みは取得しない。左右の振幅、位相、倍音を変え、native 48 kHz、2ch、float32 WAVとして保存する。長さは約1.5〜2秒で、train 12作品、validation 4作品を生成時に固定する。

manifestには生成器の版、seed、原作品ID、split、hash、sample数、レート、チャンネル、素材の権利宣言を残す。独立作品としてのID・hash重複、音声の改変、shape・レートの不一致を拒否する。完全なmanifestと音声はGit対象外に置く。

最初のCNNは10 msの非重複パッチをstride付きConv1dで処理し、1×1 Conv1dで幅128へ写す。その後、2層・4 headsの全体双方向Transformerを通す。RMSNorm、QK正規化、時間のRoPE、SwiGLUを使い、特徴列を全体平均へ潰さない。

CUDAの既定はFP32の学習パラメーターとFP16 autocast、GradScaler、Flash SDPAである。Flashでは各録音の有効部分へ個別に全体Attentionを適用し、特徴を再paddingする。CPUではFP32とmath SDPAを使い、key側に有効位置maskを渡す。二乗サイズのpadding maskは作らない。

`--backend math` で通常SDPAを明示できる。Flashを選んだ場合はそのbackendだけを有効にし、未対応環境での暗黙のfallbackは行わない。CUDAでFP32を使う場合は `--precision float32 --backend math` とする。

## 自己教師あり目的と正解の漏洩防止

10 ms・480 samplesのパッチについて、Hann窓を掛けたRFFTの実部・虚部と `log1p(abs(spectrum))` を左右別に予測する。lossは実部・虚部の平均二乗誤差と、対数振幅の平均二乗誤差の和とする。教師は波形自身から作り、意味ラベルを使わない。

既定では完全なパッチの約半分を、5位置程度のspanから選ぶ。対象パッチの両チャンネルをCNNに入る前にゼロ化し、特徴位置に学習するmask tokenを加える。少なくとも1つの隠した位置と1つの見えている完全な位置を残す。

CNNの受容野とstrideを同じ480 samplesにし、教師の窓も隣接パッチと重ねない。隠した原音を変更しても入力と予測が変わらず、対象の教師だけが変わることをテストする。将来の重複窓や広い受容野を導入する際には、同じテストを拡張する必要がある。

長さが480 samplesの倍数でない最終パッチは、実sample数と座標を保持して入力へ使うが、復元lossの対象から外す。batchのpaddingもlossから除外し、有効位置がpadding内容の影響を受けないことを確認する。

10 ms窓の周波数間隔は100 Hzであり、精密な採譜用の教師ではない。複素成分を教師に含めたことだけで、学習後の特徴が制作に必要な位相情報を十分に保持したとは判断しない。

## 更新と再開

`--steps` は累計の実optimizer更新数である。20更新の実行が終わった後、別のコマンドで21更新まで進める。

```bash
uv run --no-sync creative-agent smoke-audio-ssl \
  --config recipes/audio-ssl-smoke.json \
  --device cuda --steps 21 --resume --output runs/audio-ssl-cuda-001
```

checkpointにはmodel、AdamW、scheduler、GradScaler、実更新数、PythonとTorchの乱数、CUDA乱数、シャッフルした提示順とcursor、設定、データ署名、ソースとlockのhash、更新履歴を保存する。`.partial` に書いてflush・fsyncしてから置換し、完了したcheckpointを中途半端なファイルで上書きしない。

再開時には設定、データ、実装とlock、device種類、精度、backendの一致を検査する。既存runへの新規実行や、更新数が増えないresumeは拒否する。CPUでは中断なしの6更新と、3更新で中断して再開した6更新の重み・AdamW・提示履歴が厳密に一致することをテストする。

AMPで非有限の勾配が出た試行は重みを更新せず、scaleを下げてskipとして記録する。実更新数には加えず、skipが続く場合は失敗として停止する。正常な更新ではCNN、Transformer、復元headの各群に有限で非ゼロの勾配があり、各群の重みが実際に変わることを検査する。

これは最大1,000更新、32原音、1音源10秒以下のスモーク用runnerである。長期学習のwall-clock管理、分散学習、学習品質によるcheckpoint選択は本runnerの対象にしない。

## 成果物と確認

```text
runs/audio-ssl-cuda-001/
  data/manifest.json     # 原作品、split、hash、生成元
  data/*.wav            # 自作の48 kHzステレオ
  checkpoint.pt         # 次の更新へ再開する完全な状態
  training.jsonl        # 実更新とloss、勾配、使用作品
  tensorboard/          # loss、勾配、設定、検証用ステレオ音声
  report.json           # 最後に完了した実行の結果
```

`report.json` の `optimizer_steps`、`resumed_from_step`、`skipped_updates`、`parameter_groups_changed` を確認する。`training.jsonl` はcheckpointの履歴から復元し、TensorBoardには `purge_step` を指定して保存済み更新後の表示を再開する。

```bash
uv run --no-sync tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

`train/loss/*`、`train/gradient_norm/*`、`train/skipped_updates`、`validation/loss/*` を表示する。ステレオ音声は左右を維持したWAVのAudio summaryとして記録し、monoに変換しない。

時間はrunner内のmodel初期化から最終評価・ログ終了までを測り、依存導入・CLI起動・fixture生成と読み込みは含まない。GPUメモリはPyTorchのpeak allocatedとpeak reservedであり、GPU全体の使用量ではない。

確認されたlossの低下は処理の健全性を調べる材料である。汎化、意味理解、制作能力の達成は、正常例、独立した実録音、局所probe、音声差し替え等で別途測る。
