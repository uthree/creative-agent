# LFM3による5タスク混合学習とTensorBoard比較

SSL、キャプション、音声問題検出、発話書き起こし、audio-to-MIDIを1つの学習過程で扱う。共通の音声encoderを全タスクで更新し、テキストを返す4タスクは共通のprojectorと事前学習済みLFM3-1.7BのLoRAを更新する。診断はLLMの問題区間JSONと、従来の多ラベルheadの両方を学習する。

タスクを上の順で1更新ずつ交互に選ぶため、合計5,000更新で各タスク1,000更新となる。複数GPUプロセスを起動する構成ではない。各更新のbatchは同じタスクの2例。SSLはマスク付きスペクトルloss、テキスト4タスクは応答だけのCE、診断はさらにclip/frame BCEを加える。生のlossのスケールを自動で均等化しているわけではない。

## モデル・データ

[LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B)のsnapshot `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7` を使う。公式tokenizerとchat templateをローカルから読み、思考出力を無効にする指定を渡す。model本体はBF16で凍結し、attentionのq_proj/v_projへrank 8、alpha 16のLoRAを入れる。encoder、SSL headとmask token、診断head、projectorとLoRAはFP32の学習パラメーターで、CUDA計算にはBF16 autocastを使う。gradient checkpointingと応答位置だけのLM head計算でメモリを抑える。

音声encoderは前回SSLのstep 21から読み込める。100 Hzの連続特徴列を有効位置だけ5位置ずつ平均し、20 Hz相当の列を線形projector経由でLLMへ渡す。末尾の不完全なグループも残す。音声encoderを更新するため、runをまたぐ固定特徴cacheは使わない。

入力は最大2秒、LLMの全体上限1,024位置、生成上限160 tokens。公式generation_configを固定し、do_sample=False、EOSとpad IDを指定する。snapshot既定のrepetition penalty 1.1も有効で、prompt embeddingを使うため新たに生成したtoken列へ適用される。N1の末尾改行を保存するため、応答は正解文字列のtoken列とEOSを明示的に連結し、chat templateで教師末尾をtrimしない。指示・音声位置・assistant開始・paddingはloss対象外。

データは既存の合成器で作る。SSLとcaptionは同じ16の原音、診断は同じ原音に正常・単一・複合の9通りのviewを付ける。発話はeSpeak NGの英語16文、MIDIは旋律・和音・同音連打・無音の16音源。各タスクのsplitに加え、全タスクを横断して音声hashがtrain/validation間で共有されていないことを確認する。

| タスク | train | validation |
| --- | --- | --- |
| SSL | 12原音 | 4原音 |
| caption | 同じ12原音 | 同じ4原音 |
| diagnosis | 12原音の108 view | 4原音の36 view |
| speech transcription | 12発話 | 4発話 |
| MIDI | 12音源 | 4音源 |

これは小さな既存fixture上の学習実験。音源と正解への過適合が起こり得る。実用のASRや採譜の一般化、自然音・未知楽器・未知話者への性能を示すベンチマークとは扱わない。診断のJSON時刻は整数msへ切り下げ、補助headは元の10 ms特徴位置で評価する。

## 実行と再開

依存は `uv.lock` に固定したTransformers 5.20.2。LFM3のsnapshotはカスタムコードを含む。既存のローカルsnapshotから `.local/models/LFM3-1.7B-ac2586a/` へ構成とコードをコピーし、重みを参照した。モデル・tokenizer・コード・generation設定のSHA-256を記録し、再開時にも照合する。`trust_remote_code=True` はこのローカルの固定コードを読むための指定で、学習時に外部モデルやコードをダウンロードしない。

```bash
uv sync --locked --no-group cpu --group cuda
uv run --no-sync python - <<'PY'
from pathlib import Path
from creative_agent.multitask_data import MultitaskConfig, prepare_multitask_data
config = MultitaskConfig.load(Path('recipes/audio-multitask-lfm3.json'))
prepare_multitask_data(Path('data/multitask-001'), config, espeak='espeak-ng')
PY
HF_HOME="$PWD/.local/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run --no-sync creative-agent train-audio-multitask \
  --config recipes/audio-multitask-lfm3.json \
  --data-root data/multitask-001 \
  --model-dir .local/models/LFM3-1.7B-ac2586a \
  --encoder-checkpoint runs/audio-ssl-cuda-initial-20261101/checkpoint.pt \
  --device cuda --steps 5000 --output runs/multitask-001
```

途中から再開する場合は同じコマンドに `--resume` を加え、`--encoder-checkpoint` を外す。`--steps` は累計の完了したoptimizer更新数。再開先は保存stepより大きくする。encoder・projector・全head・LoRA、AdamW、scheduler、各タスクの順序とcursor、乱数、評価履歴を復元する。凍結LLMはsnapshotから再構成し、保存したhashに一致することを検査する。単体スモークとはcheckpointの種類を区別する。

250更新ごとにcheckpoint、500更新ごとに評価を保存する。`training.jsonl` は実更新だけを記録し、再開時にはcheckpointの履歴から書き直す。評価では乱数状態を退避して、次の学習maskとサンプル順を変えない。全学習パラメーターはcheckpointに含むが、凍結された約1.7Bの本体は重複して保存しない。

## TensorBoardで見る

```bash
uv run --no-sync tensorboard \
  --logdir runs/multitask-001/tensorboard --host 127.0.0.1 --port 6006
```

同じLANの別マシンから見る場合は、`--host` を実行ホストのLAN IPv4へ置き換える。接続URLは `http://<そのLAN IPv4>:6006`。今回の実行もLAN側のアドレスで待ち受け、HTTP応答を確認した。実際のIPとサーバーPIDはGit対象外のrun内 `tensorboard-server.json` に記録する。

SCALARSで `train/loss`、`train/<task>/loss`、勾配、タスクごとのvalidation lossと正しい音声・無音・別音声の参照NLLを見る。診断headとLLM応答はそれぞれ別に採点する。ASRはCER/WER、MIDIは形式率と音符F1を記録する。

TEXTとAUDIOは `examples/<task>/<sample_id>/` で対応付ける。入力音声 `input`、教師文 `reference`、実生成 `llm_response`、EOS/生成上限の状態が同じサンプルIDとstepで比較できる。caption・ASR・MIDIはvalidation各4件、診断は正常・単一・複合の代表各1件を生成する。診断のhead評価はvalidation全36 viewであり、生成3例のLLM評価と分母が異なる。

採譜のIMAGESは `midi_score_comparison`。上段が正解MIDI、中段が生成MIDI、下段が重ね合わせで、音高と時間の軸、凡例を付ける。正解も生成もMIDIへ書き、読み直した音符から描画する。生成形式が不正ならMIDIを作らず、画像にも赤字で形式エラーと表示する。有効な非空MIDIは倍音合成で再生用音声も記録するが、入力録音ではなく予測音符の再合成として区別する。

SSLにはLLM応答がないので、入力と正解/予測の左右別スペクトル比較を記録する。全評価のraw応答、正解、生成MIDI、PNGは `evaluation/step-xxxxxx/` にも残す。音源、教師文、MIDI、モデル・adapter・checkpoint、TensorBoardイベントはGit対象外。公開には小さな集計と実測報告だけを残す。
