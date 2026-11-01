# 音声とテキストキャプションの学習スモーク

音声特徴とLLMの接続を確認する第2段階のスモーク。音声エンコーダーと小型Qwen3を凍結し、キャプションの応答lossから線形projectorだけを更新する。第1段階のSSL checkpointからの読み込み、可変長音声、日本語の教師文、保存・再開、教師を与えない生成を扱う。

小型Qwen3はランダム初期化した実アーキテクチャで、事前学習済み重みを使わない。tokenizerは可逆なUTF-8 byte方式のfixtureであり、本番のQwen tokenizerやchat templateではない。確認するのは接続・勾配・凍結・更新・再開であり、caption生成能力の獲得ではない。

## 実行

Python 3.12とuvを使う。依存はlockに固定し、モデルとデータのダウンロードは行わない。CPUは次のコマンドで実行する。

```bash
uv sync --locked --group dev
uv run --no-sync creative-agent smoke-audio-caption \
  --config recipes/audio-caption-smoke.json \
  --device cpu --steps 20 --output runs/caption-cpu-001
```

既存のSSLエンコーダーを使うには `--encoder-checkpoint` を指定する。CUDAでは次のように実行する。

```bash
uv sync --locked --no-group cpu --group cuda
uv run --no-sync creative-agent smoke-audio-caption \
  --config recipes/audio-caption-smoke.json \
  --encoder-checkpoint runs/audio-ssl-cuda-initial-20261101/checkpoint.pt \
  --device cuda --steps 20 --output runs/caption-cuda-001
```

指定したSSL checkpointの構成、完了した更新、重みのshapeを検査し、読み込んだファイルのhashと更新数を記録する。未指定なら決定的なランダム音声エンコーダーを使い、reportで区別する。既存のSSL runや重みは上書きしない。

CUDAでは引き続き `uv run --no-sync` を使う。音声エンコーダーの既定backendはFlash、LLMはpaddingを扱うmath SDPA。学習パラメーターはFP32、計算にはFP16 autocastとGradScalerを使う。

20更新後は別プロセスから21更新目へ進められる。再開時はencoderもcaption checkpointから戻すため、`--encoder-checkpoint` を再指定しない。

```bash
uv run --no-sync creative-agent smoke-audio-caption \
  --config recipes/audio-caption-smoke.json \
  --device cuda --steps 21 --resume --output runs/caption-cuda-001
```

`--steps` は累計の実optimizer更新数。設定、データ署名、ソースとlock、精度、backend、device種類の不一致を拒否する。

## 入力と教師

既定では48 kHzステレオ・約1.5〜2秒の自作16作品を使う。train 12作品とvalidation 4作品を固定し、各作品に基音と左右の違いを説明する日本語キャプションを付ける。音声と文章は数式と生成器のパラメーターから作り、外部アセットを取得しない。

キャプションはクラスIDへの置換ではなく、任意の文字列を教師にする。自分のmanifestを使う場合は `--manifest <path>` を指定し、音声契約に加えて各assetの `caption`、任意の `prompt`、manifest直下の `caption_rights` を記録する。作品ID、生成パラメーター、captionの出所は、LLMの入力に渡さない。

既定の上限はcaption 256 bytes、prompt 128 bytes、mediaと全テキストを合わせて512位置。長い音声・教師文を切り詰めず、最初の更新前に全件の文脈上限を検査する。音声の末尾の不完全パッチも有効な入力位置として残し、batchのpaddingだけを除外する。

## 学習する部分

```text
48 kHz stereo
  → 凍結したCNNと双方向Transformer
  → 有効位置の特徴列
  → 学習する線形projector
  → 凍結したcausal Qwen3
  → キャプションとEOSのcross entropy
```

音声エンコーダーはevalとno_gradで特徴を計算し、凍結された同じ重みの特徴だけをrun内でcacheする。LLMの重みはrequires_gradをFalseにするが、LLMのforwardをno_gradで包まない。応答lossの勾配はLLMの入力まで戻し、projectorを更新する。

入力はBOS、指示、音声特徴、assistant開始の区切り、教師caption、EOSの順に構成する。教師captionとEOSだけをlabelにし、BOS・指示・media・区切り・paddingは `-100` にする。next-tokenのshiftを適用し、最初の回答tokenもprefixから予測する。

生成時は音声と指示だけを渡し、教師captionを入力しない。greedy生成の実token IDとdecode結果を記録する。byte列が不正なUTF-8になった場合は置換文字を表示し、教師文で補修しない。

## 合格条件と記録

各更新でlossとprojectorの勾配が有限で非ゼロであること、encoderとLLMにパラメーター勾配がないことを確認する。更新前後のhashで凍結重みの不変性とprojectorの変化を確認する。AMPでskipした試行を実更新として数えない。

checkpointには凍結したencoderとLLMも含めて保存するため、元のSSLファイルがなくてもcaption runを再開できる。AdamW、scheduler、GradScaler、乱数、データ順とcursor、実更新履歴、tokenizerの版も復元する。CPUでは中断なしと中断・再開後の重み・optimizer・履歴が厳密に一致することをテストする。

```text
runs/caption-cuda-001/
  data/manifest.json   # 音声とcaption、split、出所、hash
  data/*.wav          # 自作の48 kHzステレオ
  checkpoint.pt       # 凍結重み、projector、完全な学習状態
  training.jsonl      # 実更新、caption CE、勾配、教師token数
  tensorboard/        # loss、音声と参照文と実生成文のペア
  report.json         # 凍結監査、音声対照、生成、時間とVRAM
```

TensorBoardでは `train/response_ce`、`train/projector_gradient_norm`、`validation/response_ce/*` と、`validation/examples/<sample_id>/` の音声・参照文・生成文を確認できる。

validationでは正しい音声、同じ長さの無音、別のvalidation音声で教師captionのNLLを測る。別音声は元の長さを保持するため、音声内容と位置数の両方が変わる対照であり、差だけで意味理解を断定しない。validationが1作品だけの小fixtureでは別音声の値を未計測とする。

全データ、完全なmanifest、重み、TensorBoardイベントはGit対象外。公開用には小さな集計と[実測報告](audio-caption-smoke-report.ja.md)だけを残す。
