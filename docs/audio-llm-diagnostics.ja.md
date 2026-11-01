# LLMの出力能力と音声との接続を分ける比較

[8条件・8,000更新の実測結果](audio-llm-diagnostics-report.ja.md)と[匿名集計](../experiments/results/audio-llm-diagnostics-20261101.json)を公開している。

目的は、現在のLFM3構成が出力形式・長さへの適応で制約されているかを調べること。成功を任意の音源の認識能力と混同しない。

## 事前に固定する条件

- LFM3-1.7BとQwen3-1.7B（公式revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`、Apache-2.0）。[公式モデルカード](https://huggingface.co/Qwen/Qwen3-1.7B)。思考出力を無効化、greedy、比較本体ではrepetition penalty 1.0。モデルの優劣を一般化する実験ではない。
- 英語・日本語ASR、30秒・60秒の採譜を課題ごとに独立学習。SSLはLLMを使わないため比較しない。
- 元のsource-disjoint manifestを維持し、IDのSHA256順で言語／長さごとに選ぶ。ASRは訓練各8・validation各2・test各4発話。MIDIは音符ありの訓練各2・validation各1・test各2区間。静かな空列を当てるだけの成功を除く。小規模な診断集合で、最終的な汎化保証ではない。
- 共通encoderは最初の合成5k更新checkpointから取り出して凍結。新規cacheでは100 Hzの特徴を5フレーム連結して640次元・20 Hzへ変換し、同じ特徴cacheを両LLMで使う。正音声・同じ長さの無音・同じsplitの別音声を事前に計算する。
- 両LLMのLoRAとprojectorは新規初期化。既存LFMの学習済み接続は流用しない。音声条件はprojectorとLLMのq/v rank 8 LoRAのみ更新。テキスト条件はLoRAのみ更新し、通常のtoken embeddingへ入力する。
- テキストの正解情報条件：ASRは正解文を渡して転記。MIDIはvelocity,pitch,offset,onsetのCSVからN1へ並べ替え・変換。これはoracle対照であり、認識の評価ではない。音声条件と同じ出力教師を使うが、入力の長さ・難しさは異なる。
- 各条件1,000更新、batch 1、LR 3e-4、warmup 50更新、その後一定。言語／長さを交互に提示する専用乱数の順序を固定する。訓練提示数を揃え、token数と時間は別に記録する。1 seedで初期切り分けを行う。
- ASR生成上限192、MIDI 6144、文脈12288。Qwen tokenizerでは同じ240音の教師が約4300 tokenとなる事前検査を踏まえ、両モデルの上限を同時に拡張した。既存checkpointのpenalty比較は元の上限を維持する。教師と入力を切り詰めず事前検査。teacher NLLだけでなく、訓練集合全例・validation・testの自由生成を採点する。testは最終重みだけを評価しcheckpoint選択に使わない。
- 既存モデルのpenalty 1.1 / 1.0比較は同じ固定test subsetを使い、各保存checkpoint自身のencoderで特徴を再計算する。凍結共通encoderのcacheへ置き換えない。

## 判断と追加条件

テキスト条件でもLFMが失敗しQwenが成功する場合は出力能力・適応設定が候補。両者がテキストで成功し音声ではQwenだけ成功する場合は、音声表現の利用・接続への適応が候補。両者とも音声で失敗すればencoder／接続／学習条件が残る。どれもLLM本体の欠陥を単独で証明しない。

LFMが失敗した課題ではrank 32・q/k/v/oおよびMLPへのLoRA拡張（LFMのMLPはw1/w2/w3、畳み込み層のin/out線形投影も含む）も比較する。実際に該当する線形層だけに適用し、適用一覧・学習パラメータ数を保存する。rankと適用範囲を同時変更する救済条件であり、どちらの効果かは分離しない。VRAM 8 GiBでの全重みAdamW学習は本比較に含めない。

TensorBoardへ入力音声、oracle入力、正解、実際の生成文、保存MIDIから読み直した比較画像を出す。不正なMIDIは修復せず不正として採点する。入力音声が教師の代わりに見えるような表示は避ける。

## 実行

`uv run --no-sync python experiments/run_llm_diagnostics.py --help`を参照。`prepare`でmanifestと共通encoder checkpointからcacheを作成し、`decode`で既存モデルの生成設定を比較、`train`でbackbone/task/modeを指定する。出力先は既存ディレクトリへ上書きしない。データ・oracle全文・特徴・重み・イベントは除外対象のローカルrunに保存し、匿名集計と結果文書をGitへ残す。

全条件を同じGPUで順に実行する入口は次のとおり。`QWEN_MODEL_DIR`には固定revisionのローカルsnapshotを指定する。

```sh
uv run --no-sync python experiments/run_llm_comparison_suite.py \
  --output runs/audio-llm-diagnostics-concat-20261102 \
  --manifest data/public-audio-20261101/manifest-long-midi-30-60.json \
  --lfm-dir .local/models/LFM3-1.7B-ac2586a \
  --qwen-dir "$QWEN_MODEL_DIR" \
  --encoder-checkpoint runs/audio-multitask-lfm3-5000-20261101/checkpoint.pt \
  --previous-suite runs/audio-public-single-task-20261101
```

完了した`report.json`がある条件は再実行しない。途中のディレクトリを上書きする再開機能はなく、失敗した実行は別名へ退避して保存する。LoRA拡張の追加基準は、LFMの全訓練例の完全一致率が95%未満であること。未知例の結果で追加条件やcheckpointを選ばない。

診断runnerは毎更新1例を明示的に渡す。cache由来の`config.audio`に残るSSL用batch・区間長・学習率の既定値は、このrunnerの実効batch・公開音声長・LLM学習率を規定しない。実効batch 1、実際の区間長はmanifest、学習率は外側の`config.learning_rate`を使う。異なるtokenizerのNLL同士の絶対値は直接比較せず、同一モデル内の正音声／無音／別音声差を見る。

主指標と実際の生成文・MIDI画像は修復しない。補助診断として、N1で始まる応答の欠けた末尾改行だけを追加した場合の値も別に保存する。音符・順序・数値は変えず、生成上限到達も残す。これは形式要因の切り分け用で、実際の生成成功率へ混ぜない。

監査CLIに`--publish-tensorboard`を付けると、検証後の`diagnostics/`タグへ末尾改行のみの参考値、音符行の一致数、元の不正理由を追記する。元の応答、MIDIファイル、画像、主指標は変更しない。

未知例の生成文が選択した訓練例の正解を再生していないかも集計する。末尾LFだけを除いた完全一致で数え、正しい未知例の認識と少数例の記憶を区別する。TensorBoardの`diagnostics/`にもコピー数を出す。

## 保存結果の監査

全条件の学習後、次のコマンドで履歴、optimizer、特徴、入力と教師の対応、元データhash、応答、MIDI比較画像を照合する。モデルの再読込と自由生成を伴うためCUDAが必要。`--no-replay`は応答再現と元音声からの特徴再計算を省く部分監査であり、完全監査の代用にはしない。

```sh
uv run --no-sync python experiments/audit_llm_diagnostics.py \
  --suite runs/audio-llm-diagnostics-concat-20261102 \
  --manifest data/public-audio-20261101/manifest-long-midi-30-60.json \
  --lfm-dir .local/models/LFM3-1.7B-ac2586a \
  --qwen-dir "$QWEN_MODEL_DIR" \
  --encoder-checkpoint runs/audio-multitask-lfm3-5000-20261101/checkpoint.pt \
  --previous-suite runs/audio-public-single-task-20261101 \
  --summary .local/audio-llm-diagnostics-verified.json \
  --publish-tensorboard
```

新規条件のcheckpointにはLoRA・projectorとoptimizerを保存し、共通encoderは`shared_encoder.pt`に分けて保存する。音声条件の推論を再構成する際は、checkpoint内の未使用encoder初期値ではなく、この共通encoderと、cache／checkpointに保存した`frame_reduction`・`pool_frames=5`を使う。監査は元の親checkpointから共通特徴を再計算する。学習済み重み、元データ、生成全文、イベント、保存ファイルの完全なhash一覧は除外対象のrunに残す。

2026年11月1日開始の実測報告は平均方式での結果であり、変更後の精度を示さない。新規の`prepare`は親checkpointの古い圧縮設定を継承せず`concat`を選ぶ。旧平均cacheは新規学習に再利用せず、別のrunへ再生成する。[連結方式と旧成果物の扱い](audio-multitask-lfm3.ja.md#フレーム連結への変更と旧成果物)も参照。
