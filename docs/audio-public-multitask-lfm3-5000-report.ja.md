# 公開音声・MIDIを使った5,000更新の実測

2026年11月1日。**楽器名の弱教師短文では改善を確認した。発話書き起こし、ピアノ採譜、正常・複合問題の検出は、この予算では実用的な汎化に達していない。** 発話・採譜は訓練例の抜き取りでも誤りが多く、データ追加だけで十分に学習できたとは判断しない。

[取得・実行手順](audio-public-multitask.ja.md)、[匿名集計と監査結果](../experiments/results/audio-public-multitask-lfm3-5000-20261101.json)、[監査スクリプト](../experiments/audit_public_audio.py)を残した。

## 取得したデータと評価範囲

| データ | 実際に保存した音声 | 主な用途 |
| --- | ---: | --- |
| [Mini LibriSpeech / OpenSLR 31](https://www.openslr.org/31/) | train-clean-5とdev-clean-2、計2,608 FLAC、16 kHz mono | 英語の実発話と全文書き起こし |
| [JSUT v1.1](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) | basic5000から1,024 WAV、48 kHz mono | 日本語の実発話と元の書き起こし |
| [BabySlakh v2](https://zenodo.org/records/4603870) | 20曲、mixture・stem合計229 WAV、16 kHz mono。元MIDIと楽器別MIDIも保存 | 音楽SSL、孤立楽器の短文、ピアノkey-note採譜 |

実波形は合計3,861ファイル、約2.76 GB。Git除外の`data/public-audio-20261101/sources/`に保存した。配布アーカイブは一時領域へ置き、Mini LibriSpeechとBabySlakhは公式MD5、JSUTの部分取得はZIPのCRC・サイズを検証した。取得ファイルのSHA-256を記録し、学習に使用した原音・MIDIはmanifestとの一致も確認した。長さ・課題・予算で選別しており、取得した音声全てを学習したものではない。

Hugging Faceトークンは環境から読み込み、値や設定ファイルの内容を表示していない。ReazonSpeechはファイル取得に利用承認の403エラーがあり、日本語にはJSUTを使用した。JSUTの音声・文の条件を保持し、今回の重みは非商用研究・個人利用の系統として扱う。音声、教師全文、完全なmanifest、重み、TensorBoardは再配布しない。

英語は訓練27話者、validation 8話者、test 6話者を分離した。日本語は1話者のまま未知文を評価する。音楽は元MIDI UUIDで14／3／3作品を分け、全課題でその分割を共通にした。採譜も14／3／3作品から選んだ。別作品でも同じ合成bankを使うため、任意の実演・未知bankの採譜評価とは異なる。

訓練はASR 892発話（英語252・日本語640、約71.3分）、MIDI 800区間（約19.8分）、楽器短文600区間、診断736派生例、SSL 512区間。課題間の原音は重なる。診断testの24例は、音楽・英語・日本語の3原音に正常を含む8組合せを加えたもので、24の独立原音ではない。

## 学習条件

前回の合成データ5,000更新のcheckpointから、全trainable重みをwarm startした。optimizerを初期化し、今回新たに5,000更新、各課題1,000更新を実行した。batch 2で各課題2,000例ずつ提示し、各訓練集合の全例を少なくとも一度使用したことを履歴から照合した。

LFM3-1.7B本体をBF16で凍結し、rank 8のq/v LoRA、共有encoder・projector、SSL・診断headの計1,474,508パラメーターを更新した。波形は遅延decodeして明示的に48 kHzへresampleし、monoを2chへ複製する。encoderは100 Hz、LLMへ渡す系列は20 Hz、文脈1,024位置、生成上限192 token。全教師は上限内であり、final testの生成打ち切りは全課題0件だった。

実行コードは`aacdb31`、manifest SHA-256は`bbe02d91a1b0193e57c56a79b1c114a2a2ebca677c907dd4f721cd42d3e51659`。実行中に行った`a9183d5`のWindows互換修正はテキスト入出力とテストfixtureのみで、数値処理を変更していない。再現監査には`aacdb31`のソースを別ディレクトリへ展開して使った。

RTX 3070で、開始前評価・学習・終了後評価を含め約1,169.8秒（19分30秒）、PyTorchピーク割当約3.46 GiB。取得・前処理・後続の再現監査の時間は含めない。validationは0／2,500／5,000更新、固定testは開始前と最後だけで評価し、最終重みを採用した。

## 固定testの学習前後

全評価例を自由生成して採点した。CER/WERは低いほどよく、F1・一致率は高いほどよい。

| 指標 | 学習前 | 学習後 | 分母・意味 |
| --- | ---: | ---: | --- |
| 英語CER | 87.32% | 76.99% | 未知6話者の16発話、1,104文字 |
| 英語WER | 97.63% | 100.00% | 同16発話、211語。文字誤りの低下を正確な単語認識としない |
| 日本語CER | 100.00% | 93.65% | 同一話者の未知16文、441文字 |
| MIDI 音高＋onset F1 | 0 | 0.0464 | 32区間、正解195音。TP 8／FP 142／FN 187 |
| MIDI 音高＋onset・offset F1 | 0 | 0 | 同じ32区間。終了時刻までの正しい音符は確認できない |
| MIDI N1有効率 | 75.00% | 81.25% | 学習後も6/32が不正。音符列完全一致は0/32 |
| 楽器短文の完全一致 | 0/24 | 19/24（79.17%） | piano／guitar／bass各8区間。訓練最多クラスの定数基準は33.33% |
| 診断head clip macro F1 | 0.5854 | 0.1333 | 24例。全問題ありの定数基準は0.6667 |
| 診断head 時間区間macro F1 | 0 | 0 | class・ch・区間IoUで評価 |
| 診断LLM clip macro F1 | 0.2917 | 0.2698 | 実際のJSON応答。補助headの値と区別する |
| 診断LLM 時間区間macro F1 | 0 | 0.0926 | 実際のJSON応答。正常3例の誤報は3/3のまま |

診断headの正常誤報は3/3から1/3へ減ったが、問題の見落としが増え、clip F1は悪化した。正常例への誤報だけを改善として採用しない。LLMの不正JSONは0件でも、内容や時刻が正しいことは保証しない。現在の診断は「原音へ加えた人工的な加工」の検出であり、自然な制作ミスへの評価は未実施である。

楽器短文は元MIDI program由来の弱教師で、孤立stemを対象にする。人手の自由な音楽caption、複数楽器の全曲説明、音楽の良し悪しを学べたという結果ではない。

音楽8区間のSSL lossは0.005164から0.003764へ下がった。常にゼロを予測する基準値0.004189を約10.2%下回るが、対象は8 kHz以下のマスク付きスペクトルであり、この改善から高域・空間理解や発話認識を推定しない。

## 音声を根拠にしているか

全testの正解文平均token NLLは、ASRで実音声3.9750、同長無音4.1637、別音声3.9665だった。別音声でも悪化せず、正しい発話内容への依存の証拠は弱い。MIDIは1.6467／1.6718／1.6656で差が小さい。楽器短文は0.1290／0.2403／0.4400で、実音声との対応が見られた。

さらに、結果を読む前に固定したseedと例を使い、保存checkpointを別プロセスで復元して3条件を自由生成した。別音声は同言語・同課題の異なる教師から選び、入力長を揃える。下表の別音声も、元の正解に対する採点である。

| 固定probe | 実音声 | 同長無音 | 別音声 |
| --- | ---: | ---: | ---: |
| 楽器短文の一致（6例） | 6/6 | 2/6 | 3/6 |
| ASR CER（英日各4、計8例） | 77.01% | 100.27% | 83.16% |
| MIDI onset F1（8例） | 0.0220 | 0.0208 | 0 |

ASRでは別音声へ替えても4/8の応答文が変わらなかった。MIDIは文が変わるものの、実音声と無音のF1がほぼ同じ低さだった。応答が変化することだけを、正しい採譜の証拠にはしない。

訓練8例の抜き取りでも、英語2例CER 80.37%、日本語6例CER 83.89%、MIDI onset F1 0.1000、offset込み0.0667、音符列完全一致0/8だった。訓練集合全体の採点ではないが、未学習例だけの問題と決めつけず、実データ自体への学習不足も扱う必要がある。

## 保存物と検証

ローカルrunは`runs/audio-public-multitask-lfm3-5000-20261101/`。完全な生成・正解は`evaluation/`、音声差し替えは`controls/`、全更新履歴は`training.jsonl`、重み・optimizerは`checkpoint.pt`。TensorBoardの同じLAN URLで`public/.`を選ぶと、以前の`synthetic/.`と切り替えられる。`validation`・`test`・`train_probe`・`test_controls`の入力、正解、LLM応答、MIDI比較画像を確認できる。

監査では学習時コード・依存lockをGitの記録とSHA-256で照合し、原音とMIDIのhash、親checkpointと初期重みの完全一致、5,000件の履歴、optimizerのstep、全訓練例の提示、凍結backboneの不変性を確認した。encoderのoptimizerは5,000、LoRAとprojectorは4,000、診断headとSSL出力headは各1,000回。mask tokenは非SSLでもゼロ勾配が作られるため、AdamWのstateは5,000回進んでいる。

元の実音声で22例全ての生成token列を再現した。保存MIDIから描き直した14枚の比較PNGが元画像と画素一致し、TensorBoardは最後の更新5,000とAudio 102タグ・Images 30タグ・Text 237タグを確認した。表示するmediaは一部の例で、testの採点分母とは異なる。不正MIDIを正解画像で置き換えず赤く表示する。

CUDA環境は112テスト成功、CPU専用は104成功・GPU用8件スキップ。既定文字コードcp1252の再現検証も通り、[Ubuntu・WindowsのCI](https://github.com/uthree/creative-agent/actions/runs/41287785034)も成功した。

再監査する場合は、学習したソースを固定して次を実行する。重みは更新しない。

```bash
mkdir -p /tmp/creative-agent-audit-aacdb31
git archive aacdb31 src pyproject.toml uv.lock | tar -x -C /tmp/creative-agent-audit-aacdb31
PYTHONPATH=/tmp/creative-agent-audit-aacdb31/src uv run --no-sync python experiments/audit_public_audio.py \
  --run-dir runs/audio-public-multitask-lfm3-5000-20261101 \
  --manifest data/public-audio-20261101/manifest.json \
  --parent-checkpoint runs/audio-multitask-lfm3-5000-20261101/checkpoint.pt \
  --model-dir .local/models/LFM3-1.7B-ac2586a --summary .local/public-audit-repeat.json
```

## 次の比較

ユーザー指定により、**次回の採譜実験は30秒・60秒の音声区間へ拡張して比較する**。今回の0.5〜3秒・最大10音という制限を見直し、音楽的な前後関係を長く観測できる条件を試す。短さが精度不足の原因かは未確認であり、元作品のsplitを保持して短区間条件と比較する。

現状の20 Hz表現では音声だけで30秒600位置・60秒1,200位置になる。文脈1,024位置・生成192 tokenの設定は流用せず、prompt・教師音符列を含む長さとVRAMを実測して上限を決める。長い正解を切り捨てず、区間境界をまたぐ音の表現・復元を整備し、入力全区間に対する音符F1、出力打ち切り率、計算時間、ピークメモリーを記録する。分割処理が必要な場合も、30〜60秒の入力範囲と結合後の採点範囲を明記する。

まず同じデータ・同じ対象課題の更新数で、ASR単独・MIDI単独と今回の混合学習を比較する。少数の実発話・実際のレンダーを訓練集合内で学べるかも切り分ける。今回の結果から、全課題の同時学習が必要・有利とは結論できない。

その後、既に取得済みの長い発話を全文で扱える経路と音声encoderの事前学習を検討し、使用時間を増やす。採譜は30〜60秒への対応後、ペダル、多楽器、実演へ段階的に広げる。混合学習の影響、データ量、encoderの能力のどれが原因かは、今回だけでは未分離である。
