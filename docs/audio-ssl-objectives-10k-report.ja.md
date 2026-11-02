# SSL目的の③→②比較: 各10,000更新への継続

2026年11月2日 JST。[1,000更新の比較](audio-ssl-objective-comparison-report.ja.md)から生徒・EMA教師・optimizerを引き継ぎ、③「潜在予測＋音響復元」、②「潜在予測のみ」の順で各9,000更新を追加した。合計は各10,000更新、2条件で20,000更新。[匿名集計](../experiments/results/audio-ssl-objectives-10k-20261102.json)に実測値と追加診断を保存した。

## 共通probeの結果

**音響分類は両方式とも改善してほぼ同率になった。MIDI押下フレームF1では③が上回ったが、全体として大きな精度向上は確認できない。** 1 seed・小規模データ・3独立test曲での比較である。

| 条件 | 音響クラスタaccuracy ↑ | MIDI押下フレームmicro-F1 ↑ |
| --- | ---: | ---: |
| 学習前 | 26.46% | 14.69% |
| ③ 1,000更新 | 29.85% | 16.46% |
| ③ 10,000更新 | **33.33%** | **17.23%** |
| ② 1,000更新 | 31.14% | 18.53% |
| ② 10,000更新 | **33.40%** | **15.38%** |

1,000→10,000更新の変化は、③の音響accuracyが＋3.48 percentage points、MIDIフレームF1が＋0.77 points。②は音響accuracyが＋2.26 points、MIDIフレームF1が−3.16 pointsだった。10,000更新での音響accuracy差は約0.07 pointsに縮まった。

音響分類の10,000更新時点の内訳:

| 条件 | 音楽 | 英語 | 日本語 |
| --- | ---: | ---: | ---: |
| ③ | 52.41% | 8.04% | 20.46% |
| ② | 52.32% | 8.91% | 20.03% |

集計accuracyは音楽1/2・英語1/4・日本語1/4。これは64種類の低水準音響クラスタの分類で、発話のCER／WERではない。学習用headや異なる損失の絶対値を比較した結果でもない。

MIDIフレームの10,000更新時点の内訳:

| 条件 | precision | recall | TP | FP | FN | validation選択閾値 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ③ | 11.79% | 31.94% | 329 | 2,461 | 701 | 0.1 |
| ② | 14.78% | 16.02% | 165 | 951 | 865 | 0.2 |

②は予測を少なく出す動作になり、③よりprecisionは高いがrecallが低い。validationのF1は③27.78%、②34.24%で、testでは順位が異なる。少数の曲に対するreadout・閾値の汎化も比較結果に影響しており、encoderの情報量の差と一対一に解釈しない。

## 閾値と音高ランキングの追加診断

この節は結果を見た後の診断で、事前に決めた上表のスコアを差し替えるものではない。保存済みのvalidation選択readoutを固定し、学習やtestを使った係数・閾値選択を行わずに測った。

| 10,000更新のreadout | 閾値0.1のF1 | 閾値0.2のF1 | 音高micro AP ↑ |
| --- | ---: | ---: | ---: |
| ③ | 17.23% | 17.99% | 11.74% |
| ② | 15.45% | 15.38% | 11.97% |

閾値0.1に揃えてもF1の③＞②は変わらず、順位の逆転を閾値の違いだけでは説明できない。閾値に依存しないmicro average precisionは、全key-frameの連続スコアをまとめて順位付けし、同点をまとめて計算した。③と②の差は約0.23 pointsと小さい。

micro APの1,000→10,000更新の変化は③8.59%→11.74%、②11.49%→11.97%。この指標では③が長い学習で②に近づいた。学習前は7.56%。F1だけで②の音高表現が全面的に悪化したとも、APだけで実用的な採譜ができるとも判断しない。

通常入力を同じ長さの無音へ替えると、両方式とも音響accuracyは23.91%、MIDIフレームF1は0だった。音楽のクラスタaccuracyは無音でも47.77%になるため、音楽accuracyの絶対値には正解分布の偏りが大きく影響する。

## 比較条件と継続の確認

- 学習コードはcleanな`bcdb218`。学習関数、encoder、データ・probe処理は親と同じ。変更は継続と監査の処理。
- seed `20261102`、直接波形パッチの幅128・2層・4 heads、4秒以下の入力、batch 8、音楽4・英語2・日本語2。CNN・時間平均による間引き・人工混合は追加していない。
- 生徒・EMA教師・AdamWの状態と最初の1,000更新の履歴を継承した。LR 3e-4を維持し、EMA係数は既存の式に従って0.999で継続。
- codebook、音響教師の標準化、補助loss係数は親のファイルをbyte単位で継承。probeのtrain／validation／test cropも同一。
- frozen FP32 encoderから、同じ構造と探索候補の線形ridge probeをtrainで学習し、validationで選択。testを使った再選択はしていない。
- MIDI probeのtestは11 stem・3曲・880フレーム、正解1,030 key-frame。全群の音響probeは46録音・3,454フレーム。MIDIはkey-release区間を採点し、pedalや残響尾を延長しない。
- 音楽はBabySlakhの合成stem、発話はMiniLibriSpeechとJSUT。環境音、未知の録音方式、LLM ASR、生成MIDI、30／60秒の文脈学習の評価は含まない。

同じtestを1,000更新の比較から再利用しており、新しい独立データでの検証ではない。ここから本学習の設定を確定するには、独立曲・音色・実録音・seedを増やしたvalidationと最終testが必要になる。

## 検証・計算量・保存

| 条件 | 親から追加した更新 | 累計学習ループ秒数 | 追加分の秒数 | CUDA peak allocated |
| --- | ---: | ---: | ---: | ---: |
| ③ | 9,000 | 606.45 | 544.79 | 0.823 GiB |
| ② | 9,000 | 460.86 | 413.74 | 0.084 GiB |

時間は入力準備・前後向き・optimizerを含み、probe評価と保存を含まない。両者の推論encoderは同じ構造で、③のFFT教師と追加headは学習時の処理である。

全テスト169成功・任意のeSpeak NG依存1件スキップ。EMA・optimizerを含むCPU厳密再開、親のconfig・入力順・教師・学習コードの変更拒否、実音声での1,000→1,002更新の継続、APの同点処理を検証した。Ruff check／formatと文書リンクも確認した。

監査は各optimizer stepが10,000、各履歴が1〜10,000更新、追加が各9,000更新であることを確認した。最初の1,000更新の履歴は親と厳密一致。両encoder・EMA教師が更新され、③の補助headだけが更新されることも照合した。全3条件の保存encoderとreadoutから各3,454 testフレーム、計10,362フレームの予測が一致した。

`runs/audio-ssl-objectives-10k-20261102/`へ27ファイル・65,622,309 bytesを永続保存した。親の23ファイルはSHA256照合で変更がないことを確認した。容量確保で退避したのは再生成可能なPython bytecode cache約100 MBで、データ・モデル・ログは保持した。Gitにはソース、文書、匿名集計だけを残す。

[TensorBoard](http://192.168.1.12:6007/)の`OBJECTIVE_10K/`で、最終精度、入力音声、正解／予測画像を参照できる。`diagnostics/pitch_micro_AP`、`diagnostics/pitch_F1_threshold_0.1`／`0.2`は事後診断。最初の1,000更新のtrain scalarは親履歴から再記録したもので、追加の更新として数えない。親の`OBJECTIVE_3_2/`も保持した。

再現コマンドは[継続の手順](audio-ssl-objective-comparison.ja.md#1000更新から10000更新へ継続する)。追加診断は次で再現できる。

```bash
PYTHONPYCACHEPREFIX=/tmp/creative-agent-bytecode-runtime \
  .venv/bin/python experiments/diagnose_ssl_pitch_probes.py \
  --manifest data/public-audio-20261101/manifest.json \
  --suites runs/audio-ssl-objectives-20261102 runs/audio-ssl-objectives-10k-20261102 \
  --output runs/audio-ssl-objectives-10k-20261102/pitch-diagnostics.json
```

診断の再実行は最後に指定したsuiteのTensorBoardと成果物一覧を更新するため、その後に監査コマンドも再実行する。
