# 公開音声を課題ごとに単独学習した比較

2026年11月1日。**単独学習はSSLと診断の一部を改善したが、書き起こし・短区間採譜の汎化不足を解消しなかった。楽器短文は混合学習の方がよかった。** 全課題を単独化すれば解決するという結果ではない。

[実行手順](audio-public-single-task.ja.md)、[前回の混合学習](audio-public-multitask-lfm3-5000-report.ja.md)、[実測集計](../experiments/results/audio-public-single-task-20261101.json)を参照。

## 同条件で比較した5課題

同じ合成データcheckpointを初期値にし、各課題を独立したモデル・optimizerで1,000更新した。前回の混合学習も各課題の更新数は1,000回だった。短区間manifest、batch 2、基本学習率3e-4、warmup 100、seed、最終重みの採用方針を揃えた。全5条件で開始前のtest集計が前回と完全一致した。公開データで混合学習した最終重みを単独条件へ流用していない。

共有optimizerの履歴、各課題にとってのwarmupの進み方、提示順は異なる。1 seed・小規模testでの比較であり、干渉の原因を一つに特定したものではない。前回見たtestを再利用しているため、新しい未使用testでの最終保証とも区別する。

非SSL時も共有encoderのmask tokenにはゼロ勾配が作られ、AdamWのstateとweight decayは進む。これはSSLの教師更新を実行した意味ではない。非対象のSSL出力head・診断headと、SSL単独時のLLM LoRA・projectorは不変だった。

| 固定test指標 | 混合学習 | 単独学習 |
| --- | ---: | ---: |
| SSL loss（低いほどよい、音楽8区間） | 0.003764 | 0.002802 |
| 楽器短文の完全一致（24区間） | 79.17% | 62.50% |
| 診断head clip macro F1 | 0.1333 | 0.5397 |
| 診断LLM clip macro F1 | 0.2698 | 0.4444 |
| 診断LLM 時間区間macro F1 | 0.0926 | 0.1481 |
| 英語CER（16発話、低いほどよい） | 76.99% | 82.79% |
| 英語WER（同16発話） | 100.00% | 118.96% |
| 日本語CER（16発話） | 93.65% | 92.97% |
| MIDI 音高＋onset F1（32区間） | 0.0464 | 0.0306 |
| MIDI 音高＋onset・offset F1 | 0 | 0.0061 |
| MIDI N1形式の有効率 | 81.25% | 84.38% |

SSLのゼロ予測基準は0.004189で、単独学習は約33.1%下回った。対象は8 kHz以下のスペクトル再構成であり、発話や音楽の意味理解を保証しない。

診断は人工加工を加えた24例で、原音は音楽・英語・日本語の3例。LLMの正常3例は単独化後も全て誤報した。headの時間区間F1は0のまま。常に全問題ありと答えるclip F1基準0.6667を単独化後も下回る。LLMの教師token NLLは実音声0.2309、無音0.1970、別音声0.2195で、F1の改善だけから音声に根拠を持つ正しい診断とは判断しない。

楽器短文は孤立stemに対するpiano／guitar／bassの弱教師テンプレートで、人手の自由文captionではない。日本語ASRはJSUTの同一話者の未知文、英語は未知6話者の16発話という範囲を維持した。WERは挿入誤りを含むため100%を超え得る。

単独ASRの教師token NLLは実音声3.9232、無音4.2610、別音声3.9215。別音声でほぼ悪化せず、内容の聞き取りは弱い。訓練8例のprobeでも英語2例CER 79.44%、日本語6例100.00%だった。短区間MIDIも訓練8例のonset F1は0.0377、完全一致0/8。未知例だけの問題とは扱わず、実データ自体への学習不足も疑う。

## 30秒・60秒への拡張

短区間の比較とは別に、同じ初期checkpointから30秒・60秒の採譜専用モデルを1,000更新した。encoderは100 Hzの全区間をFlash Attentionで処理し、LLMへ20 Hz、30秒600位置・60秒1,200位置を渡す。文脈上限8,192、生成上限4,096 token、batch 1。実際に学習した最長教師は2,562 token、最長系列は3,804位置で、音声・教師を途中で切っていない。

訓練144区間（30秒・60秒各72）、validation／test各16区間（各長さ8）。元MIDI UUIDで14／3／3作品に分離し、testは教師に計1,674音を含む。半区間strideの候補から選び、同一split内の窓の重なりは許容する。16例を16の独立作品として数えない。原MIDIで窓と交差する全音を残し、境界では観測範囲へ押鍵区間をclipする。MIDIのkey releaseで終了を定義し、ペダルは出力対象外。同一音源bankの合成ピアノであり、任意の実演を採譜できたという評価ではない。

1区間256音を超える30秒34候補・60秒50候補は、教師を削らず除外した。訓練16例・test4例はkey-eventのない低エネルギー区間で、完全なゼロ波形だけを意味しない。testの完全一致率は常に空の音符列を返すだけでも25%になるため、音符F1と音符あり・なしの採点を併記する。

未学習モデルの初期自由生成が長引いたため、0更新で停止してログを別名保存し、長区間のみ学習前は全例のNLL評価へ切り替えた。初期F1は欠測であり0ではない。学習後は全例を自由生成し、最終重みを採用する。短区間とbatch・Attention実装・入力と教師の提示量も異なるため、長さだけの効果を分離した対照とはしない。

| 長区間の固定test | 30秒 | 60秒 |
| --- | ---: | ---: |
| 例数／正解音符数 | 8／773 | 8／901 |
| 音高＋onset F1 | 0.0066 | 0 |
| 音高＋onset・offset F1 | 0.0022 | 0 |
| N1形式の有効率 | 75.00% | 50.00% |
| 生成上限への到達 | 0/8 | 2/8 |

全16例のonset F1は0.0030（TP 3／FP 315／FN 1,671）、offset込み0.0010。有効な列は10/16、不正は6/16だった。形式が有効な出力でも音符はほぼ一致しておらず、30秒・60秒へ拡張した今回の条件も採譜精度は未達だった。教師は上限内でも、自由生成では60秒2例が4,096 tokenに達した。

完全一致4/16は全て無音例で、音符あり12例の完全一致は0/12だった。不正6例の内訳はMIDI値の範囲外1、区間外の時刻1、不正な音符行2、同音の重複・重なり2。主指標には修復後の出力を混ぜていない。

正解token NLLは実音声1.7493→1.1590に低下したが、学習後の無音1.1839・別音声1.1625との差は小さい。訓練8例のprobeでもonset F1 0.0081、offset込み0、完全一致0/8。汎化だけでなく、訓練例の正しい採譜にも未達である。

30秒・60秒を各500回提示し、教師は計1,041,391 token。batch・入力量・応答長が短区間と異なる点を結果の解釈に含める。RTX 3070の学習・評価区間は約2,071.3秒（34分31秒）、PyTorchピーク割当4.29 GiB。停止した初期生成、取得、前処理、後続の監査の時間は含めない。短区間5条件の学習・評価区間の合計は約20分33秒だった。

## 保存物と検証

ローカルrunは`runs/audio-public-single-task-20261101/`。`ssl`、`caption`、`diagnosis`、`transcription`、`midi`、`midi_30_60s`が独立条件で、各々にcheckpoint、全更新履歴、評価、TensorBoardを保存する。短区間は開始前・終了後の評価、長区間は開始前NLL・終了後自由生成を記録した。SSLはスペクトル再構成の評価で、LLM応答は生成しない。既存のLAN URLで`single_`から始まるrunを選ぶ。課題に応じた入力音声、正解、実LLM応答、保存MIDIから描いた比較画像を確認できる。

学習時ソースは短区間5条件が`d3a28bd`、長区間が`0c008a2`。各条件のソース・依存lockをGitと照合し、再現実行のファイルhashも学習時と一致させた。全6条件で親checkpointと初期trainable重みの完全一致、選択課題のみ1,000更新・他課題0更新、optimizerのstep、全訓練例の提示、凍結backboneの不変性、原音・MIDIとmanifestのhashを確認した。

[監査スクリプト](../experiments/audit_single_tasks.py)で、hash順に選んだ保存応答7例をtoken列まで再現した。MIDIでは音符ありの例を使い、長区間は30秒・60秒を各1例照合した。保存MIDIから描き直した比較PNGは計24枚が画素一致した。各TensorBoardにも1〜1,000更新が揃い、実音声・正解・応答・画像を確認できた。再現した7例は全testの採点分母とは異なる。

再監査はリポジトリのルートで、短区間のソースを固定してから長区間を照合する。どちらも重みを更新しない。

```bash
mkdir -p /tmp/creative-agent-replay-d3a28bd
git archive d3a28bd src pyproject.toml uv.lock | tar -x -C /tmp/creative-agent-replay-d3a28bd
PYTHONPATH=/tmp/creative-agent-replay-d3a28bd/src uv run --no-sync python experiments/audit_single_tasks.py \
  --suite runs/audio-public-single-task-20261101 \
  --manifest data/public-audio-20261101/manifest.json \
  --long-manifest data/public-audio-20261101/manifest-long-midi-30-60.json \
  --model-dir .local/models/LFM3-1.7B-ac2586a \
  --names ssl caption diagnosis transcription midi --summary .local/single-short-verified.json
mkdir -p /tmp/creative-agent-replay-0c008a2
git archive 0c008a2 src pyproject.toml uv.lock | tar -x -C /tmp/creative-agent-replay-0c008a2
PYTHONPATH=/tmp/creative-agent-replay-0c008a2/src uv run --no-sync python experiments/audit_single_tasks.py \
  --suite runs/audio-public-single-task-20261101 \
  --manifest data/public-audio-20261101/manifest.json \
  --long-manifest data/public-audio-20261101/manifest-long-midi-30-60.json \
  --model-dir .local/models/LFM3-1.7B-ac2586a --names midi_30_60s \
  --previous-summary .local/single-short-verified.json --summary .local/single-all-verified.json
```

CUDA環境119テスト成功、CPU環境111成功・GPU用8件スキップ。[Ubuntu・WindowsのCI](https://github.com/uthree/creative-agent/actions/runs/41292065704)も成功した。データ、重み、完全なmanifest、生成全文、TensorBoardはGitへ含めず、集計・実行と監査のコード・レシピ・説明を残す。

## 結果を踏まえた次の切り分け

単独化を全課題へ一律に採用しない。ASRと採譜は、少数の実音源を訓練集合内で正しく聞き取れるか、無音・別音声に対する自由生成が変わるかを先に確かめる。SSL単独学習したencoderの接続や、発話・採譜に適した事前学習encoderとの比較も候補になる。今回だけではencoderの情報量、接続方法、教師量、最適化のどれが支配的な原因かは分離できない。

LLM生成は全条件でgreedyとし、モデル配布の`repetition_penalty: 1.1`を継承した。N1向けの文法・時刻・同音重なりの制約や、penalty 1.0との比較は未実施で、decoder設定も切り分け候補に含める。書式の有効化だけを正しい音符認識とは扱わない。
