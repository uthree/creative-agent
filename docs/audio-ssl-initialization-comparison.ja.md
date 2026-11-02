# 潜在予測SSLの初期化を1要因ずつ比較する

2026年11月2日 JST。潜在予測のみの目的関数を維持し、残差出力層の小さい初期化と、波形パッチ投影の正弦波初期化を別々に比較する。学習loss以外の検証指標も記録する。

## 条件と固定項目

| 条件 | 残差出力層 | パッチ投影 |
| --- | --- | --- |
| baseline | 従来の初期化、係数1 | 従来のランダム初期化 |
| residual_1e4 | 初期重みだけ1e-4倍 | baselineと同一 |
| mel_sinusoidal | baselineと同一 | メル尺度のsin/cos、bias=0 |

同じseed `20261102`、各10,000更新、batch 8（音楽4・英語2・日本語2）、4秒以下、50 msの直接パッチ・幅128・Transformer 2層を用いる。データ分割・入力順・crop・mask・EMA・optimizer・LR・凍結probe例・探索予算を揃える。今回は2案を同時に適用する条件を含めない。対照も同じコードで最初から学習する。

初期stateを比較し、変更されたtensorが残差案では2層それぞれの`output.weight`と`down.weight`だけ、正弦波案では`stem.weight`と`stem.bias`だけであることを確認する。残りの重みは同一。各条件に対応する学習前encoderも同じprobeで評価し、初期特徴の違いと学習による変化を区別する。

BabySlakhの音楽・英語MiniLibriSpeech・日本語JSUTを継承する。環境音・ASRのCER/WER・完成MIDIのnote F1・30/60秒の長文脈採譜は今回の評価に含まない。音楽は合成stem、JSUTは同一話者で、1 seed・少数曲による比較である。

## 残差出力層の初期化

各Transformer blockのAttention出力投影`output.weight`と、SwiGLU末尾の`down.weight`を、通常の初期化後に`1e-4`倍する。どちらもbiasは元々ない。Q/K/V、FFNのgate/up、パッチ投影、RMSNormは変えない。追加パラメータ・forward中の係数は導入しない。保存重みのロード後に再び縮める処理もない。

[GPT-2原論文 §2.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)でも残差重みを初期化時に`1/sqrt(N)`で縮める。`N`は残差分岐数で、[TransformersのGPT-2実装](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)はAttention/MLPの出力投影の標準偏差を`initializer_range / sqrt(2 * n_layer)`にする。2層なら係数0.5であり、今回の1e-4ははるかに強い抑制で、GPT-2の初期化をそのまま再現するものではない。

最初は各blockが恒等写像に近くなるが、固定の`x + alpha*f(x)`とは学習中の制約・勾配・パラメータ更新が異なる。AdamWの更新で小さい重みが早く成長する可能性があるため、実測した残差RMS比も追う。マスク位置では初期状態の文脈混合も弱くなるので、改善を前提にしない。

## 正弦波パッチ投影

左右それぞれ32周波数×sin/cosの64成分、合計128成分を配置する。周波数は20 Hz〜20 kHzをHTKメル尺度で等間隔に配置し、50 msの非周期Hann窓を掛けた正弦・余弦を各投影行の初期値とする。パッチ内の時間原点は中央。各行のL2 normを`1/sqrt(3)`に揃え、従来の`nn.Linear`初期化の期待行normと合わせる。biasは0。重み・biasは以降も通常どおり学習する。

入力の左2,400サンプル・右2,400サンプルを分け、出力は左のsin/cos対を周波数順、その後に右の対を並べる。モデル幅は4の倍数を要求する。CNN、時間平均pooling、追加の特徴次元は導入しない。

これはメルスペクトログラムではない。[メル三角フィルタ](https://docs.pytorch.org/audio/main/generated/torchaudio.functional.melscale_fbanks.html)は周波数binの重み付き集約であり、今回の線形投影にはpower、三角帯域積分、log、位相除去はない。sin/cos対もそのまま保持する。32個の窓付き周波数フィルタなので、全FFT binの情報を網羅しない。左右同じモノラル音声では左右の初期特徴も重複する。良い初期表現になるかはprobeで比較する。

## loss以外の指標

| 指標 | 観測するものと限界 |
| --- | --- |
| 凍結encoderの音響クラスタaccuracy | train-fit 64クラスタを線形ridgeで予測。音楽1/2・英語1/4・日本語1/4。音響情報のproxyであり意味理解の精度ではない |
| MIDI押下フレームmicro F1 | 50 ms中心時刻の128音高。ridge係数・閾値はvalidationで選ぶ。note/onset精度ではない |
| MIDI micro AP | 128音高×フレームをflattenしたスコアのランキング。完全tieをまとめ、閾値の影響を補足する。APでもridgeはvalidation F1で選んだものを使う |
| 原音と無音のprobe・特徴差 | 出力が音声に依存するか。良い意味表現を保証する値ではない |
| 特徴の標準偏差・有効rank | 定数化・一部方向への集中の診断。全128次元が等分散であるべきとは仮定しない |

上のprobeを各条件の学習前・最終encoderに対して同じ例で学習・採点する。testでcheckpointや係数を選ばない。原音・無音を含め、保存encoder/readoutから全test予測を再実行し、完全一致を確認する。APは追加診断だけでなく通常の`score_probes`でも記録する。

学習中は、固定validationから各群2録音・計6録音を取り、step 0、500更新ごと、最終更新で次をFP32評価する。crop・maskは固定し、勾配・EMA・optimizer・乱数状態・学習入力順は変えない。

- 同じ教師targetに対するモデルlossと常時ゼロpredictorのloss。`relative_gain_over_zero = 1 - model_loss / zero_loss`が正ならゼロ予測より良い。ただし教師が変化するので、時点をまたぐ絶対的な精度尺度にはしない。
- `zero_loss <= 1e-8`なら`target_is_degenerate=1`として、比率を0にする。target自体が消えた場合を「良い予測」と扱わない。
- 教師の時間標準化前の録音内標準偏差、正規化後targetの標準偏差、生徒特徴・predictorの標準偏差、生徒特徴の有効rank。
- 原音と無音の特徴RMS差。同じ群の別録音との特徴RMS差は、全録音を同じ有効長に揃えて計算し、長さ・paddingの差を手掛かりにしない。
- 各層のAttention/FFN残差のRMSを、その加算直前の本流のRMSで割った値と、出力重みのRMS。マスクなし入力の完全パッチで測る。

validation診断は`latent/validation.json`とTensorBoardの`validation/`へ記録する。分散や入力依存だけを満たす不要な特徴も作れるため、凍結probeと合わせて判断する。この2つの初期化は定数解を損失から排除せず、分散・共分散ペナルティは今回追加しない。

## 実行と再現

```bash
PYTHONPYCACHEPREFIX=/tmp/creative-agent-bytecode-runtime \
  .venv/bin/python experiments/run_ssl_initializations.py \
  --manifest data/public-audio-20261101/manifest.json \
  --codebook runs/audio-patch20-comparison-20261102/codebook.pt \
  --suite runs/audio-ssl-initializations-20261102 \
  --steps 10000 --seed 20261102 --device cuda
```

順番はbaseline → residual_1e4 → mel_sinusoidal。中断後は同じコマンドで保存点から継続する。個別監査と、共通入力・mask・probe例・正解・変更tensor範囲の照合を行い、合計更新数を保存する。小さいGPUスモークは出力先と`--steps`を変更して実行できる。

汎用の`run_ssl_objectives.py prepare`でも`--residual-init-scale 1e-4`、`--patch-init mel_sinusoidal`を指定できる。既定値は従来の`1.0`と`random`のまま。古いconfigの欠落項もこれらへfallbackし、旧checkpointを読み込むと保存済み重みがそのまま使われる。実装hashを照合する厳密な継続実験には、記録されたrevisionを用いる。

データ・重み・optimizer・ログは除外対象の`runs/`に保存し、Gitには設定・ソース・小さな集計だけを残す。初期化や評価を変更した新しい実験であり、以前の10k checkpointへ途中から適用しない。
