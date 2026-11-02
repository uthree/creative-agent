# GPT-2式の残差初期化とVICReg型損失を順に比較する

2026年11月2日 JST。[前回の初期化比較](audio-ssl-initialization-report.ja.md)に続き、残差出力層をGPT-2式の正規分布で初期化する。その潜在予測だけの条件を学習・評価した後、同じ初期値からVICReg型の分散・共分散項を加えた条件を学習する。

## 初期化

Attention出力投影`output.weight`とFFN末尾`down.weight`だけを、次の正規分布で初期化する。

$$
W_{\mathrm{residual}}\sim\mathcal{N}\left(0,\left(\frac{0.02}{\sqrt{2L}}\right)^2\right)
$$

$L$ はTransformer層数。今回の2層では標準偏差0.01。通常のuniform重みに0.5を掛ける方式とも、前回の1e-4倍とも異なる。[GPT-2原論文 §2.3](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)と[TransformersのGPT-2実装](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)を参照した。

Q/K/V、FFNのgate/up、波形パッチ投影、predictor等は従来どおり。モデル全体をGPT-2へ変更するものではなく、双方向Attention・RMSNorm・SwiGLU・20 FPSの直接パッチ構成を維持する。全線形層を標準偏差0.02へ変更することもしない。

残差出力層には現在の初期化乱数stateから派生した独立generatorを使い、他の重みの乱数列を消費しない。初期化以外の重みを対照と一致させる。forward中の固定係数や新しい学習パラメータは加えず、checkpointロード後に再初期化しない。

新規latent用CLIの既定値は`--residual-init gpt2`。汎用configでこの項目が欠ける場合は`legacy`へfallbackし、古いencoderを読む際の設定を維持する。旧1e-4方式は`--residual-init legacy --residual-init-scale 1e-4`で再現できる。GPT-2方式と旧scaleの同時指定は拒否する。

## 追加損失

[VICReg原論文](https://arxiv.org/abs/2105.04906)・[公式実装](https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py)の分散・共分散項を参考にする。元のVICRegを完全再現するのではなく、既存のEMA教師・masked Smooth L1潜在予測を残した補助項である。新しいprojectorや音響復元教師は加えない。

予測の生徒はマスク付き音声を見る。正則化用には同じ音声の**マスクなし生徒encoder**をもう一度forwardし、その20 FPS・128次元出力 $z$ に勾配を流す。マスクパターンだけから統計を作る経路を避け、推論時の特徴を直接正則化する。教師とpredictorにはこの正則化項を直接掛けない。通常の潜在予測よりforward/backwardが増えるため、計算時間も別に報告する。

完全な50 msパッチのうち、元波形の左右・サンプルに沿ったRMSが`1e-4`を超えるフレームだけを選ぶ。padding、末尾の部分パッチ、無音を除外する。十分なフレームがないbatchでは正則化項を0にし、無音へ人工的な変動を強制しない。RMSの平均は選択用の統計であり、encoderの時間平均poolingではない。

選択フレームの重み $w_i$ は、音楽1/2・英語1/4・日本語1/4、その群内の録音を等重み、各録音内の有効フレームを等重みにする。欠けた群の分は再正規化し、$\sum_i w_i=1$ とする。教師targetとは異なり、各録音の時間標準化は行わない。録音間の音高等の違いも統計に残す。

$$
\mu=\sum_i w_i z_i,\qquad
C=\frac{\sum_i w_i(z_i-\mu)(z_i-\mu)^\top}{1-\sum_i w_i^2}
$$

重み付きの不偏補正を用いるが、時間フレームは相関するため、独立な大量標本を得たとは見なさない。各次元の標準偏差を1以上へ促し、非対角共分散を抑える。

$$
L_{\mathrm{var}}=\frac1D\sum_d\max\left(0,1-\sqrt{C_{dd}+10^{-4}}\right),\qquad
L_{\mathrm{cov}}=\frac1D\sum_{d\ne e}C_{de}^2
$$

$$
L=L_{\mathrm{latent}}+L_{\mathrm{var}}+0.04L_{\mathrm{cov}}
$$

係数は公式の25:25:1を参考に、潜在項を1へ揃えて1:1:0.04と事前固定する。ただし潜在項は元論文のMSEではなく既存のSmooth L1で、正則化は生徒の1 viewだけなので同じレシピではない。校正やtestでの係数探索はしない。設定・係数はplanへ保存する。

CUDAのencoderはBF16 autocast、統計・共分散行列・lossは明示的にFP32で計算する。教師は従来どおりstop-gradientとFP32 EMAで更新する。

定数表現では $C=0$ となり、潜在lossが0でも分散ペナルティは0.99となる。一方、`sqrt(var + epsilon)`では完全な定数点の勾配も0なので、そこから必ず脱出する保証ではない。分散の大きいノイズやドメイン差だけで有用性が保証されるわけでもなく、下流probeと入力置換の診断を併用する。

## 比較条件と評価

順番は`latent` → `latent_vicreg`。どちらもGPT-2残差初期化、randomパッチ投影、幅128・2層、batch 8、各10,000更新、seed `20261102`、4秒以下crop。前半のcheckpointへ後半を継続する方式ではなく、**同一初期値から各10k**を学習し、学習量と損失の違いを混同しない。前半のprobe評価と監査を終えてから後半を開始する。

共通データは従来の音楽・英語・日本語。既存のtrain/validation/test分割、入力順、crop、mask、LR、EMA、コードブック、probe例を維持する。前回の変更なし/1e-4初期化の結果も参照し、入力・評価例・正解が一致することを照合する。

学習前の共通encoderと、2条件の最終encoderを同じ凍結ridge probeで比較する。音響クラスタaccuracy、MIDI押下フレームF1とmicro AP、無音置換を主な外部指標とする。ridge係数・MIDI閾値はvalidationだけで選び、testや途中のprobeでcheckpointを選ばない。音響accuracyはクラス均等ではなく音楽1/2・英語1/4・日本語1/4の集計である。

固定validationの診断はstep 0と500更新ごと。ゼロ予測との比較、退化フラグ、raw教師と生徒の変動、有効rank、同じ長さに揃えた別音声との特徴差、各層残差RMS比に加え、両条件で`variance_penalty`・`covariance_penalty`・正則化の標本録音数/フレーム数・stdの平均/最小値を記録する。lossに正則化項が入らない対照でも同じ診断を測る。

## 実行

```bash
PYTHONPYCACHEPREFIX=/tmp/creative-agent-bytecode-runtime \
  .venv/bin/python experiments/run_gpt2_vicreg.py \
  --manifest data/public-audio-20261101/manifest.json \
  --codebook runs/audio-patch20-comparison-20261102/codebook.pt \
  --reference runs/audio-ssl-initializations-20261102/baseline \
  --suite runs/audio-gpt2-vicreg-20261102 \
  --steps 10000 --seed 20261102 --device cuda
```

中断後は同じコマンド・同じ実装revisionを使う。checkpointより先に記録されたvalidation行は再開時に除き、完了済みprobeは再fitしない。損失の全成分、optimizer step、同じ初期値、共有正解、保存encoder/readoutからの全test予測再現、TensorBoard、成果物hashを監査する。

データ・重み・ログはGit除外の`runs/`へ永続保存する。今回の容量確保では、完全一致する不変ファイルの内容・パスをSHA256照合付きで共有し、再取得可能なパッケージキャッシュだけを一時退避した。既存実験の内容は変更しない。小さなGPUスモークは別の出力先と短い`--steps`を指定し、10kの参照は指定しない。
