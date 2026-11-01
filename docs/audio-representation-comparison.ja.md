# 20 FPS CNNと浅いTransformerの自己教師あり比較

[9 run・9,000更新の実測結果](audio-representation-comparison-report.ja.md)と[匿名集計](../experiments/results/audio-ssl-cnn-comparison-20261102.json)を参照。

音楽・発話から内容を保持するencoderを学習し、LLMへ接続する前に比較する。新しい`RepresentationEncoder`は48 kHzステレオ波形から20 FPS・128次元を返す。LLM、文字起こし教師、MIDI教師はこのSSLの損失に使わない。

## 参考と実装範囲

[Conformer原論文](https://arxiv.org/abs/2005.08100)の局所的な畳み込みと大域的なAttentionの組み合わせ、pointwise GLU・depthwise convolution・SiLUを参考にする。今回のCNNは100 FPSのstemの後にkernel 15・stride 5・padding 5を置き、20 FPSへ圧縮する。正規化は時刻ごとのLayerNorm。後段は幅128・4 heads・2層の双方向TransformerとRoPEであり、Macaron FFNやblock内畳み込みを持つ原論文のConformerそのものではない。

[WavLM原論文](https://arxiv.org/abs/2110.13900)のclean音声に対する離散ラベルのmasked predictionと、別音声を混ぜたdenoisingを参考にする。今回は訓練音声だけから作った固定log-melクラスタの初段SSLとし、WavLMの事前学習済み重み、HuBERT教師の再クラスタリング、gated relative position biasは使わない。原論文の20 ms strideに対し、今回は指定の20 FPS（50 ms stride）である。WavLMの再現や公開モデル同等の性能を主張しない。

## 事前に固定する比較

| 条件 | Transformer前 | Transformer | Transformer後 |
| --- | --- | --- | --- |
| `late_concat` | 10 ms・非重複CNN stem | 100 FPS、2層 | 5フレーム連結→640から128へ投影 |
| `early_concat` | 同じstem→5フレーム連結→640から128へ投影 | 20 FPS、2層 | 正規化 |
| `cnn20` | 同じstem→pointwise GLU→depthwise CNN→LayerNorm→SiLU→pointwise投影 | 20 FPS、2層 | 正規化 |

全条件の出力は20 FPS・128次元で、同じ64クラス線形headを使う。`late_concat`は現行の処理順を模した対照であり、従来のスペクトル復元SSLやLLMを含むモデルと同一ではない。3条件とも新規初期化し、同じseedのstem・Transformer・分類headの初期重みを一致させる。CNNと連結投影でパラメータ数は異なり、実数を記録する。平均による時間圧縮は使わない。

各条件1,000更新、seed 20261102／20261103／20261104の3つ、計9 run・9,000更新。実効batch 8で音楽4・英語2・日本語2。区間は最大4秒、短い発話は有効長を維持してゼロ埋め。AdamW、LR 3e-4、warmup 50、その後一定、gradient clipping 1.0。GPUはFP32パラメータ＋BF16 autocast、SDPAの自動backend選択を使う。更新数・音声区間・donor・SNR・maskをseedごとに3条件で揃え、同じGPUで順次実行する。実時間・ピークallocated VRAMは別に記録する。

## データ・教師・情報漏れの防止

既存公開manifestの元音声から、BabySlakhの音楽、Mini LibriSpeechの英語、JSUTの日本語を使う。既存SSLタスクのレコードだけを使うとvalidation／testが音楽のみになるため、元のsource分割を維持して全3群から作り直す。

- 音楽はtrain 71録音・14作品、validation 11録音・3作品、test 14録音・3作品。同じ作品のstemを独立作品と数えない。同じ合成音源bankであり実演の評価ではない。
- 英語は252／8／16録音。元の話者分割を維持する。
- 日本語は640／8／16発話。同一女性話者のテキスト分割であり未知話者への保証はない。
- trainは元音声内で毎回区間を選ぶ。validation／testは各録音1区間を固定し、条件・seed間で共通にする。testは最終更新だけ評価し、checkpoint選択に使わない。

教師はclean音声の非重複50 ms窓のステレオ合計パワーから作る40次元log-mel。窓内の総パワーで正規化し、訓練256区間（音楽128・英語64・日本語64）の完全な窓だけで標準化と64クラスタのk-meansをfitする。教師は全runで固定し、validation／testや混合音声をfitへ入れない。

20 FPSで約40%の完全フレームを4フレームspanでmaskし、その位置だけCEを計算する。CNNの受容野は150 msなので、対象の前後1フレームもTransformerへ入る前の特徴段階で隠す。比較3条件に同じ保護範囲を適用する。mask対象の元波形だけを書き換えても、Transformerへ入る特徴が変わらないことをテストする。末尾の不完全な窓とpaddingは教師lossから除く。

50%の訓練例で、同じ群・同じtrain splitの別録音をSNR 0〜10 dBで混ぜる。教師は元のclean音声のまま。短いdonorは反復して長さを合わせる。検証は同じclean教師・maskに対してclean入力／mixed入力／無音入力を比較する。

## 評価と保存

群別のmasked CE・accuracy・クラス別recall平均、教師／予測クラス数を出す。訓練教師の頻度に基づく群別priorとも比較する。音楽0.5・英語0.25・日本語0.25の重みで集計し、録音数の偏りと区別する。自己教師の予測精度であり、ASRや採譜の認識精度ではない。

TensorBoardには全更新のloss・勾配・mask数・混合例数・LR、validationの推移、最終test、clean／mixed入力音声、固定教師ラベルと予測とmaskの比較画像を出す。SSLにはLLM応答はない。checkpoint・optimizer・元の予測列・codebook・計画・20 FPS encoder exportはGit除外対象のrunへ保存する。

```sh
uv run --no-sync python experiments/run_ssl_comparison.py prepare \
  --manifest data/public-audio-20261101/manifest.json \
  --suite runs/audio-ssl-cnn-comparison-20261102

uv run --no-sync python experiments/run_ssl_comparison.py train \
  --manifest data/public-audio-20261101/manifest.json \
  --suite runs/audio-ssl-cnn-comparison-20261102 \
  --variant cnn20 --seed 20261102
```

全9条件を順次実行する場合は、同じ引数でstageを`prepare`から`suite`へ変える。trainは各variant・seedへ実行できる。途中の確認には`--stop-after 25`を使い、同じコマンドからその指定を外して再開する。source・lock・plan・codebookを照合し、専用のmask／augmentation乱数とoptimizerを復元する。完了済みreportは同じsource・教師・scheduleを確認して再利用する。

新encoderのcheckpointは`audio_representation_encoder_v1`で、従来の100 FPSスペクトル復元モデルと区別する。LLM用projector・LoRAの再学習、従来の診断headやSSL復元教師との統合はこのencoder単体の比較には含めない。

監査は`experiments/audit_ssl_comparison.py --manifest ... --suite ... --summary ...`で実行する。履歴・optimizer更新数・seedからの初期重み・Git commitとソースhash・codebook・sampling順・TensorBoard・全保存指標を照合し、3群を含む固定test batchの予測を保存重みから再現する。30秒／60秒の実音楽でも20 FPSの出力形状と推論時間を確認する。比較表と匿名集計は実測報告へ記録した。
