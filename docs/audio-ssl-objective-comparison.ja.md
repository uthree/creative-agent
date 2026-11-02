# 直接パッチモデルのSSL目的を③→②で比較する

2026年11月2日 JST。[汎用SSL調査](audio-universal-ssl-survey.ja.md)の③「EMA潜在予測＋音響復元」、②「EMA潜在予測だけ」を、この順に同じ実データで学習する小規模比較。新しい実装と実験経路の説明であり、最終的な精度は別の実測報告に記載する。

## 固定するもの

- 非重複50 msの48 kHzステレオ波形パッチ、`Linear(4800, 128)`、幅128・2層・4 headsの双方向Transformer。CNN・時間平均による間引きは使わない。
- 各1,000更新、seed `20261102`、batch 8、4秒以下のcrop。音楽4・英語2・日本語2の比率を継続する。
- encoderと学習headの初期値、入力順、mask位置を一致させる。新規の人工混合は両条件とも無効。
- 既存の元録音・曲・話者単位のtrain／validation／test分割を継承する。probe用の固定cropも全モデルで共有する。
- AdamW、LR 3e-4、50更新warmup、勾配clip 1。CUDAではFP32パラメータとBF16 autocastを使う。

今回のデータはBabySlakhのpiano／guitar／bass stem、英語MiniLibriSpeech、日本語JSUT。環境音の学習・評価は含まない。音楽は合成音色、JSUTは同一話者であり、あらゆる実録音への汎化の実証ではない。

## 損失

両条件で、生徒と同じencoder構造のEMA教師がマスクなしの観測波形を見る。教師の出力を各録音の有効な完全フレームに沿って正規化し、stop-gradientを掛ける。生徒は投影後のパッチを隠し、線形predictorからマスク位置の教師表現をSmooth L1で予測する。教師はoptimizerに入れず、FP32のEMAで更新する。係数は0.99から最初の1,000更新で0.999へ近づける、小規模実験用の設定である。

③には次の補助教師を加える。

| 教師 | 内容 |
| --- | --- |
| 多解像度スペクトル | 20／50／200 msのHann FFTから、左右それぞれ32個の固定されたおおむね対数間隔の周波数binのlog振幅を予測 |
| 短時間エネルギー | 20 ms窓のlog RMS。左右のlog RMSを合成した録音内の相対エネルギー |
| 左右差 | 元録音が2 chの場合に限った左右log RMS差。校正例に有効な2 ch例がなければ係数0 |

軽量な近似としてFFTの全bin復元を32点の予測へ縮小している。CQTやEnCodec教師は使用しない。10 ms hopの教師を、1つの50 ms tokenから順序を保った5個のsubframeとして予測する。窓が原音の有効範囲内にあり、支持域全体が隠れていて、中心が採点mask内にあるsubframeだけにlossを掛ける。

標準化は最初の最大64例のtrain音声だけでfitする。各補助項の係数は、固定の最初のtrain batchにおけるパッチ投影重みへの勾配normを使って校正する。`0.05 × latent_norm / auxiliary_norm`を`[0.001, 1]`へ制限し、有効な勾配がない項は0とする。値と校正例はplanへ保存する。各lossは群内の有効位置で正規化し、音楽1/2・英語1/4・日本語1/4で集計する。

教師スペクトルの周波数集約と、特徴系列の時間方向の間引きは異なる。encoderの時系列は20 FPSで保持する。左右レベル差や振幅スペクトルの予測だけで位相を保存できるとは仮定しない。

## 共通の精度指標

学習損失の数値同士を比較せず、**マスクなしの凍結encoderから同じ線形ridge probeを学習し、同じtest音声で比較する**。学習前のランダムencoderも対照にする。既存のクラスタheadは両SSL条件で未使用・凍結で、probeとして流用しない。

| 指標 | 学習・採点方法 | 意味と限界 |
| --- | --- | --- |
| 音響クラスタ分類accuracy | 既存のtrain-fit 64クラスタを正解に、共通の線形readoutを学習。music／en／jaを個別に報告し、集計は1/2・1/4・1/4 | 音響的な情報を線形に読み出せるか。発話のCER／WERではなく、③のスペクトル教師にも近いproxy |
| MIDI key-active frame micro-F1 | 50 msパッチ中心時刻で押下中の128音高をmulti-label予測 | 公式MIDIをSSLに入れず、probeの教師として使用。完成した採譜のnote F1やonset精度ではない |
| 無音入力への置換 | 同じreadout・正解を維持し、入力だけを無音にして再評価 | 学習した特徴と予測が音に依存しているかを補足する |

probeのtrain例は音楽の全train sourceから4 cropずつ、英語・日本語は固定順の最大64 sourceから1 cropずつ。validation／testは各sourceの固定1 cropとする。音響分類のridge fitも群のフレーム総数で重みを補正する。MIDI probeには注釈が解釈できる音楽フレームだけを使う。

MIDIは原音cropの開始時刻を足したパッチ中心で、`onset <= center < key_release`を採点する。crop境界をまたぐ音も残す。pedalや残響尾は押下区間へ延長しない。表現豊かなMIDIなど既存parserで解釈できないsourceはMIDI指標から除き、除外数を記録する。音響指標からは除かない。

標準化はtrain特徴だけでfitし、ridge係数の候補`0.001／0.01／0.1／1`、MIDI閾値の候補`0.1／0.2／0.3／0.4／0.5`をvalidationだけで選ぶ。各モデルで同じ探索予算を使う。checkpointは事前に決めた最終更新のものを使い、testによる選択はしない。

1 seed・短時間・少数の独立曲による簡易比較なので、小さな差を確定的な優劣として扱わない。4秒cropの結果を30／60秒の長文脈採譜へ一般化しない。

## 実行

```bash
.venv/bin/python experiments/run_ssl_objectives.py suite \
  --manifest data/public-audio-20261101/manifest.json \
  --codebook runs/audio-patch20-comparison-20261102/codebook.pt \
  --suite runs/audio-ssl-objectives-20261102 \
  --steps 1000 --seed 20261102 --device cuda

.venv/bin/python experiments/audit_ssl_objectives.py \
  --suite runs/audio-ssl-objectives-20261102
```

`prepare`、`train --objective latent_acoustic`、`train --objective latent`、`evaluate`を分けても実行できる。`train --stop-after N`で中断点を保存でき、同じplanで再実行すると継続する。初期値・入力スケジュールのhash、実装hash、manifest・教師・校正統計のhashを照合する。

`checkpoint.pt`には教師・生徒・optimizer・全更新の履歴、`encoder.pt`には再利用可能なencoderを保存する。評価ではreadoutとencoderを保存重みから読み直し、全testフレームの予測が一致することを確認する。全成果物のサイズ・SHA256一覧を作り、監査では更新数、optimizer step、変更された重み、共通正解、TensorBoardを照合する。

TensorBoardにはtrain loss、EMA係数、教師の分散、勾配、共通probeのtest／無音精度、入力音声、対応するクラスタ正解／予測、音楽のMIDI押下ピアノロール正解／予測を記録する。画像は上側が正解、下側が予測。これはSSLとprobeの比較であり、LLM応答やMIDI生成モデルの学習は含まない。

永続ディスクの容量が小さいため、実行前にcheckpoint2つ・encoder2つ・atomic置換用の一時コピー・ログを含めた容量を確認する。データ・重み・特徴・ログはGitに含めない。

## 1,000更新から10,000更新へ継続する

```bash
PYTHONPYCACHEPREFIX=/tmp/creative-agent-bytecode-runtime \
  .venv/bin/python experiments/run_ssl_objectives.py suite \
  --manifest data/public-audio-20261101/manifest.json \
  --parent runs/audio-ssl-objectives-20261102 \
  --suite runs/audio-ssl-objectives-10k-20261102 \
  --steps 10000 --seed 20261102 --device cuda

PYTHONPYCACHEPREFIX=/tmp/creative-agent-bytecode-runtime \
  .venv/bin/python experiments/audit_ssl_objectives.py \
  --suite runs/audio-ssl-objectives-10k-20261102
```

別の出力先で、親の生徒・EMA教師・optimizer・更新履歴を引き継ぎ、③→②の順に各9,000更新を追加する。合計は各10,000更新であり、各11,000更新ではない。親のcheckpoint・encoder・probe・評価結果は上書きしない。

親planとcheckpointのSHA256、config・seed・manifest、最初の1,000更新のスケジュールが一致することを確認する。学習アルゴリズム・データ・probeのコードが変わった親からは継続しない。校正統計・補助loss係数・codebook・probe例は親と同じものを引き継ぎ、再校正値に置き換えない。EMAは既存の式に従い1,000更新後は0.999付近、LRはwarmup後の3e-4を維持する。

新しいTensorBoardには、親の保存履歴から最初の1,000更新のtrain scalarを再記録し、以降を追加する。再記録した履歴を新たなoptimizer更新とは数えず、監査で親の履歴との厳密一致と、合計・追加更新数を別々に確認する。10,000更新の最終encoderに、同じ候補集合からvalidationで選ぶprobeを学習し、共通testで比較する。

容量確保には再生成可能なPython bytecode cacheだけを一時領域へ退避できる。ソース、インストール済みライブラリ本体、データ、重み、実験ログは維持し、新しいbytecodeの書き込み先も一時領域へ指定する。checkpoint等の実験成果物は永続領域へ保存する。
