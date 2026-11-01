# 公開実データによる5課題の学習と汎化評価

2026年11月1日。[候補調査](public-audio-datasets-survey.ja.md)に基づき、容量を抑えて取得できるMini LibriSpeech、JSUT basic5000の一部、BabySlakhを使う。既存の音声encoder・projector・LFM3 LoRAを実データへ接続する。発話はLLMによる書き起こしを維持する。

## 取得と用途

| データ | 取得経路・検証 | 用途・条件 |
| --- | --- | --- |
| Mini LibriSpeech | [OpenSLR 31](https://www.openslr.org/31/)の`train-clean-5`と`dev-clean-2`。公式MD5と取得後SHA-256、展開ファイルの整合性を確認 | 英語の実発話・全文書き起こし。CC BY 4.0。trainは元train、validation/testは元devの話者を分ける |
| JSUT v1.1 | [著者の公式案内](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)が指定するZIPから、basic5000の1,024発話と同梱文書をHTTPS Rangeで部分取得。ZIPのCRC・サイズと取得後SHA-256を確認 | 日本語の実発話。元文をNFC化。音声は非商用研究・個人利用等に用途が限られ、原文の条件は同梱LICENCEを保持。**1話者なので未知話者の評価ではない** |
| BabySlakh v2 | [公式配布](https://zenodo.org/records/4603870)、882,818,115 bytes。公式MD5とSHA-256を照合。16 kHz monoのstemと楽器別MIDI | 音楽SSL、楽器の弱教師、ピアノのkey-note採譜。CC BY 4.0。元MIDI UUIDで14／3／3作品に分割 |

Hugging Face認証は環境から読み込み、トークンを表示・保存しない。今回ReazonSpeechはファイル取得に403の利用承認エラーがあり、JSUTを使用する。Mozilla Data Collectiveの認証情報は別であり、Hugging FaceのトークンからCommon Voiceの取得権限は得られない。

JSUTを含む実験は非商用研究・個人利用の系統として記録する。取得音声、教師全文、完全なmanifest、TensorBoard media、checkpointはGitに含めない。JSUTの文には複数のCC条件があり、音声の条件と区別する。

## データ契約

`public_audio_v1`のmanifestは原音一覧と、原音を参照する課題別区間からなる。原音のhash、rate、ch、frame数、出典、権利、分割単位、元MIDIとそのhashを記録する。複数課題が同じ原音を使ってもsplitは共通で、評価用原音をSSLへ訓練投入しない。

音声は区間ごとに遅延decodeし、SciPyの`resample_poly`で明示的に48 kHzへ変換する。monoは左右に複製するが、実ステレオ情報を持つとは扱わない。小さな16区間cacheだけを使い、全波形をRAMやfloat32の別ファイルへ複製しない。元ファイルは変更せず、加工用の余裕として区間peakを0.25へ揃える。

- **発話**：1〜9秒の完全な発話を選び、長い発話を切って全文の正解を付けない。英語は元ラベルを小文字化、日本語はNFC。train/validation/testをまたぐ同一文も除外する。
- **caption**：piano／guitar／bassの孤立stemに、元MIDI program由来の短文を付ける。3クラスの件数を揃える。**人手による自由な楽曲説明ではなく、楽器名の弱教師**である。
- **MIDI**：対応する`MIDI/SXX.mid`から実際のkey onset・key release・velocityを読み、0.5〜3秒・最大10音の区間へ変換する。全ての鍵盤が離れた境界から選び、境界で音符を短くしない。既存N1の出力は標準piano programへ正規化し、元のprogramを出典に保持する。controller数も保持するが、ペダルや音色制御を生成する課題ではない。無音の教師はこのsubsetには作らない。
- **診断**：音楽・英語・日本語の原音へ既知の3加工を加え、正常を含む8組合せを作る。加工前後を観測して確認できない例は除外する。「正常」は追加加工なしを意味し、元音源全体の品質保証ではない。monoを複製した音声で左右加工を学ぶため、本来のステレオ制作物への評価は別に必要。
- **SSL**：音楽と発話の波形を混ぜる。16 kHz原音を48 kHzへ変換しても高域情報は増えないため、復元lossの対象は8 kHz以下とする。現subsetの固定validation/testは音楽区間である。

BabySlakhには`audio_rendered: false`・`midi_saved: false`でも実ファイルが存在する箇所があった。元フラグを保存し、アーカイブのchecksum、実ファイル、音声ヘッダー、MIDIを検査して採用する。フラグの不一致・欠損・未対応MIDI・低エネルギー・観測不能な加工は件数を記録する。

既存スモーク用loaderの32原音上限は維持し、公開データは`PublicData`の別経路を使う。manifestは完成後に上書きしない。

今回固定した集合は次のとおり。ASRの訓練は英語252・日本語640発話、合計約71.3分。MIDIの訓練は約19.8分である。複数課題が原音を共有するため、課題別の時間を独立した収録時間として足し合わせない。

| 課題 | train | validation | test |
| --- | ---: | ---: | ---: |
| SSL | 512 | 8 | 8 |
| 楽器の弱教師caption | 600 | 12 | 24 |
| 正常・単一・複合診断 | 736 | 24 | 24 |
| 発話書き起こし | 892 | 16（英日各8） | 32（英日各16） |
| ピアノMIDI | 800 | 16 | 32 |

診断validation/testは、それぞれ音楽・英語・日本語の3原音に8組合せを適用したもの。24例は24の独立原音ではない。MIDIも多数の区間を同じ元作品から選んでおり、元作品数を評価例数と区別する。

## 学習・評価

[専用recipe](../recipes/audio-public-multitask-lfm3.json)は既存LFM3-1.7Bと同じencoder・rank 8 LoRAを使う。旧5,000更新の重みからwarm startでき、optimizer・scheduler・samplerは新しく作る。同一データでのresumeは別機能で、元manifest、コード・依存lock、backboneの署名を検査する。

事前に全教師と音声長・生成予約が文脈予算へ収まることを確認する。教師を黙って切り捨てず、生成上限を超える教師も拒否する。encoderを更新するため、固定音響特徴への置き換えはしない。

今回の開始予算は**新たに5,000更新、5課題へ各1,000更新、batch 2**。validationは開始前・2,500・5,000更新で、固定testと訓練8例のprobeは開始前・5,000更新で生成する。最終重みを評価し、testから良いcheckpointを選び直さない。

採点は各固定評価集合の全例で行い、TensorBoardのmedia表示件数と区別する。

- ASR：言語別CER、英語WER、完全一致率、打ち切り件数。英語はNFC・casefold・アポストロフィ以外の句読点除去・空白正規化、日本語はNFCのみ。未正規化CERも保持する。
- MIDI：N1有効率、音高＋onset／offset F1、完全一致率。key velocityは完全一致にだけ含める。不正出力を正解や正常な無音へ置き換えない。
- caption：楽器短文の完全一致、クラス別値、訓練多数クラスだけを返す基準値。
- 診断：headと実LLM応答を別々に採点し、正常・単一・複合、clip／時間区間F1、正常誤報、不正応答数を記録。「全問題あり」の単純なclip F1も比較する。
- SSL：固定maskでの8 kHz以下の復元lossと、常にゼロを返す基準値。

正解文NLLは実音声・同長無音・別音声の3条件で比較する。別音声は同じ言語・課題で長さの近い異なる教師を選び、中央cropまたは右zero-padで入力長を揃える。文章が音声に依存しているかを確認し、NLLの低下だけで汎化成功とは扱わない。

TensorBoardはvalidation／test／train_probeを分け、入力音声、正解、実LLM応答、MIDIの正解・生成を比較する画像を記録する。MIDI画像の音符は書き出したMIDIを読み直して描く。不正出力は赤く表示する。訓練probeは8例の抜き取りで、訓練集合全体の精度ではない。

## 再実行

```bash
uv sync --locked --no-group cpu --group cuda

# 生データはGit除外領域、配布アーカイブは容量に余裕のある場所へ。
uv run --no-sync creative-agent fetch-public-audio \
  --output data/public-audio-NEW/sources \
  --cache /tmp/creative-agent-public-downloads --jsut-count 1024

uv run --no-sync creative-agent prepare-public-audio \
  --data-root data/public-audio-NEW --seed 20261101

uv run --no-sync creative-agent train-audio-multitask \
  --config recipes/audio-public-multitask-lfm3.json \
  --data-root data/public-audio-NEW \
  --public-manifest data/public-audio-NEW/manifest.json \
  --model-dir .local/models/LFM3-1.7B-ac2586a \
  --warm-checkpoint runs/audio-multitask-lfm3-5000-20261101/checkpoint.pt \
  --steps 5000 --device cuda --output runs/audio-public-NEW
```

同じrunの中断から続ける場合は元の設定・manifest・outputを指定し、`--warm-checkpoint`を外して`--resume`を付ける。取得は完成アーカイブのchecksum、展開済みファイルの内容、JSUTのCRCを検証して再利用する。

今回の評価は小subset・1 seed・混合学習の試行である。英語は未知話者、日本語は同一話者の未知文、音楽は同じ合成bankからの未知元MIDIという範囲を分ける。任意の長さの音源、実演アンサンブル、未知の音源bank、無音採譜、自然な制作ミスへの性能は別の評価が必要になる。
