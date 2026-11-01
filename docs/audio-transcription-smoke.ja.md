# 発話の潜在表現をLLMへ渡す書き起こしスモーク

音声エンコーダーの潜在表現を線形projectorでLLMの入力埋め込みへ写し、LLMが書き起こし文とEOSを自己回帰生成する。キャプション学習とモデル・学習・保存処理を共有し、書き起こし専用のmanifest検査とCER/WERを加える。CTC headやCTC lossは使わない。

今回の範囲は接続のスモーク。SSLで更新済みのencoderを読み込めるが、LLMはランダム初期化の小型Qwen3、tokenizerはUTF-8 byte方式で、事前学習済みLLMの重みやtokenizerではない。encoderとLLMを凍結し、応答CEでprojectorだけを更新する。本番のASR能力や日本語認識精度の達成は合格条件に含めない。

## 実行

Python 3.12とuvを使う。音声fixtureの生成には別途[eSpeak NG](https://github.com/espeak-ng/espeak-ng/blob/master/docs/guide.md)が必要。Linuxでは配布元のパッケージ、Windowsでは公式配布を用意する。モデルやデータの自動ダウンロードはしない。外部コーパスを持ち込む場合、eSpeakは不要。

```bash
uv sync --locked --no-group cpu --group cuda
uv run --no-sync creative-agent prepare-speech-fixtures \
  --config recipes/audio-transcription-smoke.json \
  --espeak espeak-ng --output data/speech-smoke-001
uv run --no-sync creative-agent smoke-audio-transcription \
  --config recipes/audio-transcription-smoke.json \
  --manifest data/speech-smoke-001/manifest.json \
  --encoder-checkpoint runs/audio-ssl-cuda-initial-20261101/checkpoint.pt \
  --device cuda --steps 20 --output runs/transcription-cuda-001
uv run --no-sync creative-agent smoke-audio-transcription \
  --config recipes/audio-transcription-smoke.json \
  --device cuda --steps 21 --resume --output runs/transcription-cuda-001
```

`--espeak` には実行ファイルのパスも指定できる。`--manifest` を省くとPATH上の `espeak-ng` でfixtureを生成する。見つからなければ停止する。トーンを発話として代用しない。CPUでは `uv sync --locked --group dev` と `--device cpu` を使う。既存SSL checkpointの指定は任意で、未指定時はランダムencoderであることを記録する。再開時に `--encoder-checkpoint` は指定しない。

`--steps` は累計の実optimizer更新数。設定、manifestと音声のhash、ソースとlock、精度、backend、device種類を検査し、異なるタスクのcheckpointも拒否する。CPUの連続実行と中断・再開で、重み・AdamW状態・更新履歴・最終評価の厳密一致をテストする。

## 発話と正解の契約

既定レシピはオリジナルの短い英語8文をeSpeak NGの同一voiceで合成し、train 6文、validation 2文に分ける。文面と音声ファイルをsplit間で重複させない。話者は共有しており、未知話者評価にはならない。合成器の版、voice、元音声のhashとsample rateをmanifestへ残す。

合成器のmono出力を線形補間で48 kHzへ明示的にupsampleし、左右へ複製する。自然録音、native 48 kHz、独立したステレオ情報があるデータとは区別する。入力上限は3秒、全文脈512位置、教師256 UTF-8 bytes、生成32 tokens。上限超過は切り捨てず停止する。

独自manifestでは[SSLの音声契約](audio-ssl-smoke.ja.md)に加え、次を指定する。音声ファイルは前処理後の48 kHz・2chで保存し、変換内容を明記する。

```json
{
  "schema_version": 1,
  "task": "speech_transcription",
  "rights": {"origin": "利用権限のある音声の出所"},
  "transcript_rights": {"origin": "教師文の出所と利用条件"},
  "assets": [{
    "id": "utterance-unique-id",
    "split": "train",
    "path": "speech.wav",
    "sha256": "ファイルのSHA-256",
    "frames": 48000,
    "sample_rate": 48000,
    "channels": 2,
    "transcript": "利用者が確認した書き起こし文",
    "language": "ja",
    "source_kind": "recorded_speech",
    "speaker_id": "ローカル話者ID",
    "preprocessing": "例: 48 kHzのmonoをL/Rへ複製"
  }]
}
```

trainとvalidation両方が必要で、最大32音声。`source_kind` は `synthetic_speech` または `recorded_speech`。空の教師文・言語、欠けた話者や前処理宣言、NFC正規化後のsplit間の同じ教師文を拒否する。日本語を含む任意Unicode文字列を扱うが、言語と発話内容の正しさはデータ提供者が確認する。

学習入力は固定指示 `Transcribe the speech verbatim.` と音声潜在表現、assistant区切り、教師文、EOS。教師文とEOSだけをloss対象とし、指示・音声位置・区切り・paddingを除外する。因果shiftにより最初の回答tokenを教師文の前から予測する。asset ID、話者、言語、出所、任意のmanifest promptはモデルへ渡さない。生成時には教師文も渡さない。

## 評価と成果物

正しい発話、同じ長さの無音、別のvalidation発話を使った参照文のtoken NLLを測る。別音声は長さも変わるため、差だけで音声内容の理解を判断しない。教師なしのgreedy生成からCER、完全一致率を計算し、英語のみ空白区切りのWERを併記する。日本語に空白区切りのWERを適用しない。

CERはUTF-8 bytesではなくUnicode文字数を分母にし、NFC正規化のみを行う。空白・句読点・大文字小文字を保持する。コーパス全体の編集数を参照文字数または単語数で割り、挿入が多ければ1を超える。空の生成は全削除として採点する。不正なUTF-8は置換文字のまま採点し、正解文で補修しない。EOSで止まった件数と生成上限に達した件数も残す。

`runs/.../` にcheckpoint、実更新のJSONL、参照文と実生成を含むreport、TensorBoardのloss・CER/WER・音声/正解/生成の組を保存する。checkpointにはencoder、LLM、projector、optimizer、scheduler、scaler、乱数、sampler順とcursorを保存する。凍結重みのhash不変性、projectorの変化、有限で非ゼロの勾配を検査する。CUDAのAMP skipは更新数に含めない。

スモークの `status: passed` はこれらの処理が完走した意味で、正しく書き起こせた意味ではない。通常のCIでは外部合成器なしでloss mask、因果性、更新、指標、保存・再開を数値fixtureでテストし、eSpeakのある環境では実合成の追加テストも実行する。数値fixtureは音声認識品質の証拠に使わない。

発話音声、教師文を含む完全なmanifest、生成全文、合成器、モデル重みとログはGitへ含めない。公開するのは実装、レシピ、テスト、集計と実測報告。
