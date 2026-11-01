# 音源から音符列とMIDIを生成する学習スモーク

48 kHzステレオ音源を音声encoderへ通し、潜在表現をprojector経由でLLMへ渡す。LLMは音高・開始時刻・終了時刻・velocityを持つ音符列を生成し、形式と範囲が正しい出力だけを標準MIDIへ変換する。発話書き起こしと同じ応答CEの学習・保存・再開処理を使う。

今回の目的は学習経路の動作確認。encoderとランダム初期化の小型Qwen3は凍結し、projectorだけを更新する。tokenizerはUTF-8 byte方式のfixtureで、事前学習済みLLMのtokenizerではない。音符列が正しく生成できることと、学習処理が完走することを別に評価する。

## 実行

Python 3.12、uv、既存のSSL checkpointを使う。追加したMIDIライブラリはlock済みのMido 1.3.3。音源生成はNumPyで行い、外部音源・音源ライブラリ・MIDIデータ・モデルのダウンロードは不要。

```bash
uv sync --locked --no-group cpu --group cuda
uv run --no-sync creative-agent prepare-midi-fixtures \
  --config recipes/audio-midi-smoke.json --output data/midi-smoke-001
uv run --no-sync creative-agent smoke-audio-midi \
  --config recipes/audio-midi-smoke.json \
  --manifest data/midi-smoke-001/manifest.json \
  --encoder-checkpoint runs/audio-ssl-cuda-initial-20261101/checkpoint.pt \
  --device cuda --steps 20 --output runs/midi-cuda-001
uv run --no-sync creative-agent smoke-audio-midi \
  --config recipes/audio-midi-smoke.json \
  --device cuda --steps 21 --resume --output runs/midi-cuda-001
```

CPUでは `uv sync --locked --group dev` と `--device cpu` を使う。`--manifest` を省くとrun内にfixtureを作る。`--encoder-checkpoint` 未指定時はランダムencoderと記録し、再開時はcheckpointからencoderも復元するため指定しない。`--steps` は累計の実optimizer更新数。

## 入力と音符形式

既定の16音源は、単音の旋律、3音の同時和音、同音連打、無音を4件ずつ含む。trainは各3件の12音源、validationは各1件の4音源。作品IDと空でない音符スケジュールをsplit間で共有しない。空の音符列は両splitに存在するが、無音ファイルの長さを変え、同一音声hashの重複を避ける。

既知の音符スケジュールから左右で異なる倍音を合成し、48 kHzの浮動小数点WAVと正解MIDIを保存する。velocityは合成gainへ反映する。短いattack/releaseのfadeを持つが、音の支持区間はkey onset/offset内に収める。実楽器の残響、ペダル、録音遅延、複数楽器の分離は含まない。

教師は次の `N1` 形式。各行はMIDI pitch、音源先頭からのonset/offset（整数ms）、velocity。和音は開始時刻順、同時刻ならpitch順に並べる。各行は改行で終える。無音の正解は `N1` と改行だけ。

```text
N1
60 100 600 90
64 100 600 80
60 600 1000 70
```

音高は0〜127、velocityは1〜127、onsetは0以上、offsetはonsetより後で音源内。同じ音高の重複・重なりは拒否する。前のnote offと次のnote onが同時刻の連打は許容する。余分な説明、未完の行、順序違い、範囲外、不正な文字、上限超過を補修せず形式エラーにする。

短いbyte生成で複数音符を検証するため、長いJSONではなくこの形式を採用した。既定の上限は音源2秒、文脈512位置、教師256 bytes、生成64 tokens、音符16件。fixtureは最大3音・教師49 bytesで、この生成上限内に正解全文とEOSを表現できる。任意長の曲や高密度の採譜に対応した設定ではない。教師や音源を自動で切り詰めず、全件を更新前に検査する。

固定指示と潜在表現から音符列を予測する。lossは音符テキストとEOSだけに適用し、指示・音声位置・区切り・paddingを除外する。因果shiftで最初の回答tokenも採点する。MIDI、正解音符、case、作品ID、合成パラメーターは音声入力や推論promptに含めない。生成時は正解文も渡さない。

## 自分の音源と正解MIDIを使う場合

`--manifest` で[SSLの音声契約](audio-ssl-smoke.ja.md)を満たすmanifestを指定する。音声は48 kHz・2ch、音源の全長は整数ms。manifest直下に `task: "audio_to_midi"` と利用条件を記した `note_rights` を加え、各assetへ以下を加える。

```json
{
  "source_work_id": "元の曲や演奏のID",
  "duration_ms": 1600,
  "notes": [{"pitch": 60, "onset_ms": 100, "offset_ms": 600, "velocity": 90}],
  "midi_path": "reference.mid",
  "midi_sha256": "MIDIファイルのSHA-256"
}
```

MIDIを読み直した音符が宣言した `notes` と一致することも検査する。`creative_agent.midi.read_midi(path, duration_ms)` はテンポ変更を積算して整数msへ丸める。`dataclasses.asdict` でその結果をmanifest用dictへ変換できる。異なる演奏やrenderでも同じ作品なら `source_work_id` を共有し、splitをまたがせない。音声とMIDIの同期、音源と教師の利用権限は提供者が確認する。

このレシピで扱うのは1楽器の音符。MIDIはtype 0/1の同期形式を受け付け、複数channel、ドラム、非piano program、pedal/CC、pitch bend、SMPTE offset、未完のnote on/offを拒否する。対応外の情報を黙って消して教師にしない。出力はtype 0、channel 0、program 0、1 tick=1 msに固定する。program 0は再生時の既定音色で、音源から楽器を推定した結果ではない。[MidoのMIDI仕様説明](https://mido.readthedocs.io/en/latest/files/midi.html)

## 評価・MIDI出力・単体推論

validationで正しい音源、同じ長さの無音、別のvalidation音源を使った教師token NLLを測る。別音源では長さも変わり得る。また無音target同士では意味が変わらない場合もあるため、この対照だけで採譜能力を断定しない。

生成全文から有効形式率、音符列の完全一致率、pitchとonsetのF1、pitch/onset/offsetのF1を計算する。開始の許容差は50 ms、終了はmax(50 ms,参照音長の20%)。pitchはMIDI整数で完全一致を求め、最大二部マッチングで一対一に照合する。velocityは完全一致率に含め、音符F1からは除く。[mir_evalの評価条件](https://github.com/mir-evaluation/mir_eval/blob/main/mir_eval/transcription.py)と対応するが、全機能を実装した代替ではない。

不正な生成は形式エラーとして残し、有音の正解音符は未検出として集計する。無音への不正生成を「無音の正解」と数えない。無音での正解、余分な音符、形式エラーを別々に記録し、空集合のF1は0とする。生成上限に達した件数も残す。

ローカルrun内の `midi/step-000021/` にvalidationの正解MIDIと、有効な生成のMIDIを保存し、それぞれ読み直して音高・時刻・velocityを照合する。不正な生成からダミーの無音MIDIは作らない。TensorBoardには応答loss、形式率、音符F1、入力音声・正解音符文・実生成文を残す。

保存したcheckpointと1音源だけで、教師を使わず推論する経路も用意した。

```bash
uv run --no-sync creative-agent transcribe-audio-midi \
  --checkpoint runs/midi-cuda-001/checkpoint.pt \
  --audio data/midi-smoke-001/work-0008.wav \
  --device cuda --output runs/midi-inference-001
```

有効な音符列なら `transcription.mid` とreportを保存する。不正ならreportへ `status: invalid_generation` と実生成を保存し、MIDIを作らず終了code 2を返す。現在のランダム小型LLMで正しい採譜ができるとは限らない。

checkpointにはencoder・LLM・projector、AdamW・scheduler・scaler、乱数・sampler順・cursor・更新履歴を含む。凍結の不変性、projectorの更新、実step数、設定・ソース・lock・manifest・音声/MIDIのhashを検査する。CPUでは連続更新と保存再開後の重み・optimizer・履歴・最終評価の厳密一致をテストする。

音声、MIDI、完全なmanifest、正解と生成全文、checkpoint、TensorBoardログはGit除外先に保存する。公開するのは実装・設定・合成器コード・テストと小さな実測集計。`status: passed` は学習処理の合格で、採譜品質の合格ではない。
