# 公開音声の単独課題比較

2026年11月1日に全6条件を実行した。[結果と再現監査](audio-public-single-task-report.ja.md)を参照。

前回の混合学習は追加5,000更新、5課題が各1,000更新だった。今回はSSL、楽器短文、正常・複合問題の診断、英日ASR、ピアノ採譜を**独立した重み・optimizerで各1,000更新**する。全条件を前回と同じ合成データcheckpointからwarm startし、公開データで混合学習した最終重みからの追加学習にはしない。

`recipes/audio-public-single-task-lfm3.json`を使い、`train-audio-multitask --task ssl`などで対象を選ぶ。対象の課題だけを学習・評価し、非対象のheadやLoRAは勾配がなければ更新しない。保存再開時は対象課題も含めて設定を照合する。原音・作品のsplit、短区間manifest、batch 2、学習率、seedと各課題の更新数を前回に揃える。ただし共有optimizerの履歴、課題ごとのwarmup進行、サンプルの提示順は混合学習と異なるため、干渉の原因を一つに特定する実験とはしない。

## 30秒・60秒の採譜

短区間のMIDI単独学習に加え、ユーザー指定の30秒・60秒を同じ採譜専用モデルで学習する。`prepare-long-midi`は既存manifestの元作品splitと出典を保持し、ピアノstemから両方の長さを半区間strideで選ぶ。train内の窓の重なりは許容するが、作品を別splitへ移動しない。原音は複製せず、遅延decodeする。

入力と交差するkey-noteを全て正解に含め、窓の左右では観測範囲へonset／offsetをclipする。この境界規約は元の押鍵時刻を完全復元する主張ではない。最大256音を超える区間は教師を途中で切らず除外し、除外数を記録する。無音の正解も少量含める。ペダルは引き続き対象外で、同じ音源bankの合成ピアノという評価範囲も維持する。

レシピ`audio-public-single-midi-long-lfm3.json`は20 Hzの音声表現、文脈8,192、生成上限4,096 token、encoderのFlash Attention、batch 1を使う。1,000更新を30秒・60秒の集合へ行い、両方を分けて採点する。全教師と生成reserveが文脈内に入るか学習前に検証し、打ち切り率を出す。batch、入力量、教師長、Attention実装も変わるため、短区間との比較は長さだけの厳密な対照とは扱わない。

```bash
uv run --no-sync creative-agent prepare-long-midi \
  --manifest data/public-audio-20261101/manifest.json \
  --output data/public-audio-20261101/manifest-long-midi-30-60.json

uv run --no-sync python experiments/run_single_task_comparison.py \
  --output runs/audio-public-single-task-20261101 \
  --manifest data/public-audio-20261101/manifest.json \
  --long-manifest data/public-audio-20261101/manifest-long-midi-30-60.json \
  --model-dir .local/models/LFM3-1.7B-ac2586a \
  --warm-checkpoint runs/audio-multitask-lfm3-5000-20261101/checkpoint.pt \
  --baseline runs/audio-public-multitask-lfm3-5000-20261101/report.json
```

短区間は開始前・1,000更新後の固定testと訓練probeを自由生成し、同長無音・別音声の正解token NLLも比較する。長区間は未学習モデルの長文生成で更新開始が遅れたため、初期自由生成をログを残して停止し、`initial_midi_generation: false`で全例のNLLのみ測る。学習後は全例を自由生成し、初期の未実行F1を0として扱わない。TensorBoardには各条件の学習後の入力、正解、LLM応答、MIDI比較画像を残す。ASRはaudio潜在表現→LLMの方針を維持する。短区間の同じtestを採点し直すため、この追加実験自体は新しい未使用testによる最終性能保証とはしない。
