# LFM3・5タスク混合5,000更新の実測報告

実施日 2026年11月1日 JST。

事前学習済みLFM3-1.7Bを使い、SSL・caption・音声診断・発話書き起こし・MIDI採譜を混ぜて、合計5,000回の実optimizer更新を完了した。各タスク1,000更新。encoder、projector、LoRA、SSL復元head、診断headが変化し、LFM本体の凍結重みは不変だった。保存したcheckpointから再計算した検証loss・実生成・指標は厳密に一致した。

少数の訓練例では発話と採譜に完全一致が出たが、未学習例では発話CER 95.83%、採譜の終了込みF1 0だった。学習経路は動作しており、小さなfixtureへの過適合と、一般化の不足が見られる。実用のASR・採譜モデルを得たという結果ではない。

## 条件と更新

| 項目 | 条件 |
| --- | --- |
| 実装commit | `88b077ec5593a68c7c6fb719f3b7b7af909ff9ef` |
| LLM | LiquidAI/LFM3-1.7B、snapshot `ac2586a34f1a1c03329dbdcdefabd8b6478ddaa7` |
| LLMの扱い | 本体BF16凍結、q_proj/v_projへrank 8・alpha 16のLoRA |
| 音声encoder | 前回SSL step 21から初期化、幅128・2層、100 Hz |
| 音声とLLMの接続 | 有効特徴を5位置ずつ平均した約20 Hz列→線形projector |
| 学習パラメーター | 1,474,508、FP32。CUDA計算はBF16 autocast |
| optimizer | AdamW、learning rate 0.0003、100更新のwarmup |
| task選択 | SSL→caption→diagnosis→transcription→MIDIを交互に各1更新 |
| batch | 同一タスクの2例。タスクをまたいだ同時batchではない |
| 記録 | 250更新ごとにcheckpoint、500更新ごとに評価と比較メディア |
| 文脈と生成 | 全体1,024位置、生成160 tokens、公式tokenizer/chat template |

音源と教師は[手順書](audio-multitask-lfm3.ja.md)の合成器による。SSLとcaptionは同じtrain 12原音、診断はその108 view。発話は英語train 12文、MIDIはtrain 12音源。validationはそれぞれ4原音・36診断view・4発話・4 MIDI音源。SSL/caption/診断のviewを独立作品として水増しせず、全タスクを横断してtrain/validation間の同一音声hashを拒否した。

最初の5更新で全タスクを1回ずつ学習・評価・保存し、別プロセスで復元して残り4,995更新を行った。最終AdamW stepはencoder 5,000、projector/LoRA 4,000、診断headとSSL復元headは各1,000だった。mask tokenはテキスト側にもゼロ値のmask枝として含まれるため、AdamW stepは5,000であり、復元headとは区別して監査した。

再開セッションの計測時間は839.43秒（約14分）、最初の5更新と初期評価等は84.41秒。これらを合わせて約15.4分。依存導入、CLI起動、データとsnapshotの事前検査等は計時に含めない。RTX 3070でPyTorch peak allocatedは約3.35 GiB、reservedは約3.40 GiB。GPU全体の占有メモリではない。

## 検証結果

| 指標 | 初期 | 5,000更新後 |
| --- | --- | --- |
| SSL masked復元loss | 0.01724 | 0.00504 |
| captionの参照token NLL | 3.97411 | 0.69286 |
| captionの実生成完全一致 | 未集計 | 0/4件 |
| 診断headの種類macro F1（36 view） | 0.61538 | 0.56651 |
| 診断headの時刻・ch込みevent F1 | 0 | 0 |
| 正常への診断head誤警報 | 8/8件 | 1/8件 |
| 診断LLMの種類macro F1（代表3 view） | — | 0.16667 |
| 診断LLMの時刻・ch込みevent F1 | — | 0 |
| 発話CER | 1,387.50% | 95.83%（46編集/48文字） |
| 発話WER | 1,587.50% | 150%（12編集/8単語） |
| 発話の完全一致 | 0/4件 | 0/4件 |
| MIDI形式の有効率 | 0/4件 | 4/4件 |
| MIDI pitch/onset F1 | 0 | 0.18182（TP 1、FP 3、FN 6） |
| MIDI pitch/onset/offset F1 | 0 | 0（TP 0、FP 4、FN 7） |
| MIDI音符列完全一致 | 0/4件 | 1/4件、無音だけ一致 |

初期診断headは全種類を出す振る舞いで、種類F1だけなら最終値より高い。最終値を単純なF1向上とは扱わない。正常への誤警報は減ったが、区間検出は未達。最終LLM診断は3件ともJSONとして有効だったが、正常例にも誤警報を出した。LLMとheadの分母も異なる。正常の検証時間は約13.85秒しかなく、JSONの時間当たり誤警報値は短い観測からの換算値である。

初期発話の巨大な誤り率は、未調整の音声接続から長い無関係な応答が出た結果。文字数や単語数で割った編集距離は100%を超える。最終値は大幅に下がったが、正しく文字起こしできた状態ではない。CERはNFC正規化のみで空白・句読点を保持する。

最終の参照NLLは、captionが正しい音源0.69286・無音0.66722・別音源0.86050、発話が6.53967・6.61224・5.74826、MIDIが2.14087・2.40319・2.35008。MIDIでは音源を置き換えると悪化する兆候はあるが、別音源は長さも変わり得る。captionでは無音の方がよく、発話では別音源の方がよい。音声依存や理解の獲得を一律に主張できる結果ではない。

## 訓練例への適合との比較

各タスクの訓練lossについて、最初と最後の100回のタスク更新を平均した。diagnosisは応答CEと補助BCEの和で、他タスクとlossの数値を直接比較しない。

| タスク | 最初の100更新 | 最後の100更新 |
| --- | --- | --- |
| SSL | 0.01329 | 0.00429 |
| caption | 0.59626 | 0.00102 |
| diagnosis | 1.52909 | 0.63411 |
| transcription | 1.79648 | 0.02404 |
| MIDI | 0.96632 | 0.04925 |

終了後、別途訓練例の先頭または種別代表を少数だけ生成した。caption 4件は4件完全一致。発話4件はCER/WERとも0、4件完全一致。MIDIは無音・旋律・和音・連打各1件で4件完全一致、終了込みF1 1.0だった。診断の訓練代表3件はJSONが有効でも、イベント列の完全一致は0/3件だった。訓練全例の自由生成を採点した結果ではない。

この差は、訓練例に適合できても未学習例へ十分に一般化できていないことを示す。次の実験では発話・音符・音色の多様化とデータ数、区間教師の学習設計を見直す必要がある。単独タスクとの比較実験はしていないため、全タスクの混合が必要か、どの混合比がよいかは今回だけでは判断できない。

## TensorBoardと監査

TensorBoardは実行ホストのLAN IPv4でport 6006を待ち受け、HTTP応答を確認した。実際のURLとPIDはrun内 `tensorboard-server.json` に保存する。

`examples/<task>/<sample_id>/` で入力音声・正解・実LLM応答を比較できる。MIDIは書き出したMIDIを読み直し、正解／生成／重ね合わせの3段ピアノロールを表示する。有効な非空MIDIには再合成音も付けた。追加の訓練例比較は `training_fit_probe/` に分け、validationと混ぜない。SSLはLLM応答の代わりに入力音声と左右別スペクトル比較を表示する。

- 全5,000 step、タスク別1,000 step、optimizer・scheduler・JSONL・TensorBoardの更新数を照合した。
- すべての更新に訓練splitだけが提示されたことを確認した。
- LFM凍結重みとsnapshot、実行ソースとlockのhashを照合した。
- 保存した重みから全validationを再評価し、loss・指標・実生成が厳密に一致した。
- validationのLLM生成15例について、TensorBoardの48 kHz・2ch音声、正解文、実応答を照合した。
- 採譜全4例で、MIDIの音高・開始・終了・velocity、PNG画像、TensorBoard画像の一致を確認した。不正出力をMIDIへ補修していない。
- CUDA環境で98テスト成功、CPU専用環境で90成功・CUDA専用8件skip。CPUの連続実行と保存再開の厳密一致も含む。
- lint・format・lock検査と、[Linux/Windows CI](https://github.com/uthree/creative-agent/actions/runs/41283305294)が成功した。

音源、MIDI、教師と生成全文、画像、重み・adapter・checkpoint、TensorBoardログはGit対象外。公開用には[集計JSON](../experiments/results/audio-multitask-lfm3-5000-20261101.json)と本報告を残す。再実行方法とログの見方は[手順書](audio-multitask-lfm3.ja.md)を参照。
