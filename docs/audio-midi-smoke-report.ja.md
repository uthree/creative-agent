# 音源からMIDIへの採譜スモークの実測結果

実施日 2026年11月1日 JST。

音源の潜在表現をLLMへ接続し、音高・開始時刻・終了時刻・velocityの音符列を教師にした実更新、保存・別プロセスからの再開、教師なしの生成と単体推論まで動作した。MIDIの読み書きと往復一致も検証した。学習経路のスモークは合格。現在の自由生成は4件すべて形式エラーで、採譜品質は未達。

実装はcommit `e501ee0a52df04ca20a147dfcbc8abb631eb11d4`。commit前に実行し、保存された全ソースと依存lockのSHA-256が、このcommitの内容と一致することを確認した。checkpointを読み直したloss・生成文・音符指標も、runのreportと厳密に一致した。

## 合成データとモデル

| 項目 | 条件 |
| --- | --- |
| 音源 | 自作の48 kHzステレオ倍音合成、約1.600〜1.615秒 |
| 学習 | 12音源。旋律・3音和音・同音連打・無音を各3件、計21音符 |
| 検証 | 4音源。同じ4種類を各1件、計7音符 |
| 同期 | 合成スケジュールから教師とMIDIを作成。開始/終了はkeyと信号の境界 |
| split | 元作品ID・音声hash・空でない音高/開始/終了列を共有しない |
| encoder | 前回の合成トーンSSLから読み込み、幅128・2層、凍結 |
| LLM | ランダム初期化の小型Qwen3、幅64・2層、凍結 |
| 接続 | 線形projectorだけを更新、8,256パラメーター |
| 教師 | N1音符テキストとEOSの応答CE、UTF-8 byte tokenizer |
| 生成 | 教師を渡さずgreedy生成、上限64 tokens。最大教師49 bytes |
| 文脈 | 最大実測301位置、設定上限512位置 |

無音の正解 `N1` は両splitに存在する。音源ファイルは長さが異なるため同一hashにはならないが、無音を独立した音楽作品の品質評価と解釈しない。実楽器、音源ライブラリ、外部MIDI、pedal、残響、録音の時刻ずれは含まない。

## 更新と再開

| 実行 | 初期encoder | 初回実更新 | 別プロセスからの再開 | AMP skip |
| --- | --- | --- | --- | --- |
| RTX 3070、FP16、encoder Flash | SSL step 21 | 20回 | step 20→21 | 0 |
| CPU専用wheel、FP32、math | SSL step 11 | 10回 | step 10→11 | 0 |

各更新で有限・非ゼロのprojector勾配を確認した。encoderはSSL親checkpointと全tensorが一致し、LLMのhashも初期値と一致する。projectorだけが変化し、AdamW両パラメーターのstep数、学習JSONLとTensorBoardの更新数にも欠落・重複がなかった。

CUDA再開セッションは約2.44秒、PyTorch peak allocatedは36.81 MiB、reservedは50 MiB。model初期化、特徴cache、1更新、評価・生成・MIDI検査とログを含む。依存導入、CLI起動、checkpointと音声の読み込みを含まず、21更新の合計時間でもGPU全体の使用メモリでもない。

## 採譜品質とMIDI出力

| 指標 | CUDA 21更新後 | CPU 11更新後 |
| --- | --- | --- |
| 正しい音源の参照token NLL、初期→最終 | 5.59838→5.37065 | 5.59945→5.37678 |
| 無音の参照token NLL、最終 | 5.37065 | 5.37679 |
| 別音源の参照token NLL、最終 | 5.37070 | 5.37688 |
| 有効な音符形式 | 0/4件 | 0/4件 |
| 音符列の完全一致 | 0/4件 | 0/4件 |
| pitch/onset F1 | 0 | 0 |
| pitch/onset/offset F1 | 0 | 0 |
| 未検出の正解音符 | 7/7音 | 7/7音 |
| 無音の正解 | 0/1件、形式エラー1件 | 0/1件、形式エラー1件 |
| 生成上限へ到達 | 4/4件 | 4/4件 |
| モデル生成から保存できたMIDI | 0件 | 0件 |

不正な生成を無音MIDIに変換しない。形式エラーは別に数え、有音の正解音符は未検出として音符F1へ反映した。無音への誤音符数は0だが、出力自体が不正なので正常な無音判定に成功したとは扱わない。

各runの最終stepで4件の正解MIDIを出力し、Midoで読み直した音高・onset・offset・velocityが教師と一致した。有効な生成からMIDIを作る経路は、和音・連打・無音を含む既知の正解列を使った単体テストで確認している。この検証はモデルが正しく生成できた証拠には数えない。

実checkpointと和音音源だけを渡す単体CLIも実行した。教師もmanifestも指定せず、runの同じ音源のvalidation生成と一致した。不正生成のreportを保存し、MIDIを作らず終了code 2を返した。

lossの低下と微小な音源対照の差だけから採譜能力の獲得を主張しない。事前学習済みLLM/tokenizer、十分な学習と音符数・音色の多様化、独立した楽曲・実楽器での評価は後続となる。長い曲の分割と境界音符、複数楽器、pedal、楽器ごとのMIDI track、拍や調の推定も今回の対象外。

## 検証と保存

- CUDA環境で全88テスト成功。CPU専用環境で80成功、CUDA専用8件skip。既存SSL・caption・診断・発話書き起こしの回帰も含む。
- 音符範囲、重複・同音の重なり、同時和音、同音連打の同時刻note off/on、velocity、無音と音の支持区間を検査した。
- MIDIのテンポ変更とvelocity 0のnote offを読み取れること、対応外のpedal・pitch bend・program等を拒否することを検査した。
- 最大一対一マッチング、開始/終了の許容差、形式エラーを無音正解に数えないこと、正解とMIDIの不一致やhash改変を検査した。
- CPUの連続4更新と、2更新から再開した4更新の重み・AdamW・提示履歴・最終集計が厳密に一致した。
- validation全4件についてTensorBoardの48 kHz・2ch入力と正解音符文・実生成文を照合した。
- Ruff lint・format・依存lock検査と、GitHub Actionsの[LinuxとWindowsのCPU検証](https://github.com/uthree/creative-agent/actions/runs/41282121514)が成功した。

完全な音源、MIDI、manifest、音符教師と生成全文、重み、TensorBoardイベントはGit対象外。[公開用集計JSON](../experiments/results/audio-midi-smoke-20261101.json)には条件・件数・指標・hash・監査結果を残した。[手順書](audio-midi-smoke.ja.md)から再実行できる。
