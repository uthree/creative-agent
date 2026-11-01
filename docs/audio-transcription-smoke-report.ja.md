# LLMによる発話書き起こしスモークの実測結果

実施日 2026年11月1日 UTC。

発話音声の潜在表現をprojector経由でLLMへ渡し、書き起こし文の応答CE、実更新、教師なし生成、保存・別プロセスからの再開まで完走した。接続と学習処理のスモークは合格。現在の重みは書き起こし精度が未達であり、利用可能なASRモデルとは扱わない。

実装はcommit `5b344e9770dfe80575e014113036120ce755d1c5`。学習はcommit前に実施し、実行時の全ソースと依存lockのSHA-256が、このcommitの各ファイルと一致することを確認した。保存した重みからloss・実生成・CER/WERを再計算し、reportと厳密に一致した。

## 構成とデータ

| 項目 | 条件 |
| --- | --- |
| 発話 | eSpeak NG 1.52.0、en-us、150 words/min、オリジナルの英語8文 |
| 音声長 | 約1.05〜1.50秒、train 6文、validation 2文 |
| 前処理 | 元は22,050 Hz mono。線形補間で48 kHzへupsampleしL/Rへ複製 |
| split | 音声hash・ID・NFC正規化後の教師文が重複しない。話者は同じ1 voice |
| encoder | 前回の合成トーンによるSSL checkpoint、幅128・2層、凍結 |
| 接続 | 128→64の線形projector、8,256パラメーターのみ更新 |
| LLM | ランダム初期化の小型Qwen3、幅64・2層、凍結 |
| tokenizer | UTF-8 byte方式のfixture。本番の事前学習済みtokenizerではない |
| loss | 書き起こし文とEOSのnext-token CE。指示・音声位置・paddingは除外 |
| 生成 | 固定指示と音声潜在表現からgreedy生成、上限32 tokens |

元の発話はローカル合成で、外部録音は使っていない。自然発話、native 48 kHz収録、ステレオ情報、未知話者、日本語発話の認識は、この実験では検証していない。日本語の教師文の扱いは数値テストで確認しており、日本語ASRの品質評価とは区別する。

## 更新と再開

| 実行 | 初期encoder | 最初の実更新 | 別プロセスからの再開 | AMP skip |
| --- | --- | --- | --- | --- |
| RTX 3070、FP16、encoder Flash | SSL step 21 | 20回 | step 20を復元し21更新目を完了 | 0 |
| CPU専用wheel、FP32、math | SSL step 11 | 10回 | step 10を復元し11更新目を完了 | 0 |

各更新でprojectorの勾配が有限かつ非ゼロであること、encoderとLLMにパラメーター勾配がないことを検査した。checkpoint内のencoderは読み込み元SSLと全tensorが厳密に一致し、LLMのhashも初期値と一致した。projectorのhashは変化した。AdamWの両パラメーターのstep数、更新履歴、JSONL、TensorBoardのstep数にも欠落・重複がなかった。

CUDA再開セッションは約1.72秒、PyTorch peak allocatedは26.95 MiB、reservedは48 MiB。model初期化、特徴cache、1更新、検証・生成・ログを含む。依存導入、CLI起動、checkpointとデータの読み込みは計時対象外で、累計21更新の時間でもGPU全体の使用メモリでもない。

## 書き起こし品質

| 指標 | CUDA 21更新後 | CPU 11更新後 |
| --- | --- | --- |
| 正しい音声での参照token NLL、初期→最終 | 5.59537→5.40004 | 5.60773→5.40684 |
| 無音での参照token NLL、最終 | 5.40003 | 5.40681 |
| 別音声での参照token NLL、最終 | 5.39884 | 5.40551 |
| CER | 295.24%（62編集/21文字） | 295.24%（62編集/21文字） |
| WER | 100%（4編集/4単語） | 100%（4編集/4単語） |
| 書き起こし完全一致 | 0/2件 | 0/2件 |
| 生成上限へ到達 | 2/2件 | 2/2件 |
| EOSで終了 | 0/2件 | 0/2件 |

CERはUnicode NFC正規化後の文字数で計算し、空白・句読点を保持する。挿入が多いので100%を超える。WERは英語に対する空白区切り。生成の不正UTF-8は置換文字のまま採点した。値をよく見せるために正解文を補完したり、上限で止まった生成を除外したりしない。

lossは下がったが、無音や別音声への置き換えで悪化しておらず、音声内容を利用した認識の獲得を示していない。ランダムな小型LLMを凍結したこのスモークから実用精度を期待しない。次の品質検証には、事前学習済みLLMと対応tokenizerの接続、利用権限と教師を確認した発話データ、話者・文面を分離した十分な検証セットが必要になる。

## 検証と保存

- CUDA環境で全59テスト成功。CPU専用環境で52テスト成功、CUDA専用7件skip。eSpeakによる実発話生成の追加テストを含む。
- 外部合成器がない環境でも数値fixtureのテストは実行可能。通常のCPU CIでは51成功、CUDAとeSpeakを必要とする8件skip。
- CPUの連続4更新と、2更新から再開した4更新で、重み・AdamW・提示履歴・最終評価が厳密に一致した。
- Unicodeの教師、応答だけのloss mask、最初の予測への教師漏洩がないこと、split重複拒否、空の生成、CER/WER、生成上限を検査した。
- TensorBoardのvalidation全2件について、48 kHz・2chの入力音声と正解文・実生成文を、checkpointから再計算した内容に照合した。
- Ruff lint・format、依存lock検査、GitHub Actionsの[LinuxとWindowsのCPU検証](https://github.com/uthree/creative-agent/actions/runs/41281520031)が成功した。

音声、完全なmanifest、教師文と生成全文、checkpoint、TensorBoardイベントはGit除外先へ保存した。公開用の[集計JSON](../experiments/results/audio-transcription-smoke-20261101.json)には条件、件数、指標、hashと監査結果のみを残す。[手順書](audio-transcription-smoke.ja.md)から再実行できる。
