# キャプション付き音声学習スモークの実測結果

実施日 2026年11月1日 JST。

音声の自己教師あり学習で保存したエンコーダーと小型Qwen3を凍結し、テキストキャプションの応答lossでprojectorだけを学習する経路が通った。GPU・CPUとも、別プロセスで保存状態を復元して次の更新を完了した。

実装はcommit `d79af3f7f542d769cc0997602649b6e7a41f37cc`。実測はcommit前に行い、実行時の全ソースと依存lockのSHA-256がこのcommitと一致することを照合した。

## 条件と結果

| 項目 | 実測条件 |
| --- | --- |
| 入力 | native 48 kHzステレオ、約1.5〜2秒、自作16作品 |
| split | train 12作品、validation 4作品 |
| キャプション | 合成信号の基音と左右の違いを説明する日本語。生成パラメーターから作成 |
| 音声エンコーダー | 前回のSSL checkpoint。CNNと幅128・2層の双方向Transformerを凍結 |
| LM | ランダム初期化した幅64・2層の小型Qwen3を凍結。UTF-8 byte tokenizer |
| 学習対象 | 128→64の線形projector、8,256パラメーター |
| 全体 | 765,958パラメーター、最大361位置 |
| CUDA | RTX 3070、PyTorch 2.14.1+cu126、Transformers 4.57.6 |
| GPU計算 | 学習重みFP32、FP16 autocastとGradScaler。encoderはFlash、LMはmath SDPA |

CUDAではSSLのstep 21の重みから、caption学習を20回更新し、別プロセスでstep 21へ再開した。CPU専用wheelではSSLのstep 11の重みから、10回更新した後に別プロセスでstep 11へ再開した。どちらもAMP skipは0回だった。両者は元のSSL重みと更新回数が異なるため、品質を直接比較する条件ではない。

CUDAの再開セッションは約1.83秒、PyTorch peak allocatedは42.16 MiBだった。音声特徴の再計算、1更新、最終評価と生成を含み、依存導入、CLI起動、checkpoint・データの読み込みは含まない。累計21更新の総時間でも、GPU全体のメモリ使用量でもない。

## 確認できたこと

- CaptionとEOSだけをlossに含め、指示・音声prefix・区切り・paddingを除外した。next-tokenのshiftを検査し、最初の回答予測への教師文の漏洩がないことを確認した。
- 凍結LLMを通じてprojectorに有限で非ゼロの勾配が届き、projectorの重みが変わった。encoderとLLMにはパラメーター勾配がなかった。
- 全更新の履歴とAdamWのstep数を照合した。optimizerの状態はprojectorのweightとbiasの2個だけだった。
- Caption checkpointのencoderの全重みが、読み込み元SSL checkpointと厳密に一致した。LLMも開始時のhashと一致した。
- CPUで中断なしの4更新と、2更新で中断・再開した4更新の重み・AdamW・提示履歴・最終評価が厳密に一致した。
- validation 4作品の音声、参照文、実生成文を同じIDとstepでTensorBoardに記録し、保存済みの出力と一致することを確認した。音声は48 kHz・2chを維持した。
- GPU環境で全30テスト成功。CPU専用環境で26テスト成功、CUDA専用4件skip。Ruff lint・formatと依存lockの検査も成功した。
- GitHub Actionsでも[LinuxとWindowsのCPU検証](https://github.com/uthree/creative-agent/actions/runs/41279681934)が成功した。

## キャプション品質の解釈

| CUDAのvalidation条件 | Caption応答tokenの平均NLL |
| --- | --- |
| 学習前の正しい音声 | 5.61221 |
| 21更新後の正しい音声 | 5.35697 |
| 21更新後の同長の無音 | 5.35695 |
| 21更新後の別音声 | 5.35667 |

NLLは下がったが、無音や別音声でも悪化しなかった。音の内容とキャプションの対応を学べた証拠にはならない。別音声は元の長さを保持する対照なので、位置数も異なる。

教師を入力しない生成経路も実行したが、GPU・CPUとも4作品すべてで不正なUTF-8 byte列が生成され、意味のあるキャプションにはなっていない。decode時の置換文字も含めて実出力を記録し、教師文に差し替えていない。小型LMは事前学習を行っていないため、この結果は文章生成能力の合格を意味しない。

今回の到達点は、SSL音声エンコーダーからキャプション学習までの接続、回答loss、凍結、実更新、生成、保存・再開の確認である。次の品質検証では、表現を読み出すprobeと事前学習済みLLMへの接続を分けて評価する。

設定、数値、hash、照合結果は[集計JSON](../experiments/results/audio-caption-smoke-20261101.json)、再実行は[手順書](audio-caption-smoke.ja.md)を参照。音声、教師caption、全生成文、重み、完全なmanifestとTensorBoardイベントは `runs/` にのみ保存し、Gitへ含めない。
