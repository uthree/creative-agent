# 音声自己教師あり学習スモークの実測結果

実施日 2026年11月1日 JST。

48 kHzステレオ音声の生成・前処理から、マスク付きスペクトル復元、逆伝播、重み更新、保存、別プロセスでの再開まで動作した。音声ドメインの第1段階を確認する最小の学習経路として合格とする。

実装はcommit `8dc2fdde267b7bcb4d0e7db7773d3427f6390149`。実測はcommit前の作業treeで行い、記録した全ソースと依存lockのSHA-256が、このcommitの内容と一致することを照合した。

## 条件

| 項目 | 内容 |
| --- | --- |
| 音声 | 自作16作品、native 48 kHz、左右が異なるステレオ、約1.5〜2秒 |
| 分割 | train 12作品、validation 4作品。元作品を共有しない |
| モデル | 浅いCNN、幅128、全体双方向Transformer 2層・4 heads |
| パラメーター | 623,814 |
| 位置 | 10 ms間隔、今回の最大198位置 |
| 教師 | 隠した完全パッチの左右別RFFT実部・虚部・対数振幅 |
| GPU | RTX 3070 8 GB、PyTorch 2.14.1+cu126、CUDA runtime 12.6 |
| 計算 | 学習重みFP32、FP16 autocast＋GradScaler、Flash SDPA |

## 実更新と再開

| 実行 | 最初の実行 | 別プロセスでの再開 | AMP skip |
| --- | --- | --- | --- |
| CUDA | 20回の実更新 | step 20を復元しstep 21を完了 | 0 |
| CPU専用wheel | 10回の実更新 | step 10を復元しstep 11を完了 | 0 |

全更新の履歴とAdamWのstep数を照合し、CNN・Transformer・復元headの有限で非ゼロの勾配と重み変化を確認した。checkpointにはoptimizer、scheduler、GradScaler、乱数、提示順とcursorも残る。

CUDA再開セッションの所要時間は約1.26秒、peak allocatedは36.64 MiB、peak reservedは60 MiBだった。これは再開後の1更新と初期化・検証等の時間であり、累計21更新の総時間ではない。依存導入、CLI起動、データ準備は計時に含めない。メモリ値はPyTorchの管理分であり、GPU全体の使用量ではない。

## 検証

- CUDA環境で全18テスト成功。Flashとmathの出力・勾配、可変長入力、実更新、再開を確認した。
- CPU専用環境で15テスト成功。CUDA専用の3件をskipした。
- CPUで、中断なしの6更新と、3更新で中断・再開した6更新の重み・AdamW・提示履歴が厳密に一致した。
- 故意にAMP overflowを発生させ、破損した重みを保存せず、skipを実更新として数えないことを確認した。
- 隠した原音だけを変更した対照、左右と位相を変える対照、padding、末尾の不完全パッチ、未来側の入力への依存を確認した。
- TensorBoardの全実更新stepと履歴を照合し、検証音声の48 kHz・2chをデコードして確認した。
- Ruff lint・formatと依存lockの検査に成功。GitHub Actionsでも[LinuxとWindowsのCPU検証](https://github.com/uthree/creative-agent/actions/runs/41278840157)が成功した。

## 学習品質の解釈

固定したvalidation 4作品とマスクに対するCUDAモデルのlossは、初期値0.70546から21更新後0.01710へ下がった。一方、同じ教師へすべてゼロを予測する事後の基準値は0.00330だった。現在のモデルはこの単純な基準値にも達していないため、loss低下を有用な表現獲得の証拠にはしない。

今回は単純な合成音と短い非重複窓で学習処理を確認した。次の実験では、予測対象の分解能・重み付け、ゼロ等の基準値、局所probe、正常例、実録音を使い、何を特徴に保持できるかを評価する。LLM接続、講評、異常診断の精度はこの実測の対象外である。

設定、実測、ソースhash、checkpointのhash、照合結果は[公開用集計JSON](../experiments/results/audio-ssl-smoke-20261101.json)に残した。音声、完全なmanifest、重み、TensorBoardイベント、全更新履歴はGit対象外の `runs/` に保存している。

再実行は[手順書](audio-ssl-smoke.ja.md)を参照。
