# 音声の問題を検出する学習スモーク

音切れ、クリッピング、左右レベルの偏りを、時間とチャンネルを持つ多ラベル問題として学習する。正常な音声、単一の問題、複数の問題がある音声を同じタスクに含め、加工済み音声だけから検出する。原音の取得、加工と教師生成、学習、保存、再開、正常への誤警報の集計までを確認するスモークである。

## 実行

CPUでは、Python 3.12とuvを使って次を実行する。

```bash
uv sync --locked --group dev
uv run --no-sync creative-agent smoke-audio-diagnosis \
  --config recipes/audio-diagnosis-smoke.json \
  --device cpu --steps 20 --output runs/diagnosis-cpu-001
```

既存のSSLエンコーダーから開始する場合は `--encoder-checkpoint` を指定する。CUDAの例は次のとおり。

```bash
uv sync --locked --no-group cpu --group cuda
uv run --no-sync creative-agent smoke-audio-diagnosis \
  --config recipes/audio-diagnosis-smoke.json \
  --encoder-checkpoint runs/audio-ssl-cuda-initial-20261101/checkpoint.pt \
  --device cuda --steps 20 --output runs/diagnosis-cuda-001
```

CUDA環境を保持するため、以後も `uv run --no-sync` を使う。既定ではFP32の学習パラメーター、FP16 autocastとGradScaler、Flash SDPAを使う。CPUではFP32とmath SDPAを使う。`--backend math` で通常のSDPAも指定できる。

20更新後に、別プロセスで21更新目へ進められる。

```bash
uv run --no-sync creative-agent smoke-audio-diagnosis \
  --config recipes/audio-diagnosis-smoke.json \
  --device cuda --steps 21 --resume --output runs/diagnosis-cuda-001
```

再開時のencoderは診断checkpointから復元するため、`--encoder-checkpoint` は再指定しない。`--steps` はskipを除いた累計の実optimizer更新数を表す。元のSSL runは変更しない。

## 正常と複合問題のデータ

native 48 kHzステレオの自作16原音から、それぞれ9種類の音声を作る。原音は約1.5〜2秒で、全体のRMSを元の原音に揃える。左右を別々に正規化せず、1つのgainを全体へ掛けて、その値をmanifestへ記録する。

| 種類 | 1原音から作るケース |
| --- | --- |
| 正常2種類 | 元音声、滑らかなfadeを伴う意図的な休符 |
| 単一問題3種類 | 局所音切れ、局所クリッピング、片側の局所的なレベル低下 |
| 複合問題4種類 | 同じchの離れた音切れとクリッピング、同時刻の別chの音切れと偏り、同時刻の別chのクリッピングと偏り、3種類の複合 |

学習は12原音から108ケース、検証は4原音から36ケース。検証の内訳は正常8件、単一問題12件、複合問題16件となる。同じ原音の加工違いはsplitを跨がない。派生36件を独立した36録音とは数えない。

開始位置を原音ごとに変え、左右も入れ替える。複合加工は同一時間の別チャンネルと、同一チャンネルの離れた区間を扱う。任意の加工を同じ場所へ重ねられる仕様にはしていない。

操作ログをそのまま正解にせず、最終信号に音切れ、clippingの平坦な頂点、レベル低下が残っていることを確認する。元が無音の領域への操作、後段の加工で先の問題が消えた例は拒否する。これは数値的な観測可能性の検査であり、人の聴取や美的判断を保証するものではない。

正常な休符はfade付きで作り、無条件に無音区間を欠損扱いしない例にする。モデルの「問題なし」は、対象の3種類を検出しなかったという意味であり、未知の不具合がないことまでは表さない。

## 入力と教師を分ける

manifestには原音のID・hash・split、加工版のhash、演算子の版、処理区間、ch、全体gain、観測可能性を残す。正解区間は48 kHzのsample境界で、開始を含み終了を含まない範囲として保存する。チャンネル0が左、1が右。

モデルのforwardへ渡すのは加工済み波形と有効位置maskだけである。原音、正解event、操作ログ、case名、source IDを入力へ渡さない。`--manifest` で外部の同契約データを使う場合も、元作品のsplit、音声のhash、eventの範囲と既知クラスを検査する。

1位置10 msの有効特徴ごとに、左右と3クラスの二値教師を作る。重なる問題は同時に1になる。正常例は全クラス0で、別の排他的な正常クラスにはしない。繰り返す同種の問題も時間区間を別に保持できる。

## モデルと損失

既存のCNNと双方向Transformerへ線形の診断headを追加し、`[batch, positions, left/right, classes]` のlogitsを出す。各クラスは独立したsigmoidなので、複数種類を同時に陽性にできる。

```text
加工済み音声 → CNN → 全体双方向Transformer → 左右と種類のlogits
                                               ├─ 有効位置の多ラベルBCE
                                               └─ 時間とchの最大値でclipの多ラベルBCE
```

lossはclip BCEと、設定した重みを掛けた位置別BCEの和。paddingを両方のlossから除外する。閾値は既定0.5で固定し、検証結果を見て変更していない。全有効位置・全chで全クラスが閾値未満なら `no_known_problem` を返す。

CNN、Transformer、最終norm、診断headを更新し、各群へ有限で非ゼロの勾配が届くことを確認する。キャプション用の凍結接続とは異なり、診断ではencoderも学習する。元のSSL復元headとmask tokenは使わず、凍結する。今回は診断損失のみで、SSL損失との同時最適化は後続となる。

## 採点とスモークの合格条件

出力の連続した陽性位置を、種類・chごとのeventに変換する。境界は10 msの位置へ丸められ、最後は実音声の長さで止める。sample境界を48,000で割ると秒になる。

採点では種類ごとのprecision・recall・F1とmacro/micro集計、正常・単一・複合別の種類集合の完全一致、正常clipに1件以上誤警報がある割合、正常1時間あたり誤event数を出す。位置とchも含むevent採点は、同じ種類・chで時間IoU 0.5以上の候補をIoU順に一対一対応し、重複予測を重ねて加点しない。

「常に問題なし」と「常に全種類あり」の基準系を併記する。正常だけの正解率や、loss低下だけで検出品質を判定しない。正常例を多く正解しても、問題を全部見落としていればF1は0になる。

スモークの合格条件は、データと教師の契約、実更新、期待する勾配と重み変化、保存、再開、全条件の採点が通ることである。検出品質の達成とは分ける。単一の合成器と短い音声による検証なので、実録音や別実装の加工による精度は別途測る。

## 保存と確認

```text
runs/diagnosis-cuda-001/
  data/sources/       # 元音声
  data/views/         # 正常・単一・複合の入力
  data/manifest.json  # 派生、split、加工と観測可能なevent
  checkpoint.pt       # model、optimizer、scheduler、scaler、乱数と提示順
  training.jsonl      # 実更新、loss、群ごとの勾配
  predictions.json    # 検証全件の予測と正解event
  tensorboard/        # loss、F1、正常誤警報、3種類の音声とeventの例
  report.json         # 実行環境、実更新と再開、品質集計、基準系
```

checkpointは原子的な置換で保存し、設定、データ署名、ソースとlock、精度、backendの不一致を拒否する。CPUでは中断なしと中断・再開後の重み・AdamW・提示履歴が厳密に一致する。AMP overflowは更新せずskipとして数え、skipを成功した更新へ加えない。

TensorBoardでは `train/loss/*`、`train/gradient_norm/*`、`validation/clip_macro_f1`、`validation/event_macro_f1`、`validation/normal_clip_false_alarm_rate` を確認できる。正常・単一・複合の代表音声と、その実際の予測と正解eventを同じID・stepで記録する。

音声、完全なmanifest、event教師、予測、重み、ログはGit対象外に保存する。Gitにはコード、recipe、検証手順と小さな集計を残す。
