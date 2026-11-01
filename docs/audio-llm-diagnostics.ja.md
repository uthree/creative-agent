# LLMの出力能力と音声との接続を分ける比較

目的は、現在のLFM3構成が出力形式・長さへの適応で制約されているかを調べること。成功を任意の音源の認識能力と混同しない。

## 事前に固定する条件

- LFM3-1.7BとQwen3-1.7B（公式revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`、Apache-2.0）。[公式モデルカード](https://huggingface.co/Qwen/Qwen3-1.7B)。思考出力を無効化、greedy、比較本体ではrepetition penalty 1.0。モデルの優劣を一般化する実験ではない。
- 英語・日本語ASR、30秒・60秒の採譜を課題ごとに独立学習。SSLはLLMを使わないため比較しない。
- 元のsource-disjoint manifestを維持し、IDのSHA256順で言語／長さごとに選ぶ。ASRは訓練各8・validation各2・test各4発話。MIDIは音符ありの訓練各2・validation各1・test各2区間。静かな空列を当てるだけの成功を除く。小規模な診断集合で、最終的な汎化保証ではない。
- 共通encoderは最初の合成5k更新checkpointから取り出して凍結。100 Hzから5フレーム平均で20 Hzへ変換し、同じ特徴cacheを両LLMで使う。正音声・同じ長さの無音・同じsplitの別音声を事前に計算する。
- 両LLMのLoRAとprojectorは新規初期化。既存LFMの学習済み接続は流用しない。音声条件はprojectorとLLMのq/v rank 8 LoRAのみ更新。テキスト条件はLoRAのみ更新し、通常のtoken embeddingへ入力する。
- テキストの正解情報条件：ASRは正解文を渡して転記。MIDIはvelocity,pitch,offset,onsetのCSVからN1へ並べ替え・変換。これはoracle対照であり、認識の評価ではない。音声条件と同じ出力教師を使うが、入力の長さ・難しさは異なる。
- 各条件1,000更新、batch 1、LR 3e-4、warmup 50更新、その後一定。言語／長さを交互に提示する専用乱数の順序を固定する。訓練提示数を揃え、token数と時間は別に記録する。1 seedで初期切り分けを行う。
- ASR生成上限192、MIDI 4096、文脈8192。教師と入力を切り詰めず事前検査。teacher NLLだけでなく、訓練集合全例・validation・testの自由生成を採点する。testは最終重みだけを評価しcheckpoint選択に使わない。
- 既存モデルのpenalty 1.1 / 1.0比較は同じ固定test subsetを使い、各保存checkpoint自身のencoderで特徴を再計算する。凍結共通encoderのcacheへ置き換えない。

## 判断と追加条件

テキスト条件でもLFMが失敗しQwenが成功する場合は出力能力・適応設定が候補。両者がテキストで成功し音声ではQwenだけ成功する場合は、音声表現の利用・接続への適応が候補。両者とも音声で失敗すればencoder／接続／学習条件が残る。どれもLLM本体の欠陥を単独で証明しない。

LFMが失敗した課題ではrank 32・q/k/v/oおよびMLPへのLoRA拡張（LFMのMLPはw1/w2/w3、畳み込み層のin/out線形投影も含む）も比較する。実際に該当する線形層だけに適用し、適用一覧・学習パラメータ数を保存する。rankと適用範囲を同時変更する救済条件であり、どちらの効果かは分離しない。VRAM 8 GiBでの全重みAdamW学習は本比較に含めない。

TensorBoardへ入力音声、oracle入力、正解、実際の生成文、保存MIDIから読み直した比較画像を出す。不正なMIDIは修復せず不正として採点する。入力音声が教師の代わりに見えるような表示は避ける。

## 実行

`uv run --no-sync python experiments/run_llm_diagnostics.py --help`を参照。`prepare`でmanifestと共通encoder checkpointからcacheを作成し、`decode`で既存モデルの生成設定を比較、`train`でbackbone/task/modeを指定する。出力先は既存ディレクトリへ上書きしない。データ・oracle全文・特徴・重み・イベントは除外対象のローカルrunに保存し、匿名集計と結果文書をGitへ残す。
