# 音楽・発話・MIDIの公開データセット調査

調査日：2026年11月1日。公式配布、著者のリポジトリ、データカードを確認した。規模・容量は配布元の記載であり、本調査では音源アーカイブの取得や実ファイルの音声仕様検査は行っていない。容量のGB/GiBは原資料の表記に従う。利用条件は音声・注釈・コードを区別して記録する。

## 採用案

**最初はLibriSpeech、日本語Common Voice、小型のBabySlakhで取り込み経路を整備し、音楽には出典を選別したFMAとPDMXのレンダーを追加する。** 採譜の規模拡大にはSlakh2100-redux、実演ピアノの研究にはMAESTROを使う。以下の優先順位とpilot規模は、このプロジェクトへの提案である。

| 課題 | 最初の候補 | 次の拡大先 | 判断理由 |
| --- | --- | --- | --- |
| 発話→LLM書き起こし | 英語LibriSpeech、日本語Common Voice 27.0 | 条件に適合する場合にReazonSpeech | 実発話と文のペア、多話者。英語の接続確認と日本語対応を別々に測れる |
| 波形→音符列 | BabySlakhのピアノstem | Slakh2100-reduxの楽器別stem、MAESTRO | 音声と実際のレンダー用MIDI、または演奏MIDIが対応する |
| 多様な音楽のSSL | 個別条件を選別したFMA | 条件に適合するMTG-Jamendo、実録音 | MIDIを持たない実音楽も音響表現の学習に使える |
| 音符の組合せ・音色 | PDMXの選別譜面＋許諾済み音源でのレンダー | NSynth、E-GMD、GuitarSet | 正解を追跡した合成と、異なる音色・実演を段階的に加えられる |
| 音楽caption | 学習用の短区間を聴いて記述を追加。タグ由来の文は弱教師と明記 | 権利を確認した追加の音声・文章ペア | Song Describerは評価用として確保。全曲説明を2秒のcropに流用しない |
| 正常・単一・複合問題の検出 | 確認済みの音楽・発話に既知の加工を加える | 実制作ミスの独立評価、条件を確認したNISQA | 公開音源の存在だけでは、現在の3問題の区間・左右・正常ラベルは得られない |

現在の[5,000更新の実験](audio-multitask-lfm3-5000-report.ja.md)は小さな合成集合へ適合したが、未学習例での認識・採譜は未達である。作品、話者、音色の種類を増やし、音を根拠に応答が変わるかを固定評価で確認する。

## 1. 音楽の波形・説明

| データセット・一次資料 | 規模・音声・教師 | 入手と容量 | 利用条件・本プロジェクトでの扱い |
| --- | --- | --- | --- |
| **FMA** — [公式リポジトリ](https://github.com/mdeff/fma) | 全106,574曲。ジャンル・作者などのメタデータとMP3。smallは8,000曲×30秒（計約66.7時間）、8ジャンル | small 7.2 GiB、medium 22 GiB、large 93 GiB、full 879 GiB。メタデータ342 MiB。公式から配布先へリンク | メタデータCC BY 4.0、コードMIT。**録音は作者が選んだ曲別条件**で、公式は研究向けと記載。SSLの有力候補。MP3由来の帯域・圧縮痕を高品質録音の正解と混同しない。自然文captionやMIDIは揃わない |
| **MTG-Jamendo** — [公式仕様・現行条件](https://mtg.github.io/mtg-jamendo-dataset/) | 55,000曲以上の全曲音声、195種類のジャンル・楽器・ムード等のタグ。標準320 kbps MP3、別途低ビットレートmono版 | 全音声508 GB、低品質版156 GB。mood/theme subsetは152 GB／46 GB。公式スクリプトと分割TAR | メタデータCC BY-NC-SA 4.0、録音は`audio_licenses.txt`の曲別CC。さらに**配布は非商用研究・学術用途に限定、他用途はJamendoの事前許諾**との記載あり。研究候補。タグは区間を保証しない弱教師 |
| **Song Describer Dataset（SDD）** — [公式配布](https://zenodo.org/records/10072001)、[データシート](https://github.com/mulab-mir/song-describer-dataset/blob/main/docs/datasheet.md)、[仕様](https://github.com/mulab-mir/song-describer-dataset) | 706音源・1,106件の人手caption。検証済み部分は546音源・746文。音源はおおむね2分、説明は主に英語 | 音声ZIP 3.3 GB、caption CSVと音源別ライセンスを同時配布。レート・chは取り込み時に検査 | データセットCC BY-SA 4.0、録音は曲別条件。上流Jamendoの条件も別途確認。**データシートは学習利用を控え、評価に使うことを推奨**。本計画では評価候補とし、過去の訓練使用曲を除外 |
| **MusicCaps** — [Google公式データカード](https://huggingface.co/datasets/google/MusicCaps) | 5,521件、各10秒に対する音楽家の英語説明・属性。YouTube ID、開始・終了秒を収録 | 注釈CSV中心。**音声本体の一括配布ではない**。元動画の利用可否と対象区間の取得可能性が別問題 | 注釈CC BY-SA 4.0。動画の録音・楽曲の利用権まで付与する表記ではない。caption候補だが、直ちに使える波形コーパスとして取得量へ数えない |
| **MUSDB18-HQ** — [公式配布](https://zenodo.org/records/3338373) | 150曲、train 100／test 50。44.1 kHz stereo WAV。mixtureとvocals・drums・bass・other | ZIP 22.7 GB。MIDI・書き起こし・品質ラベルは付かない | **教育目的の独自条件**。商用には各権利者の明示的許可が必要。構成音源の条件も異なる。ステムを使うミックス研究候補。単一のCCライセンスとは扱わない |
| **FSD50K** — [公式配布とライセンス内訳](https://zenodo.org/records/4060432) | 一般音を含む51,197 clip・108.3時間、200クラス。44.1 kHz・16 bit **mono WAV**、0.3〜30秒。clip単位の多ラベル | 音声は分割ZIP、ラベル・ライセンス対応表も配布 | データセット全体CC BY、音源はCC0／BY／BY-NC／Sampling+が混在。商用は配布元への連絡案内あり。楽器・環境音の補助候補。タグを時間区間や音楽の良し悪しへ変換しない |

FMA smallの約66.7時間は8,000×30秒からの計算であり、破損・欠損除外後の実使用量ではない。FMA・Jamendoの全曲を「問題なし」とはみなさず、元からある歪み、意図的な左右差、無音、効果音を点検する。

## 2. 発話の波形と書き起こし

| データセット・一次資料 | 規模・音声・教師 | 入手と容量 | 利用条件・本プロジェクトでの扱い |
| --- | --- | --- | --- |
| **LibriSpeech** — [OpenSLR公式](https://www.openslr.org/12) | 英語朗読。train-clean-100／360、train-other-500で訓練計960時間。16 kHz、書き起こし付き。公式dev/testあり | train-clean-100は6.3G。各dev/testは約0.3G。HTTPミラーとchecksumがある | **CC BY 4.0**。英語ASRの最初の候補。まずtrain-clean-100から10時間を選び、その後100時間へ拡大。朗読だけで雑音・会話・日本語の性能を代表させない |
| **Common Voice Scripted Speech 27.0 日本語** — [Mozilla公式データカード](https://mozilladatacollective.com/datasets/cmu5xbhnf00fmnq07z9p3aa70)、[版管理](https://github.com/common-voice/cv-dataset) | 対象版`cv-corpus-27.0-2026-09-11`。585,330 clip、録音726.23時間、**validated 372.6時間**、7,852話者。MP3と文・メタデータ | 2026-09-17公開、14.36 GB。Mozilla Data CollectiveのDownload経路を使う。アカウント・認証等は取得時の案内に従う | データカードは**CC0-1.0**。同時に**話者の身元特定、再ホスト・再共有を禁止**と記載。配布条件も記録する。日本語ASRの最初の候補。validated総量とtrainで使用可能な量は同じではない。実レート・chは取得時確認 |
| **ReazonSpeech v2系コーパス** — [公式ライセンス表](https://research.reazon.jp/projects/ReazonSpeech/index.html)、[取得仕様](https://research.reazon.jp/projects/ReazonSpeech/howto.html)、[配布カード](https://huggingface.co/datasets/reazon-research/reazonspeech) | 日本語35,000時間。放送由来の16 kHz FLACと書き起こし。字幕由来の整列・ラベル品質を点検する | tiny 8.5時間／600 MB、small 100時間／6 GB、medium 1,000時間／65 GB、all 2.3 TB。規約同意後HFから取得、streaming案内あり | **コーパスはCDLA-Sharing-1.0、利用目的は配布元が示す情報解析に限定**。モデル・ライブラリのApache-2.0と区別する。日本語の規模拡大候補。原音を含むTensorBoardの共有範囲も、この条件を含めて決める |
| **JSUT v1.1** — [著者公式仕様・条件](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) | 日本語女性1名、約10時間。無響室収録48 kHz、文付き | 約2.7 GB、公式リンクから取得 | 文はCC BY-SA 4.0等で詳細は同梱LICENSE。**音声は学術研究・非商用研究・個人利用、再配布不可、商用は連絡**。研究での日本語接続確認には使えるが、単一話者への適合に注意 |
| **JVS** — [著者公式仕様・条件](https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus) | 日本語100話者、約30時間、公開音声**24 kHz**。通常発話・ささやき・裏声、文と自動音素整列等 | ZIP約3.5 GB。商用相談用の48 kHz・24 bit版とは区別 | 音声はJSUTと同様の用途限定・再配布制限。タグCC BY-SA 4.0、文はJSUT由来。多話者の研究候補。parallel100は同文なので、話者分割だけで未知文の評価とは言えない |
| **GigaSpeech v1.0.0** — [公式仕様と取得手順](https://github.com/SpeechColab/GigaSpeech) | 英語10,000時間の書き起こし付き、朗読・podcast・YouTube。XS 10／S 250／M 1,000／L 2,500／XL 10,000時間。16 kHzへ正規化、Opus | 公式フォーム申請後の案内、または案内先HF。元音源URLと区間メタデータあり | **コードのApache-2.0から元録音の一律の利用条件を推定しない**。申請規約・音源条件の確認が残るため初回取得から外す。英語の会話・雑音を増やす次段階候補 |

公開されているのは、自由朗読、定型朗読、放送字幕等の異なる教師である。まず通常の発話ASRを評価し、重畳会話、話者分離、歌詞認識は別の評価集合を設ける。書き起こし経路は既存方針どおり**音声潜在表現→projector→LLMの自己回帰生成**を維持する。

## 3. 波形とMIDI・音符が対応するデータ

| データセット・一次資料 | 規模・音声・教師 | 入手と容量 | 利用条件・取り込みの注意 |
| --- | --- | --- | --- |
| **BabySlakh v2** — [公式配布](https://zenodo.org/records/4603870) | Slakhの最初の20曲。**16 kHz WAV**、stemと対応MIDI。実演録音ではなく合成音源 | 882.8 MB。全Slakhより小さく、取り込み・同期・MIDI変換の確認向け | **CC BY 4.0**（[公式API](https://zenodo.org/api/records/4603870)も確認）。最初はピアノstemを選別。20曲は汎化の主評価には少なく、重複・元曲IDを確認してから独自pilot分割 |
| **Slakh2100-redux** — [公式配布](https://zenodo.org/records/4599666)、[作者のライセンス説明](https://github.com/ethman/slakh-generation/blob/master/README.md)、[公式分割仕様](https://github.com/ethman/slakh-utils/blob/master/splits/README.md) | 元版2,100曲・145時間のmixture。**reduxで使用するのは1,710曲：train 1,289／val 270／test 151**。44.1 kHz・16 bit **mono FLAC**、187 patch・34楽器クラス、stem別MIDI | 配布TAR 104.3 GB。展開・WAV化は公式概算で約500 GB。145時間は元版mixtureの値で、reduxの実使用時間や全stem時間ではない | **CC BY 4.0**。採譜の拡大候補。`omitted`の重複曲を除外。BabySlakh・Lakh・再レンダーとの元MIDI重複も確認する。複数楽器の採譜には出力形式の拡張が必要 |
| **MAESTRO v3.0.0** — [公式仕様・配布](https://magenta.withgoogle.com/datasets/maestro) | 実演ピアノ1,276演奏・198.7時間。train 159.2／val 19.4／test 20.0時間。44.1〜48 kHz・16 bit **stereo**、音声とMIDIの同期約3 ms。velocity・ペダル付き | 全ZIP 101 GB／展開120 GB。MIDIのみ56 MB／展開81 MB。SHA-256を掲載 | **CC BY-NC-SA 4.0**。非商用研究の実演評価に有力。作品単位の公式分割を維持。現在のparserはペダルを拒否するため変換設計が先。v2とv3は分割互換ではない |
| **NSynth** — [公式仕様・配布](https://magenta.withgoogle.com/datasets/nsynth) | 305,979単音、1,006音色、4秒・16 kHz mono。音高・velocity・音色等。発音3秒＋減衰1秒という生成条件 | WAV＋JSONまたはTFRecord。楽器が重ならない公式train／valid／test | **CC BY 4.0**。単音の音高・音色の補助学習向け。楽曲・多声音の採譜用MIDIペアとは異なる。発音3秒と実際の残響・減衰の終了を混同しない |
| **MusicNet v1.0** — [公式配布](https://zenodo.org/records/5120004) | 実演クラシック330録音、100万件以上の音符注釈。WAV＋録音時間軸のCSV。譜面をDTWで対応させた教師で、公式は注釈誤り約4%と推定 | 約11.1 GB。参照MIDI別途2.6 MB | 配布全体**CC BY 4.0**（[公式API](https://zenodo.org/api/records/5120004)確認）、録音はCC／PDで個別出典をmetadataに掲載。**同期教師はCSVを使い、参照MIDIの時刻を録音へ直結しない**。演奏velocityの正解は別途確認し、欠測を架空の正解で埋めない |
| **GuitarSet v1.1.0** — [公式配布](https://zenodo.org/records/3371780)、[仕様](https://guitarset.weebly.com/) | 6演奏者・360区間、約30秒ずつ。monoマイクと弦別6ch pickup、JAMSの音符・pitch contour・弦・フレット・和音・拍 | マイク音声656.9 MB＋注釈39.1 MBから開始可能。全種で8.2 GB | **CC BY 4.0**（[公式API](https://zenodo.org/api/records/3371780)確認）。6chは左右のstereoではない。同音高が別弦で重なる場合と音程変化に対応が必要。公式の既知の時刻誤り・重複音符を確認する |
| **E-GMD v1.0.0** — [公式仕様・配布](https://magenta.withgoogle.com/datasets/e-gmd) | 1,059演奏を43 kitで収録、45,537音声・444.5時間。44.1 kHz・24 bit、MIDIと約2 ms以内、velocity付き | ZIP 90 GB／展開132 GB、MIDIのみ103 MB。音符だけが目的なら公式は元Groove MIDI Datasetを推奨 | **CC BY 4.0**。ドラム採譜の追加候補。現在のparserはdrum channelを拒否。同演奏の別kitをsplit間へ分散させない。公式分割は全43 kitが各splitにあり、未知kit評価は別設計 |

Slakhは`stems/SXX.flac`と**`MIDI/SXX.mid`**を対応させる。`all_src.mid`は元Lakhの内容で、レンダー時の楽器別処理を反映したMIDIとは一致しない場合がある。`audio_rendered`・`midi_saved`・楽器情報も読む。[公式のファイル対応仕様](https://github.com/ethman/slakh-utils#at-a-glance)

## 4. MIDI・記譜データ

| データセット・一次資料 | 内容と入手 | 利用条件・本プロジェクトでの扱い |
| --- | --- | --- |
| **PDMX v9を調査対象とする** — [公式配布](https://zenodo.org/records/15571083)、[著者リポジトリ](https://github.com/pnlong/PDMX) | 25万件以上のMusicXML、可能なものにMIDI・PDF。MIDI TAR 214.4 MB、CSV 225.4 MB、subset一覧29.3 MB。PDF等を含む全体14.4 GB。波形録音は付かない | 配布レコード**CC BY 4.0**（[公式API](https://zenodo.org/api/records/15571083)確認）、コードMIT。**31,221譜で外部表示と内部の権利メタデータが不一致**。推奨の`no_license_conflict` 222,856譜を、`all_valid`・重複除去・出典確認と組み合わせる。許諾済みbankでレンダーし採譜教師を増やす候補 |
| **Lakh MIDI v0.1** — [作者公式](https://colinraffel.com/projects/lmd/) | 176,581 MIDI。45,129件にMillion Song Datasetとの対応、aligned版あり。壊れたMIDIが数千件あるとの記載。音源previewは別扱い | 配布表記**CC BY 4.0**。ただし作者は元MIDIの採譜者ではなく個別作者帰属を十分に追えないと説明。元曲・採譜の権利来歴を保持。Slakhと元データが重なるため独立評価の水増しに使わない |
| **ASAP** — [公式仕様](https://github.com/fosfrancesco/asap-dataset)、[リリース](https://github.com/fosfrancesco/asap-dataset/releases) | MusicXML／譜面MIDI、演奏MIDI、拍・拍子・調の対応。音声は別途MAESTRO v2から準備。README表では222 score・1,067 MIDI演奏・519音声演奏。冒頭などに異なる件数表記があり、採用版のmetadataで再集計する | **CC BY-NC-SA 4.0**。記譜を扱う将来課題向け。**MAESTROと音源が重なる**。READMEの版説明と実際のタグも一致しないため、使用時はrelease／commitを固定する |

現在のN1は音高・開始ms・終了ms・velocityという**演奏の音符列**である。五線譜の拍子、調号、綴り、声部、連符、タイを持つ記譜形式とは異なる。まず録音から演奏MIDIへの精度を上げ、読みやすい五線譜への変換はMusicXML等の教師を使う別段階にする。

## 5. 問題検出に流用できる範囲

**NISQA Corpus**は14,000件以上の発話に主観品質・品質次元のラベルを持ち、通信codec、packet loss、雑音、実通話等を含む。雑音感、音色変化、不連続、音量の評価候補になる。ただしclipの品質評点は、現在の「音切れ・クリッピング・左右差」の区間ラベルと同じではない。コードMIT、モデル重みBY-NC-SA、**コーパスは元発話・雑音の条件を引き継ぐ**ので、データの条件を個別確認してから追加する。[公式説明とライセンス](https://github.com/gabrielmittag/NISQA#nisqa-corpus)

今回の3問題には、まず許諾・音質を確認した原音から加工を作る。原音をsplitしてから、正常、単一、複合の派生を生成する。無音区間へのdropoutは検出可能な音切れとせず、実際に変化した区間・chと加工順序を記録する。左右差の学習・評価には、本来のstereoとmonoを複製した2chを区別する。

原音が持つ意図的歪み、パン、スタッカート、休符やfadeは、合成問題とは別に正常側の難例として検討する。真の制作ミスを人が確認した評価集合も残し、合成加工だけの成績を実制作での検出品質としない。

## 6. 既存実装に必要な変更

1. **スモーク用と実データ用のloaderを分ける。** 現在の`AudioDataset`は32原音・最長10秒・48 kHz stereoの制限があり、全波形を`.waveforms`へ読み込む。上限を外すだけでは大規模化できない。ファイル／shardとsegment indexから必要区間を遅延読み込みし、epochごとのshuffle、worker seed、取得の再開、破損隔離、checksum検査を実装する。
2. **元の音声仕様を保存する。** 16／24／44.1 kHzも受け入れる実データ用前処理を作り、必要に応じて48 kHzへ明示的にresampleする。元のrate・実効帯域・codec・ch・変換器の版を記録する。mono複製は空間情報を作らず、16 kHz→48 kHzは8 kHz超の情報を作らない。SSLでは元帯域を超えた復元を高域学習の実績とせず、帯域maskや課題別の入力仕様を設ける。
3. **音声長と教師の範囲を一致させる。** 発話は文に対応する区間を入力し、2秒へ切って全文を正解にしない。captionも説明対象の全区間を扱うか、cropを聴いて別の教師を付ける。長い音声ではencoderの計算量、20 HzのLLM入力列、文章・音符の生成予約を含めて文脈予算を測る。
4. **MIDI変換の意味を明示する。** 現parserは単一piano program・単一ch、既定16音、pedal／CC／pitch bend／drums／同音高重複を拒否する。初回は条件を満たすpiano stemを理由付きで選別。MAESTROのkey releaseとpedalによる持続を分け、後続で楽器・drums・controller・同音高の別声部を表現する。未対応イベントを黙って捨てない。
5. **crop境界と不足ラベルを扱う。** 境界をまたぐ音符の持続、開始済みのnote／pedal stateを保持し、ローカル時刻へ変換する。velocity等がないデータはその項目のlossをmaskし、固定値を測定された正解として採点しない。無音の正例と、MIDIや整列の欠損を分ける。
6. **原音の単位で分割・重複除去する。** ASRは話者・book・録音セッション、音楽は作品・作者・録音、合成は元譜面を先に固定。Slakh／Lakh、ASAP／MAESTRO、SDD／Jamendo等の横断重複を登録する。音声hashだけでなく元MIDI・作品ID・近似重複も調べる。評価音源はSSLにも訓練投入しない。
7. **ラベルのある課題だけを学習する。** 音声のみ→SSL、文付き発話→ASR、同期音符付き→採譜という適用maskを持つ。全行に5課題の教師を捏造しない。日本語は元文と正規化後の文を保存し、CERの文字正規化を固定する。
8. **権利と実験の履歴を残す。** dataset版・原音ID・取得先・取得日・原音と注釈のhash・ライセンスURLと本文hash・帰属・利用条件・派生親・splitをmanifestに保持する。用途が異なるデータ系統を識別し、checkpointにも使用manifestの署名を記録。データ変更は新runへのwarm startとし、同一データのresumeと区別する。

生データ、全manifest、変換済み音声、MIDI、モデル、キャッシュ、TensorBoardは既存どおり`data/`、`.local/`、`runs/`のGit除外領域に置く。Gitには取得・変換コード、設定、権利の出典一覧、小さな匿名集計を残す。規約で再共有を制限する音声を含むrunでは、TensorBoardの視聴者とmedia出力設定も区別する。

## 7. 最初のpilotと拡大順序

ここでの時間数は**新たな実験の提案値**で、取得済み・実行済みを示さない。

| 段階 | データ量の開始案 | 完了条件 |
| --- | --- | --- |
| 取り込み確認 | LibriSpeech train-clean-100とCommon Voiceのtrainから各数十発話。BabySlakhから利用可能なpiano stem。PDMXはmetadataと少数譜面から | checksum、decode、rate/ch、ラベル範囲、split、MIDI往復を確認。未対応・破損の件数と理由を出す |
| 実データpilot | 英語10時間、日本語10時間、選別音楽10時間。MIDIはBabySlakhで経路確認後、Slakh-redux／PDMX renderから数百作品・1〜5時間程度を目標に選別 | 固定した未学習の話者・作品で自由生成を評価。実際の作品数・時間・音符数・ラベル欠測率を報告。raw segmentの件数だけで規模を示さない |
| caption・診断を追加 | 学習用音楽の短区間に人手captionを100〜300件程度。診断の派生は原音ごとに正常・単一・複合を作る | captionの聴き直し、診断の観測可能性・正常誤報を確認。正常と3問題の各組合せを固定評価へ含める |
| 次段階 | ASR各50〜100時間、Slakh-reduxを拡大。実演採譜の研究では条件に適合するMAESTROを別系統で追加 | 新旧の固定評価で改善と忘却を確認。PDMX renderだけの改善を実演への汎化と扱わない |

初回の音源取得をLibriSpeech train-clean-100（6.3G）＋Common Voice日本語（14.36 GB）＋BabySlakh（0.883 GB）とすると、配布アーカイブは概算22 GB。小subsetだけを学習しても、配布形式によっては全アーカイブの取得が必要である。展開、元音声、変換後cache、checkpointは別に容量を見積もる。FMA smallやSlakh全体を追加する前に空き容量と取得単位を確認する。

48 kHz stereoのfloat32を全展開すると1時間あたり約1.38 GB（48,000×2×4×3,600 bytes）になる。原音圧縮ファイルと区間indexを基本とし、全データをfloat32へ複製しない。RTX 3070ではbatchをduration／token予算で決め、短い実測から更新速度とメモリを見積もる。

混合学習は現経路を出発点にできるが、**全タスク同時学習が必要・有利という比較結果はまだない**。データ量を増やした後は、ASRまたはMIDI単独と混合の短い比較を、同じ評価集合と対象タスクの更新予算で行う。各taskの更新数だけでなく、見た原音数・時間・token数・epoch相当も記録し、大規模SSLが小さな教師付き課題を押し流さないように調整する。

評価は日本語CER／英語WER、MIDIの構文有効率・音高＋onset／offset F1、無音の誤発音、診断の正常誤報・単一／複合別F1・時間区間F1を使う。captionは楽器・発声・場面の事実誤りを聴いて確認する。無音・別音声への差し替え対照と、TensorBoardでの入力・正解・実生成・MIDI画像の対比を維持する。

## 8. Aurisから引き継ぐ知見と調査の残件

隣接リポジトリの`doc/existing-datasets-survey.ja.md`、`doc/audio-text-pilot.ja.md`、`doc/transcription-scaled.ja.md`、`doc/transcription-minute.ja.md`を参照した。音声仕様や現行条件は上記の公式資料で確認し直した。

- AurisはSDD 32音源を24／4／4に分けて使用済み。**少なくとも訓練24曲と派生を新しい独立評価から外す**。以前の検証・テスト8曲も使用履歴を保持し、既にモデル選択・観察に使った集合として扱う。
- AurisではSDDの約2分の説明に音源全体を入力し、短いcropへの説明流用を避けていた。この対応関係を引き継ぐ。今回はSDDデータシートの評価推奨も採用判断に加える。
- 合成採譜を1,024曲へ増やしたAuris実験でも未学習例のF1は低く、60秒化には長い音符列の不整合もあった。曲数の増加、長時間化、音色の多様化を個別に測る。別モデル・別集合の値を今回の改善率へ混ぜない。

取得時に残る確認は、原音ヘッダーとラベルの実検査、利用可能なtrain総時間、曲別の条件選別後の残数、現在の配布経路での認証、MIDIの未対応イベント率、crop境界の整列精度である。これらを取り込みレポートとして確定させてから、本学習のデータ量と予算を更新する。
