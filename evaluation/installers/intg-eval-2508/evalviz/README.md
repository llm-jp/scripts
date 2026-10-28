# evalviz: 統合評価の結果の横断分析 UI

統合評価 (qsub.py / run_eval.sh) の出力を実行横断で比べる Web UI。
目的は、モデルの現在のスコアを知りたい人がモデル開発者に聞きに行かずに、自分で必要な情報を見つけられるようにすることである。
トップページは主要なモデルを同じ条件で比べたリーダーボードで、載せるモデルは共有の設定ファイルで決める (下記「掲載モデル」)。
自分で比較を作りたい人は、評価の出力 (実行) を選んで、llm-jp-eval・swallow・llm-jp-judge・safety-eval のスコアを複数の観点から比べ、llm-jp-eval の推論結果をサンプル単位で分類して誤りの傾向を比べる。
構成と技術スタックは agent-eval の tbviz に揃えた (サーバは Python 標準ライブラリのみ、UI は Vite + React + TypeScript + Tailwind CSS v4 + shadcn/ui + TanStack Router / Query)。
ログインノードで常駐させ、手元から `ssh -L 8766:localhost:8766 <login node>` で `http://localhost:8766/` を開く。

## 使い方

```bash
bash evalviz/build.sh                 # UI をビルドする (初回は依存も入れる。--install で入れ直し)
bash evalviz/restart.sh               # サーバを常駐 (再) 起動する。止めるのは restart.sh stop (ログインノードごとに別プロセス)
PORT=8790 bash evalviz/run.sh         # フォアグラウンドで起動する (動作確認用)
```

出力ルートは環境変数 `EVALVIZ_ROOTS` にカンマ区切りで指定する (既定は `run.sh` の `DEFAULT_ROOTS`)。
出力ルートは qsub.py の `<output_dir>` を並べた親ディレクトリで、サーバはそこから 4 段まで潜って実行を探す。
`logs`・`checkpoints*`・`offline` 等の大きなディレクトリには潜らない (`discover.py` の `SKIP_DIRS`)。

UI だけを変えたときは `build.sh` の後にブラウザを再読込すればよく、サーバのコードを変えたときは `restart.sh` が要る。
サーバは実行の要約を `~/.cache/evalviz/runs.json` に、サンプル分析を `~/.cache/evalviz/analysis/` に保存し、結果ファイルの mtime が変わらない限り読み直さない。
起動後も 3 分ごとに出力ルートを走査する。
Python は 3.9 以上で動く (ABCI のログインノードの `/usr/bin/python3` でよい)。

サーバのテストは 2 種類ある。
単体テストは qsub.py の出力レイアウトを一時ディレクトリに組み立てて確かめる。
実データの点検は全実行の要約を作り、`--samples` を付けるとサンプル分析も作って、再計算した exact_match と ool を result.json の値と突き合わせる。

```bash
cd evalviz && python3 -m unittest discover -s tests
cd evalviz && python3 -m evalviz_server check --samples --root "$EVALVIZ_ROOTS"
```

## 画面

URL がそのまま画面の状態になる。
選択中の実行 (`?runs=`) と比べる llm-jp-eval のバージョン (`?ver=`) は全画面で共有され、ページを移っても保たれる。

| パス | 内容 |
|---|---|
| `/` | リーダーボード。掲載モデルの主要指標の棒グラフと一覧表 (列見出しで並べ替え)。チェックしたモデル (無ければ全部) を「詳しく比べる」で総合比較以下の画面に送る |
| `/runs` | 実行を選ぶ。出力ルート配下の実験ディレクトリごとに畳んで並べ、開いて比較する実行にチェックを付ける (付けた順に系列色が決まる)。結果の無い実行は既定で出さない |
| `/overview` | 総合比較。評価ごとの代表値の棒グラフ、カテゴリ別の表 (値・基準との差)、出力の長さとスコアの散布図、チェックポイントの推移、llm-jp-judge と safety-eval の全観点 |
| `/datasets` | データセット別。行がデータセット、列が実行。主指標の値か基準 (先頭の実行) との差を出し、実行間の差が大きい順・基準より下がった順などで並べる |
| `/errors` | 誤りパターン。サンプルの状態の内訳、誤りの兆候、それらのデータセット別の内訳、出力の長さ、ラベルの偏り |
| `/samples` | サンプル比較。1 つのデータセットについて、行がサンプル、列が実行。結果が割れた行・状態・兆候で絞り込む |
| `/samples/<ds>/<key>` | サンプル 1 件。入力・正解・プロンプトと、全実行の推論と最終出力を並べる |
| `/runs/<run>` | 1 つの実行の評価条件 (config.json) と全スコア |

## 掲載モデル

リーダーボードに載せるモデルと比較の標準条件は、JSON の設定ファイルに書く。
既定の場所は gcg51557 で共有の `/groups/gcg51557/evaluation2/evalviz/featured.json` で (`EVALVIZ_FEATURED` で変更)、グループのメンバーが編集できる。
サーバはファイルの更新を検知して読み直すので、再起動は要らない。
書式は [featured.example.json](./featured.example.json) のとおりで、モデルごとに表示名と評価の出力ディレクトリ (`runs`) を書く。

```json
{"name": "llm-jp-4-8b-thinking (medium)", "family": "LLM-jp", "params": "8B", "type": "thinking", "note": "reasoning_effort=medium",
 "runs": ["/groups/gcg51557/experiments/.../<qsub.py の output_dir>", "/groups/.../<safety-eval だけ別に流した output_dir>"]}
```

- `runs` は評価ごとに別の投入でもよい。評価ごとに、`runs` を先頭から見て最初に結果のある実行の値を使う。
- llm-jp-eval は `standard` の条件 (バージョンとデータセットあたりのサンプル数) に合う実行の値だけを使う。条件の違う値は比べられないので載せず、ページの末尾に設定の問題として出す。
- 他組織のモデルには `"baseline": true` を付ける。リーダーボードでは LLM-jp のモデルを青、比較対象を灰で描く。
- `runs` の実行は出力ルート (`EVALVIZ_ROOTS`) の外にあってもインデックスに載る。
- 設定の点検は `python3 -m evalviz_server check --featured <file> --root ...` で行う (実行が見つからない・標準条件の結果が無いモデルを警告する)。

## データの読み方

### 実行

実行は qsub.py の出力ディレクトリ 1 つ、つまり 1 モデル × 1 回の投入である。
`llm-jp-eval/`・`llm-jp-eval_v*/`・`swallow/`・`llm-jp-judge/`・`safety-eval/` のどれかがあるディレクトリを実行とみなす。
モデル名・評価条件は qsub.py が書く `config.json` から取る。
実行名かモデルパスに `step`・`iter` 等のステップ番号があれば、名前のステップ番号を `*` にしたものを系列としてまとめ、総合比較でチェックポイントの推移を描く。

| 評価 | 読むファイル | 代表値 |
|---|---|---|
| llm-jp-eval | `llm-jp-eval/<ver>/results/result.json` (旧レイアウトの `llm-jp-eval/`・`llm-jp-eval_<ver>/` も読む) | `lang_scores` の日本語・英語の AVG。v1.4.1 は言語別が無いので `scores.AVG` |
| swallow | `swallow/results/result.json` | 英語・日本語の主要タスクのうち実行したもの (-1 でないもの) の平均 |
| llm-jp-judge | `llm-jp-judge/evaluation/score_table.json` | 各ベンチマークの代表の観点 (quality_ja の総合評価、safety_ja の違反回答率など) |
| safety-eval | `safety-eval/evaluate_count/*/<bench>_base_evaluated_*.json` | ジャッジ平均 (attempt_avg)・真実性の率・正解率 |

llm-jp-eval はバージョンごとにデータセットと指標が違うので、画面は 1 つのバージョンだけを比べる。
選択中の実行でサンプル数・chat template・reasoning parser 等が揃っていないときは、総合比較に警告を出す。
v2.1.0 の全件評価の result.json は 700MB を超えるので、先頭 32MB から `config` と `evaluation` だけを読む (`util.read_json_keys`)。

### サンプル分析

llm-jp-eval v2 の推論結果 `llm-jp-eval/<ver>/offline/<run_name>/<ds>.eval-generated.json` には、サンプルごとの生成・推論・正解が残る。
一方 result.json の records は既定で各データセットの先頭 5 件しか残らない。
そこで llm-jp-eval の回答抽出 (`answer_parser.py` と BBH の `extract_answer`) を `ljeval.py` に移植し、生成から回答を作り直して分類する。
分類の結果は exact_match の正解率と ool (選択肢外・抽出不可の率) が result.json と一致することを `check --samples` で確かめている (2026-10-21 時点で ABCI の 46 実行 × バージョンで不一致なし)。
数式の同値 (`mathematical_equivalence`) は math_verify の近似で、単位・桁区切り・`\text{}` を落として先頭の数値で比べる。
BLEU・F1 等の連続値の指標だけのデータセットは、サンプル単位の正誤を出さない。
v1.4.1 は推論結果の形式が違うので、サンプル分析の対象外である。

サンプルの状態は次のいずれかに分類する (上から順に判定)。

| 状態 | 条件 |
|---|---|
| 出力が空 | 最終出力が空白だけ (thinking モデルで推論が出力長の上限で打ち切られた場合など) |
| 回答の抽出不可 | 回答形式 (`<answer>` タグ・`\boxed{}` 等) に沿った回答が見つからない |
| 選択肢外 | 抽出した回答が選択肢 (label_list) に無い |
| 正解 / 不正解 | 抽出した回答が正解と一致するか (NFKC 正規化後の完全一致、数式は近似) |
| 正誤なし | 連続値の指標だけのデータセット |

誤りの兆候は、出力の中身から経験則で付ける印である (重複あり)。
どれも一次分類であり、判断はサンプルのページで生成を見て行う。

| 印 | 条件 |
|---|---|
| 推論の打ち切り | 推論 (reasoning_content) はあるが最終出力が空 |
| 反復 | 出力か推論の末尾 4000 バイトの zlib 圧縮率が 0.15 未満。実データで正常な出力は 0.15 を下回らなかった |
| 回答後の続き | 回答の区切り (`</answer>`・1 行目・空行) のあとに 30 字を超える生成が続く。ベースモデルが次の few-shot 例を書き始める場合など |
| 中国語の混入 | 出力か推論の末尾 2000 字に簡体字にしかない字 (们・这・说 等。日本語でも使う字は除く) が 3 字以上 |
| 英語タスクで日本語 | 英語のデータセットの出力に仮名が 5 字以上 |
| 想定外の文字種 | キリル・アラビア・インド系・タイ・ハングル等の文字が出力に 10 字以上 |
| 特殊トークンの漏れ | `<\|...\|>` や Harmony のチャネル名 (analysis / assistantfinal) が出力に残る |

サンプルは入力と正解のハッシュで識別し、実行間で同じサンプルを並べる。
サンプル数 (`--llm-jp-eval-max-num-samples`) が違う実行どうしでも、共通するサンプルは同じ行に並ぶ。

## 構成

```
evalviz/
  evalviz_server/        サーバ (Python 標準ライブラリのみ)
    discover.py          出力ルートから実行を列挙し、モデル名・評価条件・ステップ番号を取る
    featured.py          掲載モデルの設定の読み込みと、モデルごとに値を取る実行の決定
    ljeval.py            llm-jp-eval のスコアの要約と、推論結果のサンプル分析 (回答抽出の移植・状態・兆候)
    components.py        swallow・llm-jp-judge・safety-eval のスコアの要約
    index.py             実行のインデックスとディスクキャッシュ、サンプル分析のキャッシュ、定期的な再インデックス
    httpd.py, meta.py    JSON API と SPA の配信、UI に渡す定義 (状態・兆候・カテゴリの説明)
  web/                   UI (pnpm)
  tests/                 サーバの単体テスト
```

API の型 `web/src/api/types.ts` はサーバの `ljeval.py`・`components.py`・`httpd.py` の返す形と対応しているので、片方を変えたら両方を直す。
サンプル分析の判定を変えたら `ljeval.py` の `ANALYSIS_VERSION` を、要約の形を変えたら `index.py` の `SUMMARY_VERSION` を上げてキャッシュを無効にする。

開発時は API サーバを別ポートで起動し、Vite の開発サーバから中継する。

```bash
cd evalviz && python3 -m evalviz_server --port 8791 --root "$EVALVIZ_ROOTS"   # API
cd evalviz/web && pnpm dev                                                   # http://localhost:5173/ (/api を 8791 に中継。EVALVIZ_API で変更)
pnpm typecheck                                                               # ルートツリー生成 + 型検査
```
