# india-ai-model

MSCI India IMI をユニバースとするインド株 AI スコア（月次リバランス、rank IC で評価）。
データ取得（`namdb` → `data/raw`）は Ryosuke、mart 生成以降は本リポジトリのコードが担当する。
model 層は mart だけを読み、DB には接続しない。

## セットアップ

```sh
uv sync                      # Python 3.12+, pandas 3, PyTorch, LightGBM
brew install libomp          # macOS で LightGBM に必要
uv run pytest                # ユニットテスト（calendar / targets）
```

## フォルダ構造

```
data/
├─ raw/                         ← Ryosuke（追記のみ）
│  ├─ universe/year=YYYY/       date(暦月末), bid, weight
│  ├─ factor/table=core/year=YYYY/   date(暦月末), bid, 98 ファクター
│  ├─ sector/year=YYYY/         date(暦月末), bid, sector, industry_group, industry, sub_industry（GICS）
│  └─ return/kind={drtn,srtn,gp_trtn}/year=YYYY/   date, bid, ret（drtn/srtn は %、gp_trtn は小数）
└─ mart/                        ← uv run build-mart で再生成
   ├─ calendar.parquet          インド営業日、ym, td_index, is_rebal, rebal_index
   ├─ ret_daily.parquet         date, bid, drtn, srtn, gp_trtn（小数、営業日のみ、疑わしい drtn/srtn は NaN）
   ├─ sector_monthly.parquet    date(rebal), bid, sector, industry_group, industry, sub_industry
   ├─ universe_monthly.parquet  date(rebal), bid, weight
   ├─ panel_x.parquet           date(rebal), bid, ファクター生値（センチネル→NaN）
   ├─ panel_y/<target>.parquet  ターゲットのキャッシュ（date, bid, <name>, n_obs, n_days, end_date）
   ├─ factor_meta.csv           factor, source_table, category, is_binary, description, sentinel_frac
   └─ _qc_report.parquet        check, date, bid, value, note
src/india_ai_model/
├─ build_mart.py                raw → mart（QC 付き）
├─ common/
│  ├─ io.py                     raw/mart の読み書き、キー正規化、センチネル処理
│  ├─ calendar.py               営業日・リバランス日・月次データの結合（ym / asof）
│  ├─ targets.py                make_target / get_target（{kind}_{n}m[_s{skip}]）
│  ├─ features.py               rebal_index 単位のラグ、ローゼンバーグ変化率、日付内 rank
│  ├─ evaluate.py               日付別 rank IC、ICIR、分位スプレッド、自己相関、回転率
│  ├─ walkforward.py            拡大/ローリング窓、val、embargo、年1回再学習の分割
│  ├─ runner.py                 config.yaml → runs/<config名>/<run_id>/
│  ├─ ensemble.py               複数 run の z-score 平均（等ウェイト / 過去 IC 加重）
│  └─ impute.py                 masked AE による欠損補完（オプション）
└─ models/{ridge,lgbm,dnn}/model.py
configs/*.yaml                  実験ごとの設定
docs/                           実験ログ、ストーリー、run_summary.csv
runs/<config名>/<run_id>/          config.yaml, pred.parquet, pred_by_date/YYYYMMDD.parquet,
                                metrics.{parquet,csv}, per_date.parquet, fit_info.json, splits.txt
```

## 使い方

```sh
uv run build-mart                                  # mart を作る（約10秒）
uv run python -c "from india_ai_model.common.targets import build_panel_y; build_panel_y()"
uv run python -m india_ai_model.common.runner configs/ridge.yaml --run-id baseline
uv run python -m india_ai_model.common.runner configs/dnn.yaml   --run-id baseline
```

モデルは `models/<name>/model.py` に `Model(params, feature_names)` を持ち、
`fit(X_tr, y_tr, d_tr, X_va, y_va, d_va)` / `predict(X)` / `info()` を実装する。

## 実データに合わせた決定事項（引き継ぎ資料からの差分）

| 項目 | 決定 | 理由 |
|---|---|---|
| 営業日 | `kind=drtn` で非欠損 ≥ 50 銘柄かつ「リターン = 0 の銘柄比率 < 99%」の平日 | raw は月〜金カレンダーで、祝日は全銘柄 0 として入っている（27年で 349 日 ≈ 13日/年、NSE の祝日と一致） |
| リバランス日 | **月末営業日（`--rebal-lag 0`）** | factor / universe が暦月末時点の値で取得されているため、lag=1 だと1日分の先読みになる。lag は引数で変更可 |
| 月次データの結合 | `ym`（同月）をデフォルト、`asof` も選択可 | 資料通り |
| 未完了月 | リバランス日を付けない（2026-09 など） | ターゲットが揃わない |
| センチネル | `<= -9.99e8` を NaN に変換（factor: -1e9、return: -999999998） | DB の欠損表現 |
| リターン単位 | raw は %、mart は小数 | |
| `bid` | 末尾空白を strip | raw/return の bid に末尾空白あり（`'INDAAA1 '`） |
| リターンのエラー処理 | `\|drtn\| > 0.4` の日は DS2 トータルリターン（`gp_trtn`）が 5pp 以内で裏付ける場合のみ残し、それ以外は drtn/srtn を NaN（`--max-abs-ret`, `--confirm-kind`, `--confirm-tol`） | Barra には未調整のコーポレートアクションが稀にある。分割日を NaN にすれば前後の複利は正しくつながる。gp_trtn は 2026-05 で更新停止なので、それ以降は閾値のみ |
| 月次クロスチェック | ユニバース行の drtn_1m と gp_trtn_1m が 10% 超乖離 → QC `ret_monthly_mismatch` | 84k 行中 50 行 |
| panel_y | `mart/panel_y/<name>.parquet` に1ターゲット1ファイル | 列の増減が楽 |
| 学習開始 | 2007-01（ユニバースが 2003〜2006 年は 15〜46 銘柄しかない） | |

## 結果（OOS 2012-12〜2026-08、164ヶ月、拡大窓・年1回再学習・val 12ヶ月）

| | rank IC | ICIR | t | Q5−Q1 (月) | L/S 片道回転率 (月) | hl=1 平滑化後 IC / 回転率 |
|---|---|---|---|---|---|---|
| 初期ベースライン `ridge/baseline` | 0.069 | 0.46 | 5.9 | 1.33% | 0.47 | — |
| 単体最良 `dnn_peer_na_cpr_s10` | 0.078 | 0.58 | 7.4 | 1.60% | 0.60 | 0.074 / 0.33 |
| **最終候補 `ens_lgbm_dnn`**（lgbm_reg_peer_cpr + dnn_peer_na_cpr_s10 の等ウェイト） | **0.082** | **0.57** | 7.4 | **1.66%** | 0.60 | **0.078 / 0.32** |

採用した要素：GICS 業種内ピア相対化、リターン相関 top-10 ピア相対化、欠損フラグ（DNN のみ）、シード 10 本、lgbm + dnn 等ウェイト、EMA 半減期 1 ヶ月の平滑化。
不採用：attention、長期ターゲット学習、サンプル重み、相関ピアの亜種、masked AE 補完、srtn ターゲット DNN の追加。

- 全 run の指標：[docs/run_summary.csv](docs/run_summary.csv)
- 再現用の実験ログ（日付順、設定、判断基準）：[docs/experiment_log.md](docs/experiment_log.md)
- 改善の経緯の説明資料（他者向け）：[docs/model_story.md](docs/model_story.md)

### 主要な config

| config | 内容 |
|---|---|
| `ridge.yaml`, `lgbm_reg.yaml`, `dnn.yaml` | ベースライン（145 特徴量） |
| `*_peer.yaml` | ＋ GICS industry_group 内ピア相対（`peer_level`, `peer_factors`, `group_factors`） |
| `*_cpr.yaml` | ＋ リターン相関ピア（`corr_peer_factors`, `corr_peer_k`, `corr_peer_window`, `corr_peer_kind`, `corr_peer_method`, `corr_peer_weighting`） |
| `*_na.yaml` | ＋ 欠損フラグ（`missing_flags`, `missing_count`） |
| `dnn_peer_na_cpr_s10.yaml` | 単体最良（MLP、シード 10 本） |
| `dnn_attn*.yaml` | cross-sectional attention（`arch: attn`） |
| `*_w_*.yaml` | サンプル重み（`sample_weight: {date, name, time_halflife_months}`） |
| `*_ae_*.yaml` | masked AE 補完（`impute:`） |

アンサンブルは `python -m india_ai_model.common.ensemble --name <名前> --runs runs/a/exp runs/b/exp [--ic-weighted]`。

## raw に追加予定のファイル（Ryosuke 作成）

いずれも `date` は datetime64（時刻なし）、`bid` は str（末尾空白なし）、キーは列として持つ。

**`raw/sector/`**（`year=YYYY/part.parquet` でも 1 ファイルでも可）— universe と同じ暦月末日付・同じ銘柄集合

| 列 | 型 | 内容 |
|---|---|---|
| date | datetime64 | 暦月末（universe と同じ） |
| bid | str | Barra ID |
| sector | str | 中立化に使う分類コード（GICS セクターなど）。数値コードも文字列で |
| industry_group / industry | str | 任意。細かい分類があれば一緒に |

**`raw/_manifest.parquet`** — 取得ごとに 1 行追記

| 列 | 型 | 内容 |
|---|---|---|
| dataset | str | universe / sector / factor / return |
| partition | str | `table=core/year=2020`、`kind=drtn/year=2020` など（ファイルのディレクトリ） |
| fetched_at | datetime64 | 取得時刻 |
| sql_sha1 | str | 発行 SQL の sha1（定義変更の検知用） |
| n_rows, n_bid | int64 | 取得行数、銘柄数 |
| date_min, date_max | datetime64 | 取得データの日付範囲 |

build_mart は最新の fetch と実ファイルの行数を突き合わせ、`manifest_row_mismatch` / `manifest_partition_missing` / `manifest_partition_unrecorded` を QC に出す。

**`raw/factor_meta.csv`（または .parquet）** — DB のファクター定義書

| 列 | 内容 |
|---|---|
| factor | 列名（panel_x と一致） |
| category | 分類（value / momentum / ...） |
| description | 定義 |
| direction, unit, formula, source_table … | 任意。そのまま factor_meta.csv に引き継ぐ |

存在すれば `mart/factor_meta.csv` の category / description を上書きし、`meta_source=db` になる（無い行は `inferred`）。

## 外部評価用のエクスポート（`output/`）

```sh
uv run python -m india_ai_model.common.export            # 全 run → output/<score_name>/yyyymm.dat
uv run python -m india_ai_model.common.export --runs runs/ens_lgbm_dnn/exp --halflives 1 2
```

- `output/<score_name>/yyyymm.dat`：1 行目 `#bid <score_name>`、以降 `bid score`（空白区切り）。値は日付内 z-score。
- 各 run につき生スコアと EMA 平滑化版（`<name>_hl1`）を出力。`score_name` は config 名（run_id が `exp` 以外なら `<config>__<run_id>`）。
- `output/scores.csv`：score_name, description, model, target, variant, smooth_halflife, value, n_files, date_min/max, n_bid_mean, ic_drtn_1m, icir_drtn_1m, ic_drtn_3m, spread_q5q1_pct, turnover_ls, run_dir, config, exported_at
