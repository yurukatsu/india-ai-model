# 実験ログ（再現用）

対象期間：2026-09-30〜2026-10-01。全 run の指標は [run_summary.csv](run_summary.csv)（`runs/*/*/metrics.csv` から機械生成）。
run の成果物は `runs/<config名>/<run_id>/`（`config.yaml`, `pred.parquet`, `pred_by_date/`, `metrics.csv`, `metrics_smooth.csv`, `per_date.parquet`, `fit_info.json`, `splits.txt`）。

## 0. 環境と再現手順

```sh
uv sync && brew install libomp                       # Python 3.12 / pandas 3.0.6 / torch 2.14 / lightgbm 4.6
uv run pytest                                        # calendar / targets のユニットテスト 12 本
uv run build-mart                                    # data/raw -> data/mart（約 15 秒）
uv run python -c "from india_ai_model.common.targets import build_panel_y; build_panel_y(['drtn_1m','drtn_1m_s1','drtn_1m_s2','drtn_3m','srtn_1m','srtn_3m','gp_trtn_1m'])"
uv run python -m india_ai_model.common.runner configs/<name>.yaml --run-id exp
uv run python -m india_ai_model.common.ensemble --name <ens名> --runs runs/a/exp runs/b/exp
uv run python -m india_ai_model.common.evaluate runs/<name>/exp --halflives 0 1 2 3 6   # 事後の平滑化評価
```

- 再現性：DNN は `torch.manual_seed(seed)` ＋ CPU 実行。同じ config・同じ mart なら数値は一致する（シード 0〜4 の run を再実行して一致を確認済み）。
- `--run-id` が既存なら上書き拒否（`--overwrite` で明示）。初日に同名 run_id を上書きした事故があったため追加した。

## 1. データの確認事項（build_mart の前提）

| 項目 | 事実 | 対応 |
|---|---|---|
| return のカレンダー | raw は月〜金。インドの祝日は全銘柄 ret=0 の行として入っている（349 日 ≈ 13 日/年、NSE 祝日と一致） | 非欠損 ≥ 50 銘柄 かつ ret=0 比率 < 99% の日を営業日とする |
| return の欠損 | `-999999998`（drtn 668 行） | `<= -9.99e8` を NaN |
| return の単位 | drtn / srtn は %、gp_trtn は小数 | `RETURN_UNITS` で kind ごとに指定、未知 kind は中央値の大きさで推定 |
| `bid` | raw/return に末尾空白 `'INDAAA1 '` | strip |
| factor の日付 | 暦の月末（土日含む）。センチネル `-1e9` | ym 結合でリバランス日へ。センチネル → NaN |
| universe | 暦月末、2003-01〜。2003〜2006 年は 15〜46 銘柄、2007 年 148、2008 年以降 280〜680 | 学習開始 2007-01 |
| リバランス日 | 資料案は「月末営業日の 1 営業日前」だが、factor が暦月末時点の値なので lag=1 は 1 日先読み | **lag=0（月末営業日）** を採用。`--rebal-lag` で変更可 |
| 未完了月 | 2026-09 は 09-29 までしかデータがない | リバランス日を付けない |
| sector | GICS 4 階層、2000-01〜、1,071 銘柄、欠損なし | `sector_monthly` に 4 階層とも保持 |
| gp_trtn（DS2） | 小数、営業日のみ、2026-05 で更新停止 | 月次で drtn と相関 0.998、平均差 3e-5 → Barra drtn は基本的に調整済み |
| drtn の異常値 | `INDHMS1` +5,400% など。ユニバース在籍中の |ret|>50% は 26 年で 15 行 | `|drtn| > 0.4` で gp_trtn が 5pp 以内で裏付けない日を drtn/srtn とも NaN（124 行）。QC に `ret_masked`, `ret_monthly_mismatch`(50 行) |
| factor_meta | category / description は列名からの推定 | Ryosuke の定義書 `raw/factor_meta.csv` が来れば上書き（`meta_source=db`） |

ユニバース内の |drtn| > 20% の日次 474 行のうち 466 行は DS2 が 5pp 以内で裏付け（−77% の INDB7Y1 2009-01 なども実際の値）。

## 2. 共通設定（全 run）

- 評価：OOS 2012-12-31〜2026-08-31（164 リバランス日）。日付別 Spearman IC の平均、ICIR = mean/std、t = ICIR×√n。分位スプレッドは等ウェット 5 分位の Q5−Q1（drtn_1m）。
- 回転率：`turnover_ls` = 日付内 z-score をウェイト（ロング 1・ショート 1）にした L/S の `Σ|Δw|/2`（月次片道、ユニバース出入りを含む）。`turnover_top20` = 上位 20% バスケットの入替率。
- walk-forward：拡大窓（train_start 2007-01）、年 1 回再学習（first_retrain 2012-12、14 split）、val = 直近 12 ヶ月、embargo = ターゲットホライズン、val 込みの再学習なし。
- 特徴量：93 ファクター（センチネル率 > 0.5 の `ltg, ois_est, roic_est, roa_est, rdp` を除外）＋ 14 本のラグ（3, 12 ヶ月）＋ 12 本のローゼンバーグ変化率（3, 12 ヶ月）= 145 本。日付内一様 rank、欠損 0.5、バイナリ（`intwo`）は 0/1。
- ターゲット：`drtn_1m` を日付内 Gaussian rank。`min_obs_frac 0.5`。
- モデル：ridge（α ∈ {1,10,100,1e3,1e4} を val IC で選択）、lgbm 回帰（lr 0.02, leaves 15, min_data 100, ff 0.5, bf 0.8, l2 10, early stop 200 on val IC）、dnn（MLP 64-32, LayerNorm, GELU, dropout 0.2, 入力ノイズ 0.05, AdamW 1e-3 / wd 1e-4, 16 日付/バッチ, IC 損失 √N 加重, patience 20, シード 5 本の z-score 平均）。

## 3. 時系列の記録

### Day 1（2026-09-30）

| # | 内容 | run | IC / ICIR | 判断 |
|---|---|---|---|---|
| 1 | mart 構築、ターゲット、評価、walk-forward、3 ベースライン | `ridge/baseline` | 0.069 / 0.46 | リッジが基準 |
| | | `lgbm/baseline`（lambdarank, 10 分位） | 0.051 / 0.35 | early stop が 1〜13 it で止まる。不採用 |
| | | `lgbm_reg/baseline`（回帰） | 0.067 / 0.47 | lambdarank より良い。以後 lgbm は回帰 |
| | | `dnn/baseline` | 0.065 / 0.47 | ベースラインに負ける。early stop が数エポック |
| 2 | 単変量 IC（2007〜）で符号確認 | — | mom12_1 +0.056, rev3r +0.057, vola60 −0.053, csi −0.041 | パイプラインの妥当性確認 |

### Day 2（2026-10-01）

| # | 内容 | run | IC / ICIR | 判断 |
|---|---|---|---|---|
| 3 | ターゲット srtn_1m に一時切替（drtn の調整係数疑義） | `*/srtn` | ridge 0.048/0.52, lgbm 0.053/0.58, dnn 0.044/0.49 | DS2 で drtn が裏付けられたので drtn に戻す |
| 4 | sector / gp_trtn 受領、drtn マスク処理追加、drtn に戻す | `*/drtn_masked` | ridge 0.068/0.46, lgbm 0.068/0.48, dnn 0.065/0.47 | 変化なし（マスク対象がユニバース内で少ない） |
| 5 | セクター中立ターゲット | `ridge_sector/drtn_masked` | 0.069 / 0.48 | ICIR +0.02、わずか |
| 6 | **ピア相対特徴量**（GICS industry_group 内中央値との差 19 本 ＋ 業種中央値 7 本） | `ridge_peer`, `lgbm_reg_peer`, `dnn_peer` | 0.073/0.51, 0.072/0.52, **0.074/0.54** | 全モデルで +0.004〜0.009。採用。DNN がリッジを初めて上回る |
| 7 | cross-sectional attention（1 block, 2 heads, width 32） | `dnn_attn`, `dnn_attn_peer` | 0.067/0.47, 0.074/0.56 | peer なしでは MLP と同じ。peer 併用の +0.02 ICIR は後に（#12）シードノイズと判明 |
| 8 | 回転率の定義追加（top20 / bottom20 / L/S 片道）、EMA 平滑化（半減期 0,1,2,3,6） | 全 run に `metrics_smooth.csv` | hl=1 で回転率 −45%、IC −5% | hl=1 が最もコスパ良い。hl≥2 は IC の減りが速い |
| 9 | 長期ターゲット学習（drtn_3m / 6m） | `ridge_peer_3m`, `_6m`, `dnn_peer_3m` | 0.067/0.43, 0.058/0.40, 0.063/0.43 | 1m IC が大幅低下。不採用 |
| 10 | サンプル重み（√N 日付、√指数ウェイト銘柄、時間減衰 5 年） | `*_w_sqrtn`, `*_w_cap`, `*_w_time` | ridge 0.073/0.52, 0.072/0.53, 0.073/0.49; dnn cap 0.070/0.50, time 0.072/0.53 | 効果なし〜マイナス。不採用 |
| 11 | 欠損フラグ（センチネル率 0.05〜0.5 の 56 本 ＋ 欠損数） | `ridge_peer_na`, `lgbm_reg_peer_na`, `dnn_peer_na` | 0.072/0.50, 0.072/0.53, **0.076/0.54** | DNN のみプラス（3m IC 0.093、回転率 0.56 に低下）。DNN で採用 |
| 12 | 相関ピア（drtn 250 日相関 top-10 の中央値との差 19 本） | `ridge_peer_cpr`, `lgbm_reg_peer_cpr`, `dnn_peer_cpr` | **0.075/0.54**, **0.077/0.55**, **0.075/0.56** | 全モデルでプラス。採用。lgbm で最大（3m IC 0.096） |
| 13 | 11 + 12 | `dnn_peer_na_cpr`, `lgbm_reg_peer_na_cpr` | **0.077/0.59**, 0.074/0.56 | DNN は両方、lgbm は cpr のみが最良 |
| 14 | シードノイズの測定 | `dnn_peer_na_cpr_s5b`（seeds 5〜9） | 0.075 / 0.56 | 5 本アンサンブルの組替えで IC ±0.002、ICIR ±0.03、スプレッド ±0.15% |
| 15 | シード 10 本 | `dnn_peer_na_cpr_s10` | **0.078 / 0.58** | **新基準** |
| 16 | attention 再確認（10 本） | `dnn_attn_peer_na_cpr_s10` | 0.078 / 0.55 | MLP と順位相関 0.965、ICIR 低、学習 3.5 倍。不採用 |
| 17 | 相関ピアの亜種：srtn 相関 / 低ランク(10) / カーネル(pool 30, τ 0.1) | `dnn_peer_na_cpr_{srtn,lowrank,kernel}` | 0.077/0.59, 0.076/0.59, 0.076/0.57 | 全てノイズ範囲内。不採用 |
| 18 | アンサンブル（日付内 z-score 等ウェイト） | `ens_lgbm_dnn`, `ens3`, `ens3_icw` ほか | **0.082 / 0.57**, 0.082/0.57, 0.082/0.57 | IC +0.004、ICIR 横ばい。lgbm の寄与が最大（メンバー相関 0.77）。IC 加重は等ウェイトと同じ。**lgbm + dnn が最終候補** |
| 19 | masked AE 補完（64-32-64, mask 25%, 50 epochs） | `dnn_peer_na_cpr_ae_s10` | 0.078 / 0.59 | AE の復元 R² 0.70〜0.73 だが下流は不変（順位相関 0.99）。不採用 |
| 20 | srtn ターゲット DNN をアンサンブルに追加 | `dnn_srtn_peer_na_cpr_s10`, `ens_lgbm_dnn_srtn`, `ens4_srtn` | 単体 0.059/0.56（srtn ICIR 0.61）; 3 本 0.079/0.60; 4 本 0.081/0.59 | ICIR +0.03 だが IC −0.003、回転率 +0.08。平滑化後は優位性なし。不採用（運用が srtn 軸なら再考） |

### 最終状態

| | IC | ICIR | t | 3m IC | Q5−Q1 | Sharpe | 回転率 | hl=1: IC / ICIR / 回転率 |
|---|---|---|---|---|---|---|---|---|
| 初期 `ridge/baseline` | 0.069 | 0.46 | 5.9 | 0.085 | 1.33% | 0.95 | 0.47 | — |
| 単体最良 `dnn_peer_na_cpr_s10` | 0.078 | 0.58 | 7.4 | 0.094 | 1.60% | 1.19 | 0.60 | 0.074 / 0.52 / 0.33 |
| **最終候補 `ens_lgbm_dnn`** | **0.082** | **0.57** | 7.4 | **0.101** | **1.66%** | 1.18 | 0.60 | **0.078 / 0.52 / 0.32** |

## 4. 判定基準として使った数値

- シードノイズ（#14）：IC ±0.002、ICIR ±0.03。これ以内の差は「同じ」。
- 回転率と IC のトレードオフ：hl=1 の EMA で回転率 −45% に対し IC −5%。新しい特徴量が回転率を上げても、hl=1 後に基準を上回れば可。
- val IC（`fit_info.json` の `val_ic_mean`）は OOS IC とほぼ連動しないので、採否判断には使っていない（early stopping のみに使用）。

## 5. 実装上のメモ（次に触るときの注意）

- `features.build_features` は `(df, feat_cols, observed)` を返す。`observed` は補完前の欠損マスク（impute 用）。
- ピア特徴量・相関ピア・欠損フラグは**生値**（rank 化前）で計算し、ユニバースに絞った後で rank 化する。相関ピアはリバランス日 t 以前の 250 営業日のみ使用。
- `targets.make_target` の窓は `(R_{i+skip}, R_{i+skip+months}]`。`end_date <= T` でリークマスク。`panel_y/<name>.parquet` はキャッシュなので mart を再生成したら `rm -rf data/mart/panel_y`。
- モデル API：`Model(params, feature_names).fit(X_tr, y_tr, d_tr, X_va, y_va, d_va, w_tr=None)`, `.predict(X, dates)`, `.info()`。
- `runs/<config stem>/<run_id>` に保存。同名は上書き拒否。
- lgbm は macOS で `brew install libomp` が必要。
- `pred_by_date/YYYYMMDD.parquet` は `date, bid, score`（alpha_signal_evaluation の入力仕様に合わせて要調整）。

## 6. 未着手・保留

- MSCI EM での事前学習 → インド微調整（データ待ち）。attention など容量の大きいモデルはこの後で再評価。
- マクロ変数によるレジーム条件付け（データ待ち）。
- factor_meta の定義書反映、`_manifest.parquet`（学習には不要）。
- drtn の調整係数の確認（DS2 で裏付けは取れているが、2026-06 以降は閾値マスクのみ）。
- 取引コストを入れた平滑化半減期の最適化（コスト前提が必要）。
