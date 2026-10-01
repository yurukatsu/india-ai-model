"""Model-independent factor dictionary (category + short description).

Descriptions are inferred from the column names of ``table=core``; entries
marked ``(要確認)`` should be verified against the DB definition.
Which factors a model uses is decided in each model's config, not here.
"""

from __future__ import annotations

FACTOR_INFO: dict[str, tuple[str, str]] = {
    # --- price based
    "mom1": ("momentum", "1M return (short-term reversal)"),
    "mom12": ("momentum", "12M return"),
    "mom12_1": ("momentum", "12M return excluding most recent 1M"),
    "mom60": ("momentum", "60M return (long-term reversal)"),
    "vola60": ("risk", "60D return volatility"),
    "skew60": ("risk", "60D return skewness"),
    "tover": ("liquidity", "share turnover"),
    "illiq": ("liquidity", "Amihud illiquidity"),
    "csi": ("issuance", "composite share issuance (要確認)"),
    # --- Altman Z components
    "mve_tl": ("altman", "market value of equity / total liabilities"),
    "zscore": ("altman", "Altman Z-score"),
    "wk_ta": ("altman", "working capital / total assets"),
    "re_ta": ("altman", "retained earnings / total assets"),
    "ebit_ta": ("altman", "EBIT / total assets"),
    "sales_ta": ("altman", "sales / total assets"),
    # --- Beneish M components
    "mscore": ("beneish", "Beneish M-score"),
    "dsr": ("beneish", "days sales in receivables index"),
    "gmi": ("beneish", "gross margin index"),
    "aqi": ("beneish", "asset quality index"),
    "depi": ("beneish", "depreciation index"),
    "sgai": ("beneish", "SG&A index"),
    "levi": ("beneish", "leverage index"),
    # --- Ohlson O components
    "intwo": ("ohlson", "1 if net income < 0 for last two years (binary)"),
    "chin": ("ohlson", "change in net income scaled by sum of abs NI"),
    "ni_ta": ("ohlson", "net income / total assets"),
    "tl_ta": ("ohlson", "total liabilities / total assets"),
    "curr": ("ohlson", "current ratio (要確認)"),
    "ocftd": ("ohlson", "operating cash flow / total debt (要確認)"),
    # --- accruals / financing / investment
    "acc_cf": ("accruals", "accruals (cash-flow statement based)"),
    "acc_bs": ("accruals", "accruals (balance-sheet based)"),
    "xfin_cf": ("issuance", "net external financing (cash-flow based)"),
    "xfin_bs": ("issuance", "net external financing (balance-sheet based)"),
    "xfin_sp": ("issuance", "net external financing (要確認)"),
    "diss": ("issuance", "net debt issuance"),
    "eiss": ("issuance", "net equity issuance"),
    "ia": ("investment", "investment / asset growth (要確認)"),
    # --- efficiency
    "rec_to": ("efficiency", "receivables turnover"),
    "invent_to": ("efficiency", "inventory turnover"),
    "acpy_to": ("efficiency", "accounts payable turnover"),
    "ccc": ("efficiency", "cash conversion cycle"),
    "sales_bep": ("efficiency", "sales / break-even point (要確認)"),
    "npop": ("efficiency", "(要確認)"),
    # --- leverage / solvency
    "tdint": ("leverage", "total debt / interest (要確認)"),
    "td_ta": ("leverage", "total debt / total assets"),
    "eq_tl": ("leverage", "equity / total liabilities"),
    "eqr": ("leverage", "equity ratio"),
    "pti_tl": ("leverage", "pre-tax income / total liabilities"),
    # --- profitability (actual)
    "roe_act": ("profitability", "ROE (actual)"),
    "roa_act": ("profitability", "ROA (actual)"),
    "roic_act": ("profitability", "ROIC (actual)"),
    "nis_act": ("profitability", "net income / sales (actual)"),
    "ois_act": ("profitability", "operating income / sales (actual)"),
    "ocfs": ("profitability", "operating cash flow / sales"),
    "prof": ("profitability", "gross profitability (GP / assets)"),
    "dlt_gpa_5y": ("profitability_trend", "5Y change in GP / assets"),
    "dlt_roe_5y": ("profitability_trend", "5Y change in ROE"),
    "dlt_roa_5y": ("profitability_trend", "5Y change in ROA"),
    "dlt_ocfa_5y": ("profitability_trend", "5Y change in OCF / assets"),
    "dlt_gps_5y": ("profitability_trend", "5Y change in GP / sales"),
    # --- growth
    "sales_grw": ("growth", "sales growth"),
    "oi_grw": ("growth", "operating income growth"),
    "ni_grw": ("growth", "net income growth"),
    "ppe_grw": ("growth", "PP&E growth"),
    "ta_grw": ("growth", "total asset growth"),
    "invent_grw": ("growth", "inventory growth"),
    "eq_grw": ("growth", "equity growth"),
    "capex_grw": ("growth", "capex growth"),
    # --- value (actual)
    "bp_act": ("value", "book / price (actual)"),
    "ep_act": ("value", "earnings / price (actual)"),
    "dp_act": ("value", "dividend / price (actual)"),
    "sp_act": ("value", "sales / price (actual)"),
    "ebitdaev_act": ("value", "EBITDA / EV (actual)"),
    "cfp_act": ("value", "cash flow / price (actual)"),
    "ocfp": ("value", "operating cash flow / price"),
    "fcfp": ("value", "free cash flow / price"),
    "rdp": ("value", "R&D / price"),
    "maratio": ("value", "market / assets ratio (要確認)"),
    "peg_act": ("value", "PEG ratio (actual)"),
    # --- estimates
    "roe_est": ("estimate", "ROE (consensus estimate)"),
    "roa_est": ("estimate", "ROA (consensus estimate)"),
    "roic_est": ("estimate", "ROIC (consensus estimate)"),
    "nis_est": ("estimate", "net margin (consensus estimate)"),
    "ois_est": ("estimate", "operating margin (consensus estimate)"),
    "bp_est": ("estimate", "book / price (estimate)"),
    "ep_est": ("estimate", "earnings / price (estimate)"),
    "dp_est": ("estimate", "dividend / price (estimate)"),
    "sp_est": ("estimate", "sales / price (estimate)"),
    "ebitdaev_est": ("estimate", "EBITDA / EV (estimate)"),
    "cfp_est": ("estimate", "cash flow / price (estimate)"),
    "ltg": ("estimate", "long-term growth estimate"),
    "eps_grw": ("estimate", "EPS growth (estimate)"),
    "doe": ("estimate", "dispersion of estimates (要確認)"),
    # --- revisions
    "rev1p": ("revision", "1M EPS revision (percent)"),
    "rev1r": ("revision", "1M EPS revision (ratio up/down)"),
    "rev3p": ("revision", "3M EPS revision (percent)"),
    "rev3r": ("revision", "3M EPS revision (ratio up/down)"),
    "dlt_ep3": ("revision", "3M change in E/P"),
    "dlt_ep6": ("revision", "6M change in E/P"),
}


def category_of(factor: str) -> str:
    return FACTOR_INFO.get(factor, ("unknown", ""))[0]


def description_of(factor: str) -> str:
    return FACTOR_INFO.get(factor, ("unknown", ""))[1]
