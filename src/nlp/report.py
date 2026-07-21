"""Markdown / JSON / PNG report writers for the Chinese-text NLP pipeline.

Output layout (always under a caller-supplied ``out_dir``; nothing is ever
written to the repo root):

    <out_dir>/eda/eda_report.md
    <out_dir>/eda/eda_report.json
    <out_dir>/eda/plots/*.png
    <out_dir>/benchmark/benchmark_report.md
    <out_dir>/benchmark/benchmark_report.json
    <out_dir>/benchmark/plots/*.png

matplotlib is imported lazily inside the plot helpers with the ``Agg``
backend forced first, so importing this module stays cheap and headless
safe. CJK note: the default matplotlib font may lack Traditional Chinese
glyphs — the helpers request ``CJK_FONT_FAMILIES`` and let matplotlib fall
back with a warning; missing fonts never fail the report.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.nlp.eda import (
    BERT_TOKEN_LIMIT,
    MAX_CONFLICT_EXAMPLES,
    NEAR_DUP_JACCARD,
    NEAR_DUP_SHINGLE,
    SHORT_TEXT_CHARS,
    TextEdaReport,
)

if TYPE_CHECKING:
    from src.nlp.analysis.feature_selection import FeatureSelectionReport

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
EDA_SUBDIR = "eda"
BENCHMARK_SUBDIR = "benchmark"
FEATURE_SELECTION_SUBDIR = "feature_selection"
PLOTS_SUBDIR = "plots"
EDA_MD_NAME = "eda_report.md"
EDA_JSON_NAME = "eda_report.json"
BENCHMARK_MD_NAME = "benchmark_report.md"
BENCHMARK_JSON_NAME = "benchmark_report.json"
FEATURE_SELECTION_MD_NAME = "feature_selection_report.md"
FEATURE_SELECTION_JSON_NAME = "feature_selection_report.json"
HEADLINE_METRIC = "f1_macro"
MD_MAX_FINDINGS = 20
TOP_TOKEN_PLOT_N = 20
CJK_FONT_FAMILIES = ["Noto Sans CJK TC", "Microsoft JhengHei", "PingFang TC", "sans-serif"]
# Single-series chart palette (validated reference palette, slot-1 blue).
PLOT_SURFACE = "#fcfcfb"
PLOT_SERIES = "#2a78d6"
PLOT_GRID = "#e8e7e4"
PLOT_INK = "#0b0b0b"
PLOT_INK_MUTED = "#52514e"
PLOT_DPI = 120


# --------------------------------------------------------------------------- #
# Public writers
# --------------------------------------------------------------------------- #
def write_eda_report(report: TextEdaReport, out_dir: str) -> dict:
    """Write the EDA report (md + json + plots) under ``<out_dir>/eda``.

    Returns ``{"markdown": path, "json": path, "plots": [paths]}``.
    """
    if not isinstance(report, TextEdaReport):
        raise TypeError(f"report must be a TextEdaReport, got {type(report).__name__}")
    if not out_dir:
        raise ValueError("out_dir must be a non-empty path")
    eda_dir = Path(out_dir) / EDA_SUBDIR
    plots_dir = eda_dir / PLOTS_SUBDIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    json_path = eda_dir / EDA_JSON_NAME
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path = eda_dir / EDA_MD_NAME
    md_path.write_text(_eda_markdown(report), encoding="utf-8")
    plots = _eda_plots(report, plots_dir)
    return {
        "markdown": str(md_path),
        "json": str(json_path),
        "plots": [str(p) for p in plots],
    }


def write_benchmark_report(result: dict, out_dir: str) -> dict:
    """Write the benchmark report (md + json + plots) under ``<out_dir>/benchmark``.

    ``result`` follows the benchmark result schema; models whose ``error``
    is set are shown with their error and excluded from ranking and plots.
    Returns ``{"markdown": path, "json": path, "plots": [paths]}``.
    """
    if not isinstance(result, dict):
        raise TypeError(f"result must be a dict, got {type(result).__name__}")
    if not out_dir:
        raise ValueError("out_dir must be a non-empty path")
    if not result.get("models"):
        raise ValueError(
            "Benchmark result contains no models — nothing to report; "
            "run the benchmark before writing its report."
        )
    bench_dir = Path(out_dir) / BENCHMARK_SUBDIR
    plots_dir = bench_dir / PLOTS_SUBDIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    json_path = bench_dir / BENCHMARK_JSON_NAME
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path = bench_dir / BENCHMARK_MD_NAME
    md_path.write_text(_benchmark_markdown(result), encoding="utf-8")
    plots = _benchmark_plots(result, plots_dir)
    return {
        "markdown": str(md_path),
        "json": str(json_path),
        "plots": [str(p) for p in plots],
    }


def write_feature_selection_report(report: "FeatureSelectionReport", out_dir: str) -> dict:
    """Write the feature-selection report (md + json) under ``<out_dir>/feature_selection``.

    Plots are produced by :func:`run_feature_selection`; their paths are
    echoed back here. The metadata section is omitted when ``report.metadata``
    is ``None``. Returns ``{"markdown": path, "json": path, "plots": [paths]}``.
    """
    from src.nlp.analysis.feature_selection import FeatureSelectionReport

    if not isinstance(report, FeatureSelectionReport):
        raise TypeError(
            f"report must be a FeatureSelectionReport, got {type(report).__name__}")
    if not out_dir:
        raise ValueError("out_dir must be a non-empty path")
    fs_dir = Path(out_dir) / FEATURE_SELECTION_SUBDIR
    fs_dir.mkdir(parents=True, exist_ok=True)

    json_path = fs_dir / FEATURE_SELECTION_JSON_NAME
    json_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path = fs_dir / FEATURE_SELECTION_MD_NAME
    md_path.write_text(_feature_selection_markdown(report), encoding="utf-8")
    return {
        "markdown": str(md_path),
        "json": str(json_path),
        "plots": [str(p) for p in (report.plots or {}).values()],
    }


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #
def _md_table(headers, rows) -> str:
    def cell(value) -> str:
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(cell(c) for c in row) + " |")
    return "\n".join(lines)


def _fmt_float(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:.4f}"


def _fmt_seconds(value) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "—"
    return f"{value:.1f}"


# --------------------------------------------------------------------------- #
# EDA markdown
# --------------------------------------------------------------------------- #
def _eda_markdown(report: TextEdaReport) -> str:
    parts = [
        "# 中文文本 EDA 報告",
        _md_overview(report),
        _md_length(report.length),
        _md_balance(report.balance),
        _md_vocab(report.vocab),
        _md_quality(report.quality),
        _md_pii(report.pii, report.normalization),
        _md_leakage(report.leakage),
        _md_advice(report.warnings),
    ]
    return "\n\n".join(parts) + "\n"


def _md_overview(report: TextEdaReport) -> str:
    rows = [
        ("文件數", report.n_docs),
        ("任務型態", report.task_type),
        ("類別數", len(report.balance.counts)),
        ("警示數", len(report.warnings)),
    ]
    return "## 概要\n\n" + _md_table(("項目", "值"), rows)


def _md_length(length) -> str:
    rows = [
        ("平均", f"{length.mean:.1f}", f"{length.token_mean:.1f}"),
        ("中位數", f"{length.median:.1f}", f"{length.token_median:.1f}"),
        ("P95", f"{length.p95:.1f}", f"{length.token_p95:.1f}"),
        ("最大", length.max, length.token_max),
    ]
    note = (
        f"超過 BERT {BERT_TOKEN_LIMIT} token 上限的文件比例："
        f"{length.over_512_ratio:.2%}"
    )
    return "## 長度分布\n\n" + _md_table(("統計量", "字元", "Token"), rows) + "\n\n" + note


def _md_balance(balance) -> str:
    table = _md_table(("類別", "件數"), list(balance.counts.items()))
    minority = "、".join(balance.minority_classes) if balance.minority_classes else "無"
    lines = [
        f"- 不平衡比（最多/最少）：{balance.imbalance_ratio:.2f}",
        f"- 正規化熵（0-1）：{balance.entropy:.3f}",
        f"- 少數類別（少於多數類 10%）：{minority}",
    ]
    return "## 類別分布\n\n" + table + "\n\n" + "\n".join(lines)


def _md_vocab(vocab) -> str:
    stat_rows = [
        ("詞彙量", vocab.vocab_size),
        ("TTR（type-token ratio）", f"{vocab.ttr:.4f}"),
        ("Hapax 比例", f"{vocab.hapax_ratio:.4f}"),
    ]
    parts = [
        "## 詞彙",
        _md_table(("指標", "值"), stat_rows),
        "### 高頻 Token（前 20）",
        _ranked_table(vocab.top_tokens[:TOP_TOKEN_PLOT_N]),
        "### 高頻 Bigram（前 10）",
        _ranked_table(vocab.top_bigrams[:10]),
        "### 高頻 Trigram（前 10）",
        _ranked_table(vocab.top_trigrams[:10]),
    ]
    return "\n\n".join(parts)


def _ranked_table(pairs) -> str:
    if not pairs:
        return "（無資料）"
    rows = [(i + 1, token, count) for i, (token, count) in enumerate(pairs)]
    return _md_table(("排名", "Token", "次數"), rows)


def _md_quality(quality) -> str:
    near_dup_label = (
        f"近似重複配對數（{NEAR_DUP_SHINGLE} 字元 shingle "
        f"Jaccard ≥ {NEAR_DUP_JACCARD}）"
    )
    rows = [
        ("完全重複群組數", quality.exact_duplicate_groups),
        ("完全重複文件數", quality.exact_duplicate_docs),
        (near_dup_label, quality.near_duplicate_pairs),
        ("標籤衝突組數（同文不同標）", quality.label_conflicts),
        ("空白文件數", quality.empty_texts),
        (f"過短文件數（< {SHORT_TEXT_CHARS} 字元）", quality.short_texts),
    ]
    parts = ["## 品質問題", _md_table(("檢查項", "數量"), rows)]
    if quality.conflict_examples:
        listed = "；".join(str(group) for group in quality.conflict_examples)
        parts.append(f"標籤衝突列索引（最多 {MAX_CONFLICT_EXAMPLES} 組）：{listed}")
    return "\n\n".join(parts)


def _md_pii(pii, normalization) -> str:
    parts = [
        "## PII 與正規化",
        f"含 PII 文件數：{pii.docs_with_pii} / {pii.total_docs}",
        _md_table(("PII 類型", "命中數"), list(pii.counts.items())),
    ]
    if pii.findings:
        sample = pii.findings[:MD_MAX_FINDINGS]
        rows = [(f.row, f.kind, f.masked) for f in sample]
        parts.append(f"### 遮罩後樣本（前 {len(sample)} 筆）")
        parts.append(_md_table(("列", "類型", "遮罩片段"), rows))
    norm_rows = [
        ("含全形英數文件數", normalization.docs_with_fullwidth),
        ("全形英數字元總數", normalization.fullwidth_char_count),
        ("含 OCR 疑似字元文件數", normalization.docs_with_ocr_suspects),
        ("全半形數字混用文件數", normalization.mixed_width_digit_docs),
    ]
    parts.append("### 正規化檢查")
    parts.append(_md_table(("檢查項", "數量"), norm_rows))
    parts.append(_md_table(("OCR 疑似字元", "次數"), list(normalization.ocr_suspect_counts.items())))
    return "\n\n".join(parts)


def _md_leakage(leakage) -> str:
    if leakage is None:
        return "## 切分洩漏\n\n未提供資料切分（splits），未執行洩漏檢查。"
    parts = ["## 切分洩漏"]
    if leakage.pairs:
        parts.append(_md_table(("切分配對", "重複文本數"), list(leakage.pairs.items())))
    else:
        parts.append("切分不足兩組，無可比較配對。")
    if leakage.leaked_examples:
        rows = [(e["pair"], e["row_a"], e["row_b"]) for e in leakage.leaked_examples]
        parts.append(_md_table(("配對", "列 A", "列 B"), rows))
    return "\n\n".join(parts)


def _md_advice(warnings_list) -> str:
    if not warnings_list:
        return "## 建議\n\n未發現需特別處理的問題。"
    return "## 建議\n\n" + "\n".join(f"- {w}" for w in warnings_list)


# --------------------------------------------------------------------------- #
# Benchmark markdown
# --------------------------------------------------------------------------- #
def _benchmark_markdown(result: dict) -> str:
    parts = [
        "# 文本分類基準測試報告",
        _md_bench_summary(result),
        _md_bench_details(result),
        _md_bench_ranking(result),
    ]
    return "\n\n".join(parts) + "\n"


def _md_bench_summary(result: dict) -> str:
    label_space = result.get("label_space") or {}
    classes = label_space.get("classes") or []
    env_rows = [
        ("任務型態", result.get("task_type", "—")),
        ("隨機種子", result.get("seed", "—")),
        ("裝置", result.get("device", "—")),
        ("精度", result.get("precision", "—")),
        ("訓練/驗證/測試筆數",
         f"{result.get('n_train', '—')} / {result.get('n_val', '—')} / {result.get('n_test', '—')}"),
        ("類別數", len(classes)),
        ("多標籤", "是" if label_space.get("is_multilabel") else "否"),
    ]
    model_rows = []
    for model in result.get("models") or []:
        metrics = model.get("metrics") or {}
        model_rows.append((
            model.get("name", "—"),
            model.get("family", "—"),
            _fmt_float(metrics.get(HEADLINE_METRIC)),
            _fmt_seconds(model.get("train_seconds")),
            model.get("error") or "—",
        ))
    return (
        "## 執行摘要\n\n"
        + _md_table(("項目", "值"), env_rows)
        + "\n\n"
        + _md_table(("模型", "家族", HEADLINE_METRIC, "訓練秒數", "錯誤"), model_rows)
    )


def _md_bench_details(result: dict) -> str:
    parts = ["## 每模型詳細"]
    for model in result.get("models") or []:
        parts.append(f"### {model.get('name', '—')}（{model.get('family', '—')}）")
        error = model.get("error")
        if error:
            parts.append(f"執行失敗：{error}")
            continue
        info_rows = [
            ("訓練秒數", _fmt_seconds(model.get("train_seconds"))),
            ("訓練輪數", model.get("n_epochs", "—")),
            ("排名分數", _fmt_float(model.get("ranking_score"))),
        ]
        parts.append(_md_table(("項目", "值"), info_rows))
        metric_rows = _scalar_metric_rows(model.get("metrics") or {})
        if metric_rows:
            parts.append(_md_table(("指標", "值"), metric_rows))
        notes = model.get("notes") or {}
        if notes:
            parts.append("備註：" + json.dumps(notes, ensure_ascii=False))
    return "\n\n".join(parts)


def _scalar_metric_rows(metrics: dict) -> list:
    rows = []
    for key in sorted(metrics):
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue  # nested structures stay in the JSON report
        rows.append((key, f"{value:.4f}" if isinstance(value, float) else value))
    return rows


def _md_bench_ranking(result: dict) -> str:
    models = result.get("models") or []
    ok_models = [m for m in models if not m.get("error")]
    errored = [m for m in models if m.get("error")]
    parts = ["## 排名與建議"]

    ranked = _ranked_names(result, ok_models)
    if ranked:
        by_name = {str(m.get("name")): m for m in ok_models}
        rows = [
            (pos, name, _fmt_float(by_name.get(name, {}).get("ranking_score")))
            for pos, name in enumerate(ranked, start=1)
        ]
        parts.append(_md_table(("名次", "模型", "ranking_score"), rows))
        best = ranked[0]
        best_score = _fmt_float(by_name.get(best, {}).get("ranking_score"))
        parts.append(f"建議優先採用 **{best}**（ranking_score={best_score}）。")
    else:
        parts.append("所有模型皆執行失敗，無法產生排名。")
    if errored:
        listed = "；".join(f"{m.get('name', '—')}（{m.get('error')}）" for m in errored)
        parts.append(f"已排除執行失敗模型：{listed}")
    return "\n\n".join(parts)


def _ranked_names(result: dict, ok_models: list) -> list:
    """Ranking order, restricted to non-errored models.

    Uses ``result["ranking"]`` when it covers usable models, otherwise
    falls back to sorting by ``ranking_score`` descending.
    """
    ok_names = {str(m.get("name")) for m in ok_models}
    ranked = [str(name) for name in result.get("ranking") or [] if str(name) in ok_names]
    if ranked:
        return ranked
    scored = [m for m in ok_models
              if isinstance(m.get("ranking_score"), (int, float))
              and not isinstance(m.get("ranking_score"), bool)]
    scored.sort(key=lambda m: (-float(m["ranking_score"]), str(m.get("name"))))
    return [str(m.get("name")) for m in scored]


# --------------------------------------------------------------------------- #
# Feature-selection markdown
# --------------------------------------------------------------------------- #
def _feature_selection_markdown(report: "FeatureSelectionReport") -> str:
    parts = [
        "# 中文文本特徵篩選分析與建議報告",
        _md_fs_overview(report),
        _md_fs_term(report.term),
    ]
    if report.metadata is not None:
        parts.append(_md_fs_metadata(report.metadata))
    parts.append(_md_fs_recommendations(report.recommendations))
    return "\n\n".join(parts) + "\n"


def _md_fs_overview(report: "FeatureSelectionReport") -> str:
    rows = [
        ("文件數", report.n_docs),
        ("任務型態", report.task_type),
        ("含 Metadata 分析", "是" if report.metadata is not None else "否"),
    ]
    return "## 概要\n\n" + _md_table(("項目", "值"), rows)


def _md_fs_term(term) -> str:
    method_rows = []
    for method, scored in term.methods.items():
        top = "、".join(f.name for f in scored[:10])
        method_rows.append((method, top or "（無）"))
    curve_rows = [(p["k"], _fmt_float(p["f1_macro"])) for p in term.feature_count_curve]
    ngram_rows = [(k, _fmt_float(v)) for k, v in term.ngram_scores.items()]

    parts = [
        "## 詞彙特徵篩選",
        "### 方法比較表（各方法前 10 個字元特徵）",
        _md_table(("方法", "前 10 特徵"), method_rows),
        f"### 推薦保留特徵數與 n-gram\n\n"
        f"- 推薦保留特徵數（macro-F1 飽和點）：**{term.recommended_max_features}**\n"
        f"- 推薦 n-gram 設定：**{term.recommended_ngram}**",
        "特徵數 vs macro-F1：",
        _md_table(("k", "macro-F1"), curve_rows) if curve_rows else "（無曲線資料）",
        "n-gram 交叉驗證 macro-F1：",
        _md_table(("n-gram", "macro-F1"), ngram_rows) if ngram_rows else "（無資料）",
    ]
    if term.redundant_pairs:
        red_rows = [(p["term_a"], p["term_b"], _fmt_float(p["cosine"]))
                    for p in term.redundant_pairs[:MD_MAX_FINDINGS]]
        parts.append(f"### 冗餘詞對（cosine >= 0.9，前 {len(red_rows)} 組）")
        parts.append(_md_table(("特徵 A", "特徵 B", "cosine"), red_rows))
    else:
        parts.append("### 冗餘詞對\n\n未發現高度冗餘的字元特徵。")
    if term.notes:
        parts.append("備註：\n" + "\n".join(f"- {n}" for n in term.notes))
    return "\n\n".join(parts)


def _md_fs_metadata(metadata) -> str:
    rel_rows = []
    for r in metadata.relevance:
        rel_rows.append((
            r.column, r.dtype,
            _fmt_float(r.chi2) if r.chi2 is not None else "—",
            _fmt_float(r.mutual_info),
            _fmt_float(r.anova_f) if r.anova_f is not None else "—",
            _fmt_float(r.cramers_v) if r.cramers_v is not None else "—",
            _fmt_float(r.tree_importance),
        ))
    corr = metadata.correlation
    parts = [
        "## Metadata 特徵",
        "### 相關性表（欄位對標籤的相關性）",
        _md_table(("欄位", "型別", "chi2", "mutual_info", "anova_f",
                   "cramers_v", "tree_importance"), rel_rows),
    ]
    if corr.cramers_v_pairs:
        rows = [(p["col_a"], p["col_b"], _fmt_float(p["cramers_v"]))
                for p in corr.cramers_v_pairs[:MD_MAX_FINDINGS]]
        parts.append("### Cramér's V（類別-類別）")
        parts.append(_md_table(("欄位 A", "欄位 B", "Cramér's V"), rows))
    if corr.numeric_correlation_pairs:
        rows = [(p["col_a"], p["col_b"], _fmt_float(p["pearson"]))
                for p in corr.numeric_correlation_pairs[:MD_MAX_FINDINGS]]
        parts.append("### 數值相關（|pearson| >= 0.8）")
        parts.append(_md_table(("欄位 A", "欄位 B", "pearson"), rows))
    if corr.vif:
        rows = [(c, _fmt_float(v)) for c, v in corr.vif.items()]
        parts.append("### VIF（數值欄位共線性）")
        parts.append(_md_table(("欄位", "VIF"), rows))
        if corr.high_vif_columns:
            parts.append("高 VIF（>= 10）欄位：" + "、".join(corr.high_vif_columns))
    keep = "、".join(metadata.recommended_keep) if metadata.recommended_keep else "（無）"
    drop = "、".join(metadata.recommended_drop) if metadata.recommended_drop else "（無）"
    parts.append("### 建議保留與剔除")
    parts.append(f"- 建議保留：{keep}\n- 建議剔除：{drop}")
    if metadata.notes:
        parts.append("備註：\n" + "\n".join(f"- {n}" for n in metadata.notes))
    return "\n\n".join(parts)


def _md_fs_recommendations(recommendations) -> str:
    if not recommendations:
        return "## 總體建議\n\n未產生建議。"
    return "## 總體建議\n\n" + "\n".join(f"- {r}" for r in recommendations)


# --------------------------------------------------------------------------- #
# Plot helpers (lazy matplotlib, Agg forced)
# --------------------------------------------------------------------------- #
def _plt():
    """Import matplotlib lazily, force Agg, apply CJK-tolerant defaults."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = CJK_FONT_FAMILIES
    return plt


def _new_axes(plt, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(PLOT_SURFACE)
    ax.set_facecolor(PLOT_SURFACE)
    for name in ("top", "right"):
        ax.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_color(PLOT_GRID)
    ax.tick_params(colors=PLOT_INK_MUTED, labelsize=9)
    return fig, ax


def _style_axes(ax, title: str, xlabel: str, ylabel: str, grid_axis: str) -> None:
    ax.set_title(title, color=PLOT_INK, fontsize=12)
    ax.set_xlabel(xlabel, color=PLOT_INK_MUTED, fontsize=10)
    ax.set_ylabel(ylabel, color=PLOT_INK_MUTED, fontsize=10)
    ax.grid(axis=grid_axis, color=PLOT_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _save(plt, fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def _eda_plots(report: TextEdaReport, plots_dir: Path) -> list:
    plt = _plt()
    paths = []

    if report.length.char_lengths:
        fig, ax = _new_axes(plt, (8, 4.5))
        ax.hist(report.length.char_lengths, bins=30, color=PLOT_SERIES,
                edgecolor=PLOT_SURFACE, linewidth=0.6)
        _style_axes(ax, "文件長度分布（字元）", "字元數", "文件數", grid_axis="y")
        path = plots_dir / "length_hist.png"
        _save(plt, fig, path)
        paths.append(path)

    if report.balance.counts:
        classes = list(report.balance.counts)
        values = [report.balance.counts[c] for c in classes]
        fig, ax = _new_axes(plt, (8, 4.5))
        ax.bar(range(len(classes)), values, color=PLOT_SERIES,
               edgecolor=PLOT_SURFACE, linewidth=1.0, width=0.7)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        _style_axes(ax, "類別分布", "類別", "件數", grid_axis="y")
        path = plots_dir / "class_distribution.png"
        _save(plt, fig, path)
        paths.append(path)

    top = report.vocab.top_tokens[:TOP_TOKEN_PLOT_N]
    if top:
        tokens = [token for token, _ in top][::-1]  # best at the top of the axis
        counts = [count for _, count in top][::-1]
        fig, ax = _new_axes(plt, (8, 0.32 * len(tokens) + 1.6))
        ax.barh(range(len(tokens)), counts, color=PLOT_SERIES,
                edgecolor=PLOT_SURFACE, linewidth=1.0, height=0.65)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        _style_axes(ax, f"高頻 Token（前 {len(tokens)}）", "次數", "", grid_axis="x")
        path = plots_dir / "top_tokens.png"
        _save(plt, fig, path)
        paths.append(path)

    return paths


def _benchmark_plots(result: dict, plots_dir: Path) -> list:
    plottable = [
        m for m in (result.get("models") or [])
        if not m.get("error")
        and isinstance(m.get("ranking_score"), (int, float))
        and not isinstance(m.get("ranking_score"), bool)
    ]
    if not plottable:
        return []
    plt = _plt()
    ordered = sorted(plottable,
                     key=lambda m: (float(m["ranking_score"]), str(m.get("name"))))
    names = [str(m.get("name", "—")) for m in ordered]
    scores = [float(m["ranking_score"]) for m in ordered]

    fig, ax = _new_axes(plt, (8, 0.55 * len(names) + 1.8))
    ax.barh(range(len(names)), scores, color=PLOT_SERIES,
            edgecolor=PLOT_SURFACE, linewidth=1.0, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    for i, score in enumerate(scores):
        ax.text(score, i, f" {score:.4f}", va="center", color=PLOT_INK, fontsize=9)
    ax.set_xlim(0, max(scores) * 1.15 if max(scores) > 0 else 1.0)
    _style_axes(ax, "模型排名分數（ranking_score）", "ranking_score", "", grid_axis="x")
    path = plots_dir / "ranking_scores.png"
    _save(plt, fig, path)
    return [path]
