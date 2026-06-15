"""
TruthSift — Interactive Results Dashboard
Run with: python dashboard.py
Then open: http://127.0.0.1:8050
"""

import pickle, os
import numpy as np
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score,
                             accuracy_score, precision_score, recall_score, f1_score)

# ── Load saved data ─────────────────────────────────────────────────────────
DATA_FILE = "dashboard_data.pkl"
assert os.path.exists(DATA_FILE), \
    f"Run the notebook cell 36 first to generate {DATA_FILE}"

with open(DATA_FILE, "rb") as f:
    d = pickle.load(f)

MODEL_DATA = d["MODEL_DATA"]
y_va       = d["y_va"]          # may be None when using hardcoded metrics
PAPER      = d["paper"]

MODELS       = list(MODEL_DATA.keys())
KERAS_MODELS = [m for m in MODELS if MODEL_DATA[m]["history"] is not None]
METRICS      = ["Accuracy", "Precision", "Recall", "F1-Score"]

PALETTE = {
    "RF":        "#2196F3",
    "SVM":       "#FF9800",
    "SimpleNN":  "#4CAF50",
    "RNN":       "#F44336",
    "LSTM":      "#9C27B0",
    "CNN":       "#00BCD4",
    "TruthSift": "#E91E63",
}

# ── Metric helpers ───────────────────────────────────────────────────────────
def get_our_metrics():
    """Return metrics dict, using pre-computed values when pred is None."""
    out = {}
    for name, md in MODEL_DATA.items():
        if md.get("metrics"):
            m = md["metrics"]
            out[name] = {
                "Accuracy":  round(m["accuracy"],  4),
                "Precision": round(m["precision"], 4),
                "Recall":    round(m["recall"],    4),
                "F1-Score":  round(m["f1"],        4),
            }
        elif md["pred"] is not None and y_va is not None:
            p = md["pred"]
            out[name] = {
                "Accuracy":  round(accuracy_score(y_va, p),                4),
                "Precision": round(precision_score(y_va, p),               4),
                "Recall":    round(recall_score(y_va, p),                  4),
                "F1-Score":  round(f1_score(y_va, p),                      4),
            }
        else:
            out[name] = {"Accuracy": 0, "Precision": 0, "Recall": 0, "F1-Score": 0}
    return out

OUR = get_our_metrics()

# ── Plot builders ────────────────────────────────────────────────────────────

def _dark_placeholder(title, message):
    """Return an empty dark figure with a centred message."""
    fig = go.Figure()
    fig.add_annotation(text=message, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color="#585b70", size=16))
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        height=420, margin=dict(l=60, r=40, t=70, b=60),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def make_confusion_matrix(model_name):
    md = MODEL_DATA[model_name]
    if md["pred"] is None or y_va is None:
        # Build synthetic confusion matrix from hardcoded metrics
        m   = md.get("metrics", {})
        acc = m.get("accuracy",  0.97)
        pre = m.get("precision", 0.97)
        rec = m.get("recall",    0.97)
        # Estimate totals (assume balanced-ish test set of 6050)
        N = 6050
        n_fake = int(N * 0.437)   # ~2646 fake in the notebook's test set
        n_real = N - n_fake
        tp = int(rec * n_fake)
        fn = n_fake - tp
        fp = int(tp * (1 - pre) / max(pre, 1e-9))
        tn = n_real - fp
        z = np.array([[tn, fp], [fn, tp]])[::-1]
    else:
        cm = confusion_matrix(y_va, md["pred"])
        z  = cm[::-1]

    xlbl = ["Fake (1)", "Real (0)"]
    ylbl = ["Real (0)", "Fake (1)"]
    text = [[str(v) for v in row] for row in z]
    fig  = ff.create_annotated_heatmap(
        z, x=xlbl, y=ylbl, annotation_text=text,
        colorscale="Blues", showscale=True)
    fig.update_layout(
        title=dict(text=f"{model_name} — Confusion Matrix  "
                        f"(Acc={OUR[model_name]['Accuracy']:.4f})",
                   font=dict(size=16)),
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4"),
        height=420, margin=dict(l=60, r=40, t=70, b=60),
    )
    return fig


def make_roc_curves(selected=None):
    selected = selected or MODELS
    # Only include models that have actual score arrays
    available = [m for m in selected
                 if MODEL_DATA[m]["score"] is not None and y_va is not None]
    if not available:
        return _dark_placeholder("ROC Curves",
                                 "ROC curves require prediction scores.<br>"
                                 "Re-run model training cells to generate scores.")
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="#585b70", dash="dash"))
    for name in available:
        md  = MODEL_DATA[name]
        fpr, tpr, _ = roc_curve(y_va, md["score"])
        roc_auc     = auc(fpr, tpr)
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{name}  AUC={roc_auc:.4f}",
            line=dict(color=PALETTE.get(name, "#cdd6f4"), width=2.5)))
    fig.update_layout(
        title=dict(text="ROC Curves", font=dict(size=16)),
        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
        paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
        font=dict(color="#cdd6f4"),
        legend=dict(bgcolor="#181825", bordercolor="#585b70", borderwidth=1),
        height=480, margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def make_pr_curves(selected=None):
    selected = selected or MODELS
    available = [m for m in selected
                 if MODEL_DATA[m]["score"] is not None and y_va is not None]
    if not available:
        return _dark_placeholder("Precision-Recall Curves",
                                 "PR curves require prediction scores.<br>"
                                 "Re-run model training cells to generate scores.")
    fig = go.Figure()
    for name in available:
        md  = MODEL_DATA[name]
        prec, rec, _ = precision_recall_curve(y_va, md["score"])
        ap = average_precision_score(y_va, md["score"])
        fig.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{name}  AP={ap:.4f}",
            line=dict(color=PALETTE.get(name, "#cdd6f4"), width=2.5)))
    fig.update_layout(
        title=dict(text="Precision-Recall Curves", font=dict(size=16)),
        xaxis_title="Recall", yaxis_title="Precision",
        paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
        font=dict(color="#cdd6f4"),
        legend=dict(bgcolor="#181825", bordercolor="#585b70", borderwidth=1),
        height=480, margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def make_training_curves(model_name):
    hist = MODEL_DATA[model_name]["history"]
    if hist is None:
        return _dark_placeholder(
            f"{model_name} — Training Curves",
            "Training history not available.<br>"
            "Re-run model training cells to generate curves.")
    epochs = list(range(1, len(hist["loss"]) + 1))
    fig    = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=hist["loss"], mode="lines+markers",
                             name="Train Loss", line=dict(color="#4C72B0", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=hist["val_loss"], mode="lines+markers",
                             name="Val Loss",   line=dict(color="#DD8452", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=epochs, y=hist["accuracy"], mode="lines+markers",
                             name="Train Acc",  line=dict(color="#55A868", width=2), yaxis="y2"))
    fig.add_trace(go.Scatter(x=epochs, y=hist["val_accuracy"], mode="lines+markers",
                             name="Val Acc",   line=dict(color="#C44E52", width=2, dash="dash"), yaxis="y2"))
    fig.update_layout(
        title=dict(text=f"{model_name} — Training Curves", font=dict(size=16)),
        xaxis_title="Epoch",
        yaxis=dict(title="Loss",     color="#4C72B0"),
        yaxis2=dict(title="Accuracy", color="#55A868", overlaying="y", side="right"),
        paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
        font=dict(color="#cdd6f4"),
        legend=dict(bgcolor="#181825", bordercolor="#585b70", borderwidth=1),
        height=450, margin=dict(l=60, r=60, t=60, b=60),
    )
    return fig


def make_comparison(metric_idx):
    metric = METRICS[metric_idx]
    models = MODELS
    our_v  = [OUR[m][metric] for m in models]
    pap_v  = [PAPER[m][metric_idx] if (PAPER[m][metric_idx] is not None) else 0
              for m in models]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Our Results", x=models, y=our_v,
        marker_color=[PALETTE.get(m, "#4C72B0") for m in models],
        text=[f"{v:.4f}" for v in our_v],
        textposition="outside", textfont=dict(size=11),
        opacity=0.9, width=0.35, offset=-0.18,
    ))
    pap_text = [f"{v:.4f}" if v else "—" for v in pap_v]
    fig.add_trace(go.Bar(
        name="Authors' Results", x=models, y=pap_v,
        marker_color="#DD8452",
        text=pap_text,
        textposition="outside", textfont=dict(size=11),
        opacity=0.9, width=0.35, offset=0.18,
    ))
    fig.update_layout(
        title=dict(text=f"{metric} — Our Results vs Authors'", font=dict(size=16)),
        barmode="overlay",
        yaxis=dict(range=[0.88, 1.03], title="Score"),
        paper_bgcolor="#1e1e2e", plot_bgcolor="#181825",
        font=dict(color="#cdd6f4"),
        legend=dict(bgcolor="#181825", bordercolor="#585b70", borderwidth=1),
        height=480, margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def make_overview_table():
    rows = []
    for name in MODELS:
        o = OUR[name]
        p = PAPER[name]
        rows.append({
            "Model":      name,
            "Our Acc":    f"{o['Accuracy']:.4f}",
            "Paper Acc":  f"{p[0]:.4f}" if p[0] else "NEW",
            "Our Prec":   f"{o['Precision']:.4f}",
            "Paper Prec": f"{p[1]:.4f}" if p[1] else "NEW",
            "Our Recall": f"{o['Recall']:.4f}",
            "Paper Rec":  f"{p[2]:.4f}" if p[2] else "NEW",
            "Our F1":     f"{o['F1-Score']:.4f}",
            "Paper F1":   f"{p[3]:.4f}" if p[3] else "NEW",
        })
    return rows


# ── App Layout ───────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="TruthSift Dashboard"
)

SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "220px", "padding": "20px 10px",
    "backgroundColor": "#181825",
    "borderRight": "1px solid #313244",
}
CONTENT_STYLE = {
    "marginLeft": "240px",
    "padding": "24px",
    "backgroundColor": "#1e1e2e",
    "minHeight": "100vh",
}

sidebar = html.Div([
    html.Div([
        html.H4("TruthSift", style={"color": "#E91E63", "fontWeight": "bold"}),
        html.P("Results Dashboard", style={"color": "#585b70", "fontSize": "13px", "marginTop": "-8px"}),
    ], style={"marginBottom": "28px", "paddingLeft": "8px"}),

    dbc.Nav([
        dbc.NavLink("📊  Overview",           href="/",           active="exact",
                    style={"color": "#cdd6f4", "marginBottom": "4px"}),
        dbc.NavLink("🔲  Confusion Matrix",   href="/confusion",  active="exact",
                    style={"color": "#cdd6f4", "marginBottom": "4px"}),
        dbc.NavLink("📉  Training Curves",    href="/training",   active="exact",
                    style={"color": "#cdd6f4", "marginBottom": "4px"}),
        dbc.NavLink("📈  ROC Curves",         href="/roc",        active="exact",
                    style={"color": "#cdd6f4", "marginBottom": "4px"}),
        dbc.NavLink("🎯  Precision-Recall",   href="/pr",         active="exact",
                    style={"color": "#cdd6f4", "marginBottom": "4px"}),
        dbc.NavLink("⚖️  Comparison",         href="/comparison", active="exact",
                    style={"color": "#cdd6f4", "marginBottom": "4px"}),
    ], vertical=True, pills=True),
], style=SIDEBAR_STYLE)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,
])

# ── Page renderers ───────────────────────────────────────────────────────────

def page_overview():
    rows = make_overview_table()
    ts_key = "TruthSift" if "TruthSift" in OUR else MODELS[-1]
    return html.Div([
        html.H3("📊 Results Overview", style={"color": "#cdd6f4", "marginBottom": "20px"}),
        html.P("Comparison of all models — our reproduced results vs authors' reported results.",
               style={"color": "#585b70", "marginBottom": "20px"}),

        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Best Our F1", style={"color": "#585b70"}),
                html.H3(f"{max(OUR[m]['F1-Score'] for m in MODELS):.4f}",
                        style={"color": "#E91E63", "fontWeight": "bold"}),
                html.P(max(MODELS, key=lambda m: OUR[m]["F1-Score"]),
                       style={"color": "#cdd6f4", "fontSize": "12px", "margin": 0}),
            ])], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Best Our Accuracy", style={"color": "#585b70"}),
                html.H3(f"{max(OUR[m]['Accuracy'] for m in MODELS):.4f}",
                        style={"color": "#2196F3", "fontWeight": "bold"}),
                html.P(max(MODELS, key=lambda m: OUR[m]["Accuracy"]),
                       style={"color": "#cdd6f4", "fontSize": "12px", "margin": 0}),
            ])], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("TruthSift F1", style={"color": "#585b70"}),
                html.H3(f"{OUR[ts_key]['F1-Score']:.4f}",
                        style={"color": "#E91E63", "fontWeight": "bold"}),
                html.P("Our Addition", style={"color": "#cdd6f4", "fontSize": "12px", "margin": 0}),
            ])], color="dark", outline=True), width=3),

            dbc.Col(dbc.Card([dbc.CardBody([
                html.H6("Models Evaluated", style={"color": "#585b70"}),
                html.H3(str(len(MODELS)),
                        style={"color": "#4CAF50", "fontWeight": "bold"}),
                html.P("incl. TruthSift", style={"color": "#cdd6f4", "fontSize": "12px", "margin": 0}),
            ])], color="dark", outline=True), width=3),
        ], style={"marginBottom": "28px"}),

        dash_table.DataTable(
            data=rows,
            columns=[{"name": c, "id": c} for c in rows[0].keys()],
            style_table={"overflowX": "auto"},
            style_header={"backgroundColor": "#313244", "color": "#cdd6f4",
                          "fontWeight": "bold", "textAlign": "center"},
            style_cell={"backgroundColor": "#181825", "color": "#cdd6f4",
                        "textAlign": "center", "padding": "10px",
                        "border": "1px solid #313244"},
            style_data_conditional=[
                {"if": {"filter_query": "{Model} = 'TruthSift'"},
                 "backgroundColor": "#2d1b33", "color": "#E91E63", "fontWeight": "bold"},
            ],
        )
    ])


def page_confusion():
    return html.Div([
        html.H3("🔲 Confusion Matrices", style={"color": "#cdd6f4", "marginBottom": "20px"}),
        html.P("Estimated from hardcoded metrics when actual predictions are unavailable.",
               style={"color": "#585b70", "marginBottom": "10px", "fontSize": "13px"}),
        dbc.Row([
            dbc.Col([
                html.Label("Select Model:", style={"color": "#cdd6f4"}),
                dcc.Dropdown(
                    id="cm-model-select",
                    options=[{"label": m, "value": m} for m in MODELS],
                    value=MODELS[0],
                    clearable=False,
                    style={"backgroundColor": "#313244", "color": "#1e1e2e"},
                )
            ], width=4)
        ], style={"marginBottom": "20px"}),
        dcc.Graph(id="cm-graph", style={"height": "460px"}),
    ])


def page_training():
    opts = KERAS_MODELS if KERAS_MODELS else MODELS
    return html.Div([
        html.H3("📉 Training Curves", style={"color": "#cdd6f4", "marginBottom": "20px"}),
        dbc.Row([
            dbc.Col([
                html.Label("Select Model:", style={"color": "#cdd6f4"}),
                dcc.Dropdown(
                    id="train-model-select",
                    options=[{"label": m, "value": m} for m in opts],
                    value=opts[0] if opts else None,
                    clearable=False,
                    style={"backgroundColor": "#313244", "color": "#1e1e2e"},
                )
            ], width=4)
        ], style={"marginBottom": "20px"}),
        dcc.Graph(id="train-graph", style={"height": "480px"}),
    ])


def page_roc():
    return html.Div([
        html.H3("📈 ROC Curves", style={"color": "#cdd6f4", "marginBottom": "20px"}),
        dbc.Row([
            dbc.Col([
                html.Label("Select Models:", style={"color": "#cdd6f4"}),
                dcc.Checklist(
                    id="roc-model-select",
                    options=[{"label": f"  {m}", "value": m} for m in MODELS],
                    value=MODELS,
                    inline=True,
                    style={"color": "#cdd6f4"},
                    inputStyle={"marginRight": "5px", "marginLeft": "15px"},
                )
            ])
        ], style={"marginBottom": "20px"}),
        dcc.Graph(id="roc-graph", style={"height": "520px"}),
    ])


def page_pr():
    return html.Div([
        html.H3("🎯 Precision-Recall Curves", style={"color": "#cdd6f4", "marginBottom": "20px"}),
        dbc.Row([
            dbc.Col([
                html.Label("Select Models:", style={"color": "#cdd6f4"}),
                dcc.Checklist(
                    id="pr-model-select",
                    options=[{"label": f"  {m}", "value": m} for m in MODELS],
                    value=MODELS,
                    inline=True,
                    style={"color": "#cdd6f4"},
                    inputStyle={"marginRight": "5px", "marginLeft": "15px"},
                )
            ])
        ], style={"marginBottom": "20px"}),
        dcc.Graph(id="pr-graph", style={"height": "520px"}),
    ])


def page_comparison():
    return html.Div([
        html.H3("⚖️ Our Results vs Authors'", style={"color": "#cdd6f4", "marginBottom": "20px"}),
        dbc.Row([
            dbc.Col([
                html.Label("Select Metric:", style={"color": "#cdd6f4"}),
                dcc.RadioItems(
                    id="cmp-metric-select",
                    options=[{"label": f"  {m}", "value": i} for i, m in enumerate(METRICS)],
                    value=0,
                    inline=True,
                    style={"color": "#cdd6f4"},
                    inputStyle={"marginRight": "5px", "marginLeft": "15px"},
                )
            ])
        ], style={"marginBottom": "20px"}),
        dcc.Graph(id="cmp-graph", style={"height": "520px"}),
    ])


# ── Routing callback ─────────────────────────────────────────────────────────
@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(path):
    if path == "/confusion":  return page_confusion()
    if path == "/training":   return page_training()
    if path == "/roc":        return page_roc()
    if path == "/pr":         return page_pr()
    if path == "/comparison": return page_comparison()
    return page_overview()

# ── Graph callbacks ──────────────────────────────────────────────────────────
@app.callback(Output("cm-graph",    "figure"), Input("cm-model-select",    "value"))
def update_cm(model):       return make_confusion_matrix(model)

@app.callback(Output("train-graph", "figure"), Input("train-model-select", "value"))
def update_train(model):    return make_training_curves(model)

@app.callback(Output("roc-graph",   "figure"), Input("roc-model-select",   "value"))
def update_roc(selected):   return make_roc_curves(selected)

@app.callback(Output("pr-graph",    "figure"), Input("pr-model-select",    "value"))
def update_pr(selected):    return make_pr_curves(selected)

@app.callback(Output("cmp-graph",   "figure"), Input("cmp-metric-select",  "value"))
def update_cmp(metric_idx): return make_comparison(metric_idx)


if __name__ == "__main__":
    print("\n🚀 TruthSift Dashboard running at: http://127.0.0.1:8050\n")
    app.run(debug=False, host="127.0.0.1", port=8050)
