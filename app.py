import json
from datetime import datetime
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="RFM + K-Means — Painel", layout="wide")

# ================== THEME / CSS (SEM DARK) ==================
def inject_theme(accent="#4c78a8"):
    st.markdown(f"""
    <style>
    :root {{ --accent: {accent}; }}
    .kpi-box {{
      background: linear-gradient(180deg, rgba(247,249,252,0.95) 0%, rgba(247,249,252,0.95) 100%);
      border: 1px solid rgba(0,0,0,0.06);
      border-radius: 16px;
      padding: 14px 16px;
      box-shadow: 0 6px 16px rgba(0,0,0,.06);
    }}
    .kpi-title {{ font-weight: 700; color:#334; font-size:.95rem; }}
    .kpi-value {{ font-weight: 900; font-size:1.6rem; margin-top:2px; color:#334; }}
    .kpi-sub   {{ color:#667; font-size:.8rem; }}
    .badge {{
      display:inline-block; padding:3px 8px; border-radius:999px;
      background: var(--accent); color:#fff; font-weight:700; font-size:.75rem;
    }}
    .headerbar {{
      padding:10px 16px; border-radius:14px; margin-bottom:8px;
      background: linear-gradient(90deg, var(--accent) 0%, rgba(76,120,168,0.45) 60%, rgba(255,255,255,0) 100%);
      color:#fff;
    }}
    .headerbar h1 {{ color:#fff !important; font-size:1.4rem; line-height:1.2; margin:0; }}
    .card {{ border:1px solid rgba(0,0,0,.06); border-radius:14px; padding:12px 14px; background:rgba(247,249,252,0.95); }}
    </style>
    """, unsafe_allow_html=True)

PALETTES = {
    "Azul (default)": px.colors.qualitative.Set2,
    "Vibrante": px.colors.qualitative.Vivid,
    "Pastel": px.colors.qualitative.Pastel,
    "D3": px.colors.qualitative.D3
}

# ================== HELPERS ==================
@st.cache_data
def try_parse_dates(s):
    try:
        return pd.to_datetime(s, dayfirst=True, errors="coerce")
    except Exception:
        return pd.to_datetime(s, errors="coerce")

def clip_p99(df_in, cols):
    df = df_in.copy()
    for c in cols:
        try:
            p99 = df[c].quantile(0.99)
            df[c] = np.clip(df[c], None, p99)
        except Exception:
            pass
    return df

def compute_rfm(transactions, customer_col, date_col, invoice_col=None, revenue_col=None):
    tx = transactions.copy()
    tx[date_col] = try_parse_dates(tx[date_col])
    tx = tx[~tx[date_col].isna()]

    # revenue derivation if needed
    if revenue_col is None or revenue_col not in tx.columns:
        qty_cols = [c for c in tx.columns if c.lower() in ("quantity","qty","qtd")]
        price_cols = [c for c in tx.columns if "price" in c.lower() or "unit_price" in c.lower() or "preco" in c.lower()]
        if qty_cols and price_cols:
            revenue_col = "__revenue_temp__"
            tx[revenue_col] = pd.to_numeric(tx[qty_cols[0]], errors="coerce").fillna(0).astype(float) * \
                              pd.to_numeric(tx[price_cols[0]], errors="coerce").fillna(0).astype(float)

    if revenue_col and revenue_col in tx.columns:
        tx[revenue_col] = pd.to_numeric(tx[revenue_col], errors="coerce").fillna(0).astype(float)

    max_date = tx[date_col].max()
    anchor = max_date + pd.Timedelta(days=1)

    # Frequency
    if invoice_col and invoice_col in tx.columns:
        freq = tx.groupby(customer_col)[invoice_col].nunique().rename("frequency")
    else:
        freq = tx.groupby(customer_col)[date_col].nunique().rename("frequency")

    # Monetary
    if revenue_col and revenue_col in tx.columns:
        mon = tx.groupby(customer_col)[revenue_col].sum().rename("monetary").fillna(0.0)
    else:
        mon = tx.groupby(customer_col).size().rename("monetary").astype(float)

    # Recency
    last_purchase = tx.groupby(customer_col)[date_col].max()
    rec = (anchor - last_purchase).dt.days.rename("recency")

    rfm = pd.concat([rec, freq, mon], axis=1).reset_index().rename(columns={customer_col: "customer_id"})
    return rfm, anchor

def make_weights(r, f, m):
    s = max(r+f+m, 1e-9)
    return r/s, f/s, m/s

def transform_scale(rfm_df, use_log1p=True, na_strategy="drop", weights=(1,1,1)):
    X = rfm_df[["recency", "frequency", "monetary"]].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    if na_strategy == "median":
        X = X.fillna(X.median(numeric_only=True))
        mask_keep = X.index
    else:
        X = X.dropna()
        mask_keep = X.index

    X = clip_p99(X, X.columns.tolist())
    if use_log1p:
        X = np.log1p(X)

    # weights (recency invertida)
    wr, wf, wm = weights
    Xw = pd.DataFrame({
        "recency": -wr * X["recency"],
        "frequency": wf * X["frequency"],
        "monetary": wm * X["monetary"]
    }, index=X.index)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xw)
    return X_scaled, scaler, mask_keep

def evaluate_kmeans(X, k, random_state=42, n_init="auto"):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if k > 1 else np.nan
    dbi = davies_bouldin_score(X, labels) if k > 1 else np.nan
    ch  = calinski_harabasz_score(X, labels) if k > 1 else np.nan
    return km, labels, sil, dbi, ch

def dunn_index(X, labels, sample_cap=2000):
    X = np.asarray(X)
    if X.shape[0] > sample_cap:
        rng = np.random.RandomState(0)
        idx = rng.choice(X.shape[0], sample_cap, replace=False)
        X = X[idx]; labels = np.asarray(labels)[idx]
    D = squareform(pdist(X))
    uniq = np.unique(labels)
    # max intra
    diam = 0.0
    for c in uniq:
        idx = np.where(labels == c)[0]
        if len(idx) > 1:
            diam = max(diam, np.max(D[np.ix_(idx, idx)]))
    # min inter
    delta = np.inf
    for i, ci in enumerate(uniq):
        for cj in uniq[i+1:]:
            idx_i = np.where(labels == ci)[0]
            idx_j = np.where(labels == cj)[0]
            dist = np.min(D[np.ix_(idx_i, idx_j)])
            delta = min(delta, dist)
    if diam == 0 or not np.isfinite(delta):
        return np.nan
    return float(delta / diam)

def xie_beni_index(X, labels, centers):
    X = np.asarray(X); centers = np.asarray(centers)
    sq_dists = np.sum((X - centers[labels])**2, axis=1).sum()
    cd = squareform(pdist(centers)); np.fill_diagonal(cd, np.inf)
    min_center_dist2 = np.min(cd**2)
    if min_center_dist2 == 0 or not np.isfinite(min_center_dist2):
        return np.nan
    return float(sq_dists / (X.shape[0] * min_center_dist2))

def stability_bootstrap_ARIs(X, base_labels, k, n_boot=10, sample_frac=0.8, random_state=42):
    rng = np.random.RandomState(random_state)
    aris = []
    for _ in range(n_boot):
        idx = rng.choice(np.arange(X.shape[0]), size=max(2, int(sample_frac*X.shape[0])), replace=True)
        km_b = KMeans(n_clusters=k, random_state=rng.randint(0, 1_000_000), n_init="auto").fit(X[idx])
        pred_full = km_b.predict(X)
        aris.append(adjusted_rand_score(base_labels, pred_full))
    return float(np.mean(aris)), float(np.std(aris)), aris

def auto_labels(rfm_with_clusters):
    q = {c: rfm_with_clusters[c].quantile([0.25,0.5,0.75]).to_dict()
         for c in ["recency","frequency","monetary"]}
    prof = rfm_with_clusters.groupby("cluster")[["recency","frequency","monetary"]].mean()
    labels = {}
    for cl, row in prof.iterrows():
        r, f, m = row["recency"], row["frequency"], row["monetary"]
        if (m >= q["monetary"][0.75]) and (r <= q["recency"][0.25]) and (f >= q["frequency"][0.75]):
            labels[cl] = "VIP Atual"
        elif (m >= q["monetary"][0.75]) and (r > q["recency"][0.5]):
            labels[cl] = "VIP Dormindo"
        elif (m >= q["monetary"][0.5]) and (f >= q["frequency"][0.5]) and (r <= q["recency"][0.5]):
            labels[cl] = "Leal"
        elif (r >= q["recency"][0.75]) and (f <= q["frequency"][0.25]):
            labels[cl] = "Dormindo/Churn"
        else:
            labels[cl] = "Oportunidade"
    return labels

def kpi_box(title, value, sub=None):
    st.markdown(f"""
    <div class="kpi-box">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub or ""}</div>
    </div>
    """, unsafe_allow_html=True)

def fig_download_button(fig, filename, label):
    """Download PNG if kaleido exists; otherwise HTML fallback."""
    try:
        import kaleido  # noqa
        buf = fig.to_image(format="png", scale=2)
        st.download_button(label, data=buf, file_name=filename, mime="image/png")
    except Exception:
        html = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
        st.download_button(label + " (HTML)", data=html.encode("utf-8"),
                           file_name=filename.replace(".png",".html"), mime="text/html")
        st.caption("Para PNG, instale 'kaleido' no mesmo ambiente do Streamlit.")

# ================== SIDEBAR ==================
st.sidebar.header("⚙️ Configurações")
accent_choice = st.sidebar.color_picker("Cor de destaque", "#4c78a8")
inject_theme(accent_choice)
pio.templates.default = "plotly_white"

palette_name = st.sidebar.selectbox("Paleta de cores", list(PALETTES.keys()), index=0)
PALETTE = PALETTES[palette_name]

uploaded = st.sidebar.file_uploader("Base transacional (CSV/Excel)", type=["csv","xlsx"])
if uploaded is not None:
    data = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
else:
    st.info("Carregue sua base (ou use sample_data/sample_transactions.csv).")
    try:
        data = pd.read_csv("sample_data/sample_transactions.csv")
    except Exception:
        data = None
if data is None or data.empty:
    st.stop()

st.sidebar.subheader("Mapeamento")
cols = list(data.columns)
def guess(colnames, candidates):
    for c in colnames:
        cl = c.lower()
        for cand in candidates:
            if cand in cl: return c
    return None
customer_col = st.sidebar.selectbox("Customer ID", cols, index=(cols.index(guess(cols, ["customer","cliente","cust"])) if guess(cols, ["customer","cliente","cust"]) in cols else 0))
date_col     = st.sidebar.selectbox("Data da Compra", cols, index=(cols.index(guess(cols, ["date","data"])) if guess(cols, ["date","data"]) in cols else 0))
invoice_col  = st.sidebar.selectbox("Invoice (opcional)", [None]+cols, index=([None]+cols).index(guess(cols, ["invoice","fatura","nota","pedido","order_id"])) if guess(cols, ["invoice","fatura","nota","pedido","order_id"]) in cols else 0)
revenue_col  = st.sidebar.selectbox("Revenue (opcional)", [None]+cols, index=([None]+cols).index(guess(cols, ["revenue","amount","valor","total"])) if guess(cols, ["revenue","amount","valor","total"]) in cols else 0)
product_col  = st.sidebar.selectbox("Product ID (opcional)", [None]+cols, index=([None]+cols).index(guess(cols, ["product","sku","item"])) if guess(cols, ["product","sku","item"]) in cols else 0)

# Filtros globais estilo slicer
data_copy = data.copy()
data_copy[date_col] = try_parse_dates(data_copy[date_col])
dmin, dmax = data_copy[date_col].min().date(), data_copy[date_col].max().date()
date_range = st.sidebar.slider("Filtro de período", min_value=dmin, max_value=dmax, value=(dmin, dmax))
mask_date = (data_copy[date_col].dt.date >= date_range[0]) & (data_copy[date_col].dt.date <= date_range[1])
data_copy = data_copy.loc[mask_date].copy()

use_log1p   = st.sidebar.checkbox("Aplicar log1p (R,F,M)", True)
na_strategy = st.sidebar.selectbox("Faltantes", ["drop","median"], index=0)
st.sidebar.markdown("**Pesos R / F / M**")
wr = st.sidebar.slider("Peso Recency (↓)", 0.0, 3.0, 1.0, 0.1)
wf = st.sidebar.slider("Peso Frequency (↑)", 0.0, 3.0, 1.0, 0.1)
wm = st.sidebar.slider("Peso Monetary (↑)", 0.0, 3.0, 1.0, 0.1)
weights = make_weights(wr, wf, wm)

k = st.sidebar.slider("Clusters (k)", 2, 10, 4, 1)
random_state = st.sidebar.number_input("Random State", 0, 9999, 42, 1)
present_mode = st.sidebar.toggle("🎤 Modo apresentação", value=False)
compare_k = st.sidebar.toggle("Comparar com k2", value=False)
k2 = st.sidebar.slider("k2 (comparação)", 2, 10, 5, 1) if compare_k else None

# ================== HEADER ==================
st.markdown(f"""
<div class="headerbar">
  <h1>Segmentação de Clientes — RFM + K-Means</h1>
  <span class="badge">{palette_name}</span>
</div>
""", unsafe_allow_html=True)

# ================== MODELAGEM ==================
rfm, anchor = compute_rfm(data_copy, customer_col, date_col, invoice_col, revenue_col)
X_scaled, scaler, mask_keep = transform_scale(rfm, use_log1p=use_log1p, na_strategy=na_strategy, weights=weights)
rfm_used = rfm.loc[mask_keep].copy()

km, labels, sil, dbi, ch = evaluate_kmeans(X_scaled, k, random_state=random_state)
rfm_used["cluster"] = labels
label_map = auto_labels(rfm_used)
rfm_used["label"] = rfm_used["cluster"].map(label_map)

# Extra metrics
dunn = dunn_index(X_scaled, labels)
xb   = xie_beni_index(X_scaled, labels, km.cluster_centers_)
ari_mean, ari_std, _ = stability_bootstrap_ARIs(X_scaled, labels, k, n_boot=6, sample_frac=0.8, random_state=random_state)

# KPIs
c1, c2, c3, c4 = st.columns([1.3,1,1,1])
with c1:
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0 if (not isinstance(sil, (int,float)) or np.isnan(sil) or np.isinf(sil)) else float(sil),
        title={"text":"Silhouette (↑ melhor)"},
        gauge={"axis":{"range":[0,1]}, "bar":{"color":accent_choice},
               "steps":[{"range":[0,0.2],"color":"#f8d7da"},
                        {"range":[0.2,0.35],"color":"#ffe9a6"},
                        {"range":[0.35,1],"color":"#d4edda"}]}
    ))
    fig_g.update_layout(height=180, margin=dict(l=10,r=10,t=30,b=0), template=pio.templates.default)
    st.plotly_chart(fig_g, use_container_width=True)
    fig_download_button(fig_g, "kpi_silhouette.png", "📸 Baixar KPI (PNG)")
with c2: kpi_box("Clientes usados", f"{len(rfm_used):,}", f"Âncora: {anchor.date()}")
with c3: kpi_box("Clusters (k)", f"{k}", f"RandomState={random_state}")
with c4:
    top_cluster = rfm_used['cluster'].value_counts(normalize=True).sort_values(ascending=False).iloc[0]
    kpi_box("Maior cluster", f"{top_cluster*100:.1f}%", "participação")

# ================== TABS ==================
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🏁 Resumo", "🧭 Perfis", "🔍 Explorar", "📈 Tendências",
    "🛒 Produtos", "⬇️ Exportar", "🧩 Roadmap", "⚙️ Monitoramento"
])

with tab1:
    with st.expander("🧠 Dicas de interpretação", expanded=False):
        st.markdown("""
- **VIP Atual**: alto gasto e frequência, compras recentes → retenção, upsell.
- **VIP Dormindo**: alto gasto histórico, sem compras recentes → reativação.
- **Leal**: bom nível em geral → plano de fidelidade, cross-sell.
- **Dormindo/Churn**: pouco engajamento e muito tempo sem comprar → campanhas de resgate.
- **Oportunidade**: nutrição e ofertas de entrada.
""")
    st.markdown("#### Distribuição por rótulo")
    lab_counts = rfm_used['label'].value_counts().reset_index()
    lab_counts.columns = ['label','n']
    figp = px.pie(lab_counts, values='n', names='label', hole=0.45,
                  color='label', color_discrete_sequence=PALETTE, template=pio.templates.default)
    figp.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(figp, use_container_width=True)
    fig_download_button(figp, "resumo_distribuicao_rotulo.png", "📸 Baixar gráfico (PNG)")

    cA, cB, cC = st.columns(3)
    with cA: kpi_box("Dunn (↑)", f"{0 if not isinstance(dunn,(int,float)) or np.isnan(dunn) else dunn:.3f}")
    with cB: kpi_box("Xie-Beni (↓)", f"{0 if not isinstance(xb,(int,float)) or np.isnan(xb) else xb:.3f}")
    with cC: kpi_box("Estabilidade ARI", f"{ari_mean:.3f} ± {ari_std:.3f}", "bootstraps=6")

    if compare_k:
        km2, labels2, sil2, dbi2, ch2 = evaluate_kmeans(X_scaled, k2, random_state=random_state)

        # Frame LIMPO com uma única coluna 'cluster' para rotular k2
        tmp_k2 = rfm_used[["recency","frequency","monetary"]].copy()
        tmp_k2["cluster"] = labels2
        label_map2 = auto_labels(tmp_k2)

        rfm_cmp = rfm_used.copy()
        rfm_cmp["cluster_k2"] = labels2
        rfm_cmp["label_k2"] = rfm_cmp["cluster_k2"].map(label_map2)

        cX, cY = st.columns(2)
        with cX:
            st.markdown("**k atual**")
            st.write(f"Silhouette: **{sil:.3f}** | Dunn: **{dunn:.3f}** | XB: **{xb:.3f}**")
            fig1 = px.pie(lab_counts, values='n', names='label', hole=0.45, color='label',
                          color_discrete_sequence=PALETTE, template=pio.templates.default)
            st.plotly_chart(fig1, use_container_width=True)
            fig_download_button(fig1, "comparacao_k_atual.png", "📸 Baixar k atual (PNG)")
        with cY:
            st.markdown("**k2 (comparação)**")
            dunn2 = dunn_index(X_scaled, labels2)
            xb2   = xie_beni_index(X_scaled, labels2, km2.cluster_centers_)
            st.write(f"Silhouette: **{sil2:.3f}** | Dunn: **{dunn2:.3f}** | XB: **{xb2:.3f}**")
            lab2 = rfm_cmp['label_k2'].value_counts().reset_index()
            lab2.columns = ['label','n']
            fig2 = px.pie(lab2, values='n', names='label', hole=0.45, color='label',
                          color_discrete_sequence=PALETTE, template=pio.templates.default)
            st.plotly_chart(fig2, use_container_width=True)
            fig_download_button(fig2, "comparacao_k2.png", "📸 Baixar k2 (PNG)")

with tab2:
    st.markdown("#### Perfis médios por cluster")
    prof = rfm_used.groupby(["cluster","label"])[["recency","frequency","monetary"]].mean().reset_index()
    prof["n_customers"] = rfm_used.groupby("cluster").size().values
    st.dataframe(prof.sort_values("cluster").style.format({"recency":"{:.1f}","frequency":"{:.2f}","monetary":"{:.2f}"}), use_container_width=True)

    melted = prof.melt(id_vars=["cluster","label","n_customers"], value_vars=["recency","frequency","monetary"],
                       var_name="feature", value_name="value")
    figb = px.bar(melted, x="feature", y="value", color="label", barmode="group",
                  color_discrete_sequence=PALETTE, facet_col="cluster", facet_col_wrap=3,
                  title="Comparativo por cluster (médias)", template=pio.templates.default)
    st.plotly_chart(figb, use_container_width=True)
    fig_download_button(figb, "perfis_barras.png", "📸 Baixar gráfico (PNG)")

with tab3:
    st.markdown("#### Explorar clientes")
    colf1, colf2 = st.columns(2)
    with colf1:
        sel_label = st.selectbox("Rótulo", ["(todos)"] + sorted(rfm_used["label"].unique().tolist()))
    with colf2:
        min_m = st.number_input("Filtro: Monetary mínimo", min_value=0.0, value=0.0, step=10.0)
    sub = rfm_used.copy()
    if sel_label != "(todos)":
        sub = sub[sub["label"] == sel_label]
    sub = sub[sub["monetary"] >= min_m]

    cA, cB = st.columns(2)
    with cA:
        fig2 = px.scatter(sub, x="recency", y="frequency", color="label", hover_data=["customer_id"],
                          title="Recency vs Frequency", color_discrete_sequence=PALETTE, template=pio.templates.default)
        st.plotly_chart(fig2, use_container_width=True)
        fig_download_button(fig2, "explorar_rxF.png", "📸 Baixar R x F (PNG)")
    with cB:
        fig3 = px.scatter(sub, x="monetary", y="frequency", color="label", hover_data=["customer_id"],
                          title="Monetary vs Frequency", color_discrete_sequence=PALETTE, template=pio.templates.default)
        st.plotly_chart(fig3, use_container_width=True)
        fig_download_button(fig3, "explorar_mxF.png", "📸 Baixar M x F (PNG)")
    st.dataframe(sub.sort_values("monetary", ascending=False).head(1000), use_container_width=True)

with tab4:
    st.markdown("#### Tendências (mês a mês)")
    tx = data_copy.copy()
    tx[date_col] = try_parse_dates(tx[date_col])
    tx = tx[~tx[date_col].isna()]
    if revenue_col and revenue_col in tx.columns:
        tx["__rev__"] = pd.to_numeric(tx[revenue_col], errors="coerce").fillna(0).astype(float)
    else:
        if "quantity" in [c.lower() for c in tx.columns] and any(("price" in c.lower() or "unit" in c.lower()) for c in tx.columns):
            qcol = [c for c in tx.columns if c.lower()=="quantity"][0]
            pcol = [c for c in tx.columns if ("price" in c.lower() or "unit" in c.lower())][0]
            tx["__rev__"] = pd.to_numeric(tx[qcol], errors="coerce").fillna(0).astype(float) * pd.to_numeric(tx[pcol], errors="coerce").fillna(0).astype(float)
        else:
            tx["__rev__"] = 1.0
    tx = tx.merge(rfm_used[["customer_id","label"]], how="left", left_on=customer_col, right_on="customer_id")
    tx["month"] = tx[date_col].dt.to_period("M").dt.to_timestamp()
    rev_month = tx.groupby(["month","label"])["__rev__"].sum().reset_index()
    figt = px.area(rev_month, x="month", y="__rev__", color="label", color_discrete_sequence=PALETTE,
                   title="Receita por mês (empilhado por rótulo)", template=pio.templates.default)
    st.plotly_chart(figt, use_container_width=True)
    fig_download_button(figt, "tendencias_receita.png", "📸 Baixar gráfico (PNG)")

with tab5:
    st.markdown("#### Produtos (opcional)")
    if product_col and product_col in data_copy.columns:
        if revenue_col and revenue_col in data_copy.columns:
            prod_rev = data_copy.groupby(product_col)[revenue_col].sum().sort_values(ascending=False).head(20).reset_index()
            figp2 = px.bar(prod_rev, x=product_col, y=revenue_col, color=product_col, color_discrete_sequence=PALETTE,
                           title="Top 20 produtos por receita", template=pio.templates.default)
            st.plotly_chart(figp2, use_container_width=True)
            fig_download_button(figp2, "produtos_top_receita.png", "📸 Baixar gráfico (PNG)")
        qcols = [c for c in data_copy.columns if c.lower()=="quantity"]
        if qcols:
            qcol = qcols[0]
            prod_q = data_copy.groupby(product_col)[qcol].sum().sort_values(ascending=False).head(20).reset_index()
            figq = px.bar(prod_q, x=product_col, y=qcol, color=product_col, color_discrete_sequence=PALETTE,
                          title="Top 20 produtos por quantidade", template=pio.templates.default)
            st.plotly_chart(figq, use_container_width=True)
            fig_download_button(figq, "produtos_top_quantidade.png", "📸 Baixar gráfico (PNG)")
    else:
        st.info("Para habilitar esta aba, informe uma coluna de **Product ID** no mapeamento.")

with tab6:
    st.markdown("#### Exportar / Integrar")
    out = rfm_used.copy()
    out_std = out.assign(
        anchor_date=str(anchor.date()),
        period_start=str(date_range[0]),
        period_end=str(date_range[1])
    )[["customer_id","recency","frequency","monetary","cluster","label","anchor_date","period_start","period_end"]]
    csv_std = out_std.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV (padrão BI)", data=csv_std, file_name="clientes_segmentados_standard.csv", mime="text/csv")

    profiles = out.groupby(["cluster","label"])[["recency","frequency","monetary"]].mean().reset_index()
    csv_prof = profiles.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ CSV Perfis de Cluster", data=csv_prof, file_name="clusters_profiles.csv", mime="text/csv")

    if customer_col in data_copy.columns:
        tx_labeled = data_copy.copy().merge(out[["customer_id","cluster","label"]], how="left", left_on=customer_col, right_on="customer_id")
        csv_tx = tx_labeled.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ CSV Transações rotuladas", data=csv_tx, file_name="transactions_labeled.csv", mime="text/csv")

    sil_txt = f"{0 if (not isinstance(sil,(int,float)) or np.isnan(sil) or np.isinf(sil)) else float(sil):.3f}"
    rotulos = ", ".join(sorted(out['label'].unique()))
    maior_grupo = out['label'].value_counts().idxmax()
    resumo = f"""# Resumo Executivo — Segmentação RFM
- Período filtrado: {date_range[0]} → {date_range[1]}
- Data-base (âncora): **{anchor.date()}**
- Clientes utilizados: **{len(out):,}**
- k: **{k}** | Silhouette: **{sil_txt}** | Dunn: **{dunn:.3f}** | XB: **{xb:.3f}** | ARI: **{ari_mean:.3f} ± {ari_std:.3f}**
- Rótulos presentes: {rotulos}
- Maior grupo: **{maior_grupo}**
"""
    st.download_button("📝 Baixar Resumo Executivo (.md)", data=resumo.encode("utf-8"),
                       file_name="resumo_executivo.md", mime="text/markdown")

with tab7:
    st.markdown("### Roadmap (status)")
    st.markdown("""
- **Pesos configuráveis por UI** — ✅ implementado  
- **Comparação de k (k x k2)** — ✅ implementado  
- **Métricas (Dunn, Xie–Beni) e estabilidade (ARI)** — ✅ implementado  
- **Export PNG** (fallback HTML) — ✅ implementado  
- **Integração Power BI / Looker** — ✅ implementado  
- **Monitoramento (PSI)** — ✅ implementado  
""")

with tab8:
    st.markdown("### Monitoramento (Drift & Saúde dos clusters)")
    st.markdown("Mede **PSI** (Population Stability Index) entre metades do período para R, F e M.")
    def psi(ref, cur, bins=10, eps=1e-6):
        qs = np.linspace(0,1,bins+1)
        edges = np.unique(np.quantile(ref, qs))
        if edges.size < 2: return np.nan
        ref_hist, _ = np.histogram(ref, bins=edges)
        cur_hist, _ = np.histogram(cur, bins=edges)
        ref_p = ref_hist / max(ref_hist.sum(), eps)
        cur_p = cur_hist / max(cur_hist.sum(), eps)
        ref_p = np.clip(ref_p, eps, None); cur_p = np.clip(cur_p, eps, None)
        return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))
    midpoint = pd.to_datetime(str(date_range[0])) + (pd.to_datetime(str(date_range[1])) - pd.to_datetime(str(date_range[0]))) / 2
    winA_mask = try_parse_dates(data_copy[date_col]) <= midpoint
    winB_mask = try_parse_dates(data_copy[date_col]) >  midpoint
    def rfm_from_subset(df):
        rfm_s, _ = compute_rfm(df, customer_col, date_col, invoice_col, revenue_col)
        return rfm_s[["recency","frequency","monetary"]]
    rfm_A = rfm_from_subset(data_copy.loc[winA_mask]) if winA_mask.any() else rfm_used[["recency","frequency","monetary"]]
    rfm_B = rfm_from_subset(data_copy.loc[winB_mask]) if winB_mask.any() else rfm_used[["recency","frequency","monetary"]]
    psi_R = psi(rfm_A["recency"].values,   rfm_B["recency"].values)
    psi_F = psi(rfm_A["frequency"].values, rfm_B["frequency"].values)
    psi_M = psi(rfm_A["monetary"].values,  rfm_B["monetary"].values)
    cR, cF, cM = st.columns(3)
    with cR: kpi_box("PSI Recency", f"{psi_R:.3f}", "0–0.1 estável · 0.1–0.25 alerta · >0.25 drástico")
    with cF: kpi_box("PSI Frequency", f"{psi_F:.3f}", "0–0.1 estável · 0.1–0.25 alerta · >0.25 drástico")
    with cM: kpi_box("PSI Monetary", f"{psi_M:.3f}", "0–0.1 estável · 0.1–0.25 alerta · >0.25 drástico")
    report = {
        "period_start": str(date_range[0]),
        "period_end": str(date_range[1]),
        "anchor_date": str(anchor.date()),
        "k": int(k),
        "metrics": {
            "silhouette": None if (not isinstance(sil,(int,float)) or np.isnan(sil) or np.isinf(sil)) else float(sil),
            "dunn": None if (not isinstance(dunn,(int,float)) or np.isnan(dunn) or np.isinf(dunn)) else float(dunn),
            "xie_beni": None if (not isinstance(xb,(int,float)) or np.isnan(xb) or np.isinf(xb)) else float(xb),
            "ari_mean": float(ari_mean),
            "ari_std": float(ari_std),
            "psi": {"recency": float(psi_R), "frequency": float(psi_F), "monetary": float(psi_M)}
        }
    }
    st.download_button("⬇️ Relatório de monitoramento (JSON)", data=json.dumps(report, indent=2).encode("utf-8"),
                       file_name="monitoring_report.json", mime="application/json")

st.info("Dica: personalize **Pesos R/F/M**, **cor**, **paleta** e **filtros de período**. Exporte imagens (PNG) direto dos gráficos.")
