import io, math, textwrap
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="RFM + K-Means — Painel", layout="wide")

# =============== THEME / CSS LIGHT STYLING ==================
st.markdown("""
<style>
.kpi-card {
  padding: 14px 16px; border-radius: 14px;
  border: 1px solid rgba(0,0,0,0.06);
  background: linear-gradient(180deg, rgba(250,250,250,0.9) 0%, rgba(245,247,250,0.9) 100%);
  box-shadow: 0 4px 14px rgba(0,0,0,0.06);
}
.kpi-title { font-weight: 600; font-size: 0.92rem; color: #444; }
.kpi-value { font-weight: 800; font-size: 1.4rem; margin-top: 2px; }
.kpi-sub   { font-size: 0.8rem; color: #777; }
.small-note { color:#666; font-size: 0.82rem; }
.section-title { font-size: 1.25rem; font-weight: 700; margin: 0.5rem 0 0.3rem 0; }
.big-number { font-size: 1.5rem; font-weight: 800; }
</style>
""", unsafe_allow_html=True)

CB_PALETTE = px.colors.qualitative.Set2

st.title("✨ Segmentação de Clientes — RFM + K-Means")
st.caption("Visual limpo, clusters com rótulos automáticos e um roteiro pronto para apresentação.")

# ----------------------------
# Helper functions
# ----------------------------
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

    # Drop rows with invalid dates
    n_before = len(tx)
    tx = tx[~tx[date_col].isna()]
    n_dropped_dates = n_before - len(tx)

    # revenue derivation if needed
    derived_revenue = False
    if revenue_col is None or revenue_col not in tx.columns:
        qty_cols = [c for c in tx.columns if c.lower() in ("quantity","qty","qtd")]
        price_cols = [c for c in tx.columns if "price" in c.lower() or "unit_price" in c.lower() or "preco" in c.lower()]
        if qty_cols and price_cols:
            revenue_col = "__revenue_temp__"
            tx[revenue_col] = pd.to_numeric(tx[qty_cols[0]], errors="coerce").fillna(0).astype(float) * \
                              pd.to_numeric(tx[price_cols[0]], errors="coerce").fillna(0).astype(float)
            derived_revenue = True

    # Clean revenue if exists
    if revenue_col and revenue_col in tx.columns:
        tx[revenue_col] = pd.to_numeric(tx[revenue_col], errors="coerce").fillna(0).astype(float)

    max_date = tx[date_col].max()
    anchor = max_date + pd.Timedelta(days=1)

    if invoice_col and invoice_col in tx.columns:
        freq = tx.groupby(customer_col)[invoice_col].nunique().rename("frequency")
    else:
        freq = tx.groupby(customer_col)[date_col].nunique().rename("frequency")

    if revenue_col and revenue_col in tx.columns:
        mon = tx.groupby(customer_col)[revenue_col].sum().rename("monetary").fillna(0.0)
    else:
        mon = tx.groupby(customer_col).size().rename("monetary").astype(float)

    last_purchase = tx.groupby(customer_col)[date_col].max()
    rec = (anchor - last_purchase).dt.days.rename("recency")

    rfm = pd.concat([rec, freq, mon], axis=1).reset_index().rename(columns={customer_col: "customer_id"})
    meta = {"dropped_invalid_dates": int(n_dropped_dates), "derived_revenue": bool(derived_revenue)}
    return rfm, anchor, meta

def transform_scale(rfm_df, use_log1p=True, na_strategy="drop"):
    X = rfm_df[["recency", "frequency", "monetary"]].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    if na_strategy == "median":
        X = X.fillna(X.median(numeric_only=True))
        dropped = 0
        kept = len(X)
        mask_keep = X.index
    else:
        before = len(X)
        X = X.dropna()
        kept = len(X)
        dropped = before - kept
        mask_keep = X.index

    X = clip_p99(X, X.columns.tolist())
    if use_log1p:
        X = np.log1p(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, dropped, kept, mask_keep

def evaluate_kmeans(X, k, random_state=42, n_init="auto"):
    km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels) if k > 1 else np.nan
    dbi = davies_bouldin_score(X, labels) if k > 1 else np.nan
    ch  = calinski_harabasz_score(X, labels) if k > 1 else np.nan
    return km, labels, sil, dbi, ch

def safe_number(n):
    return None if (n is None or (isinstance(n, float) and (math.isnan(n) or math.isinf(n)))) else float(n)

def auto_labels(rfm_with_clusters):
    q = {}
    for c in ["recency","frequency","monetary"]:
        q[c] = rfm_with_clusters[c].quantile([0.25,0.5,0.75]).to_dict()
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

def make_kpi(title, value, sub=None):
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub or ""}</div>
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# Sidebar — Upload & Mapping
# ----------------------------
st.sidebar.header("⚙️ Configurações")

uploaded = st.sidebar.file_uploader("Base transacional (CSV/Excel)", type=["csv","xlsx"], help="Contém customer_id, data de compra, invoice_id (opcional) e receita.")
if uploaded is not None:
    data = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
else:
    st.info("Use o dataset de exemplo em `sample_data/sample_transactions.csv` para começar rapidamente.")
    try:
        data = pd.read_csv("sample_data/sample_transactions.csv")
    except Exception:
        data = None

if data is None or data.empty:
    st.stop()

st.sidebar.subheader("Mapeamento de Colunas")
cols = list(data.columns)

def guess(colnames, candidates):
    for c in colnames:
        cl = c.lower()
        for cand in candidates:
            if cand in cl:
                return c
    return None

customer_col = st.sidebar.selectbox("Customer ID", options=cols, index=(cols.index(guess(cols, ["customer","cliente","cust"])) if guess(cols, ["customer","cliente","cust"]) in cols else 0))
date_col     = st.sidebar.selectbox("Data da Compra", options=cols, index=(cols.index(guess(cols, ["date","data"])) if guess(cols, ["date","data"]) in cols else 0))
invoice_col  = st.sidebar.selectbox("Invoice/Order ID (opcional)", options=[None]+cols, index=( [None]+cols ).index(guess(cols, ["invoice","fatura","nota","order_id","pedido"])) if guess(cols, ["invoice","fatura","nota","order_id","pedido"]) in cols else 0)
revenue_col  = st.sidebar.selectbox("Revenue (opcional)", options=[None]+cols, index=( [None]+cols ).index(guess(cols, ["revenue","amount","valor","total"])) if guess(cols, ["revenue","amount","valor","total"]) in cols else 0)
product_col  = st.sidebar.selectbox("Product ID (opcional)", options=[None]+cols, index=( [None]+cols ).index(guess(cols, ["product","sku","item"])) if guess(cols, ["product","sku","item"]) in cols else 0)

use_log1p    = st.sidebar.checkbox("Aplicar log1p (R, F, M)", value=True, help="Reduz assimetria e estabiliza variância.")
na_strategy  = st.sidebar.selectbox("Faltantes em RFM", ["drop", "median"], index=0, help="Remover linhas com NaN ('drop') ou imputar mediana ('median').")
k            = st.sidebar.slider("Número de clusters (k)", min_value=2, max_value=8, value=4, step=1)
random_state = st.sidebar.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)
present_mode = st.sidebar.toggle("🎤 Modo apresentação", value=False, help="Esconde detalhes técnicos para usar em reuniões.")

with st.expander("📚 Glossário rápido", expanded=False):
    st.markdown("""
- **Recency**: dias desde a última compra (quanto **menor**, melhor).
- **Frequency**: número de compras (quanto **maior**, melhor).
- **Monetary**: valor gasto (quanto **maior**, melhor).
- **Silhouette**: separação/coerência dos clusters (0 a 1, maior é melhor).
- **Davies–Bouldin**: compacidade/separação (menor é melhor).
- **Calinski–Harabasz**: separação relativa (maior é melhor).
""")

# ================== DATA PREVIEW ==================
if not present_mode:
    st.markdown('<div class="section-title">1) Prévia dos Dados</div>', unsafe_allow_html=True)
    st.dataframe(data.head(20), use_container_width=True)

# ================== RFM & TRANSFORM ==================
st.markdown('<div class="section-title">2) RFM & Pré-processamento</div>', unsafe_allow_html=True)
rfm, anchor, meta = compute_rfm(data, customer_col, date_col, invoice_col, revenue_col)
if meta["dropped_invalid_dates"] > 0 and not present_mode:
    st.warning(f"Foram removidas {meta['dropped_invalid_dates']} linhas com datas inválidas (NaT).")

X_scaled, scaler, dropped_rows, kept_rows, mask_keep = transform_scale(rfm, use_log1p=use_log1p, na_strategy=na_strategy)
if dropped_rows > 0 and na_strategy == "drop" and not present_mode:
    st.info(f"{dropped_rows} clientes removidos por RFM com NaN/inf antes da padronização.")
if kept_rows < 2:
    st.error("Após o tratamento de faltantes, restaram menos de 2 clientes. Ajuste o mapeamento/estratégia ou verifique sua base.")
    st.stop()

# alinhar RFM à amostra utilizada
rfm_used = rfm.loc[mask_keep].copy()

# ================== K-SCAN (optional) ==================
if not present_mode:
    with st.expander("🔎 Varredura rápida k=2..8 (métricas de cluster)"):
        scan_rows = []
        for kk in range(2, 9):
            try:
                km_s, labels_s, sil_s, dbi_s, ch_s = evaluate_kmeans(X_scaled, kk, random_state=random_state)
                scan_rows.append({"k": kk, "silhouette": sil_s, "davies_bouldin": dbi_s, "calinski_harabasz": ch_s})
            except Exception:
                scan_rows.append({"k": kk, "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan})
        scan_df = pd.DataFrame(scan_rows)
        st.dataframe(scan_df.style.format({"silhouette":"{:.4f}","davies_bouldin":"{:.4f}","calinski_harabasz":"{:.2f}"}), use_container_width=True)
        fig = px.line(scan_df, x="k", y="silhouette", markers=True, title="Silhouette por k")
        st.plotly_chart(fig, use_container_width=True)

# ================== FIT & LABELS ==================
km, labels, sil, dbi, ch = evaluate_kmeans(X_scaled, k, random_state=random_state)
rfm_used["cluster"] = labels
label_map = auto_labels(rfm_used)
rfm_used["label"] = rfm_used["cluster"].map(label_map)

# ================== TABS ==================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏁 Resumo", "🧭 Perfis", "🔍 Explorar", "🛒 Produtos", "⬇️ Exportar"])

with tab1:
    st.markdown("### Resumo executivo")
    c1, c2, c3, c4 = st.columns(4)
    with c1: make_kpi("Clientes usados", f"{len(rfm_used):,}", f"Âncora: {anchor.date()}")
    with c2: make_kpi("Silhouette", f"{safe_number(sil):.3f}" if safe_number(sil) is not None else "n/a", "↑ melhor")
    with c3: make_kpi("Clusters (k)", f"{k}", f"RandomState={random_state}")
    with c4:
        top_cluster = rfm_used['cluster'].value_counts(normalize=True).sort_values(ascending=False).iloc[0]
        make_kpi("Maior cluster", f"{top_cluster*100:.1f}%", "participação")

    top_lab = rfm_used.groupby("label").size().sort_values(ascending=False).index[0]
    sil_txt = f"{safe_number(sil):.3f}" if safe_number(sil) is not None else "n/a"
    qual = "boa" if (safe_number(sil) is not None and safe_number(sil) >= 0.35) else ("moderada" if (safe_number(sil) is not None and safe_number(sil) >= 0.2) else "baixa")
    st.markdown(f"**Destaque:** o cluster mais numeroso é **{top_lab}**. Silhouette de **{sil_txt}** indica **{qual}** separação.")

    lab_counts = rfm_used['label'].value_counts().reset_index()
    lab_counts.columns = ['label','n']
    figp = px.pie(lab_counts, values='n', names='label', title="Distribuição por rótulo de cluster", color='label', color_discrete_sequence=CB_PALETTE)
    st.plotly_chart(figp, use_container_width=True)

    with st.expander("Roteiro de apresentação (sugestão)"):
        st.markdown("""
1) Contexto: objetivo da segmentação e dados usados (R, F, M).  
2) Qualidade dos clusters: comente o **Silhouette** e o equilíbrio de tamanhos.  
3) Quem é nosso público principal? Mostre o maior rótulo (ex.: **VIP Atual**) e seu perfil.  
4) Ações: para cada rótulo, cite 1–2 iniciativas (ex.: reativação, cross-sell, retenção).  
5) Próximos passos: testar **k** alternativos e validar em performance de campanhas.
""")

with tab2:
    st.markdown("### Perfis médios por cluster (com rótulos)")
    prof = rfm_used.groupby(["cluster","label"])[["recency","frequency","monetary"]].mean().reset_index()
    prof['n_customers'] = rfm_used.groupby(["cluster"]).size().values

    st.dataframe(prof.sort_values("cluster").style.format({"recency":"{:.1f}","frequency":"{:.2f}","monetary":"{:.2f}"}), use_container_width=True)

    feats = ["recency","frequency","monetary"]
    norm = prof.copy()
    for c in feats:
        m, M = prof[c].min(), prof[c].max()
        if M > m:
            norm[c] = (prof[c]-m)/(M-m)
        else:
            norm[c] = 0.5

    for _, row in norm.iterrows():
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=[row[f] for f in feats],
                                        theta=["Recency (↓ melhor)","Frequency (↑)","Monetary (↑)"],
                                        fill="toself", name=f"Cluster {int(row['cluster'])} — {row['label']}"))
        fig_r.update_layout(title=f"Perfil Radar — Cluster {int(row['cluster'])} ({row['label']})",
                            showlegend=False)
        st.plotly_chart(fig_r, use_container_width=True)

    melted = prof.melt(id_vars=["cluster","label","n_customers"], value_vars=["recency","frequency","monetary"],
                       var_name="feature", value_name="value")
    figb = px.bar(melted, x="feature", y="value", color="label", barmode="group",
                  title="Comparativo de médias por rótulo", color_discrete_sequence=CB_PALETTE)
    st.plotly_chart(figb, use_container_width=True)

with tab3:
    st.markdown("### Explorar clientes por rótulo")
    sel_label = st.selectbox("Selecione um rótulo", sorted(rfm_used["label"].unique().tolist()))
    sub = rfm_used[rfm_used["label"] == sel_label].copy()
    st.write(f"Clientes no grupo **{sel_label}**: **{len(sub):,}**")
    sc1, sc2 = st.columns(2)
    with sc1:
        fig2 = px.scatter(sub, x="recency", y="frequency", color="label", title="Recency vs Frequency",
                          hover_data=["customer_id"], color_discrete_sequence=CB_PALETTE)
        st.plotly_chart(fig2, use_container_width=True)
    with sc2:
        fig3 = px.scatter(sub, x="monetary", y="frequency", color="label", title="Monetary vs Frequency",
                          hover_data=["customer_id"], color_discrete_sequence=CB_PALETTE)
        st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(sub.sort_values("monetary", ascending=False).head(1000), use_container_width=True)

with tab4:
    st.markdown("### Produtos (opcional)")
    if product_col and product_col in data.columns:
        if revenue_col and revenue_col in data.columns:
            prod_rev = data.groupby(product_col)[revenue_col].sum().sort_values(ascending=False).head(20).reset_index()
            figp = px.bar(prod_rev, x=product_col, y=revenue_col, title="Top 20 Produtos por Receita",
                          color=product_col, color_discrete_sequence=CB_PALETTE)
            st.plotly_chart(figp, use_container_width=True)
        qcols = [c for c in data.columns if c.lower()=="quantity"]
        if qcols:
            qcol = qcols[0]
            prod_q = data.groupby(product_col)[qcol].sum().sort_values(ascending=False).head(20).reset_index()
            figq = px.bar(prod_q, x=product_col, y=qcol, title="Top 20 Produtos por Quantidade",
                          color=product_col, color_discrete_sequence=CB_PALETTE)
            st.plotly_chart(figq, use_container_width=True)
    else:
        st.info("Para habilitar esta aba, informe uma coluna de **Product ID** no mapeamento.")

with tab5:
    st.markdown("### Exportar resultados")
    out = rfm_used.copy()
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV com Segmentos", data=csv, file_name="clientes_segmentados.csv", mime="text/csv")

    sil_txt = f"{safe_number(sil):.3f}" if safe_number(sil) is not None else "n/a"
    rotulos = ", ".join(sorted(rfm_used['label'].unique()))
    maior_grupo = rfm_used['label'].value_counts().idxmax()
    resumo = f"""# Resumo Executivo — Segmentação RFM
- Data-base: **{anchor.date()}**
- Clientes utilizados: **{len(rfm_used):,}**
- k: **{k}** | Silhouette: **{sil_txt}**
- Rótulos presentes: {rotulos}
- Maior grupo: **{maior_grupo}**

## Interpretação rápida dos rótulos
- **VIP Atual**: alto gasto, frequência alta, compras recentes → priorizar retenção e upsell.
- **VIP Dormindo**: alto gasto, mas sem compras recentes → campanhas de reativação.
- **Leal**: gasto e frequência medianos, recente → nutrir e fidelizar.
- **Dormindo/Churn**: pouca compra e muito tempo sem comprar → reativação ou higienização.
- **Oportunidade**: potencial a desenvolver com campanhas específicas.
"""
    st.download_button("📝 Baixar Resumo Executivo (.md)", data=resumo.encode("utf-8"), file_name="resumo_executivo.md", mime="text/markdown")

st.success("Pronto para apresentar! Use o 'Modo apresentação' na barra lateral para um visual ainda mais limpo.")
