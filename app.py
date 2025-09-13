import io, math, textwrap, calendar
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

st.set_page_config(page_title="RFM + K-Means — Painel Premium", layout="wide")

# ================== THEME / CSS ==================
st.markdown("""
<style>
:root { --accent: #4c78a8; } /* default, pode ser alterada via sidebar */
.small { font-size:0.85rem; color:#666; }
.kpi-box {
  background: linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(247,249,252,0.95) 100%);
  border: 1px solid rgba(0,0,0,0.06);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: 0 6px 16px rgba(0,0,0,.06);
}
.kpi-title { font-weight: 700; color:#334; font-size: .95rem; }
.kpi-value { font-weight: 900; font-size: 1.6rem; margin-top:2px; }
.kpi-sub { color:#667; font-size:.8rem; }
.badge {
  display:inline-block; padding:3px 8px; border-radius:999px;
  background: var(--accent); color:#fff; font-weight:700; font-size:.75rem;
}
.headerbar {
  padding: 10px 16px; border-radius: 14px; margin-bottom: 8px;
  background: linear-gradient(90deg, var(--accent) 0%, rgba(76,120,168,0.45) 60%, rgba(255,255,255,0) 100%);
  color:#fff;
}
.headerbar h1 { color:#fff !important; }
.card { border:1px solid rgba(0,0,0,.06); border-radius:14px; padding:12px 14px; }
</style>
""", unsafe_allow_html=True)

# Color palettes
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

    # Drop rows com NaT
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

    # weights: menor recency é melhor, então invertemos sinal para deixar comparável
    wr, wf, wm = weights
    Xw = pd.DataFrame({
        "recency": -wr * X["recency"],  # invertido
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

def kpi_box(title, value, sub=None):
    st.markdown(f"""
    <div class="kpi-box">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub or ""}</div>
    </div>
    """, unsafe_allow_html=True)

# ================== SIDEBAR ==================
st.sidebar.header("⚙️ Configurações")

accent_choice = st.sidebar.color_picker("Cor de destaque", "#4c78a8", help="Aplique a cor da sua marca (como no Power BI).")
st.markdown(f"<style>:root {{ --accent: {accent_choice}; }}</style>", unsafe_allow_html=True)

palette_name = st.sidebar.selectbox("Paleta de cores", list(PALETTES.keys()), index=0)
PALETTE = PALETTES[palette_name]

uploaded = st.sidebar.file_uploader("Base transacional (CSV/Excel)", type=["csv","xlsx"])
if uploaded is not None:
    data = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
else:
    st.info("Use o dataset de exemplo em `sample_data/sample_transactions.csv`.")
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
            if cand in cl:
                return c
    return None

customer_col = st.sidebar.selectbox("Customer ID", cols, index=(cols.index(guess(cols, ["customer","cliente","cust"])) if guess(cols, ["customer","cliente","cust"]) in cols else 0))
date_col     = st.sidebar.selectbox("Data da Compra", cols, index=(cols.index(guess(cols, ["date","data"])) if guess(cols, ["date","data"]) in cols else 0))
invoice_col  = st.sidebar.selectbox("Invoice (opcional)", [None]+cols, index=([None]+cols).index(guess(cols, ["invoice","fatura","nota","pedido","order_id"])) if guess(cols, ["invoice","fatura","nota","pedido","order_id"]) in cols else 0)
revenue_col  = st.sidebar.selectbox("Revenue (opcional)", [None]+cols, index=([None]+cols).index(guess(cols, ["revenue","amount","valor","total"])) if guess(cols, ["revenue","amount","valor","total"]) in cols else 0)
product_col  = st.sidebar.selectbox("Product ID (opcional)", [None]+cols, index=([None]+cols).index(guess(cols, ["product","sku","item"])) if guess(cols, ["product","sku","item"]) in cols else 0)

# Global filters (Power BI–style)
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

# ================== RFM ==================
rfm, anchor = compute_rfm(data_copy, customer_col, date_col, invoice_col, revenue_col)
X_scaled, scaler, mask_keep = transform_scale(rfm, use_log1p=use_log1p, na_strategy=na_strategy, weights=weights)
rfm_used = rfm.loc[mask_keep].copy()

# Fit e rotular
km, labels, sil, dbi, ch = evaluate_kmeans(X_scaled, k, random_state=random_state)
rfm_used["cluster"] = labels
label_map = auto_labels(rfm_used)
rfm_used["label"] = rfm_used["cluster"].map(label_map)

# KPIs (gauge + cards)
c1, c2, c3, c4 = st.columns([1.3,1,1,1])
with c1:
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=0 if sil is None or (isinstance(sil,float) and (np.isnan(sil) or np.isinf(sil))) else float(sil),
        title={"text":"Silhouette (↑ melhor)"},
        gauge={"axis":{"range":[0,1]}, "bar":{"color":accent_choice},
               "steps":[{"range":[0,0.2],"color":"#f8d7da"},
                        {"range":[0.2,0.35],"color":"#ffe9a6"},
                        {"range":[0.35,1],"color":"#d4edda"}]}
    ))
    fig_g.update_layout(height=180, margin=dict(l=10,r=10,t=30,b=0))
    st.plotly_chart(fig_g, use_container_width=True)
with c2: kpi_box("Clientes usados", f"{len(rfm_used):,}", f"Âncora: {anchor.date()}")
with c3: kpi_box("Clusters (k)", f"{k}", f"RandomState={random_state}")
with c4:
    top_cluster = rfm_used['cluster'].value_counts(normalize=True).sort_values(ascending=False).iloc[0]
    kpi_box("Maior cluster", f"{top_cluster*100:.1f}%", "participação")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["🏁 Resumo", "🧭 Perfis", "🔍 Explorar", "📈 Tendências", "🛒 Produtos", "⬇️ Exportar"])

with tab1:

    # 👉 Dicas de interpretação (no topo da aba Resumo)
    with st.expander("🧠 Dicas de interpretação", expanded=False):
        st.markdown("""
- **VIP Atual**: alto gasto e frequência, compras recentes → retenção, upsell.
- **VIP Dormindo**: alto gasto histórico, sem compras recentes → reativação.
- **Leal**: bom nível em geral → plano de fidelidade, cross-sell.
- **Dormindo/Churn**: pouco engajamento e muito tempo sem comprar → campanhas de resgate.
- **Oportunidade**: nutrição e ofertas de entrada.
""")

    with st.expander("🧠 Dicas de interpretação (Resumo)"):
        st.markdown("""
- **Fatia maior ≠ melhor**: combine % de cada rótulo com **valor** dos grupos nas abas seguintes.
- **Silhouette** no verde (≥ 0.35) sugere separação boa; amarelo (0.20–0.35) é aceitável; vermelho pede revisão.
- Compare **k**: prefira equilíbrio de tamanhos + coerência de perfis em vez de perseguir apenas Silhouette.
""")

    st.markdown("#### Distribuição por rótulo")
    lab_counts = rfm_used['label'].value_counts().reset_index()
    lab_counts.columns = ['label','n']
    figp = px.pie(lab_counts, values='n', names='label', hole=0.45,
                  color='label', color_discrete_sequence=PALETTE)
    figp.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(figp, use_container_width=True)

    if compare_k:
        # k2 comparison
        km2, labels2, sil2, dbi2, ch2 = evaluate_kmeans(X_scaled, k2, random_state=random_state)
        rfm_cmp = rfm_used.copy()
        rfm_cmp["cluster_k2"] = labels2
        label_map2 = auto_labels(rfm_cmp.rename(columns={"cluster_k2":"cluster"}).assign(cluster=rfm_cmp["cluster_k2"]))
        rfm_cmp["label_k2"] = rfm_cmp["cluster_k2"].map(label_map2)

        cA, cB = st.columns(2)
        with cA:
            st.markdown("**k atual**")
            st.write(f"Silhouette: **{sil:.3f}**")
            fig1 = px.pie(lab_counts, values='n', names='label', hole=0.45, color='label', color_discrete_sequence=PALETTE)
            st.plotly_chart(fig1, use_container_width=True)
        with cB:
            st.markdown("**k2 (comparação)**")
            st.write(f"Silhouette: **{sil2:.3f}**")
            lab2 = rfm_cmp['label_k2'].value_counts().reset_index()
            lab2.columns = ['label','n']
            fig2 = px.pie(lab2, values='n', names='label', hole=0.45, color='label', color_discrete_sequence=PALETTE)
            st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 Dicas de interpretação"):
        st.markdown("""
- **VIP Atual**: alto gasto e frequência, compras recentes → retenção, upsell.  
- **VIP Dormindo**: alto gasto histórico, sem compras recentes → reativação.  
- **Leal**: bom nível em geral → plano de fidelidade, cross-sell.  
- **Dormindo/Churn**: pouco engajamento e muito tempo sem comprar → campanhas de resgate.  
- **Oportunidade**: nutrição e ofertas de entrada.
""")

with tab2:

    with st.expander("🧠 Dicas de interpretação (Perfis)"):
        st.markdown("""
- **Recency (↓)**: quanto menor, mais recente — olhe clusters com R baixo para ações imediatas.
- **Frequency / Monetary (↑)**: use juntos para diferenciar **Leal** vs **VIP**.
- **Barras por cluster**: procure assimetria (ex.: R alto + F baixo → risco/churn).
- Use estes perfis para desenhar **campanhas** específicas por rótulo.
""")

    st.markdown("#### Perfis médios por cluster")
    prof = rfm_used.groupby(["cluster","label"])[["recency","frequency","monetary"]].mean().reset_index()
    prof["n_customers"] = rfm_used.groupby("cluster").size().values
    st.dataframe(prof.sort_values("cluster").style.format({"recency":"{:.1f}","frequency":"{:.2f}","monetary":"{:.2f}"}), use_container_width=True)

    # barras
    melted = prof.melt(id_vars=["cluster","label","n_customers"], value_vars=["recency","frequency","monetary"],
                       var_name="feature", value_name="value")
    figb = px.bar(melted, x="feature", y="value", color="label", barmode="group",
                  color_discrete_sequence=PALETTE, facet_col="cluster", facet_col_wrap=3,
                  title="Comparativo por cluster (médias)")
    st.plotly_chart(figb, use_container_width=True)

with tab3:

    with st.expander("🧠 Dicas de interpretação (Explorar)"):
        st.markdown("""
- **Dispersões**: densidade de pontos indica concentração do perfil.
- Filtre por **Monetary mínimo** para mapear clientes-chave nos rótulos.
- Use o **hover** (customer_id) para amostrar casos e validar se o grupo faz sentido.
""")

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
                          title="Recency vs Frequency", color_discrete_sequence=PALETTE)
        st.plotly_chart(fig2, use_container_width=True)
    with cB:
        fig3 = px.scatter(sub, x="monetary", y="frequency", color="label", hover_data=["customer_id"],
                          title="Monetary vs Frequency", color_discrete_sequence=PALETTE)
        st.plotly_chart(fig3, use_container_width=True)
    st.dataframe(sub.sort_values("monetary", ascending=False).head(1000), use_container_width=True)

with tab4:

    with st.expander("🧠 Dicas de interpretação (Tendências)"):
        st.markdown("""
- **Área empilhada**: observe sazonalidade e a **troca de mix** entre rótulos ao longo do tempo.
- Quedas em **VIP Atual** ou **Leal** pedem plano de retenção; alta em **Dormindo/Churn** sinaliza reativação.
- Combine com o **filtro de período** para zoom em campanhas/eventos.
""")

    st.markdown("#### Tendências (mês a mês)")
    if product_col and product_col in data_copy.columns:
        tx = data_copy.copy()
    else:
        tx = data_copy.copy()

    # Join labels por cliente
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
                   title="Receita por mês (empilhado por rótulo)")
    st.plotly_chart(figt, use_container_width=True)

with tab5:

    with st.expander("🧠 Dicas de interpretação (Produtos)"):
        st.markdown("""
- **Top por Receita**: priorize ofertas e disponibilidade destes itens.
- **Top por Quantidade**: bons para **bundles** e tíquete de entrada.
- Cruze com rótulos (aba Explorar/Tendências) para **cross-sell** direcionado.
""")

    st.markdown("#### Produtos (opcional)")
    if product_col and product_col in data_copy.columns:
        if revenue_col and revenue_col in data_copy.columns:
            prod_rev = data_copy.groupby(product_col)[revenue_col].sum().sort_values(ascending=False).head(20).reset_index()
            figp = px.bar(prod_rev, x=product_col, y=revenue_col, color=product_col, color_discrete_sequence=PALETTE,
                          title="Top 20 produtos por receita")
            st.plotly_chart(figp, use_container_width=True)
        qcols = [c for c in data_copy.columns if c.lower()=="quantity"]
        if qcols:
            qcol = qcols[0]
            prod_q = data_copy.groupby(product_col)[qcol].sum().sort_values(ascending=False).head(20).reset_index()
            figq = px.bar(prod_q, x=product_col, y=qcol, color=product_col, color_discrete_sequence=PALETTE,
                          title="Top 20 produtos por quantidade")
            st.plotly_chart(figq, use_container_width=True)
    else:
        st.info("Para habilitar esta aba, informe uma coluna de **Product ID** no mapeamento.")

with tab6:

    with st.expander("🧠 Dicas de interpretação (Exportar)"):
        st.markdown("""
- Use o **CSV** para ativar campanhas por rótulo (e medir uplift).
- O **Resumo Executivo** é o roteiro: contexto → k escolhido → insights → próximas ações.
- Registre a métrica de qualidade (Silhouette) para comparação futura.
""")

    st.markdown("#### Exportar")
    out = rfm_used.copy()
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar CSV com Segmentos", data=csv, file_name="clientes_segmentados.csv", mime="text/csv")

    sil_txt = f"{0 if sil is None or (isinstance(sil,float) and (np.isnan(sil) or np.isinf(sil))) else float(sil):.3f}"
    rotulos = ", ".join(sorted(rfm_used['label'].unique()))
    maior_grupo = rfm_used['label'].value_counts().idxmax()
    resumo = f"""# Resumo Executivo — Segmentação RFM
- Período filtrado: {data_copy[date_col].min().date()} → {data_copy[date_col].max().date()}
- Data-base (âncora): **{anchor.date()}**
- Clientes utilizados: **{len(rfm_used):,}**
- k: **{k}** | Silhouette: **{sil_txt}**
- Rótulos presentes: {rotulos}
- Maior grupo: **{maior_grupo}**

## Interpretação rápida
- **VIP Atual**: alto gasto, frequência alta, compras recentes → retenção e upsell.
- **VIP Dormindo**: alto gasto histórico, sem compras recentes → reativação.
- **Leal**: bom nível geral → fidelização e cross-sell.
- **Dormindo/Churn**: pouco engajamento → resgate/higienização.
- **Oportunidade**: nutrição e ofertas de entrada.
"""
    st.download_button("📝 Baixar Resumo Executivo (.md)", data=resumo.encode("utf-8"),
                       file_name="resumo_executivo.md", mime="text/markdown")

st.info("Dica: brinque com **Pesos R/F/M**, filtros de **período** e a **paleta/cor** para deixar o painel com a cara do seu trabalho.")
