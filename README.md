
# Dashboard de Segmentação — RFM + K-Means

Painel Streamlit **apresentável** e **didático** para segmentação via **RFM** + **K-Means**, agora com **todos os itens do roadmap implementados**:
- Pesos de **R/F/M** configuráveis por UI
- **Comparação de k** lado a lado
- Métricas extras (**Dunn**, **Xie–Beni**) e **estabilidade** por **ARI** (bootstraps)
- **Exportação de imagens (PNG)** dos gráficos
- **Tema dark** com um clique
- **Integração** com **Power BI / Looker Studio** via CSV padronizado
- **Monitoramento de drift** (PSI de R/F/M) + relatório JSON

---

## 1) Como Rodar
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Se quiser validar rápido, use `sample_data/sample_transactions.csv`.

---

## 2) Dados de Entrada
- **Obrigatório**: `customer_id`, `order_date`  
- **Recomendado**: `invoice_id`; `revenue` (ou o app deriva `quantity*unit_price`)  
- **Opcional**: `product_id` para habilitar a aba Produtos  
- Datas são parseadas automaticamente; linhas inválidas são descartadas.

---

## 3) Fluxo Analítico
1. **RFM** (Recency, Frequency, Monetary) a partir da base filtrada pelo **slicer de período**  
2. Tratamento: `drop`/`median`, **clipping P99**, `log1p` (opcional), **StandardScaler**  
3. **Pesos R/F/M** (Recency invertido)  
4. **K-Means** (k ajustável) → KPIs e **rótulos automáticos** por quartis  
5. **Métricas**: Silhouette, **Dunn**, **Xie–Beni**, **ARI** (estabilidade)  
6. Visuais **estilo Power BI** (donut, área empilhada, facets) com **download PNG**  
7. **Exportar/Integrar**: CSV padrão para BI (clientes, rótulos, período/âncora), perfis por cluster, transações rotuladas  
8. **Monitoramento**: PSI de R/F/M entre metades do período filtrado + **JSON** de relatório

---

## 4) Guia Rápido da Interface
- **Sidebar**: tema (light/dark), cor da marca, paleta, filtros de período, mapeamento e pesos R/F/M.  
- **🏁 Resumo**: KPIs (incl. Silhouette gauge), distribuição por rótulo, **Dunn**, **XB**, **ARI**, comparação k vs k2.  
- **🧭 Perfis**: médias por cluster/rótulo, **barras facetadas** por cluster.  
- **🔍 Explorar**: dispersões R×F / M×F com filtro por rótulo e Monetary mínimo.  
- **📈 Tendências**: área empilhada por rótulo (mês a mês).  
- **🛒 Produtos**: top por receita/quantidade (se `product_id`).  
- **⬇️ Exportar**: CSV padrão BI, perfis, transações rotuladas e **Resumo Executivo**.  
- **🧩 Roadmap**: agora marcado como **implementado**.  
- **⚙️ Monitoramento**: PSI para R/F/M + **JSON**.

---

## 5) Integração com Power BI / Looker
Use **clientes_segmentados_standard.csv** (schema abaixo) como **fonte**:
```
customer_id, recency, frequency, monetary, cluster, label, anchor_date, period_start, period_end
```
Opcionalmente, **transactions_labeled.csv** traz cada transação já com `cluster`/`label`.

---

## 6) Exportação de Imagens
Todos os gráficos principais têm botão **📸 Baixar PNG** (usa `kaleido`). Útil para slides e relatórios.

---

## 7) Monitoramento de Drift
A aba **⚙️ Monitoramento** calcula **PSI** para **R/F/M** comparando duas janelas do período filtrado.  
Regras de bolso: `0–0.1` estável · `0.1–0.25` alerta · `>0.25` drástico.  
Baixe o **monitoring_report.json** para histórico e auditoria.

---

## 8) Notas Técnicas
- **Dunn**: mínimo da distância interclusters / máximo diâmetro intracluster (amostragem se N>2000).  
- **Xie–Beni**: compacidade/ separação (↓ melhor).  
- **ARI (estabilidade)**: *bootstraps* com `predict` no conjunto completo e comparação ao rótulo base.  
- Tema **dark** troca CSS e `plotly` template.  
- Export PNG via `fig.to_image(format="png")`.

---

## 9) Troubleshooting
- **Sem kaleido** → `pip install kaleido` (já está no `requirements.txt`).  
- **Datas não reconhecidas** → padronize `YYYY-MM-DD` (ou deixe o parser dia/mês).  
- **Lento com bases muito grandes** → aumente `sample_cap` na função `dunn_index` ou reduza `n_boot` do ARI.

---

## 10) Licença
Uso acadêmico/empresarial livre — cite a fonte ao publicar.

---

### Observação sobre exportação de imagens (PNG)
- O painel usa **kaleido** para gerar PNG. Se o pacote não estiver instalado *no mesmo ambiente* do Streamlit, os botões de download **caem automaticamente para HTML** (arquivo interativo).
- Para habilitar PNG:
  - Windows (venv ativo): `pip install --upgrade kaleido`
  - macOS/Linux: `pip install --upgrade kaleido`
