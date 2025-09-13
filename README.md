
# Dashboard de Segmentação de Clientes (RFM + K-Means) — Versão Apresentável

Este projeto entrega um **painel Streamlit** pronto para apresentar e operar uma segmentação de clientes via **RFM** com **K-Means**, incluindo **modo apresentação**, **KPIs claros**, **rótulos automáticos** (VIP Atual, Leal, Dormindo/Churn, etc.), **gráficos intuitivos** (pizza, barras e radar), exportação de resultados e **resumo executivo** para download.

---

## 1) Visão Geral
- **Objetivo**: identificar grupos de clientes com comportamentos semelhantes para ações de CRM (retenção, reativação, upsell/cross-sell).
- **Base**: transações com `customer_id`, `order_date`, e (opcionalmente) `invoice_id`, `revenue`, `product_id`, `quantity`, `unit_price`.
- **Saídas**: clusters + rótulos, KPIs de qualidade (Silhouette/DBI/CH), perfis por grupo, CSV de clientes segmentados, markdown de resumo executivo.

---

## 2) Arquitetura do Projeto
```
rfm_dashboard/
├─ app.py                # Aplicativo Streamlit
├─ requirements.txt      # Dependências
├─ README.md             # Este guia
└─ sample_data/
   └─ sample_transactions.csv  # Dataset de exemplo
```

---

## 3) Como Executar
```bash
# (opcional) criar ambiente
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# instalar libs
pip install -r requirements.txt

# rodar
streamlit run app.py
```
Abra o link no terminal (ex.: `http://localhost:8501`). Para validar rapidamente, use `sample_data/sample_transactions.csv`.

---

## 4) Dados de Entrada (Mapeamento)
- **Obrigatório**: `customer_id`, `order_date`.
- **Recomendado**: `invoice_id` (conta frequência por pedido).
- **Monetário**: se **não** houver `revenue`, o app tenta derivar via `quantity * unit_price`.
- **Opcional**: `product_id` (habilita aba “Produtos”).
- **Datas**: o app tenta reconhecer automaticamente (dia/mês/ano ou ISO). Linhas com datas inválidas são removidas com aviso.

---

## 5) Metodologia Analítica
### 5.1 RFM
- **Âncora**: `anchor = max(order_date) + 1 dia`.
- **Recency**: `anchor - última_compra_cliente` (em dias) → **menor é melhor**.
- **Frequency**: nº de **invoice_id** distintos por cliente (fallback: nº de **dias de compra** distintos).
- **Monetary**: soma de `revenue` por cliente (fallback: tamanho do grupo).

### 5.2 Pré-processamento
- **Datas inválidas**: removidas (contabilizadas).
- **Faltantes/inf**: configurável na UI:
  - `drop`: remove linhas com NaN/inf.
  - `median`: imputa a **mediana** em cada métrica.
- **Outliers**: *clipping* em **P99** por coluna (R, F, M).
- **Transformação**: `log1p` (opcional) para reduzir assimetria.
- **Escalonamento**: `StandardScaler`.

### 5.3 Clusterização
- **Algoritmo**: `KMeans` (k ajustável), `k-means++`, `random_state` configurável.
- **Métricas**:
  - **Silhouette** (↑ melhor; regra de bolso: ≥ 0.35 = boa, ≥ 0.20 = moderada).
  - **Davies–Bouldin** (↓ melhor).
  - **Calinski–Harabasz** (↑ melhor).
- **Varredura k** (opcional): painel mostra um sweep k=2..8.

### 5.4 Rótulos Automáticos (heurística por quartis)
- Calcula médias R/F/M por cluster e compara com quartis globais para rotular:
  - **VIP Atual**: M ≥ Q3, F ≥ Q3, R ≤ Q1
  - **VIP Dormindo**: M ≥ Q3, R > Q2
  - **Leal**: M ≥ Q2, F ≥ Q2, R ≤ Q2
  - **Dormindo/Churn**: R ≥ Q3, F ≤ Q1
  - **Oportunidade**: caso geral
- **Observação**: é uma heurística explicável e simples de ajustar (ver Seção 9).

---

## 6) Guia da Interface
- **Sidebar**: upload da base, mapeamento de colunas, opções (`log1p`, faltantes, k, random_state) e **🎤 Modo apresentação**.
- **🏁 Resumo**: KPIs, pizza por rótulo, destaque automático e roteiro de fala.
- **🧭 Perfis**: tabela de médias por cluster/rótulo, **radar** (perfil polar) e barras comparativas.
- **🔍 Explorar**: filtros por rótulo com dispersões 2D (R×F, M×F) e tabela dos clientes.
- **🛒 Produtos**: top por receita/quantidade (se `product_id` estiver mapeado).
- **⬇️ Exportar**: CSV final + **Resumo Executivo (.md)** pronto para copiar/colar.

---

## 7) Interpretação & Ações (Playbook)
- **VIP Atual**: alto valor/recência/frequência → retenção, programas VIP, bundles premium.
- **VIP Dormindo**: alto valor histórico, sem compras recentes → campanhas de reativação, cupons alvo.
- **Leal**: bons níveis recentes, mas não top → fidelização, cross-sell complementar.
- **Dormindo/Churn**: pouca compra e muito tempo parado → campanhas de resgate; se não reativar, higienização.
- **Oportunidade**: base a desenvolver → onboarding guiado, ofertas de entrada, prova social.

---

## 8) Exportações
- **CSV** com: `customer_id`, R, F, M, `cluster`, `label`.
- **Resumo Executivo (.md)**: inclui âncora, k, Silhouette, rótulos presentes e maior grupo.

---

## 9) Como Personalizar
- **Pesos/normalização**: ajuste manual das colunas antes do K-Means (ex.: dar mais peso a `monetary`).
- **Regras de rótulo**: edite a função `auto_labels` no `app.py` mudando os limiares (Q1/Q2/Q3) ou adicionando regras de negócio.
- **Visual**: altere a paleta `CB_PALETTE` e o CSS no bloco `<style>` do `app.py`; substitua por cores da sua marca.
- **Algoritmo**: pode trocar `KMeans` por `GaussianMixture`, `MiniBatchKMeans` ou `HDBSCAN` (exige ajustes e novas métricas).

---

## 10) Qualidade & Validação — Checklist
- [ ] Mapeamento de colunas correto (ID, Data, Receita).  
- [ ] Datas reconhecidas (sem excesso de `NaT`).  
- [ ] RFM coerente (medianas/quantis plausíveis).  
- [ ] **Silhouette** x **equilíbrio** dos clusters validado.  
- [ ] Rótulos fazem sentido de **negócio**.  
- [ ] CSV e Resumo Executivo conferidos com stakeholders.

---

## 11) Exercícios
- **Básico**: suba sua base e descreva em 3–5 frases o perfil de cada rótulo.  
- **Intermediário**: compare k = 3..7, registre métricas e escolha k defendendo a decisão.  
- **Avançado**: altere as regras de rótulo para dar mais peso a `Monetary` e teste o impacto nas ações de CRM.

---

## 12) Resumo Técnico
RFM (Recency/Frequency/Monetary) → limpeza de datas → tratamento de faltantes (`drop` ou `median`) → clipping P99 → (opcional) `log1p` → `StandardScaler` → `KMeans` (k ajustável) → métricas (**Silhouette**, **DBI**, **CH**) → heurística de rótulos por quartis → painel com KPIs/visuais → exportações.

---

## 13) Troubleshooting
- **ValueError: NaN no KMeans**: selecione `median` ou revise mapeamento/colunas com muitos vazios.
- **Datas não reconhecidas**: normalize o formato ou garanta `dayfirst=True` se necessário.
- **Poucos clientes após limpeza**: revise filtros/mapeamento; evite k muito alto com amostras pequenas.
- **Gráficos vazios**: verifique se as colunas mapeadas existem/possuem dados.

---

## 14) Roadmap de Melhorias
- Pesos configuráveis por UI (ex.: sliders para R/F/M).  
- Comparação lado a lado entre dois valores de **k**.  
- Métricas adicionais (Dunn, XB) e **estabilidade** via reamostragem.  
- Exportação de imagens dos gráficos (PNG) e tema “dark”.  
- Integração com **Power BI / Looker Studio** via CSV padronizado.  
- Pipeline de **monitoramento de drift** e atualização periódica dos clusters.

---

## 15) Licença & Créditos
Uso acadêmico/empresarial livre; cite a fonte quando publicar os resultados.

---

## 16) Referências
- scikit-learn: KMeans, métricas (Silhouette, DBI, CH)  
- Literatura de RFM e segmentação de clientes (artigos e guias de marketing analytics)

---

## 17) Changelog
- 2025-09-13 — README estruturado; modo apresentação; rótulos automáticos; radar; resumo executivo; tratamento robusto de NaN/inf.
