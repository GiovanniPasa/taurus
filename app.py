import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------
# Dados de boi e bezerro
# -------------------------

arquivo = "dados.xlsx"

boi      = pd.read_excel(arquivo, sheet_name="BoiGordo")
bezerro  = pd.read_excel(arquivo, sheet_name="Bezerro")

boi.columns     = boi.columns.str.strip()
bezerro.columns = bezerro.columns.str.strip()

boi["Data"]     = pd.to_datetime(boi["Data"],     dayfirst=True)
bezerro["Data"] = pd.to_datetime(bezerro["Data"], dayfirst=True)

boi     = boi.rename(columns={"Arroba BRL": "boi_brl",     "Arroba USD": "boi_usd"})
bezerro = bezerro.rename(columns={"Arroba BRL": "bezerro_brl", "Arroba USD": "bezerro_usd"})

df = pd.merge(boi, bezerro, on="Data", how="inner")

# -------------------------
# Busca de benchmarks (API)
# -------------------------

@st.cache_data(ttl=86400, show_spinner="Buscando dados de benchmarks...")
def fetch_benchmarks(start="2000-01-01"):
    try:
        import yfinance as yf
        from bcb import sgs

        raw_ibov  = yf.download("^BVSP", start=start, auto_adjust=True, progress=False)
        raw_sp500 = yf.download("^GSPC", start=start, auto_adjust=True, progress=False)

        ibov  = raw_ibov["Close"].squeeze().rename("ibov")
        sp500 = raw_sp500["Close"].squeeze().rename("sp500")

        # CDI mensal — BCB SGS série 4391 (% ao mês)
        cdi_raw = sgs.get({"CDI": 4391}, start=start)
        cdi_raw["fator"] = 1 + cdi_raw["CDI"] / 100
        cdi_index = cdi_raw["fator"].cumprod().rename("cdi_index")

        # USD/BRL diário — BCB SGS série 1
        # A API do BCB limita séries diárias a janelas de 10 anos; busca em fatias e concatena.
        def _bcb_daily_chunked(code, name, start_str):
            s = pd.Timestamp(start_str)
            end = pd.Timestamp.today()
            partes = []
            while s < end:
                e = min(s + pd.DateOffset(years=9, months=6), end)
                chunk = sgs.get({name: code},
                                start=s.strftime("%Y-%m-%d"),
                                end=e.strftime("%Y-%m-%d"))
                if not chunk.empty:
                    partes.append(chunk[name])
                s = e + pd.Timedelta(days=1)
            return pd.concat(partes).drop_duplicates()

        usdbrl = _bcb_daily_chunked(1, "usdbrl", start).rename("usdbrl")

        # Reindexar para calendário contínuo (ffill preenche fins de semana/feriados)
        all_dates = pd.date_range(start=start, end=pd.Timestamp.today(), tz=None)

        def prep(s):
            """Remove timezone e normaliza para meia-noite antes de reindexar."""
            if getattr(s.index, "tz", None) is not None:
                s = s.tz_convert(None)
            s.index = s.index.normalize()
            return s.reindex(all_dates).ffill()

        ibov      = prep(ibov)
        sp500     = prep(sp500)
        usdbrl    = prep(usdbrl)
        cdi_index = prep(cdi_index)

        return ibov, sp500, usdbrl, cdi_index, None

    except Exception as e:
        return None, None, None, None, str(e)


# -------------------------
# Sidebar — Parâmetros
# -------------------------

st.sidebar.header("Parâmetros")

arrobas_bezerro      = st.sidebar.slider("Arrobas do bezerro",        5,   10,  7)
arrobas_boi          = st.sidebar.slider("Arrobas do boi gordo",      15,  25, 19)
ciclo_anos           = st.sidebar.slider("Tempo do ciclo (anos)",      1,   3,  2)
percentual_investidor = st.sidebar.slider("Parcela do investidor (%)", 10, 100, 35, step=5)

dias_ciclo = ciclo_anos * 365

moeda = st.sidebar.toggle("Mostrar valores em USD", value=False)

if moeda:
    preco_boi     = "boi_usd"
    preco_bezerro = "bezerro_usd"
    label_moeda   = "USD"
else:
    preco_boi     = "boi_brl"
    preco_bezerro = "bezerro_brl"
    label_moeda   = "BRL"

# -------------------------
# Simulação do ciclo
# -------------------------

# Lookup por data real (shift por linha deslocaria ~3 anos em vez de 2 com dados de dias úteis)
_boi_lookup = df[["Data", preco_boi]].rename(columns={"Data": "data_venda", preco_boi: "boi_futuro"})
df["data_venda"] = df["Data"] + pd.Timedelta(days=dias_ciclo)
df = pd.merge_asof(
    df.sort_values("data_venda"),
    _boi_lookup.sort_values("data_venda"),
    on="data_venda",
    direction="nearest",
    tolerance=pd.Timedelta(days=10),
)
df = df.drop(columns=["data_venda"]).sort_values("Data").reset_index(drop=True)

df["custo_total"]  = arrobas_bezerro * df[preco_bezerro]
df["receita_total"] = arrobas_boi    * df["boi_futuro"]

df["margem"]              = df["receita_total"] - df["custo_total"]
df["receita_investidor"]  = df["custo_total"] + df["margem"] * (percentual_investidor / 100)
df["retorno_ciclo"]       = (df["receita_investidor"] - df["custo_total"]) / df["custo_total"]
df["retorno_anual"]       = (1 + df["retorno_ciclo"]) ** (1 / ciclo_anos) - 1

df = df.dropna(subset=["retorno_anual"])

# -------------------------
# Benchmarks por janela
# -------------------------

ibov_s, sp500_s, usdbrl_s, cdi_idx, bench_erro = fetch_benchmarks()

benchmarks_ok = ibov_s is not None

if benchmarks_ok:
    df["sell_date"] = df["Data"] + pd.Timedelta(days=dias_ciclo)

    # Converte Ibovespa e S&P 500 para a moeda escolhida no toggle
    # USDBRL=X: quantos BRL valem 1 USD
    if moeda:   # USD → Ibov divide por USDBRL; SP500 já está em USD
        ibov_bench  = ibov_s / usdbrl_s
        sp500_bench = sp500_s
    else:       # BRL → Ibov já está em BRL; SP500 multiplica por USDBRL
        ibov_bench  = ibov_s
        sp500_bench = sp500_s * usdbrl_s

    def retorno_janela(serie, buy_dates, sell_dates):
        buy_idx  = pd.DatetimeIndex(buy_dates.values).normalize()
        sell_idx = pd.DatetimeIndex(sell_dates.values).normalize()
        buys  = serie.reindex(buy_idx,  method="ffill").values
        sells = serie.reindex(sell_idx, method="ffill").values
        return (sells / buys) - 1

    df["ret_ibov"]  = retorno_janela(ibov_bench,  df["Data"], df["sell_date"])
    df["ret_sp500"] = retorno_janela(sp500_bench, df["Data"], df["sell_date"])
    df["ret_cdi"]   = retorno_janela(cdi_idx,     df["Data"], df["sell_date"])

    df["ret_ibov_anual"]  = (1 + df["ret_ibov"])  ** (1 / ciclo_anos) - 1
    df["ret_sp500_anual"] = (1 + df["ret_sp500"]) ** (1 / ciclo_anos) - 1
    df["ret_cdi_anual"]   = (1 + df["ret_cdi"])   ** (1 / ciclo_anos) - 1

# -------------------------
# Estatísticas — Percentis
# -------------------------

st.header("📊 Estatísticas do Retorno")
st.caption(f"Parcela do investidor: {percentual_investidor}% do lucro")

percentis = df["retorno_anual"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("P5",      f"{percentis.iloc[0]*100:.1f}%")
col2.metric("P25",     f"{percentis.iloc[1]*100:.1f}%")
col3.metric("Mediana", f"{percentis.iloc[2]*100:.1f}%")
col4.metric("P75",     f"{percentis.iloc[3]*100:.1f}%")
col5.metric("P95",     f"{percentis.iloc[4]*100:.1f}%")

# -------------------------
# Taxa de superação
# -------------------------

st.subheader("% dos ciclos em que o boi superou o benchmark")

prob_loss = (df["retorno_ciclo"] < 0).mean()

if benchmarks_ok:
    df_valid  = df.dropna(subset=["ret_cdi_anual", "ret_ibov_anual", "ret_sp500_anual"])
    beat_cdi   = (df_valid["retorno_anual"] > df_valid["ret_cdi_anual"]).mean()
    beat_ibov  = (df_valid["retorno_anual"] > df_valid["ret_ibov_anual"]).mean()
    beat_sp500 = (df_valid["retorno_anual"] > df_valid["ret_sp500_anual"]).mean()

    bc1, bc2, bc3, bc4 = st.columns(4)
    bc1.metric("Bateu o CDI (BRL)",               f"{beat_cdi*100:.1f}%")
    bc2.metric(f"Bateu o Ibovespa ({label_moeda})", f"{beat_ibov*100:.1f}%")
    bc3.metric(f"Bateu o S&P 500 ({label_moeda})",  f"{beat_sp500*100:.1f}%")
    bc4.metric("Prob. de perda",   f"{prob_loss*100:.1f}%")
else:
    st.warning(f"Benchmarks indisponíveis (verifique a conexão): {bench_erro}")
    st.metric("Prob. de perda", f"{prob_loss*100:.1f}%")

# -------------------------
# Histograma
# -------------------------

st.header("Distribuição de Retornos")
bins = st.slider("Número de bins", 10, 100, 40)

fig = px.histogram(
    df, x="retorno_anual", nbins=bins,
    title=f"Distribuição de Retornos — {percentual_investidor}% do lucro ({label_moeda})"
)
fig.update_xaxes(tickformat=".0%")
st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Retorno anualizado do boi
# -------------------------

st.header("Retorno anualizado do ciclo pecuário")

fig2 = px.line(
    df, x="Data", y="retorno_anual",
    labels={"retorno_anual": "Retorno anualizado"},
    title=f"Retorno anualizado do ciclo ({arrobas_bezerro}@ → {arrobas_boi}@) — {percentual_investidor}% do lucro em {label_moeda}"
)
fig2.add_scatter(
    x=df["Data"],
    y=df[preco_bezerro],
    name=f"Arroba do bezerro ({label_moeda})",
    yaxis="y2",
    line=dict(color="darkorange", dash="dot"),
)
fig2.update_layout(
    yaxis=dict(tickformat=".0%", title="Retorno anualizado"),
    yaxis2=dict(
        title=f"Arroba do bezerro ({label_moeda})",
        overlaying="y",
        side="right",
        showgrid=False,
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig2.update_xaxes(title_text="Data de realização do aporte", range=["2000-01-01", df["Data"].max()])
st.plotly_chart(fig2, use_container_width=True)

# -------------------------
# Comparação com Benchmarks
# -------------------------

if benchmarks_ok:
    st.header("Comparação com Benchmarks (mesma janela de ciclo)")

    df_comp = df[["Data", "retorno_anual", "ret_cdi_anual", "ret_ibov_anual", "ret_sp500_anual"]].copy()
    df_comp = df_comp.rename(columns={
        "retorno_anual":   f"Boi ({percentual_investidor}% do lucro)",
        "ret_cdi_anual":   "CDI (BRL)",
        "ret_ibov_anual":  f"Ibovespa ({label_moeda})",
        "ret_sp500_anual": f"S&P 500 ({label_moeda})",
    })
    df_melt = df_comp.melt(id_vars="Data", var_name="Ativo", value_name="Retorno Anualizado")

    fig3 = px.line(
        df_melt, x="Data", y="Retorno Anualizado", color="Ativo",
        title=f"Retorno anualizado por janela de {ciclo_anos} ano(s)"
    )
    fig3.update_yaxes(tickformat=".0%")
    fig3.update_xaxes(title_text="Data de realização do aporte", range=["2000-01-01", df["Data"].max()])
    st.plotly_chart(fig3, use_container_width=True)

# -------------------------
# Auditoria de ciclo
# -------------------------

st.header("Auditoria de Ciclo")
st.caption("Selecione a data de venda do boi gordo para inspecionar o ciclo correspondente.")

# Datas disponíveis no dataset de boi gordo (original, sem dropna)
boi_audit = boi.set_index("Data").sort_index()
bezerro_audit = bezerro.set_index("Data").sort_index()

data_min = boi_audit.index.min().date()
data_max = boi_audit.index.max().date()

data_venda = st.date_input(
    "Data de venda do boi gordo",
    value=data_max,
    min_value=data_min,
    max_value=data_max,
)

data_venda_ts  = pd.Timestamp(data_venda)
data_compra_ts = data_venda_ts - pd.Timedelta(days=dias_ciclo)

# Encontrar data mais próxima disponível em cada série
def preco_mais_proximo(serie_df, coluna, data):
    idx = serie_df.index.get_indexer([data], method="nearest")[0]
    data_real = serie_df.index[idx]
    valor = serie_df[coluna].iloc[idx]
    return data_real, valor

data_venda_real,  arroba_boi_venda    = preco_mais_proximo(boi_audit,     preco_boi,     data_venda_ts)
data_compra_real, arroba_bezerro_compra = preco_mais_proximo(bezerro_audit, preco_bezerro, data_compra_ts)

valor_boi     = arrobas_boi      * arroba_boi_venda
custo_bezerro = arrobas_bezerro  * arroba_bezerro_compra
margem        = valor_boi - custo_bezerro
parcela_inv   = margem * (percentual_investidor / 100)
retorno_inv   = parcela_inv / custo_bezerro
ciclo_real    = (data_venda_real - data_compra_real).days / 365
ret_anual_inv = (1 + retorno_inv) ** (1 / ciclo_real) - 1 if ciclo_real > 0 else float("nan")

st.markdown(f"**Venda:** {data_venda_real.date()}  |  **Compra do bezerro:** {data_compra_real.date()} ({ciclo_real:.1f} anos de ciclo)")

col_v, col_b = st.columns(2)

with col_v:
    st.subheader("Venda")
    st.metric(f"Arroba do boi gordo ({label_moeda})", f"{arroba_boi_venda:,.2f}")
    st.metric(f"Valor do boi ({arrobas_boi}@)", f"{valor_boi:,.2f}")

with col_b:
    st.subheader(f"Compra (há {ciclo_anos} ano(s))")
    st.metric(f"Arroba do bezerro ({label_moeda})", f"{arroba_bezerro_compra:,.2f}")
    st.metric(f"Custo do bezerro ({arrobas_bezerro}@)", f"{custo_bezerro:,.2f}")

st.divider()

col_m1, col_m2, col_m3 = st.columns(3)
col_m1.metric("Margem bruta",                  f"{margem:,.2f} {label_moeda}", delta=f"{(margem/custo_bezerro)*100:.1f}% do custo")
col_m2.metric(f"Parcela do investidor ({percentual_investidor}%)", f"{parcela_inv:,.2f} {label_moeda}")
col_m3.metric("Retorno anualizado (investidor)", f"{ret_anual_inv*100:.1f}%")
