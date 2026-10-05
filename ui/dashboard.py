"""Arbiter cost dashboard: reads the JSONL request log. Run with `make dashboard`."""
import json
import os

import pandas as pd
import plotly.express as px
import streamlit as st

from arbiter.router import model_list, token_cost

TIERS = ["simple", "standard", "complex"]
COLORS = {"simple": "#2a9d8f", "standard": "#e9c46a", "complex": "#e76f51"}
LOG_PATH = os.environ.get("LOG_PATH", "logs/requests.jsonl")

st.set_page_config(page_title="Arbiter", layout="wide")


@st.cache_data(ttl=30)
def load_logs(path: str) -> pd.DataFrame:
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # torn line from a concurrent write
    df = pd.DataFrame(rows)
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def premium_cost(row) -> float:
    """What this request would have cost on the tier-3 model."""
    return token_cost(model_list()["tier3"], row["tokens_in"], row["tokens_out"])


st.title("Arbiter — routing cost dashboard")
df = load_logs(LOG_PATH)
if df.empty:
    st.info("No data yet. Send requests to /generate or run scripts/generate_sample_logs.py.")
    st.stop()

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total requests", f"{len(df):,}")
k2.metric("Total estimated cost", f"${df['cost_usd'].sum():.4f}")
k3.metric("Cache hit rate", f"{df['cache_hit'].mean():.1%}")
k4.metric("Avg cost / request", f"${df['cost_usd'].mean():.6f}")

c1, c2 = st.columns(2)
cost_by_tier = df.groupby("tier")["cost_usd"].sum().reindex(TIERS, fill_value=0).reset_index()
c1.subheader("Cost per tier")
c1.plotly_chart(
    px.bar(cost_by_tier, x="tier", y="cost_usd", color="tier", color_discrete_map=COLORS),
    use_container_width=True,
)
c2.subheader("Tier distribution")
c2.plotly_chart(
    px.pie(df, names="tier", color="tier", color_discrete_map=COLORS,
           category_orders={"tier": TIERS}),
    use_container_width=True,
)

c3, c4 = st.columns(2)
by_hour = df.assign(hour=df["timestamp"].dt.hour).groupby(["hour", "tier"]).size()
by_hour = by_hour.rename("requests").reset_index()
c3.subheader("Requests by hour of day (UTC)")
c3.plotly_chart(
    px.line(by_hour, x="hour", y="requests", color="tier", markers=True,
            color_discrete_map=COLORS),
    use_container_width=True,
)
lat = df.groupby("tier")["latency_ms"].quantile([0.5, 0.95]).unstack()
lat = lat.rename(columns={0.5: "p50", 0.95: "p95"}).reset_index().melt(
    id_vars="tier", var_name="percentile", value_name="latency_ms")
c4.subheader("Latency by tier")
c4.plotly_chart(
    px.bar(lat, x="tier", y="latency_ms", color="percentile", barmode="group",
           category_orders={"tier": TIERS}),
    use_container_width=True,
)

st.subheader("Cost savings vs all-premium (tier-3 model)")
live = df[~df["cache_hit"] & df["success"]]  # cache hits log 0 tokens, so can't be re-priced
if live.empty:
    st.caption("No LLM-served requests yet.")
else:
    live = live.assign(premium=live.apply(premium_cost, axis=1))
    table = live.groupby("tier").agg(actual=("cost_usd", "sum"), all_premium=("premium", "sum"))
    table = table.reindex([t for t in TIERS if t in table.index])
    table.loc["TOTAL"] = table.sum()
    table["savings_pct"] = (1 - table["actual"] / table["all_premium"]) * 100
    st.dataframe(
        table.style.format({"actual": "${:.6f}", "all_premium": "${:.6f}",
                            "savings_pct": "{:.1f}%"}),
        use_container_width=True,
    )
    st.caption("Excludes cache hits (they cost $0 and carry no token counts).")

st.subheader("Last 50 requests")
st.dataframe(df.sort_values("timestamp", ascending=False).head(50), use_container_width=True)
