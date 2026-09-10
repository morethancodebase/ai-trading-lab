import pandas as pd
import numpy as np
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

df = pd.read_csv("reports/day9_trend_continuation/trade_journal.csv")
wr = lambda s: (s > 0).mean()
print("ROWS:", len(df))

print("\n=== EXIT REASON breakdown ===")
print(df.groupby("exit_reason").agg(
    n=("net_pnl", "size"), gross=("gross_pnl", "sum"),
    charges=("charges_total", "sum"), net=("net_pnl", "sum"),
    avgR=("r_multiple", "mean"), winrate=("net_pnl", wr)))

print("\n=== DIRECTION breakdown ===")
print(df.groupby("direction").agg(
    n=("net_pnl", "size"), gross=("gross_pnl", "sum"),
    charges=("charges_total", "sum"), net=("net_pnl", "sum"),
    avgR=("r_multiple", "mean"), winrate=("net_pnl", wr)))

print("\n=== R-multiple distribution ===")
print(df["r_multiple"].describe())
print("Winners avg R:", round(df[df.net_pnl > 0].r_multiple.mean(), 2),
      "Losers avg R:", round(df[df.net_pnl <= 0].r_multiple.mean(), 2))
n_win = (df.net_pnl > 0).sum()
print("Winners:", n_win, "Losers:", (df.net_pnl <= 0).sum())
print("Big winners (R>=2):", (df.r_multiple >= 2).sum(),
      " R>=3:", (df.r_multiple >= 3).sum(), " R>=5:", (df.r_multiple >= 5).sum())

print("\n=== Charges vs gross ===")
print("Total gross:", round(df.gross_pnl.sum(), 2),
      "Total charges:", round(df.charges_total.sum(), 2),
      "Net:", round(df.net_pnl.sum(), 2))
print("Avg charges/trade:", round(df.charges_total.mean(), 2),
      "Avg gross/trade:", round(df.gross_pnl.mean(), 2))

print("\n=== Winners vs losers charges ===")
print("Winner avg charges:", round(df[df.net_pnl > 0].charges_total.mean(), 2),
      "Loser avg charges:", round(df[df.net_pnl <= 0].charges_total.mean(), 2))
print("Winner avg turnover:", round((df[df.net_pnl > 0].buy_value + df[df.net_pnl > 0].sell_value).mean(), 2))
print("Loser avg turnover:", round((df[df.net_pnl <= 0].buy_value + df[df.net_pnl <= 0].sell_value).mean(), 2))

print("\n=== Qty distribution (oversized positions) ===")
print(df["qty"].describe())
print("Trades with qty>500:", (df.qty > 500).sum(),
      "| >1000:", (df.qty > 1000).sum(),
      "| >2000:", (df.qty > 2000).sum())
print("Net P&L from high-qty (>1000) trades:", round(df[df.qty > 1000].net_pnl.sum(), 2),
      " count:", (df.qty > 1000).sum())
print("Net P&L from low-qty (<=500) trades:", round(df[df.qty <= 500].net_pnl.sum(), 2),
      " count:", (df.qty <= 500).sum())

print("\n=== Setup flags (green / hammer) ===")
print(df.groupby(["green_setup", "hammer_setup"]).agg(
    n=("net_pnl", "size"), net=("net_pnl", "sum"),
    avgR=("r_multiple", "mean"), winrate=("net_pnl", wr)))

print("\n=== Entry-time analysis ===")
df["entry_hour_min"] = df["entry_time"].astype(str).str[11:16]
print(df.groupby("entry_hour_min").agg(
    n=("net_pnl", "size"), net=("net_pnl", "sum"),
    avgR=("r_multiple", "mean"), winrate=("net_pnl", wr)).head(20))

print("\n=== What if we cap notional? simulate cost impact ===")
# Turnover drives costs. Simulate capping qty so notional <= some cap.
df["notional"] = df[["buy_value", "sell_value"]].max(axis=1)
print("Notional quantiles:") 
print(df["notional"].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).round(0))
print("If we had 1/5th the turnover (cap qty), charges would be ~", round(df.charges_total.sum() / 5, 2),
      " net would be ~", round(df.gross_pnl.sum() - df.charges_total.sum() / 5, 2))
