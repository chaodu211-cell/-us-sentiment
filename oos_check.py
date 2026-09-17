# -*- coding: utf-8 -*-
"""样本前检验：把**当前参数原样冻结**，跑在 2017-10 之前的历史上。

为什么需要它：仓库里所有标定数字都出自 2017-10 ~ 至今这一段，且"两段子样本同号"
在 SELL_W / CROWD_NARROW / LEV_W 三处都是**筛选条件**而非事后验证，所以后段数字
本身也参与过选参数。真正的样本外只能来自标定窗口之前的历史。

用法（必须在本机跑，云端容器的出口策略把行情源全挡了）：
    python3 fetch_sp500_yahoo.py --years 22      # 回补到 2004，raw/ 约 90MB
    python3 fetch_vix_dgs10.py                   # VIX / DGS10 / DFII10 也要够长
    python3 oos_check.py

本脚本**只读 raw/**，不写 data.json、不碰 dashboard.html，也不改任何参数。

【状态：未经端到端验证】提交它的容器里没有 raw/（43MB 不进 git，云端在 actions/cache 里），
所以只验证过三件事：语法、对 engine 的 15 个引用全部存在、没有 raw/ 时干净报错。
pipeline() 是照着 engine.main() 的接线手抄的一份，**存在与 main() 走样的风险**——
main() 以后改了接线，这里不会自动跟着改。第一次跑出结果后，请拿
2017-10 之后那段与线上 data.json 的 alerts.counts 对一遍：
    hot 66 / hot_bear 21 / cold 29 / cold_soft 13 / crowd 43
对不上就是这份抄写走样了，先修它再信 2011-2017 那段的数字。

—— 三条必须先读的限制，否则会把结论读歪 ——
1. 杠杆腿在 2010 年之前不存在。15 只杠杆 ETF 里 10 只是 2008-11 ~ 2010-03 才上市的
   （TQQQ 2010-02-11、SOXL/SOXS 2010-03、UPRO 2009-06，只有 SSO/QLD/SDS 是 2006）。
   加 252 日分位预热，杠杆温度最早约 2011 年年中才完整。而它在红点温度里占 2/9、
   在蓝点温度里占 1/6。所以 2006-2010 跑出来的**不是当前模型**，是一个降级变体
   （compose/compose_sell 会按剩余权重归一）。本脚本把 2011 前后分开报，别混着读。
2. 幸存者偏差在 2006-2017 上比在 2017-2026 上严重得多。用今天的 503 只成分股回溯到
   2006，等于把 2008 年破产/被并购/跌出指数的公司全部剔除——而那批公司正是当年
   宽度崩坏的主要贡献者。所以那几年的上涨占比/站上MA20/上涨拥挤度会系统性偏高
   （= 偏热），红点偏多、蓝点偏少。下面的 coverage 表把每年的有效样本数打出来。
3. 事件数才是有效样本量，不是天数。63 日前瞻窗口重叠的触发段算一个事件：
   样本内红点 66 天/23 段其实只有 8 个独立事件。日级均值的标准误按天算会严重低估。
"""
import argparse, os, sys, numpy as np, pandas as pd
import engine as E

_ap = argparse.ArgumentParser()
_ap.add_argument("--raw", default=None,
                 help="改用另一个原始数据目录（样本前检验用 raw_long，"
                      "由 fetch_long_history.py 抓出来）。默认用 engine 的 raw/。")
_ARGS, _ = _ap.parse_known_args()
if _ARGS.raw:
    # engine 的各 loader 都读模块级的 RAW，改它即可整体改向；不改 engine 源文件，
    # 免得日常流水线跟着受影响。
    E.RAW = os.path.abspath(_ARGS.raw)
    if not os.path.isdir(E.RAW):
        sys.exit(f"目录不存在：{E.RAW}")
    print(f"原始数据目录：{E.RAW}")

H = 63          # 前瞻窗口，与所有标定注释一致
SPLIT = "2017-10-01"   # 标定窗口起点：此前 = 样本前，此后 = 样本内


def pipeline():
    """完整复刻 engine.main() 的接线，但不截断、不落盘。"""
    rawdf, meta, spy, lev_info = E.build_indicators()
    dirs = E.direction(spy, rawdf.index)
    pct, adj, temp_raw, temp_adj = E.compose(rawdf, dirs)
    lev_pct, lev_temp = E.leverage_monitor(rawdf)
    temp_sell = E.compose_sell(adj, rawdf, lev_temp=lev_temp)
    have = temp_adj.dropna().index
    if len(have) == 0:
        sys.exit("温度序列为空：raw/ 里的历史不够长")

    fast_cols = {"turnover": "_turnover_daily", "advancing": "_advancing_daily",
                 "leverage": "_leverage_daily"}
    temp_fast = None
    if all(v in rawdf.columns for v in fast_cols.values()):
        rf = rawdf.copy()
        for k, v in fast_cols.items():
            rf[k] = rawdf[v]
        _, _, _, tf = E.compose(rf, dirs, fast=True)
        temp_fast = tf.reindex(have)

    vix = E.load_vix(have)
    crowd_pct = pd.DataFrame({c: E.rolling_pct(rawdf[c]).reindex(have)
                              for c in ["top2", "leverage"] if c in rawdf.columns})
    if "_topshare" in rawdf.columns:
        crowd_pct["topshare"] = E.rolling_pct(rawdf["_topshare"]).reindex(have)

    rr_full, nom_full = E.load_real_rate(), E.load_nominal_rate()
    rp_full = E.repricing_regime(rr_full, nom_full)
    rp = (None if rp_full is None
          else E.align_to(rp_full.astype(float), have).fillna(0).astype(bool))

    ndx = E.load("QQQ")
    al = E.build_alerts(temp_adj.reindex(have), adj.reindex(have), vix, temp_fast,
                        temp_sell=(temp_sell.reindex(have) if temp_sell is not None else None),
                        crowd_pct=crowd_pct if len(crowd_pct.columns) else None,
                        ndx=(ndx["close"].reindex(have) if ndx is not None else None),
                        narrow=(rawdf["_narrow_score"].reindex(have)
                                if "_narrow_score" in rawdf.columns else None),
                        repricing=rp)
    px = ndx["close"].reindex(have) if ndx is not None else None
    return have, al, px, rawdf, lev_temp


def episodes(idx_pos):
    """把触发日按 63 日不重叠聚成独立事件。"""
    eps = []
    for i in idx_pos:
        if eps and i - eps[-1][-1] <= H:
            eps[-1].append(i)
        else:
            eps.append([i])
    return eps


def report(dates, flags, px, lo, hi, tag):
    fwd = px.shift(-H) / px - 1
    win = (dates >= pd.Timestamp(lo)) & (dates <= pd.Timestamp(hi))
    if win.sum() == 0:
        print(f"\n{tag}: 窗口内无数据"); return
    base = fwd[win].dropna()
    print(f"\n{tag}  交易日 {int(win.sum())}  "
          f"基准 63 日 {base.mean()*100:+.2f}% / {(base<0).mean()*100:.0f}%为负")
    print(f"  {'信号':10s} {'天':>4s} {'段':>3s} {'事件':>4s} "
          f"{'日级均值':>9s} {'为负':>5s} {'事件级均值':>11s} {'最差事件':>9s}")
    for k, v in flags.items():
        m = pd.Series(v, index=dates) & win
        pos = np.where(m.values)[0]
        if len(pos) == 0:
            print(f"  {k:10s} {0:4d}   -    -          -     -           -         -")
            continue
        segs, eps = episodes_split(pos), episodes(pos)
        day = fwd.iloc[pos].dropna()
        first = fwd.iloc[[e[0] for e in eps]].dropna()
        print(f"  {k:10s} {len(pos):4d} {len(segs):3d} {len(eps):4d} "
              f"{day.mean()*100:+8.2f}% {(day<0).mean()*100:4.0f}% "
              f"{first.mean()*100:+10.2f}% {first.min()*100:+8.1f}%")


def episodes_split(pos):
    segs = []
    for i in pos:
        if segs and i == segs[-1][-1] + 1:
            segs[-1].append(i)
        else:
            segs.append([i])
    return segs


def coverage_by_year(lev_temp, dates):
    """每年有多少只成分股真有数据，以及杠杆温度哪年才算得出来。

    有效个股数是从 raw/ 的个股面板直接数的（当日收盘非空的只数），不依赖 engine
    内部的中间列。它同时暴露两件事：早年覆盖不足（读数噪声大），以及幸存者偏差的
    量级——数出来的 500 多只是**今天**的成分股，2008 年真实存在但已消失的公司
    一个都不在里面。"""
    stocks = E.load_many(E.STOCKS)
    px = pd.DataFrame({t: d["close"] for t, d in stocks.items()}).reindex(dates)
    n = px.notna().sum(axis=1)
    # 杠杆 ETF **当年真有几只在**——只报"杠杆温度有没有值"会骗人：
    # 15 只里 10 只是 2008-11~2010-03 才上市的，2007-2008 只有 SSO/QLD/SDS 三只
    # （做多 2 只、做空 1 只）。那种情况下多空比照样算得出数，notna 也是 100%，
    # 但它和用 10+5 只算出来的**不是同一个指标**。分母必须显性化。
    levs = E.load_many(E.LEV_ETFS)
    lpx = pd.DataFrame({t: d["close"] for t, d in levs.items()}).reindex(dates)
    nl = lpx.notna().sum(axis=1)
    nlong = lpx[[c for c in lpx.columns if c in E.LEV_LONG]].notna().sum(axis=1)
    print("\n分年可用性（看清哪几年的读数根本不该信）")
    print(f"  {'年':6s} {'有效个股数':>10s} {'杠杆ETF只数':>12s} {'其中做多':>9s} {'杠杆温度有值':>12s}")
    for y in sorted({d.year for d in dates}):
        sel = [d for d in dates if d.year == y]
        lv = (f"{lev_temp.reindex(sel).notna().mean()*100:.0f}%"
              if lev_temp is not None else "0%")
        mark = "  ← 降级" if nl.reindex(sel).mean() < len(E.LEV_ETFS) * 0.8 else ""
        print(f"  {y:<6d} {n.reindex(sel).mean():10.0f} {nl.reindex(sel).mean():12.1f} "
              f"{nlong.reindex(sel).mean():9.1f} {lv:>12s}{mark}")


if __name__ == "__main__":
    dates, flags, px, rawdf, lev_temp = pipeline()
    if px is None:
        sys.exit("缺 raw/QQQ.csv，无法算前瞻收益")
    print(f"raw/ 覆盖 {dates[0].date()} ~ {dates[-1].date()}（{len(dates)} 个交易日）")
    coverage_by_year(lev_temp, dates)
    report(dates, flags, px, "2006-01-01", "2010-12-31", "【样本前 A：2006-2010】杠杆腿缺失，是降级变体")
    report(dates, flags, px, "2011-01-01", "2017-09-30", "【样本前 B：2011-2017】完整模型，这段才是真·样本外")
    report(dates, flags, px, SPLIT, "2026-12-31", "【样本内：2017-10 至今】标定窗口，用于对照")
