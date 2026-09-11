# -*- coding: utf-8 -*-
"""独立复核：分位数、反向、等权、区间划分"""
import sys
import numpy as np, pandas as pd
from scipy import stats as sps
import engine as E

ok = True
def chk(name, cond, extra=""):
    global ok
    print(("  PASS  " if cond else "  FAIL  ") + name + ("  " + extra if extra else ""))
    if not cond: ok = False

print("1) 滚动分位 vs scipy.percentileofscore('mean')")
rng = np.random.default_rng(7)
s = pd.Series(rng.normal(size=600).cumsum())
mine = E.rolling_pct(s, window=252, min_periods=252)
# 独立实现：对每个 t，取 t-251..t 共252个点，算当前值的百分位
ref = []
for i in range(len(s)):
    if i < 251: ref.append(np.nan); continue
    w = s.values[i-251:i+1]
    ref.append(sps.percentileofscore(w, w[-1], kind="mean"))
ref = pd.Series(ref)
d = (mine - ref).abs().max()
chk("与 scipy 结果一致", d < 1e-9, f"最大偏差={d:.2e}")
chk("窗口=252：第252个点(idx251)才有值", (not np.isnan(mine.iloc[251])) and np.isnan(mine.iloc[250]))

print("2) 边界情形")
inc = pd.Series(np.arange(300, dtype=float))
p = E.rolling_pct(inc, 252, 252).dropna()
chk("单调递增序列分位恒为最高", np.allclose(p.values, (251+0.5)/252*100), f"值={p.iloc[0]:.4f}")
dec = pd.Series(np.arange(300, 0, -1, dtype=float))
p2 = E.rolling_pct(dec, 252, 252).dropna()
chk("单调递减序列分位恒为最低", np.allclose(p2.values, 0.5/252*100), f"值={p2.iloc[0]:.4f}")

print("3) ERP 反向 + 等权合成")
idx = pd.date_range("2024-01-01", periods=400, freq="B")
raw = pd.DataFrame({k: rng.normal(size=400).cumsum() for k in E.ORDER}, index=idx)
pct, adj, temp, temp_adj = E.compose(raw)
chk("输出6个指标", list(pct.columns) == E.ORDER, str(list(pct.columns)))
erp_plain = E.rolling_pct(raw["erp"])
chk("ERP 分位已取反 (100-pct)", np.allclose((100 - erp_plain).dropna(), pct["erp"].dropna()))
man = pct[E.ORDER].mean(axis=1, skipna=False)   # 引擎要求六项齐备，缺一不出温度
chk("温度=六项分位等权平均", np.allclose(man.dropna(), temp.dropna()))
chk("温度落在 0-100", float(temp.min()) >= 0 and float(temp.max()) <= 100, f"[{temp.min():.1f},{temp.max():.1f}]")
chk("无方向序列时修正=原始", np.allclose(temp.dropna(), temp_adj.dropna()))

print("3b) 方向修正")
dd = pd.Series(np.linspace(-1, 1, len(idx)), index=idx)
p2, a2, t2, t2a = E.compose(raw, dd)
for c in E.ORDER:
    if c in E.DIRECTIONAL:
        dev = (p2[c] - 50).clip(lower=0) if E.DIR_HALF else (p2[c] - 50)
        chk(f"{c} 已按方向定向", np.allclose((50 + dev*dd).dropna(), a2[c].dropna()))
    else:
        chk(f"{c} 未被改动", np.allclose(p2[c].dropna(), a2[c].dropna()))
chk("修正后温度仍在 0-100", float(t2a.min()) >= 0 and float(t2a.max()) <= 100, f"[{t2a.min():.1f},{t2a.max():.1f}]")
d0 = pd.Series(0.0, index=idx)
_, a0, _, _ = E.compose(raw, d0)
chk("方向=0 时活跃度记为中性50", all(np.allclose(a0[c].dropna(), 50.0) for c in E.DIRECTIONAL))

print("4) 任一指标缺失则不出温度")
raw2 = raw.copy(); raw2.loc[raw2.index[-1], "top2"] = np.nan
_, _, t3, _ = E.compose(raw2)
chk("末日有缺失 -> 温度为空", np.isnan(t3.iloc[-1]))

print("5) 情绪区间划分")
# 边界随 BANDS 走（参照系改动后已从 20/40/60/80 重标定为 28/46/61/76）
b = E.BANDS
names = ["极度恐慌", "偏冷", "中性", "偏热", "极度贪婪"]
cases = [(0, names[0]), (b[0]-0.1, names[0]), (b[0], names[1]), (b[1]-0.1, names[1]),
         (b[1], names[2]), (b[2]-0.1, names[2]), (b[2], names[3]), (b[3]-0.1, names[3]),
         (b[3], names[4]), (100, names[4])]
chk(f"十个边界值全部正确（边界 {'/'.join(f'{x:.0f}' for x in b)}）",
    all(E.regime(v)==lab for v,lab in cases),
    " ".join(f"{v}->{E.regime(v)}" for v,lab in cases if E.regime(v)!=lab) or "")

print("6) 新参照系")
chk("绝对锚点映射：中锚→50", abs(E.abs_map(pd.Series([53.0]), (35,53,70)).iloc[0] - 50) < 1e-9)
chk("绝对锚点映射：低于低锚截断为0", E.abs_map(pd.Series([10.0]), (35,53,70)).iloc[0] == 0)
chk("绝对锚点映射：高于高锚截断为100", E.abs_map(pd.Series([99.0]), (35,53,70)).iloc[0] == 100)
chk("绝对映射与年份无关（同值同读数）",
    E.abs_map(pd.Series([60.0]*2), (35,53,70)).nunique() == 1)
_e = E.expanding_pct(pd.Series(np.arange(1, 401, dtype=float)), min_periods=252)
chk("扩张分位：单调递增序列恒为最高", _e.dropna().min() > 99.0, f"min={_e.dropna().min():.2f}")
chk("扩张分位无前视：第252点才有值", _e.iloc[:251].isna().all() and np.isfinite(_e.iloc[251]))

print("\n" + ("全部通过" if ok else "存在失败项"))


# ---- 新增：贴现率重估状态 & 熊市状态 ----
def t_repricing():
    import pandas as pd, numpy as np, engine as E
    idx = pd.bdate_range("2020-01-01", periods=600)
    # 前 200 天在 -1.0 附近，随后线性升到 +1.5，再回落
    v = np.concatenate([np.full(200, -1.0), np.linspace(-1.0, 1.5, 250), np.linspace(1.5, -0.5, 150)])
    rr = pd.Series(v, index=idx)
    on = E.repricing_regime(rr)
    chk("重估状态：横盘期不触发", on.iloc[:200].sum() == 0)
    chk("重估状态：实际利率升过门槛后自动退出",
        on.iloc[E.RR_WIN + 260:E.RR_WIN + 300].sum() == 0)
    chk("重估状态：上行段内确实触发", on.sum() > 0, f"共 {int(on.sum())} 天")
    # 滞回：一旦进入，短暂回落不应立刻熄灭
    v2 = v.copy(); v2[300:315] = v2[300] - 0.30
    on2 = E.repricing_regime(pd.Series(v2, index=idx))
    chk("重估状态：滞回让短暂回落不熄灭", bool(on2.iloc[305]))
    chk("重估状态：无数据时返回 None", E.repricing_regime(None) is None)


def t_bear():
    import pandas as pd, numpy as np, engine as E
    idx = pd.bdate_range("2020-01-01", periods=500)
    up = pd.Series(np.linspace(100, 200, 500), index=idx)
    chk("熊市状态：单调上涨从不成立", not E.bear_regime(up).any())
    dn = pd.Series(np.linspace(200, 100, 500), index=idx)
    b = E.bear_regime(dn)
    chk("熊市状态：单调下跌在均线成型后恒成立", b.iloc[E.BEAR_MA + E.BEAR_SLOPE:].all())
    chk("熊市状态：均线未成型前不判定（无前视）", b.iloc[:E.BEAR_MA - 1].sum() == 0)


def t_alert_exclusive():
    import pandas as pd, numpy as np, engine as E
    idx = pd.bdate_range("2018-01-01", periods=800)
    t = pd.Series(np.random.RandomState(0).uniform(0, 100, 800), index=idx)
    px = pd.Series(np.linspace(300, 150, 800), index=idx)
    rr = pd.Series(np.linspace(-1.2, 0.2, 800), index=idx)
    vix = pd.Series(35.0, index=idx)
    al = E.build_alerts(t, pd.DataFrame(index=idx), vix=vix, temp_fast=t, ndx=px, real_rate=rr)
    chk("红点与熊市反弹互斥", not (al["hot"] & al["hot_bear"]).any())
    chk("实心蓝点与空心蓝点互斥", not (al["cold"] & al["cold_soft"]).any())


# ---- 新增：减仓温度（SELL_W 加权合成）----
def t_compose_sell():
    import pandas as pd, numpy as np, engine as E
    idx = pd.bdate_range("2020-01-01", periods=200)
    adj = pd.DataFrame({c: pd.Series(np.linspace(10, 90, 200), index=idx) for c in E.ORDER})
    raw = pd.DataFrame({"_narrow_neutral": pd.Series(np.linspace(90, 10, 200), index=idx)})
    t = E.compose_sell(adj, raw)
    # 手算：只有 SELL_W 里的项参与，权重归一
    w = E.SELL_W; tot = sum(w.values())
    man = sum((raw["_narrow_neutral"] if k == "narrow" else adj[k]) * v for k, v in w.items()) / tot
    chk("减仓温度=SELL_W 加权平均", np.allclose(t.dropna(), man.dropna()))
    chk("减仓温度不含 ERP 与上涨占比",
        "erp" not in E.SELL_W and "advancing" not in E.SELL_W, str(sorted(E.SELL_W)))
    chk("减仓温度落在 0-100", float(t.min()) >= 0 and float(t.max()) <= 100,
        f"[{t.min():.1f},{t.max():.1f}]")
    # 任一分项缺失 -> 当日不出温度（与综合温度同一约定）
    adj2 = adj.copy(); adj2.loc[adj2.index[-1], "top2"] = np.nan
    chk("末日分项缺失 -> 减仓温度为空", np.isnan(E.compose_sell(adj2, raw).iloc[-1]))
    raw2 = raw.copy(); raw2.loc[raw2.index[-1], "_narrow_neutral"] = np.nan
    chk("末日上涨拥挤度缺失 -> 减仓温度为空", np.isnan(E.compose_sell(adj, raw2).iloc[-1]))
    chk("拿不到上涨拥挤度时返回 None", E.compose_sell(adj, pd.DataFrame(index=idx)) is None)
    # 无前视：改动第 i 天之后的值，不能影响第 i 天的读数
    adj3 = adj.copy(); adj3.iloc[120:] = 5.0
    chk("减仓温度无前视", np.allclose(E.compose_sell(adj3, raw).iloc[:120].dropna(),
                                      t.iloc[:120].dropna()))


def t_narrow_neutral():
    """_narrow_neutral 的两层 where：下跌市填 50，历史不足保留 NaN。"""
    import pandas as pd, numpy as np
    idx = pd.bdate_range("2020-01-01", periods=10)
    sc = pd.Series([np.nan, np.nan, 70.0, 80.0, 90.0, 60.0, 55.0, 40.0, 30.0, 20.0], index=idx)
    up = pd.Series([True] * 5 + [False] * 5, index=idx)     # 后半段为下跌市
    out = sc.where(up, 50.0).where(sc.notna())              # 与 engine 中同一行表达式
    chk("上涨拥挤度中性版：历史不足处仍为 NaN", bool(out.iloc[:2].isna().all()))
    chk("上涨拥挤度中性版：上涨市保留原值", np.allclose(out.iloc[2:5], sc.iloc[2:5]))
    chk("上涨拥挤度中性版：下跌市填 50", np.allclose(out.iloc[5:], 50.0))


def t_alert_wiring():
    """红点走减仓温度、蓝点走 COLD_TH、黑框走 CROWD_TOP2/CROWD_NARROW —— 阈值确实被接上了。"""
    import pandas as pd, numpy as np, engine as E
    n = 60
    idx = pd.bdate_range("2021-01-01", periods=n)
    calm = pd.Series(50.0, index=idx)
    px = pd.Series(np.linspace(100, 160, n), index=idx)     # 上涨，排除熊市反弹干扰
    # 红点：主温度压在 50（远低于 BANDS[3]），只有减仓温度越过 SELL_TH
    ts = pd.Series(E.SELL_TH + 5, index=idx)
    al = E.build_alerts(calm, pd.DataFrame(index=idx), ndx=px, temp_sell=ts)
    chk("红点由减仓温度触发（主温度仅50）", bool(al["hot"].iloc[E.PERSIST:].all()))
    ts2 = pd.Series(E.SELL_TH - 5, index=idx)
    al2 = E.build_alerts(calm, pd.DataFrame(index=idx), ndx=px, temp_sell=ts2)
    chk("减仓温度低于门槛则不触发红点", not al2["hot"].any())
    # 退回旧行为：不给减仓温度时，红点看主温度 > BANDS[3]
    hot_old = E.build_alerts(pd.Series(E.BANDS[3] + 5, index=idx), pd.DataFrame(index=idx), ndx=px)
    chk("未提供减仓温度时退回主温度口径", bool(hot_old["hot"].iloc[E.PERSIST:].all()))
    # 蓝点：快温度落在 COLD_TH 两侧
    vix = pd.Series(float(E.VIX_COLD), index=idx)
    hit = E.build_alerts(calm, pd.DataFrame(index=idx), vix=vix,
                         temp_fast=pd.Series(E.COLD_TH - 1, index=idx), ndx=px)
    miss = E.build_alerts(calm, pd.DataFrame(index=idx), vix=vix,
                          temp_fast=pd.Series(E.COLD_TH + 1, index=idx), ndx=px)
    chk("蓝点门槛用 COLD_TH（低于则触发）", bool(hit["cold"].any()))
    chk("蓝点门槛用 COLD_TH（高于则不触发）", not miss["cold"].any())
    chk("VIX 差一点就不触发蓝点",
        not E.build_alerts(calm, pd.DataFrame(index=idx), vix=pd.Series(E.VIX_COLD - 0.1, index=idx),
                           temp_fast=pd.Series(E.COLD_TH - 1, index=idx), ndx=px)["cold"].any())
    # 黑框：TOP2 抱团分位 > CROWD_TOP2 且 上涨拥挤度评分 > CROWD_NARROW，两项刚好跨过门槛
    cp    = pd.DataFrame({"top2": pd.Series(E.CROWD_TOP2 + 1, index=idx)})
    cp_lo = pd.DataFrame({"top2": pd.Series(E.CROWD_TOP2 - 1, index=idx)})
    nr    = pd.Series(E.CROWD_NARROW + 1, index=idx)
    nr_lo = pd.Series(E.CROWD_NARROW - 1, index=idx)
    chk("黑框门槛用 CROWD_TOP2/CROWD_NARROW（越过则触发）",
        bool(E.build_alerts(calm, pd.DataFrame(index=idx), crowd_pct=cp, narrow=nr, ndx=px)["crowd"]
             .iloc[E.PERSIST:].all()))
    chk("黑框：上涨拥挤度不够则不触发",
        not E.build_alerts(calm, pd.DataFrame(index=idx), crowd_pct=cp,
                           narrow=nr_lo, ndx=px)["crowd"].any())
    chk("黑框：TOP2 抱团分位不够则不触发",
        not E.build_alerts(calm, pd.DataFrame(index=idx), crowd_pct=cp_lo,
                           narrow=nr, ndx=px)["crowd"].any())
    chk("黑框：拿不到上涨拥挤度时不触发",
        not E.build_alerts(calm, pd.DataFrame(index=idx), crowd_pct=cp, ndx=px)["crowd"].any())
    # 橙线已取消：规则表里不应再有它，build_alerts 也不应再产出该键
    chk("橙线预警已取消（ALERTS 中无 narrow）",
        not any(r["key"] == "narrow" for r in E.ALERTS))
    chk("橙线预警已取消（build_alerts 不产出 narrow）",
        "narrow" not in E.build_alerts(calm, pd.DataFrame(index=idx), crowd_pct=cp, narrow=nr, ndx=px))
    # 确认天数：第 PERSIST 天才置位，之前为假（无前视）
    chk(f"红点需连续 {E.PERSIST} 日才置位",
        (not al["hot"].iloc[:E.PERSIST - 1].any()) and bool(al["hot"].iloc[E.PERSIST - 1]))


for f in (t_repricing, t_bear, t_alert_exclusive, t_compose_sell, t_narrow_neutral, t_alert_wiring):
    f()
print("\n新增用例全部通过" if ok else "\n新增用例有失败")


# ---- 杠杆温度（leverage.py）----
def t_leverage():
    import pandas as pd, numpy as np, engine as E, leverage as LV

    idx = pd.bdate_range("2020-01-01", periods=400)
    def frame(px, vol):
        return pd.DataFrame({"close": px, "rawclose": px, "volume": vol}, index=idx)

    print("\n11) 杠杆ETF两条腿")
    # 做多 100×10=1000，做空 50×10=500，指数ETF 20×250=5000
    longs = {"TQQQ": frame(pd.Series(100.0, index=idx), pd.Series(10.0, index=idx))}
    shorts = {"SQQQ": frame(pd.Series(50.0, index=idx), pd.Series(10.0, index=idx))}
    base = {"SPY": frame(pd.Series(20.0, index=idx), pd.Series(250.0, index=idx))}
    raw, meta = LV.letf_components(idx, longs, shorts, base)
    chk("多空比 = 做多成交额/杠杆总成交额", np.allclose(raw["ratio"].dropna(), 1000/1500*100))
    chk("交易强度 = 杠杆总成交额/指数ETF成交额", np.allclose(raw["intensity"].dropna(), 1500/5000*100))
    chk("两条腿都给了未平滑当日值", {"ratio_daily", "intensity_daily"} <= set(raw))
    # 成交额用未复权价：拆股当天价格减半、成交量翻倍，成交额应连续
    px2 = pd.Series([100.0]*200 + [50.0]*200, index=idx)
    vol2 = pd.Series([10.0]*200 + [20.0]*200, index=idx)
    raw2, _ = LV.letf_components(idx, {"TQQQ": frame(px2, vol2)}, shorts, base)
    chk("拆股不制造跳变（成交额用未复权价×成交量）",
        abs(raw2["ratio_daily"].iloc[199] - raw2["ratio_daily"].iloc[200]) < 1e-9)
    # 缺做空篮子时只剩强度，且分子只含做多
    raw3, meta3 = LV.letf_components(idx, longs, {}, base)
    chk("缺做空篮子：不出多空比", "ratio" not in raw3)
    chk("缺做空篮子：强度退化为只含做多", np.allclose(raw3["intensity"].dropna(), 1000/5000*100))

    print("12) 杠杆温度的合成")
    p = pd.DataFrame({"ratio": pd.Series(80.0, index=idx), "intensity": pd.Series(20.0, index=idx)})
    w = LV.LEV_W
    man = (80.0 * w["ratio"] + 20.0 * w["intensity"]) / (w["ratio"] + w["intensity"])
    chk(f"按 LEV_W 加权（多空比:强度 = {w['ratio']:g}:{w['intensity']:g}）",
        np.allclose(LV.temperature(p).dropna(), man), f"读数={LV.temperature(p).iloc[-1]:.1f}")
    chk("方向腿权重更高（3:1，选法见 leverage.py 注释）", w["ratio"] > w["intensity"])
    p2 = p.copy(); p2.loc[p2.index[-1], "intensity"] = np.nan
    chk("缺一项时按剩余权重归一（不补 50）", abs(LV.temperature(p2).iloc[-1] - 80.0) < 1e-9)
    chk("两项全缺 -> 空", bool(LV.temperature(pd.DataFrame(index=idx)).isna().all()))
    chk("权重就是 LEV_W 里写的那两项", sorted(LV.LEV_W) == ["intensity", "ratio"])
    # 多空比那一腿在 leverage.py 里算、在 engine 里被当成第六个分项显示，两边的平滑窗口
    # 必须一致，否则页面上的"杠杆多空比"与进温度的那条线会是两条不同的线。
    chk("两处平滑窗口没分叉（LEV_SMOOTH == TO_SMOOTH）", LV.LEV_SMOOTH == E.TO_SMOOTH,
        f"{LV.LEV_SMOOTH} vs {E.TO_SMOOTH}")

    print("13) 接线：减仓温度那一格换成杠杆温度")
    adj = pd.DataFrame({c: pd.Series(np.linspace(10, 90, len(idx)), index=idx) for c in E.ORDER})
    rawd = pd.DataFrame({"_narrow_neutral": pd.Series(50.0, index=idx)})
    lev = pd.Series(np.linspace(90, 10, len(idx)), index=idx)
    base_t = E.compose_sell(adj, rawd)
    swap_t = E.compose_sell(adj, rawd, lev_temp=lev)
    w = E.SELL_W["leverage"] / sum(E.SELL_W.values())
    chk("杠杆那一格确实被顶替（差值=权重×两者之差）",
        np.allclose((swap_t - base_t).dropna(), (w * (lev - adj["leverage"])).dropna()))
    chk("不传 lev_temp 时行为与旧版一致", np.allclose(base_t.dropna(),
        sum((rawd["_narrow_neutral"] if k == "narrow" else adj[k]) * v
            for k, v in E.SELL_W.items()).dropna() / sum(E.SELL_W.values())))
    chk("杠杆温度全空时该日不出减仓温度",
        bool(np.isnan(E.compose_sell(adj, rawd, lev_temp=pd.Series(np.nan, index=idx)).iloc[-1])))

    print("14) 杠杆温度无前视")
    rngv = np.random.default_rng(11)
    rawdf = pd.DataFrame({"leverage": pd.Series(rngv.normal(70, 5, len(idx)), index=idx),
                          "_lev_intensity": pd.Series(rngv.normal(20, 4, len(idx)), index=idx)})
    _, t1 = E.leverage_monitor(rawdf)
    tampered = rawdf.copy(); tampered.iloc[300:] = 999.0
    _, t2 = E.leverage_monitor(tampered)
    chk("改动第 300 天之后的输入不影响之前的读数",
        np.allclose(t1.iloc[:300].dropna(), t2.iloc[:300].dropna()))
    chk("分位与页面其余分项同尺（走 engine.rolling_pct）",
        np.allclose(E.leverage_monitor(rawdf)[0]["ratio"].dropna(),
                    E.rolling_pct(rawdf["leverage"]).dropna()))
    chk("温度落在 0-100", bool(t1.dropna().between(0, 100).all()))

t_leverage()
print("\n杠杆温度用例全部通过" if ok else "\n杠杆温度用例有失败")


# ---- 增量合并 ----
def t_merge():
    import fetch_sp500 as FS
    loc = [[f"2026-08-{d:02d}", f"{100+d}", f"{100+d}", "1000"] for d in range(28, 0, -1)]
    # 1) 正常追加：新区间与本地重叠且刻度一致
    fresh = [["2026-09-01", "130", "130", "1000"]] + loc[:5]
    m, full = FS.merge_rows(loc, fresh)
    chk("增量：正常追加不触发全量", not full)
    chk("增量：新日期被并入", m[0][0] == "2026-09-01")
    chk("增量：旧历史被保留", len(m) == len(loc) + 1)
    chk("增量：结果按日期新到旧", all(m[i][0] > m[i+1][0] for i in range(len(m)-1)))
    # 2) 拆股：重叠区间复权价被整体改写 → 必须全量重拉
    split = [[r[0], f"{float(r[1])/4:g}", r[2], r[3]] for r in loc[:5]]
    _, full = FS.merge_rows(loc, split)
    chk("增量：复权刻度变化触发全量", full)
    # 3) 容差内的浮点噪声不该误判
    noise = [[r[0], f"{float(r[1])*(1+1e-6):g}", r[2], r[3]] for r in loc[:5]]
    _, full = FS.merge_rows(loc, noise)
    chk("增量：微小浮点差异不触发全量", not full)
    # 4) 本地太旧、与新区间毫无重叠 → 不能硬接
    gap = [["2027-01-05", "200", "200", "1000"]]
    _, full = FS.merge_rows(loc, gap)
    chk("增量：无重叠时拒绝拼接、退回全量", full)

t_merge()
print("\n增量用例全部通过" if ok else "\n增量用例有失败")

# 退出码：任一 chk 失败即非零。CI 靠这个判定成败——
# 没有它的话，脚本打印一堆 FAIL 仍然 exit 0，workflow 会一路绿灯放行。
if not ok:
    print("\n有失败项，退出码 1")
sys.exit(0 if ok else 1)
