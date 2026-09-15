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


# ---- 新增：实际利率重估期 & 熊市状态 ----
def _rr_series(seed=3):
    """构造一条有"水位维度"的实际利率序列，形状照着 2015-2023 的真实走法：

        常态高位(1.0) → 压到低位(-1.0)并长期趴着 → 从低位快速修复到 1.6 → 高位横盘

    只有"从被压低的水位往上快速修复"那一段才该被判成重估期。横盘段要带噪声——
    分位是相对量，常数序列的变动全为 0、分位恒等于 50，测不出东西。
    返回 (序列, 各段的起止下标)。
    """
    import numpy as np, pandas as pd
    rng = np.random.RandomState(seed)
    n1, n2, n3, n4 = 600, 550, 250, 300
    hi = 1.0 + rng.normal(0, 0.03, n1).cumsum() * 0.08          # 常态高位
    drop = np.linspace(hi[-1], -1.0, 60)                         # 快速压低（利率在跌）
    low = -1.0 + rng.normal(0, 0.03, n2 - 60).cumsum() * 0.06    # 低位长期趴着
    up = low[-1] + np.linspace(0, 2.6, n3)                       # 从低位快速修复
    top = up[-1] + rng.normal(0, 0.03, n4).cumsum() * 0.06       # 高位横盘：涨势停了
    v = np.concatenate([hi, drop, low, up, top])
    idx = pd.bdate_range("2016-01-01", periods=len(v))
    seg = {"low": (n1 + n2 - 200, n1 + n2), "up": (n1 + n2, n1 + n2 + n3),
           "top": (n1 + n2 + n3, len(v))}
    return pd.Series(v, index=idx), seg


def t_repricing():
    import pandas as pd, numpy as np, engine as E
    rr, seg = _rr_series()
    nom = rr + 2.0        # 名义与实际同向同幅 → 名义闸门恒通过
    on = E.repricing_regime(rr, nom)
    spd, lvl = E.real_rate_speed_pct(rr), E.real_rate_level_pct(rr)
    a, b = seg["up"]

    warm = E.RR_WIN + E.RR_PCT_MIN
    chk("重估期：分位预热完成前不判定（无前视）", on.iloc[:warm].sum() == 0)
    chk("重估期：从低位快速修复的那一段确实触发", on.iloc[a:b].sum() > 0,
        f"共 {int(on.iloc[a:b].sum())} 天")

    # 只看水位不行：低位横盘段水位分位很低，但利率没在涨，不该触发
    la, lb = seg["low"]
    chk("重估期：水位低但没在涨 → 不触发（只看水位会把 2020-03 判反）",
        on.iloc[la:lb].sum() == 0,
        f"低位段水位分位中位 {np.nanmedian(lvl.iloc[la:lb]):.0f}，触发 {int(on.iloc[la:lb].sum())} 天")

    # 水位退出：高位横盘段利率停在最高处（旧口径靠绝对水位退出，这里靠水位分位）
    ta, tb = seg["top"]
    tail = on.iloc[ta + E.RR_WIN:]
    chk("重估期：水位修复到位后退出", not tail.any(),
        f"高位横盘段仍为真 {int(tail.sum())} 天，水位分位 {lvl.iloc[-1]:.0f}")

    # 只看速度不行：同样的上涨幅度发生在"水位已经很高"时不该触发（2018Q4 那种）
    hi_rise = pd.Series(np.concatenate([
        1.0 + np.random.RandomState(1).normal(0, 0.03, 900).cumsum() * 0.05,
        np.linspace(0, 0.8, 300)]), index=pd.bdate_range("2016-01-01", periods=1200))
    hi_rise.iloc[900:] += hi_rise.iloc[899]
    on_hi = E.repricing_regime(hi_rise, hi_rise + 2.0)
    s_hi, l_hi = E.real_rate_speed_pct(hi_rise), E.real_rate_level_pct(hi_rise)
    chk("重估期：同样的涨速发生在高水位时不触发（2018Q4 那种）", not on_hi.iloc[900:].any(),
        f"涨速分位最高 {np.nanmax(s_hi.iloc[900:]):.0f}，水位分位最低 {np.nanmin(l_hi.iloc[900:]):.0f}，"
        f"触发 {int(on_hi.iloc[900:].sum())} 天")

    # 滞回：进入后短暂回落不应立刻熄灭（2022-03 俄乌避险那种闪断）
    v2 = rr.copy(); i0 = a + 60
    v2.iloc[i0:i0 + 15] -= 0.45
    on2 = E.repricing_regime(v2, nom)
    chk("重估期：滞回让短暂回落不熄灭", bool(on.iloc[i0]) and bool(on2.iloc[i0 + 8]))

    # 名义同向闸门：实际利率照样快速上行，但名义在下行（2020-03 通缩恐慌那种）
    nom_dn = pd.Series(np.linspace(3.0, 1.0, len(rr)), index=rr.index)
    chk("重估期：名义利率下行时不触发（通缩式上行不算重估）",
        not E.repricing_regime(rr, nom_dn).any())

    # 绝对下限：把涨幅整体压扁，分位再高也不该入场
    flat = (rr - rr.iloc[0]) * 0.02 + rr.iloc[0]
    chk("重估期：涨幅达不到绝对下限时不入场（死水利率环境的噪声）",
        not E.repricing_regime(flat, flat + 2.0).any())

    chk("重估期：无数据时返回 None", E.repricing_regime(None) is None)
    # 拿不到名义利率时整条闸门不启用（蓝点全记实心），而不是退化成不做这项检查
    chk("重估期：缺名义利率时整条闸门不启用", E.repricing_regime(rr) is None)


def t_rr_pct_series():
    """两条分位列本身：不能被 NaN 压低，不能有前视。"""
    import pandas as pd, numpy as np, engine as E
    rr, _ = _rr_series()
    spd = E.real_rate_speed_pct(rr)
    lvl = E.real_rate_level_pct(rr)
    warm = E.RR_WIN + E.RR_PCT_MIN
    chk("涨速分位：预热期无读数", spd.iloc[:warm - 1].isna().all())
    chk("水位分位：预热期无读数", lvl.iloc[:E.RR_LVL_MIN - 1].isna().all())
    # 单调上行段末端的 6 个月变动必是历史最大 → 分位应贴近 100；
    # 若 rolling 窗口把 NaN 计入分母，这里会明显低于 100（就是 dropna 要修的 bug）
    chk("涨速分位：历史最快的一段读数接近 100", spd.max() > 99.0, f"max={spd.max():.2f}")
    chk("水位分位：历史最高的一段读数接近 100", lvl.max() > 99.0, f"max={lvl.max():.2f}")
    # 无前视：截断序列后，公共区段的读数必须逐点一致
    cut = len(rr) - 60
    for name, fn in [("涨速分位", E.real_rate_speed_pct), ("水位分位", E.real_rate_level_pct)]:
        full, part = fn(rr).iloc[:cut].dropna(), fn(rr.iloc[:cut]).dropna()
        chk(f"{name}：无前视（截断后公共区段不变）",
            len(full) == len(part) and np.allclose(full.values, part.values, atol=1e-9))


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
    rp = pd.Series(np.arange(800) % 3 == 0, index=idx)   # 状态序列由 repricing_regime 预先算好
    vix = pd.Series(35.0, index=idx)
    al = E.build_alerts(t, pd.DataFrame(index=idx), vix=vix, temp_fast=t, ndx=px, repricing=rp)
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


def t_topshare():
    """前 CROWD_TOP_FRAC 成交额个股占比：k 的取法、占比算法，以及它不能进任何温度。"""
    import pandas as pd, numpy as np, engine as E

    def share(vols, frac):
        """独立实现（排序取前 k 求和），与 engine 的向量化版本对照"""
        v = np.array(sorted([x for x in vols if x > 0], reverse=True), dtype=float)
        k = max(1, int(np.ceil(frac * len(v))))
        return v[:k].sum() / v.sum() * 100.0

    # k = ceil(frac * 当日有效样本数)：100 只、2% -> 前 2 只
    vols = list(range(1, 101))
    chk("k = ceil(frac × 有效样本数)", abs(share(vols, 0.02) - (100 + 99) / sum(vols) * 100) < 1e-9)
    # 不足一只时向上取整到 1
    chk("样本少时 k 至少为 1", abs(share([5.0, 3.0, 2.0], 0.02) - 50.0) < 1e-9)
    # 只数固定时，越集中占比越高（单调性）
    flat = share([10.0] * 50, 0.02)
    conc = share([500.0] + [10.0] * 49, 0.02)
    chk("越集中读数越高", conc > flat, f"{flat:.1f} -> {conc:.1f}")

    # 下划线列不得进入六项蓝点温度 —— 这是"只改黑框"的全部安全性所在
    idx = pd.date_range("2024-01-01", periods=400, freq="B")
    rng = np.random.default_rng(3)
    raw = pd.DataFrame({k: rng.normal(size=400).cumsum() for k in E.ORDER}, index=idx)
    raw["_topshare"] = rng.normal(size=400).cumsum()
    pct, adj, temp, temp_adj = E.compose(raw)
    chk("_topshare 不进分位表（compose 跳过下划线列）", "_topshare" not in pct.columns)
    base = E.compose(raw.drop(columns=["_topshare"]))[3]
    chk("_topshare 存在与否不改变蓝点温度", np.allclose(temp_adj.values, base.values, equal_nan=True))
    # 红点温度同理：SELL_W 里没有它
    chk("红点温度权重表不含 topshare", "topshare" not in E.SELL_W and "_topshare" not in E.SELL_W)
    chk("黑框抱团门槛与上涨拥挤度门槛是两把尺子（分位 vs 评分）",
        E.CROWD_TOP_FRAC > 0 and 0 < E.CROWD_TOP2 <= 100)


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
    # 抱团闸门优先读 topshare（前 2% 个股成交额占比），取不到才退回 top2。
    # 两列给相反的值，看哪一列说了算。
    both_hi = pd.DataFrame({"top2":      pd.Series(E.CROWD_TOP2 - 1, index=idx),
                            "topshare":  pd.Series(E.CROWD_TOP2 + 1, index=idx)})
    both_lo = pd.DataFrame({"top2":      pd.Series(E.CROWD_TOP2 + 1, index=idx),
                            "topshare":  pd.Series(E.CROWD_TOP2 - 1, index=idx)})
    chk("黑框抱团闸门优先用 topshare（topshare 够则触发，哪怕 top2 不够）",
        bool(E.build_alerts(calm, pd.DataFrame(index=idx), crowd_pct=both_hi,
                            narrow=nr, ndx=px)["crowd"].iloc[E.PERSIST:].all()))
    chk("黑框抱团闸门优先用 topshare（topshare 不够则不触发，哪怕 top2 够）",
        not E.build_alerts(calm, pd.DataFrame(index=idx), crowd_pct=both_lo,
                           narrow=nr, ndx=px)["crowd"].any())
    chk("缺 topshare 时退回 top2（与旧版行为一致）",
        bool(E.build_alerts(calm, pd.DataFrame(index=idx), crowd_pct=cp,
                            narrow=nr, ndx=px)["crowd"].iloc[E.PERSIST:].all()))
    # 橙线已取消：规则表里不应再有它，build_alerts 也不应再产出该键
    chk("橙线预警已取消（ALERTS 中无 narrow）",
        not any(r["key"] == "narrow" for r in E.ALERTS))
    chk("橙线预警已取消（build_alerts 不产出 narrow）",
        "narrow" not in E.build_alerts(calm, pd.DataFrame(index=idx), crowd_pct=cp, narrow=nr, ndx=px))
    # 确认天数：第 PERSIST 天才置位，之前为假（无前视）
    chk(f"红点需连续 {E.PERSIST} 日才置位",
        (not al["hot"].iloc[:E.PERSIST - 1].any()) and bool(al["hot"].iloc[E.PERSIST - 1]))


for f in (t_repricing, t_rr_pct_series, t_bear, t_alert_exclusive, t_compose_sell,
          t_narrow_neutral, t_topshare, t_alert_wiring):
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
