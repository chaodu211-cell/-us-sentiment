# -*- coding: utf-8 -*-
"""独立复核：分位数、反向、等权、区间划分"""
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


for f in (t_repricing, t_bear, t_alert_exclusive):
    f()
print("\n新增用例全部通过" if ok else "\n新增用例有失败")


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
