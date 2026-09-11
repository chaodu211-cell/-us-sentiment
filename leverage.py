# -*- coding: utf-8 -*-
"""
杠杆温度 —— 杠杆ETF成交额的两个维度，合成一个 0-100 的读数。

口径参照 J.P. Morgan《Flows & Liquidity: Has investors leveraged peaked?》(2026-06-24)
对散户杠杆的测量思路（研报量的是杠杆ETF的资金流，免费源拿不到日频份额数据，
这里改用成交额）：

  · 多空比   = 做多杠杆ETF成交额 ÷ 杠杆ETF总成交额        —— 加杠杆的**方向**
  · 交易强度 = 杠杆ETF总成交额 ÷ 指数ETF(SPY+QQQ)成交额    —— 用杠杆包装交易的**强度**

两项各取 252 日滚动分位后按 3:1 加权（权重依据见 LEV_W）。方向是主项，
强度负责否决：多空比爆表但没人真往杠杆产品里砸钱时，把读数压下去。

试过并放弃的（别再试第二遍）：
· 对冲基金杠杆（研报 Fig5 的波动率比）：用 HDG 替代 HFRI，σ63(HDG)/σ63(SPY)。
  原始值秩相关 -0.309 看着强，转成分位后是 -0.006 / +0.151——符号都翻了。
  HDG 是 beta 复制产品，它的波动率比测的是自己的股票 beta，不是杠杆。
· 银行/券商杠杆（研报 Fig7）：σ63(IAI)/σ63(SPY)，同一个毛病。
· 风险平价隐含杠杆（研报 Fig6）：σ63(RPAR)/σ63(无杠杆基准)。曲线复刻得很准
  （峰值 2026-05-20，与研报"mid-May 创十年新高"对得上），但分位对未来 63 日 QQQ 的
  秩相关是 +0.000，进温度只会稀释，只展示又占地方，已删。
· 保证金净借款（研报 Fig4）：FINRA 的 WAF 按云厂商 IP 段封，CI 上实测全 403，
  只能手工下载，已删。

注释里的实测数字：2016-09 ~ 2026-09 日线，前瞻标的 QQQ。

本文件只产出**原始量**，不算分位、不做合成——分位一律由 engine.rolling_pct 统一计算，
页面上所有数字必须共用同一把尺子。
"""
import numpy as np
import pandas as pd

# ---------- 原始量的平滑 ----------
# 与 engine.TO_SMOOTH 保持一致（5 日）：两个分项都是成交额比值，日频噪声大，
# 且现行的"杠杆多空比"就是 5 日均，换成别的天数会让新旧两条线不可比。
LEV_SMOOTH = 5

# ---------- 合成权重 ----------
# 多空比（方向）3 : 交易强度（强度）1。
#
# —— 这里犯过一次错，记下来 ——
# 第一版取的是 1:1，依据是"这个因子自己对未来 63 日 QQQ 的秩相关"在两段子样本里都稳
#   （1:1 是 -0.255/-0.144，3:1 是 -0.375/-0.010）。**这个选法是错的**：这个因子不是拿来
# 单独用的，它是喂进减仓温度的一格，真正该优化的是**红点的实际表现**，不是因子自身的秩相关。
# 换成 1:1 上线后，2018-08-28~30 与 2019-12 两段红点消失了——而那是全样本里最值钱的两次
# （触发后 63 日 -11.3% 与 -15.4%）。原因不是"强度否决了它们"，恰恰相反，那几天两条腿都高
# （2018-08-28：多空比 82.7、强度 71.6），是**取平均把分位向 50 压缩**，77.2 分刚好让减仓温度
# 滑到门槛 75 以下。两个分位取平均，天然到不了单腿能到的极端值——换腿却不重标门槛，
# 等于把红点悄悄调严了（81天→63天）。
#
# 改按红点实测扫权重（线上 data.json，2017-10~2026-09，前瞻 QQQ，门槛 75 连 3 日；
# 基准 +5.18%/26%为负。"段均"的段间标准误约 ±1.4pp，差 1pp 以内的都在噪声里）：
#   多空比:强度   段/天    段均      日均/为负     前段     后段   六次真顶命中
#      1:0      26/81  -1.85%  -1.79%/59%  -5.40%  +2.49%   6/6   ← 原口径，后段几乎失效
#      5:1      23/72  -2.47%  -2.94%/65%  -5.47%  +1.04%   6/6
#      3:1      23/66  -2.39%  -3.34%/70%  -5.33%  +0.38%   6/6   ← 取这个
#      2:1      21/63  -2.11%  -3.09%/67%  -5.20%  +0.35%   5/6
#      1:1      16/63  -1.55%  -2.75%/68%  -4.63%  -0.38%   4/6   ← 上一版，漏掉 2018-08/2019-12
#      1:2      16/71  -1.67%  -2.38%/65%  -4.63%  -0.18%   4/6
#      0:1      15/95  +1.26%  -0.30%/57%  -1.28%  +0.54%   3/6   ← 只用强度，垮掉
# 3:1 在日均、为负率上是全表最好，六次真顶全中，且仍保住了强度腿的否决能力——
# 2025-09-12（多空比 99.8、强度 0.2，后续 +6.8%）在 3:1 下读 74.9，进不了红点；
# 2025 年的红点从原口径的 12 天降到 2 天。门槛也不是卡出来的：
#   >72 -2.50%/63%负、>73 -2.56%/64%、>75 -3.34%/70%、>78 -3.89%/76%、>80 -2.94%/67%。
#
# 描述侧的代价可以接受：2023-06-14 那个"多空比 99.8 分位＝十年最热"的荒谬读数，
# 1:1 压到 51.4、3:1 压到 75.6——没有 1:1 那么干净，但全年 ≥80 分位的天数仍从 141 天降到 13 天
# （2025 年从 65 天降到 10 天），泡沫顶 2021-11-19 仍读 98.9。
#
# 试过但放弃：**合成后再取一次分位**（想把平均压缩掉的量纲还回去）。看着优雅，实测是灾难：
# 段均从 -2.39% 变成 +0.27%、为负率掉到 50%，两次假信号全部放回来。原因是再取分位会把
# "强度只有 0.2 分位"这个**绝对信息**抹掉——否决能力正来自绝对水平，不来自它在自己历史中的排名。
LEV_W = {"ratio": 3.0, "intensity": 1.0}

# ---------- 分位参照系 ----------
# "roll" = 252 日滚动（与页面其余所有分位同尺），"exp" = 扩张窗口（跟截至当日的全部历史比）。
# 扫描（当时按两项等权测的，对未来 63 日 QQQ 的秩相关，全/前段/后段）：
#   126日 -0.105 / -0.221 / -0.032      252日 -0.166 / -0.255 / -0.144
#   504日 -0.163 / -0.312 / -0.182      756日 -0.185 / -0.636 / -0.229
#   扩张  -0.205 / -0.339 / -0.257
# 换到红点上（减仓温度>75 连3日，触发后 63 日）：
#   252日 64天/17段 -2.67% / 67%为负     扩张 63天/16段 -4.02% / 75%为负
# 扩张窗口确实更强，仍然选 252，两个理由：
#   1) 这个温度要和另外几项加权平均进减仓温度。尺子不同就不能相加——扩张口径的 80
#      与 252 口径的 80 不是同一件事，混在一起的加权平均没有语义。
#   2) 扩张窗口本来要修的那个病（参照系自己在晃，2023-06-14 杠杆多空比原始 66.0% 却读到
#      99.8 分位），**加上交易强度这一项之后已经自己好了**：同一天强度分位只有 3.0，
#      合成温度 51.4，不再是"十年最热"。而 2021-11-19 泡沫顶两个口径都读 98 以上。
# 想切回扩张口径只改这一行（engine 会照着选 rolling_pct / expanding_pct）。
LEV_REF = "roll"

BASE_ETFS = ("SPY", "QQQ")   # 交易强度的分母：指数ETF成交额


def dollar_volume(df):
    """成交额 = 未复权收盘价 × 成交量。用未复权价是为了让拆股前后的成交额连续。"""
    return df["rawclose"] * df["volume"]


def _sum_dv(frames, idx):
    if not frames:
        return None
    return pd.DataFrame({t: dollar_volume(d) for t, d in frames.items()}).reindex(idx).sum(axis=1, min_count=1)


def letf_components(idx, longs, shorts, base):
    """杠杆ETF通道的两个原始量（研报 Fig 1/2 的免费替代）。

    longs/shorts: {ticker: df} 做多/做空杠杆ETF
    base:         {ticker: df} 指数ETF（SPY/QQQ），作为交易强度的分母

    返回 (raw dict, meta dict)。raw 里带 `_daily` 后缀的是未平滑当日值（供快口径/展示用）。

    —— 为什么是这两个量 ——
    研报量的是杠杆ETF的**资金流**（AUM 变化），免费源拿不到日频份额数据，拿不到 AUM。
    能免费拿到的是成交额，于是拆成两个可观测的维度：
      · 多空比 = 做多杠杆ETF成交额 ÷ 杠杆ETF总成交额 —— 加杠杆的**方向**
      · 交易强度 = 杠杆ETF总成交额 ÷ 指数ETF(SPY+QQQ)成交额 —— 用杠杆包装交易的**强度**
    这两个维度实测是互补的（见 LEV_W 注释的权重扫描：单用任何一个都有整整一段样本失效）。
    直观上也说得通：2023 年做多占比很高但没人真的往杠杆产品里挤（强度分位 3.0），
    2021 年则是两个同时爆表。
    """
    raw, meta = {}, {}
    dvl = _sum_dv(longs, idx)
    dvs = _sum_dv(shorts, idx)
    dvb = _sum_dv(base, idx)

    if dvl is not None and dvs is not None:
        tot = (dvl + dvs).replace(0, np.nan)
        ratio = dvl / tot * 100.0
        raw["ratio_daily"] = ratio
        raw["ratio"] = ratio.rolling(LEV_SMOOTH, min_periods=1).mean()
        meta["ratio"] = (f"做多杠杆ETF成交额 / 杠杆ETF总成交额（{len(longs)}只做多、{len(shorts)}只做空，"
                         f"{LEV_SMOOTH}日均）")

    if dvl is not None and dvb is not None:
        lev_total = dvl if dvs is None else (dvl + dvs)
        inten = lev_total / dvb.replace(0, np.nan) * 100.0
        raw["intensity_daily"] = inten
        raw["intensity"] = inten.rolling(LEV_SMOOTH, min_periods=1).mean()
        meta["intensity"] = (f"杠杆ETF总成交额 / 指数ETF({'+'.join(sorted(base))})成交额（{LEV_SMOOTH}日均）"
                             + ("" if dvs is not None else "；缺做空杠杆ETF，分子只含做多"))
    return raw, meta


def build(idx, frames):
    """产出杠杆温度的两条原始量。

    frames: {ticker: df}，需要做多/做空杠杆ETF与 SPY/QQQ。
    返回 (raw: dict[str, Series], meta: dict[str, str], notes: dict[str, str])
    notes 记录"为什么某一条没有"，直接显示到页面上，避免静默缺项。
    """
    from engine import LEV_LONG, LEV_SHORT      # 延迟导入：避免与 engine 形成导入环
    pick = lambda ts: {t: frames[t] for t in ts if t in frames}
    raw, meta = letf_components(idx, pick(LEV_LONG), pick(LEV_SHORT), pick(BASE_ETFS))
    notes = {}
    if "ratio" not in raw:
        notes["ratio"] = "缺做多或做空杠杆ETF行情"
    if "intensity" not in raw:
        notes["intensity"] = "缺指数ETF(SPY/QQQ)行情"
    return raw, meta, notes


def temperature(pct):
    """按 LEV_W 加权合成杠杆温度。

    pct: DataFrame，列名是 LEV_W 的键，值是已经算好的分位（0-100）。
    缺项按剩余项的权重重新归一——与综合温度"缺一不出"的约定不同，这里必须容忍缺项：
    多空比与交易强度的历史长度不同，且交易强度依赖 SPY/QQQ 是否当日到齐。
    但两项全缺时返回全 NaN（不能凭空造一个 50）。
    """
    cols = [c for c in LEV_W if c in pct.columns]
    if not cols:
        return pd.Series(np.nan, index=pct.index)
    w = pd.Series({c: LEV_W[c] for c in cols}, dtype=float)
    X = pct[cols]
    num = (X * w).sum(axis=1, min_count=1)
    den = X.notna().mul(w, axis=1).sum(axis=1)
    return num / den.replace(0, np.nan)
