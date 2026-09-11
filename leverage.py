# -*- coding: utf-8 -*-
"""
杠杆温度 —— 按 J.P. Morgan《Flows & Liquidity: Has investors leveraged peaked?》
（Global Markets Strategy, 2026-06-24）的杠杆测量框架重建。数据全部免费公开、无需注册。

研报的做法不是"找一个杠杆指标"，而是**按持有人分层**，每一层各用一个可观测的代理量：

| 研报 | 层 | 它的口径 | 数据源 | 本模块怎么做 |
|---|---|---|---|---|
| Fig 1/2 | 散户 | 杠杆ETF的资金流与再平衡流（AUM $247bn） | Bloomberg | 杠杆ETF**成交额**的多空比 + 交易强度（免费日频） |
| Fig 3 | 散户 | OCC 小单（<10张）看涨期权净买入 | OCC | **放弃**：无免费日频源 |
| Fig 4 | 散户 | NYSE 保证金净借款 ÷ 标普500市值 | FINRA/NYSE | 可选月频分项（`raw/_margin.csv`） |
| Fig 5 | 对冲基金 | 基金收益波动率 ÷ 标的收益波动率 | HFR/PivotalPath | **试过并放弃**（见下"放弃清单"） |
| Fig 6 | 风险平价 | 基金收益波动率 ÷ 风险平价基准波动率 | Bloomberg | RPAR ÷ 固定权重基准，展示用读数 |
| Fig 7 | 银行 | 交易利润波动率 ÷ 资产波动率 | OCC | **试过并放弃** |
| Fig 8/9 | 企业/家庭 | 债务占GDP、净利息占经营现金流 | BIS/Fed | 不做：季频慢变量，研报自己的结论是"不构成脆弱性" |

—— 一条要先说清楚的负面结论 ——
研报的标题问的是"杠杆见顶了吗"，正文的落点是**回落本身是利空**（"signs of retreat …
presenting a potential headwind"）。这条在本模块能拿到的免费代理上**测不出来**。
把"见顶退潮"写成事件规则（63日内峰值≥Th、当前较峰值回落≥D、且温度仍≥floor），
在 2017-10~2026-09 的 QQQ 上扫了 12 组参数（Th∈{75,80,85}×D∈{6,10}×floor∈{50,60}）：
触发后 63 日收益全部落在 +4.3% ~ +5.0%，而同期基准是 +5.18% —— 没有边际。
反倒是**维持高位**有区分度（温度≥85 且仍贴近 63 日峰值：+1.95%、33%为负）。
所以本模块输出的是**水平温度**，不是"拐点信号"；页面上会把回落幅度作为描述信息列出，
但不拿它触发任何预警。研报的那句话在这里没有得到验证，别当成已验证的东西用。

—— 放弃清单（试过、测过、不要再试第二遍）——
· 对冲基金杠杆（研报 Fig5 的波动率比）：用 HDG（ProShares 对冲基金复制ETF）替代 HFRI，
  σ63(HDG)/σ63(SPY)。原始值对未来 63 日秩相关 -0.309 看着很强，但那是**水平趋势**造成的假象：
  转成 252 日滚动分位后是 -0.006（全样本）、+0.151（2021-03 起）——符号还翻正了。
  放进温度会把合成从 -0.235 拉到 -0.125。原因不难理解：HDG 是 beta 复制产品，
  它的波动率比测的是自己的股票 beta，不是对冲基金的杠杆。
· 银行/券商杠杆（研报 Fig7）：σ63(IAI)/σ63(SPY)，同一个毛病，分位口径 -0.041 / +0.085。
· 风险平价的**逆波动率基准**版本：把基准换成 SPY/TLT/GLD/DBC 按 252 日逆波动率月度再平衡，
  分位秩相关 -0.159，比忠实口径（+0.000）"好看"得多——但它画出来的曲线与研报 Fig6 对不上
  （研报的峰在 2026-05，逆波动率口径的峰在 2023-24）。它测的多半是股债相对波动的 regime，
  不是风险平价基金的杠杆。宁可要一个诚实的零，不要一个来路不明的 -0.159。

—— 本模块与 engine.py 的分工 ——
本文件只产出**原始量**（raw series），不算分位、不做合成。分位一律由 engine.rolling_pct
统一计算——页面上所有数字必须共用同一把尺子，否则加权平均在语义上就不成立。

—— 注释里那些实测数字是怎么来的 ——
在 2016-09 ~ 2026-09 的日线上跑的，前瞻标的是 QQQ。开发这版时的环境取不到 stockanalysis.com
（出口被封），行情改从另一家免费源重建；校验方式是拿重建数据重算仓库现有的杠杆多空比，
与 data.json 里存着的那条序列逐日对比：**相关 0.9994、平均绝对差 0.16 个百分点**（2242 个交易日）。
所以结论可信，但小数点后会有出入——用流水线自己的数据重跑一遍，秩相关可能是 -0.25 而不是 -0.255。
交易强度那一项的分母（SPY+QQQ 成交额）没有这样的对照物，不同数据商对 ETF 成交量的口径
（合并 vs 单一交易所）可能差一个常数；好在分位对常数缩放免疫，只有页面上那个百分比会平移。
"""
import csv
import os
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
# 扫描（两项等权，对未来 63 日 QQQ 的秩相关，全/前段/后段）：
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

# ---------- 风险平价隐含杠杆（研报 Fig 6）----------
VOL_WIN = 63          # 研报写的是"3-month rolling volatility"，取 63 个交易日
RP_FUND = "RPAR"      # Advanced Research Risk Parity ETF，2019-12 上市，免费日线
# 无杠杆基准：按 RPAR 自己公布的目标配置（股 25%、TIPS 35%、长债 35%、金 10%、商品 15%，
# 合计 120% 即其约 1.2 倍杠杆）取同样的比例、归一到 100%。
# 这样构造出来的比值 = 基金实际波动 ÷ 同样资产不加杠杆的波动 ≈ 隐含杠杆倍数。
# 校验：2020-2026 均值 1.48，与"目标 1.2 倍 + TIPS/长债久期敞口"量级相符；
# 峰值出现在 **2026-05-20（2.12）**，与研报"mid-May 创十年新高"完全对上；
# 截至 2026-09-09 回落到 1.66，也对应研报说的"recent weeks 在退"。
# 这条曲线是**描述**，不是信号：它的分位对未来 63 日 QQQ 的秩相关是 +0.000，
# 放进温度只会稀释（三项合成 -0.235 → -0.191），所以只展示、不参与合成。
RP_BENCH = {"SPY": 25.0, "TIP": 35.0, "TLT": 35.0, "GLD": 10.0, "DBC": 15.0}
RP_REQUIRED = ("SPY", "TLT", "TIP")   # 缺这三条腿中任何一条就不出读数（权重结构会失真）

# ---------- 保证金净借款（研报 Fig 4）----------
# FINRA 月频保证金统计，研报的口径：净借款 = 融资借方余额 − (现金账户贷方 + 融资账户贷方)。
# 研报再除以标普500总市值；免费口径拿不到逐日总市值，用指数点位代替（见 margin_gauge 注释）。
MARGIN_FILE = "_margin.csv"
# 发布滞后：FINRA 的月末数据通常在次月第 3~4 周才公布。不模拟这个滞后就是前视——
# 回测里会在月末当天就用上一个月后才知道的数字。取 25 个自然日，偏保守。
MARGIN_LAG_DAYS = 25

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


def rp_leverage(idx, frames):
    """风险平价隐含杠杆（研报 Fig 6 的忠实复刻）。

    = σ63(RPAR 日收益) / σ63(无杠杆基准日收益)

    frames: {ticker: df}，需要 RPAR 与 RP_BENCH 里的腿。
    返回 (Series, meta) 或 (None, 说明为什么没有)。
    """
    if RP_FUND not in frames:
        return None, f"缺 {RP_FUND} 行情"
    legs = [t for t in RP_BENCH if t in frames]
    missing_req = [t for t in RP_REQUIRED if t not in legs]
    if missing_req:
        return None, "缺基准腿：" + "/".join(missing_req)

    px = pd.DataFrame({t: frames[t]["close"] for t in legs}).reindex(idx)
    rets = px.pct_change()
    w = pd.Series({t: RP_BENCH[t] for t in legs}, dtype=float)
    w = w / w.sum()
    # 固定权重、不再平衡权重漂移：用加权日收益直接合成基准收益，等价于每日再平衡的
    # 固定权重组合。这里要的是"同样资产不加杠杆会有多大波动"，每日再平衡是最干净的定义。
    bench_ret = (rets[legs] * w).sum(axis=1, min_count=len(legs))
    fund_ret = frames[RP_FUND]["close"].reindex(idx).pct_change()

    ann = np.sqrt(252.0)
    v_fund = fund_ret.rolling(VOL_WIN, min_periods=VOL_WIN).std() * ann
    v_bench = bench_ret.rolling(VOL_WIN, min_periods=VOL_WIN).std() * ann
    s = v_fund / v_bench.replace(0, np.nan)
    meta = (f"σ{VOL_WIN}({RP_FUND}) / σ{VOL_WIN}(基准)，基准＝"
            + "、".join(f"{t} {RP_BENCH[t]:g}%" for t in legs) + "（归一到100%，即去掉基金自身杠杆）")
    return s, meta


def read_margin(raw_dir, fname=MARGIN_FILE):
    """读 raw/_margin.csv，并把金额统一换算成**十亿美元**。

    列：日期, 融资借方余额, 现金账户贷方余额, 融资账户贷方余额（三个金额列单位要一致）。
    允许只有前两列——那时净借款退化为融资余额本身，meta 里会写明。
    允许有表头、新到旧或旧到新排列。

    单位自动识别：FINRA 各期文件有的以美元、有的以百万美元给数，而融资借方余额的量级
    是已知的（近十年在 4000 亿~1.2 万亿美元之间）。按借方余额中位数判断：
      > 1e10 → 美元；> 1e4 → 百万美元；否则 → 十亿美元。
    判错会让整条曲线差 1000 倍，但**分位不受影响**（同比例缩放不改变排序），
    受影响的只有页面上显示的那个绝对数。
    """
    p = os.path.join(raw_dir, fname)
    if not os.path.exists(p):
        return None
    rows = []
    with open(p, encoding="utf-8-sig", newline="") as f:
        # 用 csv 模块而不是 split(",")：FINRA 的导出常把金额写成 "1,234,567,890"（带引号和千分位），
        # 硬切逗号会把一个数拆成四个字段，整张表静默变空。
        for parts in csv.reader(f):
            parts = [x.strip() for x in parts if x.strip() != ""]
            if len(parts) < 2:
                continue
            d = pd.to_datetime(parts[0], errors="coerce")
            if pd.isna(d):
                continue          # 表头或空行
            vals = []
            for x in parts[1:4]:
                try:
                    vals.append(float(x.replace("$", "").replace(",", "").replace('"', "")))
                except ValueError:
                    vals.append(np.nan)
            if not vals or not np.isfinite(vals[0]):
                continue
            while len(vals) < 3:
                vals.append(np.nan)
            rows.append((d, vals[0], vals[1], vals[2]))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["date", "debit", "credit_cash", "credit_margin"])
    df = df.drop_duplicates("date").sort_values("date").set_index("date")
    med = float(df["debit"].median())
    scale = 1e-9 if med > 1e10 else (1e-3 if med > 1e4 else 1.0)   # → 十亿美元
    return df * scale


def margin_gauge(idx, mdf, spx=None, lag_days=MARGIN_LAG_DAYS):
    """保证金净借款读数（研报 Fig 4）。

    净借款 = 融资借方余额 − (现金账户贷方 + 融资账户贷方)。两个贷方列缺失时退化为融资余额。
    返回 (level, ratio, meta)：
      level 净借款本身（十亿美元，页面上显示这个数，人能读）
      ratio 净借款 ÷ 标普500指数点位（**分位算在这条上**，剔除指数涨跌带来的机械抬升）
    取不到数据时返回 (None, None, 原因)。

    —— 分母为什么不是市值 ——
    研报除的是标普500总市值。免费源拿不到逐日总市值（要全成分股股本），这里除以
    **指数点位**代替：市值 = 点位 × 除数，而除数随回购/成分调整缓慢漂移，
    所以这条比值跨越多年时带系统性偏移，**不能当成研报那张 1997 年起的图来读**，
    几年尺度内的相对高低才是可用的。spx 缺失时分位就算在净借款本身上。

    —— 发布滞后 ——
    FINRA 月末数据次月第 3~4 周才公布。每个观测值从"月末 + lag_days 自然日"起才可见，
    之前的日子留空。没有这一步就是前视：回测会在月末当天就用上一个月后才知道的数字。
    """
    if mdf is None or len(mdf) == 0:
        return None, None, "无 raw/_margin.csv"
    credits = mdf[["credit_cash", "credit_margin"]].sum(axis=1, min_count=1)
    net = mdf["debit"] - credits.fillna(0.0)
    has_credit = bool(credits.notna().any())

    avail = pd.Series(net.values, index=mdf.index + pd.Timedelta(days=lag_days))
    avail = avail[~avail.index.duplicated(keep="last")].sort_index()
    level = avail.reindex(idx.union(avail.index)).ffill().reindex(idx)

    meta = ("净借款＝融资借方余额 − (现金账户贷方 + 融资账户贷方)" if has_credit else
            "融资借方余额（缺贷方数据，未做净额）")
    ratio = level
    if spx is not None:
        ratio = level / spx.reindex(idx).replace(0, np.nan)
        meta += "，分位算在「净借款 ÷ 标普500指数点位」上（研报除的是总市值，见代码注释）"
    meta += f"；FINRA 月频，滞后 {lag_days} 天后才可见"
    return level, ratio, meta


def build(idx, frames, spx=None, raw_dir=None):
    """一次性产出杠杆监测需要的全部原始量。

    frames: {ticker: df}，至少要有做多/做空杠杆ETF与 SPY/QQQ；RPAR 及基准腿可选。
    返回 (raw: dict[str, Series], meta: dict[str, str], notes: dict[str, str])
    notes 记录"为什么某一项没有"，直接显示到页面上，避免静默缺项。
    """
    from engine import LEV_LONG, LEV_SHORT      # 延迟导入：避免与 engine 形成导入环
    pick = lambda ts: {t: frames[t] for t in ts if t in frames}

    raw, meta = letf_components(idx, pick(LEV_LONG), pick(LEV_SHORT), pick(BASE_ETFS))
    notes = {}
    if "ratio" not in raw:
        notes["ratio"] = "缺做多或做空杠杆ETF行情"
    if "intensity" not in raw:
        notes["intensity"] = "缺指数ETF(SPY/QQQ)行情"

    rp, rp_meta = rp_leverage(idx, frames)
    if rp is None:
        notes["rp"] = rp_meta
    else:
        raw["rp"] = rp
        meta["rp"] = rp_meta

    if raw_dir:
        level, ratio, mg_meta = margin_gauge(idx, read_margin(raw_dir), spx)
        if level is None:
            notes["margin"] = mg_meta
        else:
            raw["margin"] = level          # 展示用：十亿美元
            raw["margin_ratio"] = ratio    # 算分位用：÷ 指数点位
            meta["margin"] = mg_meta
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
