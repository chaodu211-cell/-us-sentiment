# -*- coding: utf-8 -*-
"""
美股交易情绪温度计 —— 指标合成引擎

六个指标 -> 252日滚动分位 -> ERP反向 -> 等权平均 -> 蓝点温度
数据源无关：只要 raw/<TICKER>.csv 存在（date,close,volume，新到旧），就能跑。
"""
import json, os, math
from datetime import datetime
import numpy as np
import pandas as pd

import leverage as LV   # 杠杆温度：研报口径的多层杠杆监测（见 leverage.py 顶部注释）

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.json")

WINDOW      = 252   # 滚动分位窗口：252 个交易日
MIN_PERIODS = 252   # 要求完整窗口，不做短窗口近似
ADV_SMOOTH  = 10    # 上涨占比的平滑窗口（主口径；日频原始值噪声极大）
TO_SMOOTH   = 5     # 换手率/杠杆多空比平滑窗口（主口径）
# 另有一条"当日口径"温度：三个平滑项一律取当日值，不平滑。它不对外显示为蓝点温度，
# 只供蓝点预警使用——恐慌见底通常是 1-3 天的插针，平滑与多日确认会把这种尖峰削平。
MA_WIN      = 20    # 均线窗口

# ERP 口径。加息周期会让 ERP=E/P-10Y 被单边压缩，252日窗口把两年的利率趋势
# 当成情绪波动（2022-10 熊市底 ERP 反向分位读到 94.2 = "极贵"）。
#   "erp"   原始口径：仅用 ERP 的滚动分位。判别力最强（秩相关 -0.126），但慢熊失真。
#   "blend" 折中：ERP 分位与 E/P 分位各半。E/P 不含利率，不受加息周期污染；
#           保留 ERP 的股债相对信息。2022 底 31.5 → 23.9，秩相关 -0.126 → -0.099。
#   "ep"    仅用 E/P：2022 底降到 16.3（进恐慌区），但秩相关掉到 -0.071 且五档收益失去单调。
#
# —— 2026-09 复核：blend 撤销，退回 "erp" ——
# 上面那段只看了"合成温度"的秩相关（-0.126 → -0.099，看着像小让步），漏了看**分项本身**。
# 直接测这一个因子对未来 63 日 QQQ 的秩相关（2017-10 起 2180 个交易日）：
#     纯 ERP 分位 -0.120   |   blend 分位 -0.007   ← 不是"钝化"，是归零
# 高低两端读数几乎一样（最高10%的日子 +4.34%、最低10% +4.70%，基准 +5.19%），
# 等于往温度里掺了 1/6 的白噪声。原因是掺进来的那半个 E/P 分位自身秩相关约 +0.11，
# 与 ERP 那半**方向相反**，不是中性稀释而是反向对冲，正好把信号抵消掉。
# 两端都因此受损：ERP 最低10%的日子（=最便宜）纯口径后续 +7.91%/23%为负，
# blend 只有 +4.70%/35%为负——蓝点想要的"估值到位"信息也被抹掉了。
# 加息周期的失真交给 repricing_regime()（实际利率重估期闸门）去处理，那个实测有效
# （空心蓝 100% 为负 vs 实心蓝 0% 为负），不需要在因子层面再修一次、还修坏。
# 代价：2022-10 熊市底 ERP 反向分位会重新读到 90+ 的"极贵"。这是**描述**失真，
# 已由熊市反弹预警与空心蓝点在**信号**层面兜住，不再牺牲因子本身的判别力去换它。
ERP_MODE = "erp"

# ---------- 参照系 ----------
# 滚动252日分位的毛病：尺子自己在晃。实测各分项"窗口均值摆幅 / 自身标准差"：
#   杠杆多空比 3.50、ERP 2.64、TOP2 2.60、换手率 1.85 —— 参照系比被测物晃得还厉害；
#   上涨占比 0.90、站上MA20 0.83 —— 这两个本身平稳。
# 后果：2023 年红点 21 天（全样本最多），2021 年 0 天——恰好把泡沫顶说成不热、
# 把熊市后的正常说成最热，因为 2023 的参照窗口装的全是 2022 熊市。
#
# 现在分两类处理：
#   REF_ABS  跨年可比的有界量 → 不做分位，用固定锚点把水平值映射到 0-100，参照系完全不动
#   REF_EXP  水平会漂移的量   → 用扩张窗口分位（跟截至当日的全部历史比，无前视）
#   ERP 例外：仍用 252 日。它的漂移主要来自利率周期，用扩张窗口会拿今天的 ERP 跟
#   零利率年代比，2025-04 恐慌读数从 17 掉到 30，反而钝化。
# —— 实测结论：改参照系能修好"描述"，却修坏了"信号"，最终选择退回全 252 日 ——
# 试过的方案（宽度用固定锚点绝对映射 + 换手率/TOP2/杠杆用扩张窗口分位）确实做到了：
#   2023 年红点 21 天 → 0 天；温度与未来3个月收益的秩相关 -0.114 → -0.176；
#   五区间占比回到 4/17/39/34/6。
# 但红点触发后纳指 +3个月 +5.56%，基准 +5.22% —— 判别力完全消失，且 71 天里 52 天挤在
# 2020 年（扩张窗口"永不遗忘"，早年极值把后来的门槛永久抬高）。长滚动窗口也救不回来：
#   504日 +5.25%、756日 +5.20%、1008日 +6.89%，均无区分度；1260日虽有 +2.95% 但需5年预热。
# 也就是说，跟过去一年比在"描述市场绝对状态"上是荒谬的（把2021说成不热、2023说成最热），
# 但在"预测未来三个月"上反而更有用——决定短期均值回归的本就是相对最近regime有多绷。
# 代码路径保留，改回下面三行即可切换：
#   REF_ABS = {"advancing": (35.0, 53.0, 70.0), "ma20": (10.0, 60.0, 100.0)}
#   REF_ABS_FAST = {"advancing": (25.0, 53.0, 80.0)}
#   REF_EXP = ["turnover", "top2", "leverage"] ; BANDS = [28.0, 46.0, 61.0, 76.0]
REF_ABS = {}
REF_ABS_FAST = {}
REF_EXP = []
REF_ROLL = ["turnover", "top2", "advancing", "ma20", "leverage", "erp"]

BANDS = [20.0, 40.0, 60.0, 80.0]

# ---------- 红点温度：与展示用的蓝点温度分开 ----------
# 为什么要拆：蓝点温度被同时要求做两件相反的事——标顶和标底。实测这两件事靠的不是同一批因子。
# 现行红点（蓝点温度>80 连3日）是四个减仓信号里最弱的一个：
#   红点 +63日 +3.18% / 34%为负   黑框 +0.35% / 38%   橙线 +0.67% / 39%   熊反 -11.33% / 87%
#   （基准 +5.19% / 27%为负）——旗舰信号还不如两个辅助信号。
# 而且换阈值救不回来：>70 是 +4.54%、>80 是 +3.18%、>85 是 +2.75%，整条曲线平的，
# 说明问题在合成本身，不在门槛。三个具体病因：
#   1. ERP 那一项当时是 blend，秩相关 -0.007（已在上面修掉）
#   2. 上涨占比与站上MA20占比相关系数 0.84，是同一个东西量两遍，
#      名义各占 1/6、合计吃掉温度方差的 38%，而它俩恰是六项里最弱的（-0.023 / -0.047）
#   3. 判别力最强的因子根本不在温度里：上涨拥挤度秩相关 -0.348，是第二名（换手率 -0.153）的两倍多
# 于是红点温度只保留"标顶真的有用"的四类，并把上涨拥挤度放进来给最高权重：
#   TOP2抱团 x2、杠杆多空比 x2、上涨拥挤度 x3、站上MA20 x1、换手率 x1
# 上涨占比不入（与MA20重复），ERP 不入（估值是慢变量，对3个月内的顶没有分辨力：
# 最高10%的日子 +4.02%、最低10% +7.91%，两端差异来自"便宜"那一侧，对减仓无用）。
#
# 权重不是拍的：在 7 因子 x 0-3 档权重的 16384 种组合里，筛出"两段子样本秩相关同号"的
# 15575 个合格组合后按"顶部十分位未来63日收益"排序，前 12 名全是同一个形状——
# TOP2 + 杠杆 + 重仓上涨拥挤度 + 一个宽度项；ERP 一次都没进过前列，换手率大多缺席或垫底。
# 敏感性：把任一权重 ±1（含把 MA20 或换手率整项删掉），触发后 +63日 都落在 -1.5% ~ -2.5%、
# 为负率 55%~65% 之间，没有一处是靠某个特定权重撑住的。
# 【2026-09 更新】"leverage" 这一格喂进去的东西换了：从"杠杆多空比"单项换成 leverage.py 的
# **杠杆温度**（多空比与杠杆ETF交易强度各半）。权重不变，仍是 2.0。换的原因与实测数字写在
# compose_sell 的注释里——一句话版本：多空比在 2022 年之后秩相关翻正（+0.062），
# 是这套减仓信号后段衰减的主要来源，补上交易强度后两段同号（-0.255 / -0.144）。
SELL_W = {"top2": 2.0, "leverage": 2.0, "narrow": 3.0, "ma20": 1.0, "turnover": 1.0}
SELL_TH = 75.0   # 红点门槛。扫描：>72 → 33段/-2.19%/61%为负；>75 → 26段/-1.79%/59%；
                 # >78 → 11段/-2.78%/65%。取 75：段数与现行红点相当的前提下质量最好。
# 试过但放弃：TOP2 改成"水平与63日变化各半"（单因子秩相关确实从 -0.143 升到 -0.171），
# 放进合成后 >75 是 16段/-2.46%/65%为负，看着更漂亮，但**整个 2018 年一天都不触发**
# （2018-01 与 2018-10 两个真顶全丢），减仓窗口覆盖率也从 13/18 掉回 12/18。
# 原因是 2018 年抱团度是高位但没有加速，变化率口径看不见它。故仍用纯水平分位。
#
# 效果（2017-10 ~ 2026-09，前瞻 QQQ）：
#   现行红点   56天/18段  +21日 +0.42%  +63日 +3.18%  63日回撤 -6.20%  34%为负
#   新  红点   81天/26段  +21日 -4.06%  +63日 -1.79%  63日回撤 -12.26%  59%为负
# 最能说明问题的是分年分布——现行红点的病是"把2023说成最热、把2021说成不热"：
#   年份      2018 2019 2020 2021 2022 2023 2024 2025 2026
#   现行红点     9    1   14    0    0   21    8    0    0   ← 2021泡沫顶0天、2023熊市后21天
#   新  红点     9    3   18   14    0    0   20   12    5   ← 2021补上、2023归零
# 对"未来63日内跌幅≥8%"窗口的覆盖率 12/18 → 13/18，且深度提升明显
# （2021-10~2022-01 从 7 天变 22 天，2019-12~2020-03 从 27 天变 40 天）。
#
# 【必读的局限】减仓侧的边际在后半段明显衰减：
#   2017-2021  44天  +63日 -5.40%（该段基准 +5.71%）  82%为负
#   2022-2026  37天  +63日 +2.49%（该段基准 +4.69%）  32%为负
# 两段方向一致（都低于基准），但优势从 11 个百分点缩到 2 个。全样本的 -1.79% 是被前半段
# 拉出来的。别按全样本数字设仓位。

SECTORS = {
 "信息技术": ["AAPL","MSFT","NVDA","ON","ZBRA"],
 "通信服务": ["GOOGL","META","OMC","NWSA","DIS"],
 "可选消费": ["AMZN","HD","DRI","PHM","BBY"],
 "金融":     ["JPM","V","CINF","RJF","ALL"],
 "医疗保健": ["LLY","JNJ","CRL","DGX"],
 "工业":     ["CAT","RTX","SNA"],
 "日常消费": ["PG","COST","MKC"],
 "能源":     ["XOM","CVX","HAL","OKE"],
 "公用事业": ["NEE","SO","NI"],
 "房地产":   ["PLD","AMT","KIM"],
 "原材料":   ["LIN","SHW","PKG"],
}
SECTOR_ETFS = {"XLK":"信息技术","XLC":"通信服务","XLY":"可选消费","XLF":"金融","XLV":"医疗保健",
               "XLI":"工业","XLP":"日常消费","XLE":"能源","XLU":"公用事业","XLRE":"房地产","XLB":"原材料"}
# 杠杆ETF：做多/做空各一篮，覆盖纳指、标普、小盘、半导体、金融、道指
LEV_LONG  = ["TQQQ","UPRO","SPXL","SSO","QLD","TNA","SOXL","FAS","TECL","UDOW"]
LEV_SHORT = ["SQQQ","SPXS","SDS","TZA","SOXS"]
LEV_ETFS  = LEV_LONG + LEV_SHORT

# 若存在 sectors.json（由 fetch_sp500_yahoo.py 生成），优先用它组建全成分股篮子
_sj = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sectors.json")
if os.path.exists(_sj):
    _m = json.load(open(_sj, encoding="utf-8"))
    _byc = {}
    for _t, _s in _m.items():
        _byc.setdefault(_s, []).append(_t)
    if sum(len(v) for v in _byc.values()) >= 20:
        SECTORS = {k: sorted(v) for k, v in sorted(_byc.items())}

STOCKS = [t for v in SECTORS.values() for t in v]
SECTOR_OF = {t: s for s, v in SECTORS.items() for t in v}


# ---------- 数据加载 ----------
def load(ticker):
    p = os.path.join(RAW, f"{ticker}.csv")
    if not os.path.exists(p):
        return None
    ncol = len(pd.read_csv(p, header=None, nrows=1).columns)
    if ncol >= 4:
        df = pd.read_csv(p, header=None, names=["date", "close", "rawclose", "volume"])
    else:
        ncol = len(open(p).readline().split(","))
    if ncol >= 4:
        df = pd.read_csv(p, header=None, names=["date", "close", "rawclose", "volume"])
    else:
        df = pd.read_csv(p, header=None, names=["date", "close", "volume"])
        df["rawclose"] = df["close"]
        df["rawclose"] = df["close"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[pd.to_numeric(df["close"], errors="coerce").notna()]
    df["close"] = df["close"].astype(float)
    df["rawclose"] = pd.to_numeric(df["rawclose"], errors="coerce").fillna(df["close"]).astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(float)
    df["rawclose"] = pd.to_numeric(df["rawclose"], errors="coerce").fillna(df["close"]).astype(float)
    df = df.drop_duplicates(subset="date").sort_values("date").set_index("date")
    return df


def load_many(tickers):
    out = {}
    for t in tickers:
        d = load(t)
        if d is not None and len(d) > 30:
            out[t] = d
    return out


# ---------- 滚动分位 ----------
def rolling_pct(s, window=WINDOW, min_periods=MIN_PERIODS):
    """当前值在过去 window 个观测中的百分位（0-100）。
    采用 (小于本值的个数 + 0.5*等于本值的个数) / N * 100，标准 percentileofscore 'mean' 口径。"""
    s = pd.Series(s).astype(float)
    def f(x):
        cur = x[-1]
        n = len(x)
        if n < 2 or not np.isfinite(cur):
            return np.nan
        less = np.sum(x < cur)
        eq = np.sum(x == cur)
        return (less + 0.5 * eq) / n * 100.0
    return s.rolling(window, min_periods=min_periods).apply(f, raw=True)


def expanding_pct(s, min_periods=MIN_PERIODS):
    """扩张窗口分位：当前值在"截至当日的全部历史"中的百分位。无前视。"""
    s = pd.Series(s).astype(float)
    def f(x):
        cur = x[-1]
        if len(x) < 2 or not np.isfinite(cur):
            return np.nan
        prev = x[:-1]
        return (np.sum(prev < cur) + 0.5 * np.sum(prev == cur)) / len(prev) * 100.0
    return s.expanding(min_periods=min_periods).apply(f, raw=True)


def abs_map(s, anchors):
    """固定锚点分段线性映射：lo→0, mid→50, hi→100，超出部分截断。
    没有任何滚动参照，同一个水平值在任何年份读数相同。"""
    lo, mid, hi = anchors
    x = pd.Series(s).astype(float)
    out = np.where(x <= mid, (x - lo) / (mid - lo) * 50.0,
                   50.0 + (x - mid) / (hi - mid) * 50.0)
    out = np.clip(out, 0.0, 100.0)
    return pd.Series(np.where(np.isfinite(x), out, np.nan), index=x.index)


# ---------- 六个指标 ----------
def complete_through(idx, stocks, spy):
    """返回"输入齐备"的最后一个交易日。

    判定条件（全部满足才算齐备）：
      · SPY 当日有数据（交易日轴与ERP的基准）
      · 成分股当日有效样本 ≥ 全样本的 90%
      · 做多、做空杠杆ETF各自当日至少有一半标的有数据
    数据源对不同标的的更新时间不一致，最新一两天常出现"成分股齐了、杠杆ETF还没到"，
    此时若照常计算，5日/10日平滑会用前几天的值把缺口填上，产出误导性的"最新"读数。
    """
    def cover(tickers):
        got = load_many(tickers)
        if not got:
            return pd.Series(0.0, index=idx)
        m = pd.DataFrame({t: d["close"] for t, d in got.items()}).reindex(idx).notna()
        return m.sum(axis=1) / len(tickers)

    ok = pd.Series(True, index=idx)
    if spy is not None:
        ok &= spy["close"].reindex(idx).notna()
    if stocks:
        px = pd.DataFrame({t: d["close"] for t, d in stocks.items()}).reindex(idx)
        ok &= (px.notna().sum(axis=1) / len(stocks) >= 0.90)
    ok &= (cover(LEV_LONG) >= 0.5)
    ok &= (cover(LEV_SHORT) >= 0.5)

    good = ok[ok].index
    if len(good) == 0:
        raise SystemExit("没有任何交易日的输入是齐备的")
    last = good[-1]
    dropped = int((idx > last).sum())
    if dropped:
        print(f"  ⓘ 数据完整性闸门：最新 {dropped} 个交易日输入不齐（多为杠杆ETF尚未更新），"
              f"本次计算截至 {last.strftime('%Y-%m-%d')}")
    return last


def build_indicators():
    stocks = load_many(STOCKS)
    etfs = load_many(list(SECTOR_ETFS))
    levs = load_many(LEV_ETFS)
    spy = load("SPY")

    if not stocks and not etfs:
        raise SystemExit("raw/ 下没有可用数据")

    # 统一交易日轴：用覆盖最全的来源
    ref = spy if spy is not None else (list(etfs.values())[0] if etfs else list(stocks.values())[0])
    idx = ref.index

    # 数据完整性闸门：数据源对不同标的的更新时间不一致（杠杆ETF往往比成分股晚几小时）。
    # 若最新交易日的输入不齐，平滑窗口(min_periods=1)会用旧数据把缺口悄悄补上，
    # 产出一个"看似最新、实则掺了隔夜陈数据"的温度。此处直接把交易日轴截断到
    # 最后一个各类输入都齐备的交易日，宁可晚一天，也不发布半成品。
    idx = idx[idx <= complete_through(idx, stocks, spy)]

    raw = {}
    meta = {}

    # --- 1. 换手率：篮子成交额 / 篮子市值 ---
    if stocks:
        shares = load_shares()
        dv = pd.DataFrame({t: d["rawclose"] * d["volume"] for t, d in stocks.items()}).reindex(idx)
        px_all = pd.DataFrame({t: d["close"] for t, d in stocks.items()}).reindex(idx)
        if shares:
            mc = pd.DataFrame({t: d["close"] * shares[t] for t, d in stocks.items() if t in shares}).reindex(idx)
            turnover = dv.sum(axis=1, min_count=1) / mc.sum(axis=1, min_count=1) * 100
            meta["turnover"] = "篮子日成交额 / 篮子总市值"
        else:
            # 无股本数据：按等份额市值归一，剔除大盘价格水平漂移对成交额的机械抬升
            turnover = dv.sum(axis=1, min_count=1) / px_all.sum(axis=1, min_count=1)
            meta["turnover"] = "篮子日成交额 / 篮子价格和（等份额近似换手率；无公开股本数据）"
        raw["_turnover_daily"] = turnover
        raw["turnover"] = turnover.rolling(TO_SMOOTH, min_periods=1).mean()

    # --- 2. TOP2 行业成交额占比 ---
    # 样本足够时用成分股实际成交额按行业汇总（"行业成交额"的本义）；否则退回行业ETF成交额
    if stocks and len(stocks) >= 60:
        sdv = {}
        for sec, tks in SECTORS.items():
            cols = [t for t in tks if t in stocks]
            if not cols:
                continue
            sdv[sec] = pd.DataFrame(
                {t: stocks[t]["rawclose"] * stocks[t]["volume"] for t in cols}
            ).reindex(idx).sum(axis=1, min_count=1)
        sdv = pd.DataFrame(sdv)
        share = sdv.div(sdv.sum(axis=1, min_count=1), axis=0)
        raw["top2"] = share.apply(lambda r: r.nlargest(2).sum() * 100 if r.notna().sum() >= 2 else np.nan, axis=1)
        meta["top2"] = f"{len(stocks)}只成分股按{len(sdv.columns)}个GICS行业汇总成交额，最大两个行业的合计占比"
    elif len(etfs) >= 6:
        sdv = pd.DataFrame({SECTOR_ETFS[t]: d["rawclose"] * d["volume"] for t, d in etfs.items()}).reindex(idx)
        share = sdv.div(sdv.sum(axis=1, min_count=1), axis=0)
        raw["top2"] = share.apply(lambda r: r.nlargest(2).sum() * 100 if r.notna().sum() >= 2 else np.nan, axis=1)
        meta["top2"] = f"{len(etfs)}个行业ETF成交额中最大两个行业的合计占比"

    # --- 3. 上涨个股占比（10日平滑）---
    if stocks:
        px = pd.DataFrame({t: d["close"] for t, d in stocks.items()}).reindex(idx)
        up = (px.diff() > 0)
        valid = px.diff().notna()
        adv = up.sum(axis=1) / valid.sum(axis=1).replace(0, np.nan) * 100
        raw["_advancing_daily"] = adv
        raw["advancing"] = adv.rolling(ADV_SMOOTH, min_periods=min(3, ADV_SMOOTH)).mean()
        meta["advancing"] = ("篮子内上涨家数占比（当日值，未平滑）" if ADV_SMOOTH <= 1
                             else f"篮子内上涨家数占比（{ADV_SMOOTH}日均，原始日频噪声过大）")

        # --- 4. 站上MA20占比 ---
        ma20 = px.rolling(MA_WIN, min_periods=MA_WIN).mean()
        above = (px > ma20)
        raw["ma20"] = above.sum(axis=1) / ma20.notna().sum(axis=1).replace(0, np.nan) * 100
        meta["ma20"] = f"篮子内收盘价站上{MA_WIN}日均线的家数占比"

        # --- 4b. 上涨拥挤度（不进入蓝点温度，单独作为第四类预警）---
        # 蓝点温度有个结构性盲区：宽度差一律读成"降温"。但指数创新高的同时宽度崩坏，
        # 恰恰是最经典的顶部形态之一（2024年12月：纳指从505涨到525、站上MA20占比却
        # 掉了44个百分点，温度只有35-64，红点黑框都没响）。这个评分专门抓这种形态。
        eq_ret = px.pct_change().mean(axis=1)              # 等权组合日收益（截面均值）
        eq = (1.0 + eq_ret.fillna(0)).cumprod()
        cw = spy["close"].reindex(idx) if spy is not None else None
        nq = load("QQQ")
        nq = nq["close"].reindex(idx) if nq is not None else cw
        if cw is not None and nq is not None:
            narrow = (cw / cw.shift(NT_WIN)) / (eq / eq.shift(NT_WIN)) - 1.0   # 市值加权跑赢等权
            brd_chg = raw["ma20"] - raw["ma20"].shift(NT_WIN)                  # 宽度变化
            dd = (nq / nq.cummax() - 1.0) * 100.0                              # 距峰值回撤
            near = ((dd + NT_DD) / NT_DD * 100.0).clip(0, 100)                 # 越靠近峰值越高
            sc = (rolling_pct(narrow) + rolling_pct(-brd_chg) + near) / 3.0
            raw["_narrow_score"] = sc.where(cw.pct_change(NT_WIN) > 0)  # 只在上涨市里成立
            # 供红点温度合成用的版本：下跌市里记为中性 50 而不是缺失。
            # 两者的区别很重要——预警用的 _narrow_score 必须在下跌市里"无定义"（上涨拥挤度
            # 本来就只在上涨市成立，那是它的语义）；但合成用的不能是 NaN，因为 compose_sell
            # 要求各项齐备，一个 NaN 会让整个下跌市的红点温度消失。下跌市读 50 语义上也对：
            # 那时候该说话的是熊市反弹预警，不是上涨拥挤度。
            # 注意两层 where 的分工：外层把"下跌市"填成 50，内层把"历史不足、分位还算不出来"
            # 的日子保留为 NaN——后者是真的没有读数，不能假装中性。
            raw["_narrow_neutral"] = sc.where(cw.pct_change(NT_WIN) > 0, 50.0).where(sc.notna())
            raw["_narrow_gap"] = narrow * 100.0
            raw["_breadth_chg"] = brd_chg
            raw["_ndx_dd"] = dd

    # --- 5. 杠杆资金多空比 ---
    # 美股无日频融资买入额（FINRA 只有月频余额、滞后数周，且其 WAF 按云厂商 IP 段封，
    # CI 上取不到）。这里用杠杆ETF的多空成交额之比：
    # 崩盘时做空杠杆ETF成交额激增、比值塌陷；狂热时相反。方向内生，无需外部修正。
    longs, shorts = load_many(LEV_LONG), load_many(LEV_SHORT)
    # 多空比本身的计算已搬到 leverage.py（LV.letf_components）——它是杠杆温度的一条腿，
    # 两处各算一遍必然会分叉。这里只负责取值与命名，口径与旧版逐日一致。
    lev_frames = dict(longs)
    lev_frames.update(shorts)
    for t in LV.BASE_ETFS:
        if t not in lev_frames:
            d = load(t)
            if d is not None:
                lev_frames[t] = d
    lv_raw, lv_meta, lv_notes = LV.build(idx, lev_frames)
    lev_info = {"meta": lv_meta, "notes": lv_notes}

    if "ratio" in lv_raw:
        raw["_leverage_daily"] = lv_raw["ratio_daily"]
        raw["leverage"] = lv_raw["ratio"]
        meta["leverage"] = (f"做多杠杆ETF成交额 / 杠杆ETF总成交额（{len(longs)}只做多、{len(shorts)}只做空，"
                            f"覆盖纳指/标普/小盘/半导体/金融/道指）")
    elif levs and (stocks or etfs):
        ldv = pd.DataFrame({t: d["rawclose"] * d["volume"] for t, d in levs.items()}).reindex(idx).sum(axis=1, min_count=1)
        base = pd.Series(0.0, index=idx)
        if stocks:
            base = base.add(pd.DataFrame({t: d["rawclose"] * d["volume"] for t, d in stocks.items()}).reindex(idx).sum(axis=1, min_count=1), fill_value=0)
        if etfs:
            base = base.add(pd.DataFrame({t: d["rawclose"] * d["volume"] for t, d in etfs.items()}).reindex(idx).sum(axis=1, min_count=1), fill_value=0)
        raw["leverage"] = (ldv / base.replace(0, np.nan) * 100).rolling(TO_SMOOTH, min_periods=1).mean()
        meta["leverage"] = "杠杆ETF成交额 / 篮子总成交额（回退口径：缺做空杠杆ETF数据）"

    # --- 5b. 杠杆温度的第二条腿：交易强度 ---
    # 以 `_lev_` 前缀存进 rawdf —— 下划线开头的列 compose() 会跳过，
    # 所以六项蓝点温度、蓝点温度（当日口径）、蓝点的标定一律不受影响（这是有意为之，见 main()）。
    for k, v in lv_raw.items():
        if k not in ("ratio", "ratio_daily"):
            raw[f"_lev_{k}"] = v

    # --- 6. 风险溢价 ERP ---
    got = build_erp(idx, spy)
    if got is not None:
        erp, epy = got
        raw["erp"] = erp
        if ERP_MODE in ("blend", "ep"):
            raw["_ep"] = epy          # 供 compose 合成用，不进入六项均值
        meta["erp"] = {
            "erp":   "标普500盈利收益率(E/P) − 10年期美债收益率",
            "blend": "ERP(E/P−10Y) 分位与 E/P 分位各取一半——E/P 不含利率，抵消加息周期对 ERP 的单边压缩",
            "ep":    "标普500盈利收益率 E/P（不减利率，完全规避加息周期干扰）",
        }[ERP_MODE]

    return pd.DataFrame(raw).reindex(idx), meta, spy, lev_info


def _load_rate(fname):
    """财政部日频收益率 CSV（date,value，新到旧）→ 完整历史序列（%）。取不到返回 None。"""
    p = os.path.join(RAW, fname)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, header=None, names=["date", "v"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    s = df.dropna().drop_duplicates("date").sort_values("date").set_index("date")["v"]
    return s if len(s) >= RR_WIN + 20 else None


def align_to(s, idx):
    """把日频外部序列对齐到交易日索引（前向填充，不外推）。s 为 None 时返回 None。"""
    return None if s is None else s.reindex(idx.union(s.index)).ffill().reindex(idx)


def load_real_rate(idx=None):
    """10 年期 TIPS 实际收益率（%）。取不到返回 None，宏观闸门自动不启用。

    **idx=None 时返回完整历史序列**，这不是可有可无的便利参数：重估期要算两条滚动分位
    （6 个月变动的 3 年分位、水位的 5 年分位），都必须在完整 raw 序列上算完再截断到
    展示窗口。若先截断再算，展示窗口开头约一年半全是 NaN，那段时间的蓝点会因为
    "没数据"而不是"判断为不符合"被记成实心——2018-12 那次正好落在这里，
    上一版就是栽在这上面（见 repricing_regime 的注释）。
    raw/_dfii10.csv 由 fetch_vix_dgs10.py 回补 11 年，够预热。
    """
    return align_to(_load_rate("_dfii10.csv"), idx) if idx is not None else _load_rate("_dfii10.csv")


def load_nominal_rate(idx=None):
    """10 年期名义美债收益率（%）。与 build_erp 用的是同一个 raw/_dgs10.csv。"""
    return align_to(_load_rate("_dgs10.csv"), idx) if idx is not None else _load_rate("_dgs10.csv")


def load_vix(idx):
    """CBOE VIX 收盘（日频）。取不到则返回 None，蓝点预警退回仅看温度。"""
    p = os.path.join(RAW, "_vix.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, header=None, names=["date", "vix"])
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["vix"] = pd.to_numeric(df["vix"], errors="coerce")
    s = df.dropna().drop_duplicates("date").sort_values("date").set_index("date")["vix"]
    return s.reindex(idx.union(s.index)).ffill().reindex(idx)


def load_shares():
    p = os.path.join(RAW, "_shares.csv")
    if not os.path.exists(p):
        return {}
    df = pd.read_csv(p, header=None, names=["ticker", "shares"])
    return {r.ticker.strip().upper(): float(r.shares) for r in df.itertuples() if np.isfinite(r.shares)}


def build_erp(idx, spy):
    """ERP = E/P − 10Y。EPS 用标普500 TTM 每股收益（月频，前向填充），指数用 SPY 收盘价换算。"""
    eps_p = os.path.join(RAW, "_sp500_eps.csv")
    y_p = os.path.join(RAW, "_dgs10.csv")
    if spy is None or not os.path.exists(eps_p) or not os.path.exists(y_p):
        return None
    eps = pd.read_csv(eps_p, header=None, names=["date", "eps"])
    eps["date"] = pd.to_datetime(eps["date"], errors="coerce")
    eps = eps.dropna().drop_duplicates("date").sort_values("date").set_index("date")["eps"].astype(float)
    y = pd.read_csv(y_p, header=None, names=["date", "y"])
    y["date"] = pd.to_datetime(y["date"], errors="coerce")
    y = y.dropna()
    y = y[pd.to_numeric(y["y"], errors="coerce").notna()]
    y = y.drop_duplicates("date").sort_values("date").set_index("date")["y"].astype(float)

    eps_d = eps.reindex(idx.union(eps.index)).ffill().reindex(idx)
    y_d = y.reindex(idx.union(y.index)).ffill().reindex(idx)
    # SPY ≈ 标普500 / 10
    spx = spy["close"].reindex(idx) * 10.0
    ep = eps_d / spx * 100.0          # 盈利收益率 %
    return ep - y_d, ep                # (ERP %, E/P %)


# ---------- 合成 ----------
# 展示用：哪些分项对外显示的是平滑值，以及对应的未平滑当日值字段
SMOOTHING = {k: (f"{w}日均", col) for k, w, col in [
    ("turnover",  TO_SMOOTH,  "_turnover_daily"),
    ("advancing", ADV_SMOOTH, "_advancing_daily"),
    ("leverage",  TO_SMOOTH,  "_leverage_daily"),
] if w > 1}

LABELS = {
    "turnover":  ("换手率", "成交活跃度"),
    "top2":      ("TOP2行业成交额占比", "资金抱团度"),
    "advancing": ("上涨个股占比", "普涨程度"),
    "ma20":      ("个股站上MA20占比", "中期趋势宽度"),
    "leverage":  ("杠杆资金多空比", "加杠杆方向"),
    "erp":       ("风险溢价 ERP", "估值性价比（反向）"),
}
ORDER = ["turnover", "top2", "advancing", "ma20", "leverage", "erp"]
# 方向不可知的"活跃度"类指标：恐慌与狂热都会让它们飙升，需按市场方向定向
# 方向不可知的"活跃度"类指标：恐慌与狂热都会让它们飙升，需按市场方向定向。
# 杠杆指标改用多空比后已自带方向（与20日收益秩相关 +0.44），不再需要外部修正。
DIRECTIONAL = ["turnover"]
DIR_WIN  = 20    # 用于判断方向的收益窗口（交易日）
DIR_SCALE = 1.5  # tanh 的软化系数：越大越保守（方向不明确时更接近中性）
# 仅让"高于常态"的活跃度带方向：低于常态的活跃度不含方向信息，记为中性(50)。
# 与"全修正"（正负偏离都带方向）相比，半修正在 2016-2026 样本上五档收益完全单调，
# 秩相关也更强(-0.075 vs -0.065)，且避免了"缩量下跌=偏热"这种反直觉读数。
DIR_HALF = True


def direction(spy, idx):
    """市场方向系数 d ∈ [-1, +1]：
    强势上涨 → +1（活跃＝贪婪）；急跌 → -1（活跃＝恐慌）；横盘 → ~0（活跃不含方向信息）。
    用 tanh 平滑，避免在符号翻转处产生跳变。"""
    if spy is None:
        return None
    px = spy["close"].reindex(idx).ffill()
    r = px.pct_change(DIR_WIN)
    sd = r.rolling(WINDOW, min_periods=60).std()
    return np.tanh(r / (DIR_SCALE * sd.replace(0, np.nan)))


# ---------- 预警规则 ----------
# —— 蓝点的两个门槛 ——
# 蓝点是全系统最好的信号，骨架不要动：VIX 与温度必须**同时**到位，是交集在起作用而不是权重。
# 实测（非重估期）：VIX≥30 单独 +63日 +17.89%，快温<20 单独 +16.73%，两者相与 +26.68%。
# 只调了两处标定，第一处是被 ERP 口径变更逼出来的：
#   1) ERP 改纯口径后快温度整条曲线右移（最大偏移 7.9 分），旧的 <20 在新单位下约等于 <27，
#      门槛必须重标。扫描 VIX≥30 下的新快温：<18 → 13段/+29.09%，<20 → 15段/+28.64%，
#      <22 → 16段/+27.98%，<25 → 16段/+27.49%/0%为负，<28 → 19段/+25.63%/3%，<30 → 20段/+24.87%/5%。
#      取 25：曲线平坦段的中部，9 年 16 段一次都没亏过（最差 +6.8%），不卡在边缘。
#   2) VIX 门槛曾放宽到 25 补覆盖，2026-09 按用户要求改回 30——取更少的段数换零亏损：
#        VIX≥30 → 16段 +27.49% / 0%为负 / 覆盖6段  ← 现行
#        VIX≥26 → 19段 +24.32% / 3%为负 / 覆盖6段
#        VIX≥25 → 21段 +23.28% / 5%为负 / 最差-4.1% / 覆盖7段
#        VIX≥24 → 25段 +20.67% / 8%为负
#      代价：按"相对6个月高点回撤≥8%"数的 16 个可加仓窗口，覆盖从 7 段退回 6 段，
#      触发频率约 2.3/年 → 1.8/年。
# 试过但**放弃**的两条：
#   · 把 VIX 改成滚动分位（"自适应"）：反而更差（+26.68% → +17~19%）。绝对阈值优于相对分位——
#     VIX 30 是恐慌的绝对刻度，拿它跟过去一年比等于把平静期的小波动也算成恐慌。别"现代化"这条。
#   · 外挂一条"二档蓝点"（回撤>10% & 站上MA20<15% & VIX分位>70）补 VIX 不到门槛的中级回调：
#     不加闸门时 2022 年后半段 -0.29%/62%为负（16 天全在 2022 熊市里）；加熊市闸门后 2022 年
#     之后一次都不触发，等于零样本外证据；而且它并没覆盖当初想补的 2023-10/2024-08/2024-09
#     （那三次站上MA20占比分别是 15.2/28.7/43.5，都够不着 <15）。整条放弃——
#     剩下的漏点多数是"VIX 没到、情绪也根本没洗盘"，那不该由情绪指标负责。
# 顺带：ERP 改纯口径本身就补回了两次大漏——2020-02（-13.2%）与 2024-08（-13.6%），
# 这两次 VIX 早就过了 30，卡住的是旧口径下偏高的快温度。不用加任何新规则。
VIX_COLD = 30   # 蓝点预警的附加条件：VIX ≥ 此值
COLD_TH  = 25.0 # 蓝点的蓝点温度（当日口径）门槛（原先直接借用 BANDS[0]=20，现与展示分档解耦：
                # 展示分档是给人看的语义，信号门槛是标定出来的，两者不该被同一个数字绑死）
NDX_DD_COLD = 0     # 蓝点附加条件：纳指自峰值回撤 ≥ 此百分比（0 = 不启用）。
                    # 试过 13：十年只挡掉一天（2026-03-27，快温度18.5、VIX31.1、回撤仅11.3%），
                    # 而那次后续3个月涨了28.9%，等于唯一一次生效是挡掉了好信号，故关闭。
                    # 峰值取截至当日的累计最高（cummax），不含前视；改回非零即可重新启用。
# —— 实际利率重估期（蓝点的宏观闸门）——
# 蓝点是"恐慌到位"的信号，它只对**冲击式**下跌有效：跌得快、VIX 炸、几周内跌完。
# 2022 年那种由实际利率抬升驱动的估值重估是另一回事——问题不在分子（情绪）而在分母
# （贴现率），情绪指标再低也没用，因为跌的原因还没结束。
#
# 判据两条同时成立，缺一不可：
#   快    6 个月变动排进它自己过去 3 年的前 15%（RR_PCT_TH）——重估正在进行时
#   还没完 水位仍在自己过去 5 年分布的下半部（RR_LVL_TH）——估值仍有压缩空间
# 外加一条排除项：同期名义 10Y 也得在上行（见下）。
#
# —— 2026-09 改版：两条门槛都从**绝对值**换成**自身分位** ——
# 旧式是 实际利率 < 0.5% 且 6 个月上行 > 0.25pp，两个数都是零利率年代的刻度。
# 实际利率现在 2.6%，"< 0.5%" 永远不可能再成立，整条闸门等于装饰品——未来任何一次
# 2.6% → 3.5% 的重估都不会被识别。绝对涨幅门槛同样不跨期可比，"快"这把尺子自己在漂：
#     2018 最大 +0.44 | 2019 最大 +0.23 | 2022 中位 +1.03、最大 +2.20 | 2026 至今最大 +0.71
# 换成分位之后两条都随利率环境自动伸缩：水位比的是"相对自己过去五年高不高"，
# 而不是"比 0.5% 高不高"。真正回答了旧口径失效的问题，同时保住了它有效的那部分。
#
# —— 血的教训：这一条别再只看速度 ——
# 本次改版第一版**删掉了水位项**，只留速度分位。合并上线后才发现它把三个最好的蓝点
# 打成了空心：2018-12-21（后63日 +21.9%）、2018-12-24（+24.2%）、2026-03-27（+28.9%）。
# 原因是纯速度分不开 2018Q4 与 2022Q1——两段的 6 个月上行幅度几乎一样：
#     2018-10-05  实际利率 1.05%  6个月 +0.40pp      → 两个月后是极好的蓝点
#     2022-01-06  实际利率 -0.70% 6个月 +0.45pp      → 之后三个月全是亏钱的蓝点
# 差 5 个基点，任何纯速度判据（绝对涨幅或分位）都做不出区分。真正的差别在水位：
#     2018-10 实际利率在自己五年分布的 99.9 分位（重估早就走完，那是加息周期的尾声）
#     2022-01 在 27.5 分位（重估刚开始，估值还有一大截要压）
# 反过来，**只看水位也不行**：单独用"水位 < 50 分位"会把 2020-03 疫情底那 14 天
# 全部打成空心（那时实际利率确实在低位，但它在**跌**不在涨）。两条缺一不可。
#
# 名义同向闸门（RR_NEED_NOMINAL）：实际利率上行有两种成因，只有一种算重估。
#   2020-03 疫情底实际利率 6 个月 +0.32，但名义 10Y 是 −0.72、盈亏平衡通胀塌了 1pp——
#   涨的原因是通缩恐慌不是紧缩。当前参数下水位项已经先一步挡住了它（61.8 分位 > 50），
#   所以这一条**样本内不吃劲**；留着是因为它挡的那类错误最致命（把史上最好的蓝点判反），
#   而且水位门槛取到 60 时 2020-03 只差 1.8 个分位就漏过来，它是那个角上的唯一保险。
#   故取不到 _dgs10.csv 时整条闸门不启用（蓝点全记实心），而不是退化成只看分位。
#
# RR_MIN_RISE：分位是相对的，分布平稳时永远有 15% 的日子在 85 分位之上——利率死水一潭
#   的年代，6 个月挪 5bp 也能"排进前 15%"。入场再加一个绝对下限挡掉这种噪声。
#   只加在**入场**不加在退出：加在退出会引入闪断（2022-03 俄乌避险把 6 个月变动压回
#   0.3 以下，那段最糟的蓝点会只剩第一天被标出来，正是滞回要修的问题）。
#   这一条样本内同样不吃劲，防的是样本里没有的场景。
#
# 实测（2017-10 至今 42 个蓝点日 / 7 次事件，触发后 63 日 QQQ）：
#   空心 13 天  −8.69% / 100%为负   |  实心 29 天  +27.49% / 0%为负
# 参数是一整片平坦区，不是调出来的点：速度窗口 3年/5年 × 速度门槛 80/85/90 ×
# 水位门槛 40/50/60 共 18 组，分组结果**完全一致**（这次是在生产用的完整原始序列上
# 扫的；上一版的平坦区结论扫在截断过的序列上，是错的）。
# 判别列的间距也大：13 个坏点的水位分位落在 25.7~30.7，7 个好点全部 61.8~100。
#
# 局限没变：这个宏观状态在样本里只出现过一次（2021H2–2022），等于零样本外验证。
# 故不屏蔽蓝点，只降级为空心蓝点（分批 / 半仓），保留信息而不假装确定。
# 另外要诚实的一点：水位项意味着这条闸门只在"实际利率从被压低的水位往上修复"时才亮。
# 它比旧的 0.5% 能跨利率环境（比的是自己过去五年，不是一个固定数字），但仍然
# **没有**在非零利率环境里被验证过——样本里没有那种情形。
RR_WIN      = 126   # 变动窗口：6 个月 ≈ 126 个交易日
RR_PCT_WIN  = 756   # 速度的比较窗口：跟自己过去 3 年 ≈ 756 个交易日比
RR_PCT_MIN  = 252   # 分位的最小样本数（不足则不判定，无前视）
RR_PCT_TH   = 85    # 进入：6 个月变动的滚动分位 > 此值
RR_PCT_EXIT = 50    # 退出：该分位跌回此值以下（滞回）
RR_MIN_RISE = 0.20  # 入场的绝对下限（百分点）：排名再靠前也得先真的涨过这么多
RR_LVL_WIN  = 1260  # 水位的比较窗口：跟自己过去 5 年 ≈ 1260 个交易日比
RR_LVL_MIN  = 252
RR_LVL_TH   = 50    # 进入：水位分位 < 此值（重估还没走完）
RR_LVL_EXIT = 70    # 退出：水位分位 > 此值（重估走完了）
RR_NEED_NOMINAL = True   # 要求名义 10Y 同期同向上行（见上）

# —— 熊市反弹预警（补上 2022 年缺失的减仓信号）——
# 熊市里滚动分位的参照系全是低值，反弹再猛温度也顶不到 80，于是整个 2022 年一个红点都没有。
# 对策不是改分位口径（换扩张窗口会把 2020-03 的蓝点和几乎所有红点一起抹掉），
# 而是**在确认的下行趋势里单独降低过热门槛**：跌破 200 日均线、且 200 日均线本身在下行时，
# 温度 > BEAR_TH 连续 3 日即预警。实测 9 年只触发 15 天 / 4 段，全部落在 2022-06 至 2023-01，
# 触发日后 3 个月平均 -11.3%、87% 为负（对照：标准红点 +3.2%、34% 为负）。
BEAR_MA    = 200   # 趋势基准均线
BEAR_SLOPE = 21    # 均线斜率的回看天数（均线较 21 日前更低 = 下行）
BEAR_TH    = 60    # 熊市中的过热门槛（标准红点是 80）

# 上涨拥挤度参数
NT_WIN = 63     # 回看窗口（约3个月）：上涨拥挤的顶是慢慢形成的，不能用短窗口
NT_DD = 6.0     # 距峰值回撤在此百分比内才算"仍在高位"（0%回撤=100分，-6%及以下=0分）
NT_TH = 85      # 【当前未被任何规则使用】原橙线预警的阈值。橙线已于 2026-09 取消，
                # 其"上涨拥挤度"的信息并入黑框（见下）。保留此常量供回退。
                # NT_WIN / NT_DD 仍在使用——它们是 _narrow_score 的计算参数，与本阈值无关。
# 黑框（2026-09 定版）= TOP2 抱团分位 > CROWD_TOP2 且 上涨拥挤度 > CROWD_NARROW。
# 同时取消了橙线预警——它原本只看上涨拥挤度一项，与本条高度同源，留着是两个名字讲同一件事。
#
# 两项口径不同，别混为一谈：TOP2 是 252 日滚动分位，上涨拥挤度是 0-100 的合成分。
# 副作用（必须知道）：上涨拥挤度只在**上涨市**里有定义（指数 NT_WIN 日为跌时是 NaN，
# 见 _narrow_score），因此黑框在下跌市中永不触发。作为"抱团顶部形态"这是合理的，
# 但它不是一个"任何时候都能报抱团"的信号。
#
# 标定依据（QQQ 63 交易日远期，连 3 日确认，基准 +5.19% / 27%为负 / 触发后回撤 -5.87%）：
# TOP2 × 上涨拥挤度是一整片平坦的有效区，不是尖峰——上涨拥挤度 65~80 四列 × TOP2 70~90 五行，
# 日级收益全部落在 -0.21% ~ -1.63%，为负 48~63%，触发后 63 日最大回撤 -11.9% ~ -13.7%。
# 取 80/75 而非矩阵里数字更好看的格子，理由是两段子样本同号：
#   前段 2017-2021  -1.73% / 68%为负      后段 2022-2026  -0.39% / 54%为负
# 上涨拥挤度取 75 而不是 70，正因为 70 的后段会翻正（+0.47%）。
# 全样本 56 天 / 11 段（约 1.2 次/年），日级 -0.84% / 59%为负 / 回撤 -12.28%。
#
# **要打的折扣**：只有 11 段，且集中在 2020、2024、2026 三年，2017/2018/2019/2021/
# 2022/2023 六年零触发——2021 泡沫顶和 2022 熊市它都没响。最近一段 2026-05-07~06-03
# 后 63 日 +4.16%，是错报。参数仍是在同一段历史上调出来的，平坦区只降低过拟合风险，消不掉。
CROWD_TOP2   = 80  # 黑框：TOP2 行业成交额占比的 252 日滚动分位门槛
CROWD_NARROW = 75  # 黑框：上涨拥挤度门槛（不是分位，是评分本身）
CROWD_LEV    = 80  # 【当前未被任何规则使用】曾用于黑框的杠杆分位门槛，保留供回退。
# 为什么杠杆不进黑框：杠杆**有**独立预测力——控制过去 63 日动量后，它对未来 63 日收益的
# 偏相关在各子时期是 -0.19(全) / -0.34(2017-20) / -0.59(2021) / -0.43(2023-26)，
# 2021 与 2023-26 两段甚至强于上涨拥挤度。但它与过去收益的秩相关高达 +0.50~+0.72（强烈跟涨），
# 任何"高位切分"都会把"因为涨了所以杠杆高"（无信息）和"杠杆异常地高"（有信息）混在一起，
# 阈值化即毁掉该信息：滚动分位 >=80 覆盖 27.4% 的交易日、回撤 -6.48%（基准 -5.87%）；
# 绝对水平跨期漂移，剔除 2021 后 >80 是 +3.58%；z-score 在 2023-26 偏相关归零。
# 结论是**连续地用**——它已在 SELL_W 里以 leverage=2.0 进入红点温度，那才是正确位置。
# 另注：高杠杆的后果是"后续涨得少"，不是"跌得更深"（下跌深度 -11.10% vs 全样本 -11.46%）
# 也不是"波动更大"（未来 63 日年化波动 21.4% vs 21.9%）。
# 三类预警各用各的参数——顶是慢过程、底是快事件，用同一套过滤必然顾此失彼：
#   红点：情绪见顶是多日堆积，平滑口径 + 3日确认，滤掉毛刺（实测触发后3个月 +3.3%，基准 +5.4%）
#   蓝点：恐慌见底是插针，当日口径 + 不确认，宁可多叫几次也别错过（3日确认的代价实测中位 -0.52%）
#   黑框：抱团+杠杆共振本身是持续状态，保留3日确认，否则触发段过碎
PERSIST = 3   # 默认确认天数（红点/黑框）
PERSIST_COLD = 1
PERSIST_NARROW = 3   # 【当前未被任何规则使用】原橙线的确认天数，保留供回退。
                     # 试过 5 日：段数 17→12，但 2024-12 那次整个丢失——12/12–12/17 与
                     # 12/23–12/27 两段各只有 4 天，被 12/18 FOMC 急跌打断。而那次正是
                     # 本预警最有代表性的一次（触发后3个月 -10.8%）。判别力也随天数变弱：
                     # 3日 +0.20%、4日 +0.75%、5日 +1.25%（基准 +5.19%，越低越有区分度）。

def _p(n):
    return f"连续 {n} 个交易日" if n > 1 else "当日成立即触发"

ALERTS = [
    {"key": "hot",   "name": "红点预警", "mark": "dot",  "color": "#CE5A4E", "persist": PERSIST,
     "desc": f"红点温度 > {SELL_TH:.0f}，{_p(PERSIST)}。红点温度＝TOP2抱团×2、杠杆温度×2、"
             f"上涨拥挤度×3、站上MA20×1、换手率×1 的加权分位（页面顶部显示的就是它）"},
    {"key": "hot_bear", "name": "熊市反弹预警", "mark": "dot", "color": "#CE5A4E", "hollow": True,
     "persist": PERSIST,
     "desc": f"已跌破下行的 {BEAR_MA} 日均线（确认的下行趋势）且蓝点温度 > {BEAR_TH}，{_p(PERSIST)}。"
             f"熊市里滚动分位的参照系全是低值，反弹再猛也顶不到 {BANDS[3]:.0f}——此条专门补熊市反弹高点"},
    {"key": "cold",  "name": "蓝点预警", "mark": "dot", "color": "#3D7FB8", "persist": PERSIST_COLD,
     "desc": f"蓝点温度的当日口径（换手率、上涨占比、杠杆多空比取当日值不平滑）"
             f"< {COLD_TH:.0f} 且 VIX ≥ {VIX_COLD}"
             + (f"、且纳指自峰值回撤 ≥ {NDX_DD_COLD:.0f}%" if NDX_DD_COLD else "")
             + f"，{_p(PERSIST_COLD)}"},
    {"key": "cold_soft", "name": "空心蓝点（宏观逆风）", "mark": "dot", "color": "#3D7FB8",
     "hollow": True, "persist": PERSIST_COLD,
     "desc": f"蓝点条件成立，但同时处于实际利率重估期：10 年期实际利率的 6 个月变动"
             f"落在它自己过去 3 年分布的 {RR_PCT_TH} 分位以上（涨得快），"
             f"同时水位仍在自己过去 5 年分布的 {RR_LVL_TH} 分位以下（重估还没走完），"
             f"且同期名义 10Y 也在上行（是紧缩不是通缩）。带滞回。"
             f"此时下跌由分母驱动，情绪见底不等于价格见底，建议分批而非满仓"},
    {"key": "crowd", "name": "黑框预警", "mark": "box",  "color": "#0B0F16", "persist": PERSIST,
     "desc": f"TOP2行业成交额占比 > {CROWD_TOP2} 分位 且 上涨拥挤度 > {CROWD_NARROW}，{_p(PERSIST)}。"
             f"即「钱挤进少数几个行业，同时指数靠少数股票撑在高位、宽度已经崩坏」。"
             f"下跌市中上涨拥挤度无定义，故本预警只在上涨市出现"},
]



def real_rate_speed_pct(rr):
    """实际利率 6 个月变动，在它自己过去 RR_PCT_WIN 天分布中的分位（0-100）。

    先 dropna 再排名：chg 开头有 RR_WIN 个 NaN，而 rolling_pct 的窗口按长度计数，
    窗口里混着 NaN 会让分母虚高、分位被系统性压低（NaN 永远不计入"小于本值"）。
    """
    chg = (rr - rr.shift(RR_WIN)).dropna()
    if chg.empty:
        return pd.Series(np.nan, index=rr.index)
    return rolling_pct(chg, window=RR_PCT_WIN, min_periods=RR_PCT_MIN).reindex(rr.index)


def real_rate_level_pct(rr):
    """实际利率水位，在它自己过去 RR_LVL_WIN 天分布中的分位（0-100）。

    这一列回答的是"重估走完了没有"：水位仍在自己五年分布的下半部＝估值还有压缩空间。
    用分位而不是绝对水位，是为了跨利率环境——旧口径的 0.5% 是零利率年代的刻度。
    """
    return rolling_pct(rr, window=RR_LVL_WIN, min_periods=RR_LVL_MIN)


def repricing_regime(rr, nominal=None):
    """实际利率重估期（带滞回；rr 为 None 时返回 None）。

    进入：**快** 6 个月变动的 3 年滚动分位 > RR_PCT_TH 且绝对涨幅 > RR_MIN_RISE
          **还没完** 水位的 5 年滚动分位 < RR_LVL_TH
          **是紧缩不是通缩** 同期名义 10Y 也在上行
    退出：速度分位跌回 RR_PCT_EXIT 以下，或水位分位升过 RR_LVL_EXIT（重估走完了）。
          绝对下限只管入场，不进退出条件（见常量区注释）。

    两条判据缺一不可，别只留一条：只看速度分不开 2018Q4（好蓝点）与 2022Q1（坏蓝点），
    两段 6 个月上行只差 5bp；只看水位会把 2020-03 疫情底整段判反。详见常量区注释。

    用滞回而不是逐日判定，是因为纯逐日会闪断：2022 年 3 月俄乌避险把实际利率短暂压回
    2021 年的水位，6 个月变动一度归零，若逐日判定，2/28–3/14 这段最糟的蓝点里
    只有第一天会被标记出来。重估有没有走完是个状态，不是某一天的斜率。

    **rr 必须是完整历史序列**，不能是已经截断到展示窗口的那条——分位要预热
    RR_WIN + RR_PCT_MIN ≈ 一年半（见 load_real_rate 的说明）。上一版就是因为在截断过的
    序列上标定，2018 年整段藏在预热期后面看不见，上线后才发现把那两个蓝点判反了。

    RR_NEED_NOMINAL 为真时 nominal 是**必需**的，拿不到就返回 None（整条闸门不启用，
    蓝点全记实心）——而不是退化成不做这项检查。
    """
    if rr is None or (RR_NEED_NOMINAL and nominal is None):
        return None
    spd = real_rate_speed_pct(rr).values
    lvl = real_rate_level_pct(rr).values
    rise = (rr - rr.shift(RR_WIN)).values
    if RR_NEED_NOMINAL:
        # NaN 比较得 False = 名义方向存疑时不入场，偏保守的那一侧
        n = align_to(nominal, rr.index)
        nom_up = ((n - n.shift(RR_WIN)) > 0).values
    else:
        nom_up = np.ones(len(rr), dtype=bool)
    on = np.zeros(len(rr), dtype=bool)
    state = False
    for i, (q, v, c, up) in enumerate(zip(spd, lvl, rise, nom_up)):
        ok = np.isfinite(q) and np.isfinite(v)
        if ok:
            if state:
                if q < RR_PCT_EXIT or v > RR_LVL_EXIT:
                    state = False
            elif q > RR_PCT_TH and v < RR_LVL_TH and up and c > RR_MIN_RISE:
                state = True
        on[i] = state
    return pd.Series(on, index=rr.index)


def bear_regime(px):
    """确认的下行趋势：收盘低于 BEAR_MA 日均线，且该均线本身在下行。"""
    if px is None:
        return None
    ma = px.rolling(BEAR_MA, min_periods=BEAR_MA).mean()
    return (px < ma) & (ma - ma.shift(BEAR_SLOPE) < 0)


def build_alerts(temp, pct, vix=None, temp_fast=None, crowd_pct=None, ndx=None,
                 narrow=None, repricing=None, temp_sell=None):
    """→ {key: [bool, ...]}，与 temp 索引对齐。

    temp      主口径温度（三项平滑），用于页面展示与熊市反弹预警
    temp_sell 红点温度（SELL_W 加权），仅用于红点；缺省时红点退回用主口径 > BANDS[3]，
              行为与旧版一致（compose_sell 拿不到上涨拥挤度时就是这种情况）
    temp_fast 蓝点温度（当日口径）（同样六项等权，但三个平滑项取当日值），仅用于蓝点；
              缺省时蓝点退回用主口径，行为与旧版一致。
    repricing 实际利率重估期的布尔序列（由 repricing_regime 在**完整历史**的利率
              序列上算好再传进来，本函数不自己加载数据——分位要预热一年半，拿截断过的
              序列算会把展示窗口开头判成"无数据"）。缺省时蓝点不做宏观分级（全记为实心）。
    """
    t = temp
    tc = t if temp_fast is None else temp_fast.reindex(t.index)
    cold = (tc < COLD_TH)
    if vix is not None:
        cold = cold & (vix.reindex(t.index) >= VIX_COLD)
    if ndx is not None and NDX_DD_COLD:
        # 回撤闸门（NDX_DD_COLD=0 时不启用）：情绪和 VIX 都到位、但指数还在高位附近时，
        # 多半是盘中恐慌而不是真正的底
        px = ndx.reindex(t.index).ffill()
        drawdown = (px / px.cummax() - 1.0) * 100.0     # ≤0，峰值为截至当日的累计最高
        cold = cold & (drawdown <= -NDX_DD_COLD)
    px = ndx.reindex(t.index).ffill() if ndx is not None else None
    # 红点走红点温度；拿不到时退回旧行为（主温度 > BANDS[3]）
    ts = t if temp_sell is None else temp_sell.reindex(t.index)
    hot = (ts > (BANDS[3] if temp_sell is None else SELL_TH))
    bear = bear_regime(px)
    if bear is None:
        bear = pd.Series(False, index=t.index)
    bear = bear.reindex(t.index).fillna(False)
    # 熊市反弹与标准红点互斥：温度同时越过 80 时只记标准红点，避免同一天两个记号叠在一起
    hot_bear = bear & (t > BEAR_TH) & ~hot
    # 蓝点分级：处于实际利率重估期的记为空心蓝点，两者互斥
    rp = (pd.Series(False, index=t.index) if repricing is None
          else repricing.reindex(t.index).fillna(False).astype(bool))
    out = {"hot": hot, "hot_bear": hot_bear, "cold": cold & ~rp, "cold_soft": cold & rp}
    # 黑框 = TOP2 抱团分位 > CROWD_TOP2 且 上涨拥挤度 > CROWD_NARROW。
    # TOP2 那一项单独用 252 日滚动分位：扩张窗口"永不遗忘"，2020-21 的极值会把后来的
    # 抱团永久挡在高分位之外——实测扩张口径下 2022 年之后再没触发过，等于失明。
    top2 = None
    if crowd_pct is not None and "top2" in crowd_pct.columns:
        top2 = crowd_pct["top2"]
    elif "top2" in pct.columns:
        top2 = pct["top2"]
    if narrow is None or top2 is None:
        out["crowd"] = pd.Series(False, index=t.index)
    else:
        # 上涨拥挤度在下跌市为 NaN（见上方注释），fillna(False) 后自然不触发
        out["crowd"] = (top2.reindex(t.index) > CROWD_TOP2) & (narrow.reindex(t.index) > CROWD_NARROW)
    out = {k: v.reindex(t.index).fillna(False) for k, v in out.items()}

    # 持续性约束：条件连续成立 N 日才置位，N 由每条规则自带（顶用3日、底用1日）。
    # 只在第 N 日及之后为真——每个信号当日即可判定，不含前视。
    per = {r["key"]: r.get("persist", PERSIST) for r in ALERTS}
    res = {}
    for k, v in out.items():
        n = per.get(k, PERSIST)
        res[k] = v if n <= 1 else (v.rolling(n, min_periods=n).sum() == n).fillna(False)
    return res


def regime(t):
    if t is None or not np.isfinite(t): return None
    names = ["极度恐慌", "偏冷", "中性", "偏热", "极度贪婪"]
    for b, n in zip(BANDS, names):
        if t < b:
            return n
    return names[-1]


def compose(rawdf, dirseries=None, fast=False):
    """返回 (原始口径分位, 方向修正后分位, 原始温度, 修正温度)
    fast=True 时上涨占比改用更宽的锚点（喂给它的是日频未平滑值，波动更大）。"""
    pct = pd.DataFrame(index=rawdf.index)
    abs_ref = dict(REF_ABS, **(REF_ABS_FAST if fast else {}))
    for c in rawdf.columns:
        if c.startswith("_"):
            continue      # _ep 供 ERP 合成用；_*_daily 是未平滑的当日值
        if c in abs_ref:
            p = abs_map(rawdf[c], abs_ref[c])          # 固定锚点，参照系不动
        elif c in REF_EXP:
            p = expanding_pct(rawdf[c])                # 扩张窗口分位
        elif c == "erp":
            # ERP 仍用 252 日滚动（扩张窗口会拿今天的ERP跟零利率年代比，钝化恐慌读数）
            p = rolling_pct(rawdf[c])
            if ERP_MODE == "ep" and "_ep" in rawdf.columns:
                p = rolling_pct(rawdf["_ep"])
            elif ERP_MODE == "blend" and "_ep" in rawdf.columns:
                p = 0.5 * p + 0.5 * rolling_pct(rawdf["_ep"])
            p = 100.0 - p                              # 高ERP=股票便宜=情绪冷
        else:
            p = rolling_pct(rawdf[c])
        pct[c] = p
    cols = [c for c in ORDER if c in pct.columns]
    pct = pct[cols]
    temp_raw = pct.mean(axis=1, skipna=False)     # 等权，任一缺失则不出温度

    if dirseries is None:
        return pct, pct.copy(), temp_raw, temp_raw
    adj = pct.copy()
    d = dirseries.reindex(pct.index)
    for c in cols:
        if c in DIRECTIONAL:
            # 偏离中位的幅度不变，方向由市场决定；d≈0 时回到 50（中性）
            dev = pct[c] - 50.0
            if DIR_HALF:
                dev = dev.clip(lower=0)      # 低于常态的活跃度不表态
            adj[c] = 50.0 + dev * d
    temp_adj = adj.mean(axis=1, skipna=False)
    return pct, adj, temp_raw, temp_adj


def compose_sell(adj, rawdf, lev_temp=None):
    """红点温度：按 SELL_W 加权合成，只喂给红点预警，不作为页面展示的蓝点温度。

    分项取**方向修正后**的分位（与蓝点温度同源，保证两个数可比），上涨拥挤度取
    _narrow_neutral（下跌市填 50 的那一版）。任一分项缺失则当日不出红点温度——
    与蓝点温度一样，宁可不出，也不用半套输入产出一个看似正常的读数。
    数据不足以合成时返回 None，调用方退回用蓝点温度，行为与旧版一致。

    lev_temp: 杠杆温度（leverage.py 的两项合成分位）。给了就顶替 SELL_W["leverage"]
    那一格，不给则退回原来的"杠杆多空比"单项，行为与旧版完全一致。

    —— 为什么要换掉这一格 ——
    杠杆多空比是原红点温度里衰减最狠的一项：它对未来 63 日 QQQ 的秩相关
    2017-10~2021-12 是 -0.407，2022-01~2026-09 变成 **+0.062**（符号都反了）。
    补上交易强度腿之后（LEV_W 现为多空比:强度 = 3:1，选法见 leverage.py 的长注释），
    红点的实测对照（红点温度>75 连3日，触发后 63 日 QQQ，基准 +5.18%/26%为负）：
        原口径（纯多空比）  26段/81天  段均 -1.85%  日均 -1.79%/59%为负
                            前段 -5.40%   后段 **+2.49%**   六次真顶命中 6/6
        杠杆温度 3:1        23段/66天  段均 -2.39%  日均 **-3.34%/70%为负**
                            前段 -5.33%   后段 +0.38%       六次真顶命中 6/6
    门槛不是卡出来的：>72 -2.50%/63%负、>73 -2.56%/64%、>75 -3.34%/70%、
    >78 -3.89%/76%、>80 -2.94%/67%，SELL_TH 维持 75 不动。

    —— 曾经改错过一次，别改回去 ——
    2026-09 第一版取 1:1，结果 2018-08-28~30 与 2019-12 两段红点消失（触发后 63 日
    -11.3% 与 -15.4%，全样本最值钱的两次）。那不是强度腿否决了它们——那几天两条腿都高，
    是两个分位取平均把读数向 50 压缩，红点温度刚好滑到 75 以下。换腿而不重标门槛，
    等于把红点悄悄调严（81天→63天）。3:1 把这两段找回来，同时 2025 年秋天那批
    （强度 0.2~13 分位、后续 3 个月 +6.8%/+4.5%/+2.7%）仍被挡在外面：12 天降到 2 天。
    """
    parts, weights = [], []
    for c, w in SELL_W.items():
        if c == "narrow":
            col = rawdf.get("_narrow_neutral")
        elif c == "leverage" and lev_temp is not None:
            col = lev_temp
        else:
            col = adj.get(c)
        if col is None:
            return None
        parts.append(col)
        weights.append(w)
    X = pd.concat(parts, axis=1)
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    t = pd.Series((X.values * w).sum(axis=1), index=X.index)
    return t.where(X.notna().all(axis=1))


def leverage_monitor(rawdf):
    """杠杆温度：把 leverage.py 产出的原始量转成分位并合成。

    返回 (pct: DataFrame[ratio,intensity], temp: Series)。
    分位一律走本模块的 rolling_pct / expanding_pct（由 LV.LEV_REF 选），
    与页面其余分位共用同一把尺子——杠杆温度要与 TOP2、上涨拥挤度等项加权平均，尺子不同则不可加。
    注意 ratio 这一腿直接取 rawdf["leverage"]，与"六个分项"里显示的杠杆多空比是同一个数，
    不另算一遍。
    """
    pf = expanding_pct if LV.LEV_REF == "exp" else rolling_pct
    cols = {}
    if "leverage" in rawdf.columns:
        cols["ratio"] = pf(rawdf["leverage"])
    if "_lev_intensity" in rawdf.columns:
        cols["intensity"] = pf(rawdf["_lev_intensity"])
    pct = pd.DataFrame(cols, index=rawdf.index)
    return pct, LV.temperature(pct)


LEV_LABELS = {
    "ratio":     ("杠杆资金多空比", "加杠杆的方向", "%", "做多占杠杆ETF总成交额"),
    "intensity": ("杠杆ETF交易强度", "用杠杆包装交易的强度", "%", "占指数ETF成交额"),
}
def _ser(s, idx, nd=2):
    return [None if not np.isfinite(v) else round(float(v), nd) for v in s.reindex(idx).values]


def build_lev_block(rawdf, lev_pct, lev_temp, lev_info, idx):
    """组装 data.json 里的 leverage_monitor 块（页面「杠杆温度」那一节的全部输入）。"""
    meta, notes = lev_info["meta"], lev_info["notes"]
    t = lev_temp.reindex(idx)
    last_t = t.dropna()
    block = {
        "temperature": round(float(last_t.iloc[-1]), 1) if len(last_t) else None,
        "regime": regime(float(last_t.iloc[-1])) if len(last_t) else None,
        "as_of": last_t.index[-1].strftime("%Y-%m-%d") if len(last_t) else None,
        "ref": ("扩张窗口" if LV.LEV_REF == "exp" else f"{WINDOW}日滚动"),
        "weights": dict(LV.LEV_W),
        "used_in_sell": "leverage" in SELL_W,
        "smooth": LV.LEV_SMOOTH,
        "components": [],
        "notes": notes,
        "series": {"temperature": _ser(t, idx, 1)},
    }
    src = {"ratio": rawdf.get("leverage"), "intensity": rawdf.get("_lev_intensity")}
    for k in LV.LEV_W:
        if k not in lev_pct.columns:
            continue
        name, desc, unit, rawlab = LEV_LABELS[k]
        p, r = lev_pct[k].reindex(idx), src[k].reindex(idx)
        block["components"].append({
            "key": k, "name": name, "desc": desc, "unit": unit, "raw_label": rawlab,
            "weight": LV.LEV_W[k],
            "raw": round(float(r.dropna().iloc[-1]), 1) if r.notna().any() else None,
            "pct": round(float(p.dropna().iloc[-1]), 1) if p.notna().any() else None,
            "method": meta.get(k, ""),
        })
        block["series"][k + "_pct"] = _ser(p, idx, 1)

    return block


# ---------- 页面「分项」列表 ----------
# 只管展示：列出哪几条线、每条显示什么。**不改任何计算**——蓝点温度仍是 ORDER 那六项等权，
# 红点温度仍按 SELL_W 加权，两者的口径与门槛标定都没动。
# 与温度成分的差别有两处，tag 里写明了：上涨拥挤度与 VIX 不进蓝点温度；
# 上涨个股占比进蓝点温度但不单独列（与站上MA20 相关 0.84，列两行是同一件事量两遍）。
PANEL = [
    ("crowd_up", "上涨拥挤度",        "指数靠少数股票撑在高位", "红点"),
    ("top2",     "TOP2行业成交额占比", "资金抱团度",           "红点·蓝点"),
    ("lev",      "杠杆温度",          "加杠杆的方向与强度",     "红点·蓝点"),
    ("erp",      "风险溢价",          "估值性价比（已反向）",   "蓝点"),
    ("turnover", "换手率",            "成交活跃度",            "红点·蓝点"),
    ("ma20",     "站上MA20占比",      "中期趋势宽度",          "红点·蓝点"),
    ("advancing", "上涨个股占比",     "当日普涨程度",          "蓝点"),
    ("vix",      "VIX指数",           "隐含波动率（已反向）",   "蓝点闸门"),
]


def build_panel(rawdf, adj, lev_pct, lev_temp, vix, idx):
    """→ data.json 的 panel：七行，每行 {名称, 当前值文字, 分位, 分位历史, 归属}。"""
    def last(s):
        v = s.reindex(idx).dropna()
        return float(v.iloc[-1]) if len(v) else None
    def fmt(v, suf="", nd=1):
        return "—" if v is None else f"{v:.{nd}f}{suf}"

    turn = rawdf.get("turnover")
    turn_idx = None
    if turn is not None:
        med = float(turn.reindex(idx).dropna().median())
        turn_idx = (last(turn) / med * 100.0) if med else None

    pct_src = {
        "crowd_up": rawdf.get("_narrow_neutral"),
        "top2": adj.get("top2"),
        "lev": lev_temp,
        "erp": adj.get("erp"),
        "turnover": adj.get("turnover"),
        "ma20": adj.get("ma20"),
        "advancing": adj.get("advancing"),
        "vix": (100.0 - rolling_pct(vix)) if vix is not None else None,
    }
    val = {
        "crowd_up": ("—" if "_ndx_dd" not in rawdf else fmt(last(rawdf.get("_ndx_dd")), "%")),
        "top2": fmt(last(rawdf.get("top2")), "%"),
        "lev": (f"多空比 {fmt(last(lev_pct['ratio']) if 'ratio' in lev_pct else None, '', 0)}"
                f" · 强度 {fmt(last(lev_pct['intensity']) if 'intensity' in lev_pct else None, '', 0)}"),
        "erp": fmt(last(rawdf.get("erp")), "%", 2),
        "turnover": ("—" if turn_idx is None else f"指数 {turn_idx:.0f}"),
        "ma20": fmt(last(rawdf.get("ma20")), "%"),
        "advancing": fmt(last(rawdf.get("advancing")), "%"),
        "vix": fmt(last(vix)),
    }
    sub = {"crowd_up": "纳指距峰值；评分见右栏，下跌市无定义", "top2": "当日占比", "lev": "两腿分位（3:1 加权）",
           "erp": "E/P − 10年期美债", "turnover": "100＝十年中位", "ma20": "当日占比", "advancing": f"{ADV_SMOOTH}日均", "vix": "收盘"}
    out = []
    for key, name, desc, tag in PANEL:
        p = pct_src.get(key)
        if p is None:
            continue
        p = p.reindex(idx)
        out.append({"key": key, "name": name, "desc": desc, "tag": tag,
                    "value": val[key], "sub": sub[key],
                    "pct": (round(float(p.dropna().iloc[-1]), 1) if p.notna().any() else None),
                    "series": _ser(p, idx, 1)})
    return out


def main():
    rawdf, meta, spy, lev_info = build_indicators()
    dirs = direction(spy, rawdf.index)
    pct, adj, temp_raw, temp_adj = compose(rawdf, dirs)
    lev_pct, lev_temp = leverage_monitor(rawdf)
    temp_sell = compose_sell(adj, rawdf, lev_temp=lev_temp)
    have = temp_adj.dropna()
    if len(have) == 0:
        raise SystemExit("温度序列为空")

    last = have.index[-1]
    cols = list(pct.columns)
    out = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "as_of": last.strftime("%Y-%m-%d"),
        "window": WINDOW, "min_periods": MIN_PERIODS,
        "bands": BANDS,
        "reference": {"abs": {k: list(v) for k, v in REF_ABS.items()},
                      "expanding": REF_EXP, "rolling": REF_ROLL},
        "dir_win": DIR_WIN, "directional": DIRECTIONAL,
        "temperature": round(float(temp_adj.loc[last]), 1),
        "temperature_raw": round(float(temp_raw.loc[last]), 1),
        "temperature_sell": (round(float(temp_sell.loc[last]), 1)
                             if temp_sell is not None and last in temp_sell.index
                             and np.isfinite(temp_sell.loc[last]) else None),
        "sell_weights": dict(SELL_W), "sell_threshold": SELL_TH,
        "regime": regime(float(temp_adj.loc[last])),
        "regime_raw": regime(float(temp_raw.loc[last])),
        "direction": round(float(dirs.loc[last]), 3) if dirs is not None else None,
        "coverage": {
            "stocks": len(load_many(STOCKS)),
            "sector_etfs": len(load_many(list(SECTOR_ETFS))),
            "lev_long": len(load_many(LEV_LONG)),
            "lev_short": len(load_many(LEV_SHORT)),
            "sectors": len(SECTORS),
            "history_start": rawdf.index[0].strftime("%Y-%m-%d"),
            "history_end": rawdf.index[-1].strftime("%Y-%m-%d"),
            "history_days": int(len(rawdf)),
        },
        "indicators": [],
        "series": {
            "dates": [d.strftime("%Y-%m-%d") for d in have.index],
            "temperature": [round(float(v), 2) for v in have.values],
            "temperature_raw": [None if not np.isfinite(v) else round(float(v), 2)
                                for v in temp_raw.reindex(have.index).values],
            "temperature_sell": ([None if not np.isfinite(v) else round(float(v), 2)
                                  for v in temp_sell.reindex(have.index).values]
                                 if temp_sell is not None else None),
            "direction": [None if not np.isfinite(v) else round(float(v), 3)
                          for v in dirs.reindex(have.index).values] if dirs is not None else None,
        },
    }
    if spy is not None:
        s = spy["close"].reindex(have.index)
        out["series"]["spx"] = [None if not np.isfinite(v) else round(float(v) * 10, 1) for v in s.values]
    ndx = load("QQQ")
    if ndx is not None:
        s = ndx["close"].reindex(have.index)
        out["series"]["ndx"] = [None if not np.isfinite(v) else round(float(v), 2) for v in s.values]
        out["ndx_label"] = "纳斯达克100 · QQQ"

    # 蓝点温度（当日口径）：同样六项等权，但换手率/上涨占比/杠杆多空比取当日值不平滑。
    # 只喂给蓝点预警（抓恐慌插针），不作为页面展示的蓝点温度。
    fast_cols = {"turnover": "_turnover_daily", "advancing": "_advancing_daily",
                 "leverage": "_leverage_daily"}
    temp_fast = None
    if all(v in rawdf.columns for v in fast_cols.values()):
        rawdf_fast = rawdf.copy()
        for k, v in fast_cols.items():
            rawdf_fast[k] = rawdf[v]
        _, _, _, temp_fast_full = compose(rawdf_fast, dirs, fast=True)
        temp_fast = temp_fast_full.reindex(have.index)
        out["series"]["temperature_fast"] = [None if not np.isfinite(v) else round(float(v), 2)
                                             for v in temp_fast.values]
        out["temperature_fast"] = (round(float(temp_fast.loc[last]), 1)
                                   if np.isfinite(temp_fast.loc[last]) else None)

    # 预警：按规则标记每个交易日
    vix = load_vix(have.index)
    crowd_pct = pd.DataFrame({c: rolling_pct(rawdf[c]).reindex(have.index)
                              for c in ["top2", "leverage"] if c in rawdf.columns})
    # 宏观闸门：在**完整历史**的利率序列上算状态，再对齐到展示窗口（见 load_real_rate）
    rr_full, nom_full = load_real_rate(), load_nominal_rate()
    rr = align_to(rr_full, have.index)
    rp_full = repricing_regime(rr_full, nom_full)
    rp = (None if rp_full is None
          else align_to(rp_full.astype(float), have.index).fillna(0).astype(bool))
    al = build_alerts(temp_adj.reindex(have.index), adj.reindex(have.index), vix, temp_fast,
                      temp_sell=(temp_sell.reindex(have.index) if temp_sell is not None else None),
                      crowd_pct=crowd_pct if len(crowd_pct.columns) else None,
                      ndx=(ndx["close"].reindex(have.index) if ndx is not None else None),
                      narrow=(rawdf["_narrow_score"].reindex(have.index)
                              if "_narrow_score" in rawdf.columns else None),
                      repricing=rp)
    # 杠杆资金多空比：原始值 + 252 日滚动分位。分位直接取 crowd_pct 里那一列，
    # 与黑框的判定同源——否则页面上会出现两个口径不同的"杠杆分位"互相打架。
    if "leverage" in rawdf.columns:
        lv = rawdf["leverage"].reindex(have.index)
        out["series"]["leverage"] = [None if not np.isfinite(v) else round(float(v), 1)
                                     for v in lv.values]
        if "leverage" in crowd_pct.columns:
            out["series"]["leverage_pct"] = [None if not np.isfinite(v) else round(float(v), 1)
                                             for v in crowd_pct["leverage"].values]
    if rr is not None:
        out["series"]["real_rate"] = [None if not np.isfinite(v) else round(float(v), 2) for v in rr.values]
        # 状态序列直接下发，页面不再自己重算一遍规则——同一条规则写两份迟早会走样，
        # 何况 3 年滚动分位需要展示窗口之外的历史，前端根本算不出来。
        if rp is not None:
            out["series"]["repricing"] = [bool(x) for x in rp.values]
        nom = align_to(nom_full, have.index)
        spd = align_to(real_rate_speed_pct(rr_full), have.index)
        lvl = align_to(real_rate_level_pct(rr_full), have.index)
        def _v(s2, nd=2):
            if s2 is None or not np.isfinite(s2.loc[last]):
                return None
            return round(float(s2.loc[last]), nd)
        out["macro"] = {
            "real_rate": _v(rr), "chg": _v(rr - rr.shift(RR_WIN)),
            "nominal": _v(nom), "nominal_chg": _v(None if nom is None else nom - nom.shift(RR_WIN)),
            "pct": _v(spd, 1), "lvl_pct": _v(lvl, 1),
            "repricing": bool(rp.loc[last]) if rp is not None else False,
            "pct_th": RR_PCT_TH, "pct_exit": RR_PCT_EXIT, "pct_win": RR_PCT_WIN,
            "lvl_th": RR_LVL_TH, "lvl_exit": RR_LVL_EXIT, "lvl_win": RR_LVL_WIN,
            "win": RR_WIN,
            "need_nominal": bool(RR_NEED_NOMINAL and nom_full is not None),
        }
    bpx = ndx["close"].reindex(have.index).ffill() if ndx is not None else None
    br = bear_regime(bpx)
    if br is not None:
        br = br.reindex(have.index).fillna(False)
        out["series"]["bear"] = [bool(x) for x in br.values]
        out["bear"] = {"now": bool(br.loc[last]), "ma": BEAR_MA, "th": BEAR_TH}
    if "_narrow_score" in rawdf.columns:
        ns = rawdf["_narrow_score"].reindex(have.index)
        out["series"]["narrow_score"] = [None if not np.isfinite(v) else round(float(v), 1)
                                         for v in ns.values]
        out["narrow_score"] = (round(float(ns.loc[last]), 1)
                               if np.isfinite(ns.loc[last]) else None)
        out["narrow_detail"] = {
            "gap": round(float(rawdf["_narrow_gap"].loc[last]), 2)
                   if np.isfinite(rawdf["_narrow_gap"].loc[last]) else None,
            "breadth_chg": round(float(rawdf["_breadth_chg"].loc[last]), 1)
                           if np.isfinite(rawdf["_breadth_chg"].loc[last]) else None,
            "dd": round(float(rawdf["_ndx_dd"].loc[last]), 1)
                  if np.isfinite(rawdf["_ndx_dd"].loc[last]) else None,
            "win": NT_WIN, "threshold": NT_TH,
        }
    if vix is not None:
        out["series"]["vix"] = [None if not np.isfinite(v) else round(float(v), 2) for v in vix.values]
        out["vix_threshold"] = VIX_COLD
    out["alerts"] = {"rules": ALERTS, "persist": PERSIST,
                     "flags": {k: [bool(x) for x in v.values] for k, v in al.items()},
                     "counts": {k: int(v.sum()) for k, v in al.items()}}

    # ---------- 杠杆温度（见 leverage.py）----------
    out["leverage_monitor"] = build_lev_block(rawdf, lev_pct, lev_temp, lev_info, have.index)
    # ---------- 页面「分项」列表（展示用，不参与任何计算）----------
    out["panel"] = build_panel(rawdf, adj.reindex(have.index), lev_pct.reindex(have.index),
                               lev_temp, vix, have.index)

    for c in cols:
        name, desc = LABELS[c]
        d_ = {
            "key": c, "name": name, "desc": desc,
            "raw": round(float(rawdf[c].loc[last]), 4) if np.isfinite(rawdf[c].loc[last]) else None,
            "smooth": SMOOTHING[c][0] if c in SMOOTHING else None,
            "raw_today": (round(float(rawdf[SMOOTHING[c][1]].loc[last]), 4)
                          if c in SMOOTHING and SMOOTHING[c][1] in rawdf.columns
                          and np.isfinite(rawdf[SMOOTHING[c][1]].loc[last]) else None),
            "pct": round(float(adj[c].loc[last]), 1),
            "pct_unsigned": round(float(pct[c].loc[last]), 1),
            "inverted": c == "erp",
            "signed": c in DIRECTIONAL,
            "method": meta.get(c, ""),
            "pct_series": [None if not np.isfinite(v) else round(float(v), 1)
                           for v in adj[c].reindex(have.index).values],
            "raw_series": [None if not np.isfinite(v) else round(float(v), 4)
                           for v in rawdf[c].reindex(have.index).values],
        }
        if c in DIRECTIONAL:
            d_["pct_series_unsigned"] = [None if not np.isfinite(v) else round(float(v), 1)
                                         for v in pct[c].reindex(have.index).values]
        out["indicators"].append(d_)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print(f"蓝点温度 {out['temperature']} ({out['regime']})  未做方向修正 {out['temperature_raw']} ({out['regime_raw']})  截至 {out['as_of']}")
    if out.get("temperature_sell") is not None:
        print(f"红点温度 {out['temperature_sell']}（门槛 {SELL_TH:.0f}，权重 "
              + " ".join(f"{k}x{v:g}" for k, v in SELL_W.items()) + "）"
              + (f"  蓝点温度(当日口径) {out['temperature_fast']}（蓝点门槛 {COLD_TH:.0f} 且 VIX≥{VIX_COLD}）"
                 if out.get("temperature_fast") is not None else ""))
    lm = out.get("leverage_monitor") or {}
    if lm.get("temperature") is not None:
        print(f"杠杆温度 {lm['temperature']}（{lm['regime']}，"
              + "＋".join(f"{c['name']}{c['pct']}" for c in lm["components"]) + f"，{lm['ref']}分位）")
    for k, why in (lm.get("notes") or {}).items():
        print(f"  ⓘ 杠杆分项 {k} 缺席：{why}")
    print(f"样本 {out['coverage']['stocks']} 成分股 / {out['coverage']['sectors']} 行业，历史 {out['coverage']['history_days']} 交易日，温度序列 {len(have)} 点")
    for i in out["indicators"]:
        tag = " [方向修正]" if i["signed"] else (" [已反向]" if i["inverted"] else "")
        extra = f"  未修正={i['pct_unsigned']}" if i["signed"] else ""
        print(f"  {i['name']:<22} 原始={i['raw']}  分位={i['pct']}{extra}{tag}")
    return out


if __name__ == "__main__":
    main()
