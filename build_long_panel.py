# -*- coding: utf-8 -*-
"""把 stooq 的长历史拼到 stockanalysis 的生产序列前面，输出 raw_long/。

为什么要拼而不是直接用 stooq：
  · stockanalysis（主源）range 封顶 10Y，最早只到 2016-09——2006-2017 它给不了。
  · 但所有标定、线上 data.json、以及页面上现有的信号归类，都是在 stockanalysis
    的口径上做出来的。整盘换成 stooq 会让近十年的读数**静悄悄地**跟着变，
    等于在补历史的同时改了现状。
所以：**2016-09 之后逐行照抄 stockanalysis，一个数都不动**；2016-09 之前用 stooq，
并按重叠区间量出的比例缩放到 stockanalysis 的刻度上，使接缝处不产生凭空的涨跌幅。

—— 口径问题1（成交额）怎么修 ——
engine 里 rawclose 的全部 7 处用途都是 `rawclose * volume`，即**成交额**，
没有一处单独用原始价。所以要修的不是"补一列原始价"，而是"让成交额落在同一刻度上"。
stooq 的 3 列格式把 rawclose 当成复权价，如果它的 volume 未随拆股缩放，
成交额在拆股前会被系统性低估——换手率、抱团度、黑框全都受影响。
本脚本不去猜 stooq 的复权约定，而是**在重叠区间上直接量**：
    k_dv = median( (stooq 收盘×成交量) / (stockanalysis 原始价×成交量) )
再用 k_dv 把 2016 前的 stooq 成交额缩放过去。k_dv 若在重叠区间内漂移（尤其跨过
拆股日），说明两边的复权约定不一致，脚本会把这些票单独列出来。

用法：
    python3 fetch_sp500.py            # 先把 raw/ 灌满 stockanalysis 的 10 年
    python3 build_long_panel.py       # 读 raw/ + raw_long_stooq/，写 raw_long/
    python3 oos_check.py --raw raw_long
"""
import argparse, os, sys
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def read3or4(path):
    """engine.load 的读法：4 列 date,close(复权),rawclose(原始),volume 或 3 列 date,close,volume。"""
    n = len(open(path).readline().split(","))
    if n >= 4:
        df = pd.read_csv(path, header=None, names=["date", "close", "rawclose", "volume"])
    else:
        df = pd.read_csv(path, header=None, names=["date", "close", "volume"])
        df["rawclose"] = df["close"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    for c in ("close", "rawclose", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["close"])


def splice(sa, st, tail=10):
    """sa=stockanalysis（权威），st=stooq（补历史）。返回拼好的 4 列表 + 诊断。

    tail: 在**接缝那一端**取 tail 天量比例。

    —— 这里踩过一次，别改回去 ——
    第一版写的是 ov[-tail:]，即重叠区间的**最后** tail 天。但接缝是 sa.index.min()，
    在重叠区间的**开头**；重叠有十年，末尾离接缝十年远。
    stockanalysis 做分红复权、stooq 不做（或口径不同），两条序列的比例会随分红
    逐年累积漂移——MO 那种 7% 股息率的票十年能差近一倍。拿 2026 年量出来的比例去
    缩放 2016 年之前的段，接缝处就炸开：实测 531 只里 109 只跳变 >6σ，
    最差 HPE -46.8%、MO -25.8%、MMM -26.6%，全是高股息股。
    所以必须在接缝那一端取。而且窗口不能宽：比例在窗口内本身就在漂，取中位数等于
    把锚点放到窗口中点，离接缝还差半个窗口——实测 120 天窗口仍有 2.9σ 的残余跳变。
    所以锚点取**接缝当天**，只用开头 tail 天的中位数做异常保护：接缝当天的比例
    若偏离该中位数超过 5%，说明那天有坏价格，才退回用中位数。
    """
    ov = sa.index.intersection(st.index)
    d = {"overlap": len(ov)}
    if len(ov) < 60:
        return None, {**d, "err": "重叠不足 60 天"}
    last = ov[:tail] if len(ov) > tail else ov   # 接缝在重叠区间的**开头**

    r_px = (st.loc[last, "close"] / sa.loc[last, "close"]).replace([np.inf, -np.inf], np.nan).dropna()
    dv_sa = sa["rawclose"] * sa["volume"]
    dv_st = st["close"] * st["volume"]
    r_dv = (dv_st.loc[last] / dv_sa.loc[last]).replace([np.inf, -np.inf], np.nan).dropna()
    # 这个守卫查的是"锚点窗口里有没有可用样本"，不是重叠长度（重叠已在上面查过 60 天）。
    # 锚点窗口只有 tail 天，别再拿 30 去卡它——那会把所有票判成失败。
    if len(r_px) < 3 or len(r_dv) < 3:
        return None, {**d, "err": "接缝附近有效样本不足"}

    def _anchor(ratio):
        """锚点取接缝当天；那天异常就退回窗口中位数。"""
        med = float(ratio.median())
        if len(ratio) == 0 or not np.isfinite(med) or med == 0:
            return np.nan
        first = float(ratio.iloc[0])
        if not np.isfinite(first) or abs(first / med - 1.0) > 0.05:
            return med
        return first

    k_px, k_dv = _anchor(r_px), _anchor(r_dv)
    if not (np.isfinite(k_px) and np.isfinite(k_dv) and k_px and k_dv):
        return None, {**d, "err": "接缝比例算不出来"}
    # 漂移度：整个重叠区间的比例相对中位数的离散程度。稳 = 两源复权约定一致。
    r_px_all = (st["close"].reindex(ov) / sa["close"].reindex(ov)).replace([np.inf, -np.inf], np.nan).dropna()
    r_dv_all = (dv_st.reindex(ov) / dv_sa.reindex(ov)).replace([np.inf, -np.inf], np.nan).dropna()
    d["px_drift"] = float((r_px_all.quantile(.95) - r_px_all.quantile(.05)) / abs(k_px)) if k_px else np.nan
    d["dv_drift"] = float((r_dv_all.quantile(.95) - r_dv_all.quantile(.05)) / abs(k_dv)) if k_dv else np.nan
    d["k_px"], d["k_dv"] = k_px, k_dv

    seam = sa.index.min()
    pre = st[st.index < seam].copy()
    if len(pre) == 0:
        return sa.copy(), {**d, "pre_rows": 0}

    # 复权价缩放到 stockanalysis 的刻度
    pre_close = pre["close"] / k_px
    # 成交额缩放后，反解出一对 (rawclose, volume)：保留 stooq 的 volume，
    # 让 rawclose 承担缩放——因为 engine 只用两者的乘积。
    pre_dv = (pre["close"] * pre["volume"]) / k_dv
    pre_vol = pre["volume"].replace(0, np.nan)
    pre_raw = pre_dv / pre_vol

    out = pd.DataFrame({"close": pre_close, "rawclose": pre_raw, "volume": pre_vol}).dropna()
    out = out[out.index < seam]

    # 接缝质量怎么判：**不能**看"接缝当天涨跌幅大不大"——那天本来就有真实的市场涨跌，
    # 拿它跟日波动比会把正常行情误判成假跳变（上一版就是这么写的，合成数据上
    # 0%/3%/7%/10% 股息全都报同一个 -1.97%/1.6σ，那其实是真实收益）。
    # 正确判据：拼出来的接缝收益，理论上恒等于 stooq 自己那天的收益——
    #   拼接收益 = sa[seam] / (st[seam-1]/k_px) - 1，而 k_px = st[seam]/sa[seam]
    #            = st[seam]/st[seam-1] - 1 = stooq 自己的收益
    # 所以两者之差就是缩放误差，锚点落在接缝当天时应当 ~0；只有退回中位数
    # （接缝当天有坏价格）时才会非零。
    joined = pd.concat([out["close"], sa["close"]]).sort_index()
    ret = joined.pct_change()
    pos = joined.index.get_indexer([seam])[0]
    st_ret = st["close"].pct_change()
    if pos > 0 and seam in st_ret.index and np.isfinite(st_ret.loc[seam]):
        d["seam_ret"] = float(ret.iloc[pos])
        d["seam_err"] = float(ret.iloc[pos] - st_ret.loc[seam])
    d["pre_rows"] = int(len(out))
    return out, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa", default="raw", help="stockanalysis 目录（权威，2016-09 起）")
    ap.add_argument("--stooq", default="raw_long_stooq", help="stooq 目录（补 2016 之前）")
    ap.add_argument("--out", default="raw_long")
    ap.add_argument("--seam-err", type=float, default=0.005,
                    help="接缝缩放误差超过此值就告警（默认 0.5%%）")
    a = ap.parse_args()

    SA = os.path.join(BASE, a.sa); ST = os.path.join(BASE, a.stooq); OUT = os.path.join(BASE, a.out)
    for p in (SA, ST):
        if not os.path.isdir(p):
            sys.exit(f"目录不存在：{p}")
    os.makedirs(OUT, exist_ok=True)

    import engine as E
    syms = list(E.STOCKS) + list(E.SECTOR_ETFS) + list(E.LEV_ETFS) + ["SPY", "QQQ"]
    syms = list(dict.fromkeys(syms))

    rows, only_sa, only_st, errs = [], [], [], []
    for s in syms:
        psa, pst = os.path.join(SA, f"{s}.csv"), os.path.join(ST, f"{s}.csv")
        has_sa, has_st = os.path.exists(psa), os.path.exists(pst)
        if not has_sa and not has_st:
            errs.append((s, "两边都没有")); continue
        if not has_st:                      # 只有生产源：照抄，历史就短一截
            df = read3or4(psa); only_sa.append(s)
            _write(OUT, s, df); continue
        if not has_sa:                      # 只有 stooq：无从校准，原样写入并记名
            df = read3or4(pst); only_st.append(s)
            _write(OUT, s, df); continue
        pre, d = splice(read3or4(psa), read3or4(pst))
        if pre is None:
            errs.append((s, d.get("err", "?"))); continue
        _write(OUT, s, pre, verbatim=psa); d["sym"] = s; rows.append(d)

    # 复制利率/VIX 等下划线开头的辅助文件
    for fn in os.listdir(SA):
        if fn.startswith("_"):
            import shutil; shutil.copy(os.path.join(SA, fn), os.path.join(OUT, fn))

    r = pd.DataFrame(rows)
    print(f"拼接完成：{len(rows)} 只两源拼接，{len(only_sa)} 只仅生产源，"
          f"{len(only_st)} 只仅 stooq，{len(errs)} 只失败")
    if len(r):
        print(f"\n重叠区间中位 {r['overlap'].median():.0f} 天，补进去的历史中位 {r['pre_rows'].median():.0f} 天")
        print("\n—— 两源复权约定是否一致（漂移越小越一致）——")
        for col, name in (("px_drift", "复权价比例"), ("dv_drift", "成交额比例")):
            q = r[col].describe(percentiles=[.5, .9, .99])
            print(f"  {name}  中位 {q['50%']:.3f}  90分位 {q['90%']:.3f}  最大 {q['max']:.3f}")
        bad_dv = r[r["dv_drift"] > 0.5].sort_values("dv_drift", ascending=False)
        print(f"\n  成交额比例漂移 > 0.5 的：{len(bad_dv)} 只"
              + (f"，最差几只 {list(bad_dv['sym'].head(8))}" if len(bad_dv) else "（没有）"))
        if "seam_err" in r:
            e = r["seam_err"].abs()
            bad = r.assign(_e=e)[e > a.seam_err].sort_values("_e", ascending=False)
            print(f"\n—— 接缝缩放误差（拼接收益 vs stooq 自身收益，应 ~0）——")
            print(f"  中位 {e.median():.2e}   90分位 {e.quantile(.9):.2e}   最大 {e.max():.2e}")
            print(f"  误差 > {a.seam_err:.1%} 的：{len(bad)} 只"
                  + (f"，最差几只 {[(x.sym, f'{x._e*100:.1f}%') for x in bad.head(8).itertuples()]}"
                     if len(bad) else "（没有，拼接干净）"))
    if only_st:
        print(f"\n  仅 stooq、无从校准的 {len(only_st)} 只：{only_st[:10]}")
    if errs:
        print(f"  失败 {len(errs)} 只：{errs[:8]}")
    print(f"\n下一步：python3 oos_check.py --raw {a.out}")


def _write(outdir, sym, df, verbatim=None):
    """verbatim 给了就先逐字节照抄那个文件，再把 df（接缝前）追加在后面。

    照抄而不是重新格式化，是因为「2016-09 之后一个数都不动」必须是字面意义上的：
    绕一趟 pandas 再用 %g 写回去会掉到 6 位有效数字（实测相对偏差 ~1e-6）。
    那点误差不会改变分位排序，但"没动过"和"几乎没动过"是两件事，
    生产序列只该有一个版本。
    """
    path = os.path.join(outdir, f"{sym}.csv")
    with open(path, "w") as f:
        if verbatim:
            txt = open(verbatim).read()
            f.write(txt if txt.endswith("\n") else txt + "\n")
        for d, r_ in df.sort_index(ascending=False).iterrows():
            f.write(f"{d:%Y-%m-%d},{r_['close']:.10g},{r_['rawclose']:.10g},{r_['volume']:.0f}\n")


if __name__ == "__main__":
    main()
