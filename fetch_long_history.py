# -*- coding: utf-8 -*-
"""为样本前检验（oos_check.py）抓 20+ 年日线，写到 raw_long/。

为什么要单独写一个：现有两个抓取脚本都拿不到 2017-10 之前的完整面板。
2026-09-16 在本机实测过：
  · stockanalysis.com（主源）range 封顶 10Y——传 MAX/ALL/25Y 一律被当成 1Y。
    实测 10Y 最早 2016-09-15，够日常用，但比标定窗口只早一年，做不了样本前检验。
  · stooq.com 返回 Cloudflare 的 JS 验证页，裸 HTTP 拿不到 CSV。
  · fetch_sp500_yahoo.py 裸 urllib 打 Yahoo，56 个标的全数 HTTP 429。
    而且它的名单走 Wikipedia，解析失败会**静默回退到内置 43 只篮子**，
    还不含 QQQ 与除 TQQQ 外的 14 只杠杆 ETF——即使跑通也算不出前瞻收益和杠杆温度。
所以这里改用 yfinance（它自己处理 Yahoo 的 crumb/cookie 与退避），
名单直接取 engine.SECTORS 的 503 只，**不走 Wikipedia**——宁可整个失败，
也不要再出现"悄悄用 43 只票算宽度"那种能跑完但结论全错的情况。

用法：
    pip install yfinance
    python3 fetch_long_history.py                 # 默认 2004 起，写 raw_long/
    python3 fetch_long_history.py --start 2004-01-01 --workers 4
    python3 fetch_long_history.py --resume        # 断点续传，跳过已写好的

它**不碰 raw/**，所以不会影响 data.json / dashboard.html 的日常更新。
利率序列另外补：python3 fetch_vix_dgs10.py --years 22

—— 输出格式 ——
engine.load() 认两种：4 列 `日期,复权收盘,原始收盘,成交量` 或 3 列 `日期,收盘,成交量`，
**按日期新到旧排序**。这里写 4 列：复权价供收益/均线用，原始价供成交额用
（leverage.py 的用例明确要求成交额走未复权价×成交量，否则拆股会造出假跳变）。
"""
import argparse, os, sys, time
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "raw_long")


def targets():
    """engine 要读的全部标的。任何一类缺失都会让某个分项算不出来，所以一次凑齐。"""
    import engine as E
    t = list(E.STOCKS) + list(E.SECTOR_ETFS) + list(E.LEV_ETFS) + ["SPY", "QQQ"]
    seen, out = set(), []
    for x in t:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def to_rows(df):
    """yfinance 的一只票 → engine 格式的行（新到旧）。"""
    need = {"Close", "Volume"}
    if df is None or df.empty or not need.issubset(df.columns):
        return []
    adj = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    rows = []
    for d, a, c, v in zip(df.index, adj, df["Close"], df["Volume"]):
        if pd.isna(a) or pd.isna(c) or pd.isna(v):
            continue
        rows.append([str(d)[:10], f"{float(a):g}", f"{float(c):g}", f"{float(v):.0f}"])
    rows.sort(reverse=True)
    return rows


def write(sym, rows):
    with open(os.path.join(OUT, f"{sym}.csv"), "w") as f:
        f.write("\n".join(",".join(r) for r in rows) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2004-01-01")
    ap.add_argument("--end", default=None)
    ap.add_argument("--batch", type=int, default=40, help="每批下载多少只")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume", action="store_true", help="跳过 raw_long/ 里已有且非空的")
    ap.add_argument("--min-ok", type=float, default=0.90,
                    help="成功率低于此值就以非零码退出（默认 0.90）——"
                         "宁可失败也别让 oos_check 拿着残缺面板算出'能看的'假结论")
    a = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("缺 yfinance：pip install yfinance")

    os.makedirs(OUT, exist_ok=True)
    syms = targets()
    if a.resume:
        todo = [s for s in syms
                if not os.path.exists(os.path.join(OUT, f"{s}.csv"))
                or os.path.getsize(os.path.join(OUT, f"{s}.csv")) < 100]
    else:
        todo = list(syms)
    print(f"目标 {len(syms)} 只（成分股 + 行业ETF + 杠杆ETF + SPY/QQQ），本次待抓 {len(todo)} 只")
    print(f"区间 {a.start} ~ {a.end or '今天'}，每批 {a.batch}，并发 {a.workers}")

    ok, fail, t0 = [], [], time.time()
    for i in range(0, len(todo), a.batch):
        batch = todo[i:i + a.batch]
        try:
            df = yf.download(batch, start=a.start, end=a.end, auto_adjust=False,
                             actions=False, group_by="ticker", threads=a.workers,
                             progress=False)
        except Exception as e:
            fail += [(s, str(e)[:50]) for s in batch]
            print(f"  批次 {i//a.batch+1} 整批失败：{str(e)[:70]}")
            continue
        for s in batch:
            try:
                sub = df[s] if isinstance(df.columns, pd.MultiIndex) else df
                rows = to_rows(sub)
            except Exception as e:
                rows = []
            if len(rows) > 30:
                write(s, rows); ok.append(s)
            else:
                fail.append((s, f"only {len(rows)} rows"))
        print(f"  {min(i+a.batch,len(todo))}/{len(todo)}  成功 {len(ok)}  失败 {len(fail)}  "
              f"用时 {time.time()-t0:.0f}s")

    print(f"\n完成：{len(ok)} 成功 / {len(fail)} 失败，用时 {time.time()-t0:.0f}s")
    if fail:
        print(f"  失败样例：{fail[:6]}")
    have = [s for s in syms if os.path.exists(os.path.join(OUT, f"{s}.csv"))]
    rate = len(have) / len(syms)
    print(f"  raw_long/ 现有 {len(have)}/{len(syms)} 只（{rate:.0%}）")
    # 单独点名这几类：缺了它们，对应的分项会整条算不出来，而不是"少几只票"
    import engine as E
    for name, need in (("杠杆ETF", E.LEV_ETFS), ("行业ETF", list(E.SECTOR_ETFS)),
                       ("SPY/QQQ", ["SPY", "QQQ"])):
        miss = [s for s in need if s not in have]
        print(f"  {name:8s} {len(need)-len(miss)}/{len(need)}" + (f"  缺：{miss}" if miss else ""))
    if rate < a.min_ok:
        sys.exit(f"\n成功率 {rate:.0%} < {a.min_ok:.0%}，不要拿这份面板跑 oos_check。"
                 f"\n用 --resume 重跑补齐（Yahoo 限流时隔几分钟再试）。")
    print(f"\n下一步：python3 fetch_vix_dgs10.py --years 22")
    print(f"        python3 oos_check.py --raw raw_long")


if __name__ == "__main__":
    main()
