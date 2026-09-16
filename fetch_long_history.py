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
改用 yfinance 后仍被限流：2026-09-16 实测，40 只一批、并发 4，**第一批就整批 429**\n（YFRateLimitError），531 只全灭。不是配额耗尽，是之前那轮裸 urllib 的 56 个 429\n已经把这个 IP 标记了。所以这里逐只顺序抓、默认间隔 1.5s、撞限流按 30/60/120/300s\n退避，并且**连续失败 8 只就停**——被限流时磨完 531 只毫无意义。冷却后 --resume 续。\n名单方面用 yfinance（它自己处理 Yahoo 的 crumb/cookie），
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
    ap.add_argument("--sleep", type=float, default=1.5,
                    help="每只之间的间隔秒数（默认 1.5）。Yahoo 对突发请求很敏感，"
                         "2026-09 实测 40 只并发 4 会被整批 429。")
    ap.add_argument("--resume", action="store_true", help="跳过 raw_long/ 里已有且非空的")
    ap.add_argument("--max-consec-fail", type=int, default=8,
                    help="连续失败这么多只就放弃并退出（默认 8）。"
                         "被限流时继续磨完 531 只毫无意义，只是白等。")
    ap.add_argument("--min-ok", type=float, default=0.90,
                    help="成功率低于此值就以非零码退出（默认 0.90）——"
                         "宁可失败也别让 oos_check 拿着残缺面板算出'能看的'假结论")
    a = ap.parse_args()

    try:
        import yfinance as yf
        from yfinance.exceptions import YFRateLimitError
    except ImportError:
        sys.exit("缺 yfinance：pip3 install -U yfinance")

    os.makedirs(OUT, exist_ok=True)
    syms = targets()
    if a.resume:
        todo = [s_ for s_ in syms if not _done(s_)]
    else:
        todo = list(syms)
    print(f"目标 {len(syms)} 只（成分股 + 行业ETF + 杠杆ETF + SPY/QQQ），本次待抓 {len(todo)} 只")
    print(f"区间 {a.start} ~ {a.end or '今天'}，逐只顺序抓，间隔 {a.sleep}s")
    if not todo:
        print("都抓好了。"); _summary(syms, a.min_ok); return

    ok, fail, t0, consec = [], [], time.time(), 0
    # 撞限流时的退避梯度；走完还被限就退出，等冷却后 --resume 续
    BACKOFF = [30, 60, 120, 300]
    bo = 0
    for n, sym in enumerate(todo, 1):
        try:
            df = yf.Ticker(sym).history(start=a.start, end=a.end,
                                        auto_adjust=False, actions=False)
            rows = to_rows(df)
            if len(rows) > 30:
                write(sym, rows); ok.append(sym); consec = 0; bo = 0
            else:
                fail.append((sym, f"only {len(rows)} rows")); consec += 1
        except YFRateLimitError:
            consec += 1
            if bo < len(BACKOFF):
                w = BACKOFF[bo]; bo += 1
                print(f"  [{n}/{len(todo)}] {sym} 被限流，退避 {w}s 后重试…")
                time.sleep(w)
                try:
                    df = yf.Ticker(sym).history(start=a.start, end=a.end,
                                                auto_adjust=False, actions=False)
                    rows = to_rows(df)
                    if len(rows) > 30:
                        write(sym, rows); ok.append(sym); consec = 0
                        continue
                except Exception:
                    pass
            fail.append((sym, "rate limited"))
        except Exception as e:
            fail.append((sym, str(e)[:50])); consec += 1

        if consec >= a.max_consec_fail:
            print(f"\n连续 {consec} 只失败，停。已成功 {len(ok)} 只。")
            print("Yahoo 多半已经把这个 IP 标记了——等 30-60 分钟，或换个网络"
                  "（手机热点会换 IP），再用 --resume 续抓。")
            break
        if n % 25 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  成功 {len(ok)}  失败 {len(fail)}  "
                  f"用时 {time.time()-t0:.0f}s")
        time.sleep(a.sleep)

    print(f"\n本轮：{len(ok)} 成功 / {len(fail)} 失败，用时 {time.time()-t0:.0f}s")
    if fail:
        print(f"  失败样例：{fail[:6]}")
    _summary(syms, a.min_ok)


def _done(sym):
    p = os.path.join(OUT, f"{sym}.csv")
    return os.path.exists(p) and os.path.getsize(p) > 100


def _summary(syms, min_ok):
    """两道闸门：关键标的逐个硬卡 + 个股总成功率。

    关键标的必须单独卡，**不能只看总成功率**：531 只里 503 只是成分股，哪怕 ETF
    一只都没抓到，总成功率仍有 503/531 = 95%，照样能过 90% 的闸门——而那样跑出来
    杠杆温度、行业集中度、前瞻收益全是空的，oos_check 却会"正常"出一张表。
    这正是 fetch_sp500_yahoo.py 那次"静默回退到 43 只篮子"的同一类错误，别重蹈。
    """
    import engine as E
    have = {s for s in syms if _done(s)}
    crit = (("杠杆ETF", E.LEV_ETFS), ("行业ETF", list(E.SECTOR_ETFS)),
            ("SPY/QQQ", ["SPY", "QQQ"]))
    stocks = [s for s in E.STOCKS if s in have]
    rate = len(stocks) / len(E.STOCKS)
    print(f"  raw_long/ 现有 {len(have)}/{len(syms)} 只"
          f"（其中成分股 {len(stocks)}/{len(E.STOCKS)} = {rate:.0%}）")
    bad = []
    for name, need in crit:
        miss = [s for s in need if s not in have]
        print(f"  {name:8s} {len(need)-len(miss)}/{len(need)}" + (f"  缺：{miss}" if miss else ""))
        if miss:
            bad.append(f"{name} 缺 {len(miss)} 只")
    if bad:
        sys.exit(f"\n关键标的不全（{'；'.join(bad)}），不要拿这份面板跑 oos_check——"
                 f"\n缺杠杆ETF 则杠杆温度算不出（红点温度占 2/9、蓝点温度占 1/6），"
                 f"缺 QQQ 则前瞻收益无从算起。"
                 f"\n等限流冷却后：python3 fetch_long_history.py --resume")
    if rate < min_ok:
        sys.exit(f"\n成分股成功率 {rate:.0%} < {min_ok:.0%}，不要拿这份面板跑 oos_check。"
                 f"\n等限流冷却后：python3 fetch_long_history.py --resume")
    print(f"\n下一步：python3 fetch_vix_dgs10.py --years 22")
    print(f"        python3 oos_check.py --raw raw_long")


if __name__ == "__main__":
    main()
