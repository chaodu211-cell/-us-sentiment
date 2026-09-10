#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标普500全成分股日线拉取 —— 仅用标准库，无需 pip install。

前提：网络出口白名单需放行
    query1.finance.yahoo.com
    query2.finance.yahoo.com
    en.wikipedia.org        （取成分股名单与 GICS 行业；不放行则用 --tickers 手工指定）

用法：
    python3 fetch_sp500_yahoo.py                 # 拉全部成分股，约 10 分钟
    python3 fetch_sp500_yahoo.py --years 11      # 指定历史长度
    python3 fetch_sp500_yahoo.py --workers 6     # 并发数（Yahoo 对高并发会限流，建议 ≤8）
    python3 fetch_sp500_yahoo.py --check         # 只做连通性自检，不拉数据

输出：raw/<TICKER>.csv，四列 date,adjclose,close,volume（新到旧），
      以及 sectors.json（标的 → GICS 行业），engine.py 直接可用。
"""
import argparse, json, os, re, ssl, sys, time, urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CHART = "https://query{host}.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d&events=div%2Csplit"

# 白名单未放行 Wikipedia 时的兜底：本项目已验证过的分行业篮子
FALLBACK = {
    "AAPL": "信息技术", "MSFT": "信息技术", "NVDA": "信息技术", "ON": "信息技术", "ZBRA": "信息技术",
    "GOOGL": "通信服务", "META": "通信服务", "OMC": "通信服务", "NWSA": "通信服务", "DIS": "通信服务",
    "AMZN": "可选消费", "HD": "可选消费", "DRI": "可选消费", "PHM": "可选消费", "BBY": "可选消费",
    "JPM": "金融", "V": "金融", "CINF": "金融", "RJF": "金融", "ALL": "金融",
    "LLY": "医疗保健", "JNJ": "医疗保健", "CRL": "医疗保健", "DGX": "医疗保健",
    "CAT": "工业", "RTX": "工业", "SNA": "工业",
    "PG": "日常消费", "COST": "日常消费", "MKC": "日常消费",
    "XOM": "能源", "CVX": "能源", "HAL": "能源", "OKE": "能源",
    "NEE": "公用事业", "SO": "公用事业", "NI": "公用事业",
    "PLD": "房地产", "AMT": "房地产", "KIM": "房地产",
    "LIN": "原材料", "SHW": "原材料", "PKG": "原材料",
}
# GICS 英文行业 → 中文（engine.py 使用中文键）
GICS_CN = {
    "Information Technology": "信息技术", "Communication Services": "通信服务",
    "Consumer Discretionary": "可选消费", "Financials": "金融", "Health Care": "医疗保健",
    "Industrials": "工业", "Consumer Staples": "日常消费", "Energy": "能源",
    "Utilities": "公用事业", "Real Estate": "房地产", "Materials": "原材料",
}
# 除成分股外仍需拉取的标的：行业ETF（行业成交额集中度）、SPY（指数/ERP）、TQQQ（杠杆需求）
EXTRA = {t: "_ETF" for t in
         ["XLK", "XLC", "XLY", "XLF", "XLV", "XLI", "XLP", "XLE", "XLU", "XLRE", "XLB", "SPY", "TQQQ"]}

CTX = ssl.create_default_context()


def get(url, timeout=30, retries=4):
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403, 404):
                raise                      # 白名单拒绝或标的不存在，重试无意义
            time.sleep(1.5 * (i + 1))
        except Exception as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise last


def sp500_constituents():
    """从 Wikipedia 取成分股 + GICS 行业；失败则回退到内置篮子。"""
    try:
        html = get(WIKI)
    except Exception as e:
        print(f"  ! 取 Wikipedia 失败（{e}），回退到内置 {len(FALLBACK)} 只篮子")
        return dict(FALLBACK), False
    m = re.search(r'id="constituents".*?<tbody>(.*?)</tbody>', html, re.S)
    if not m:
        print("  ! Wikipedia 表格结构变化，回退到内置篮子")
        return dict(FALLBACK), False
    out = {}
    for row in re.findall(r"<tr>(.*?)</tr>", m.group(1), re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        if len(cells) < 4:
            continue
        def clean(x):
            x = re.sub(r"<[^>]+>", "", x)
            return re.sub(r"&amp;", "&", x).strip()
        sym, sector = clean(cells[0]), clean(cells[2])
        if re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", sym):
            out[sym.replace(".", "-")] = GICS_CN.get(sector, sector)
    if len(out) < 400:
        print(f"  ! 只解析出 {len(out)} 只，疑似异常，回退到内置篮子")
        return dict(FALLBACK), False
    return out, True


def fetch_one(sym, p1, p2):
    """返回 (sym, rows, err)。rows = [[date, adjclose, close, volume], ...] 新到旧。"""
    err = None
    for host in (1, 2):
        try:
            txt = get(CHART.format(host=host, sym=sym, p1=p1, p2=p2))
            j = json.loads(txt)
            res = (j.get("chart") or {}).get("result")
            if not res:
                err = ((j.get("chart") or {}).get("error") or {}).get("description") or "empty"
                continue
            r0 = res[0]
            ts = r0.get("timestamp") or []
            q = (r0.get("indicators", {}).get("quote") or [{}])[0]
            adj = (r0.get("indicators", {}).get("adjclose") or [{}])[0].get("adjclose")
            close, vol = q.get("close"), q.get("volume")
            if not ts or close is None:
                err = "no data"
                continue
            rows = []
            for i, t in enumerate(ts):
                c = close[i]
                v = vol[i] if vol else None
                a = adj[i] if adj else c
                if c is None or v is None or a is None:
                    continue
                d = datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d")
                rows.append([d, f"{a:g}", f"{c:g}", f"{v:.0f}"])
            rows.sort(reverse=True)
            if rows:
                return sym, rows, None
            err = "all rows null"
        except urllib.error.HTTPError as e:
            err = f"HTTP {e.code}"
            if e.code == 403:
                err += "（很可能是出口白名单未放行 query%d.finance.yahoo.com）" % host
                break
        except Exception as e:
            err = str(e)[:80]
    return sym, None, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=11)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tickers", default="", help="逗号分隔，指定则跳过 Wikipedia")
    ap.add_argument("--check", action="store_true", help="只做连通性自检")
    a = ap.parse_args()

    if a.check:
        print("连通性自检：")
        for u in ["https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d",
                  "https://query2.finance.yahoo.com/v8/finance/chart/SPY?range=5d&interval=1d",
                  WIKI]:
            host = re.sub(r"^https://([^/]+)/.*", r"\1", u)
            try:
                get(u, timeout=15, retries=1)
                print(f"  ✅ {host}")
            except Exception as e:
                print(f"  ❌ {host}  {e}")
        return

    os.makedirs(RAW, exist_ok=True)
    if a.tickers:
        universe = {t.strip().upper(): "未分类" for t in a.tickers.split(",") if t.strip()}
        full = False
    else:
        print("取标普500成分股名单…")
        universe, full = sp500_constituents()
    print(f"  成分股 {len(universe)} 只{'（Wikipedia 全量）' if full else '（内置篮子）'}")

    targets = dict(universe)
    targets.update(EXTRA)
    p2 = int(time.time())
    p1 = p2 - a.years * 366 * 86400
    print(f"拉取 {len(targets)} 个标的 × {a.years} 年日线，并发 {a.workers}…")

    ok, fail, t0 = [], [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch_one, s, p1, p2): s for s in targets}
        for n, f in enumerate(as_completed(futs), 1):
            sym, rows, err = f.result()
            if rows:
                with open(os.path.join(RAW, f"{sym}.csv"), "w") as fh:
                    fh.write("\n".join(",".join(r) for r in rows) + "\n")
                ok.append(sym)
            else:
                fail.append((sym, err))
            if n % 25 == 0 or n == len(targets):
                print(f"  {n}/{len(targets)}  成功 {len(ok)}  失败 {len(fail)}  用时 {time.time()-t0:.0f}s")

    sectors = {s: universe[s] for s in ok if s in universe}
    with open(os.path.join(BASE, "sectors.json"), "w") as fh:
        json.dump(sectors, fh, ensure_ascii=False, indent=1)

    print(f"\n完成：{len(ok)} 成功 / {len(fail)} 失败，用时 {time.time()-t0:.0f}s")
    if ok:
        import csv as _csv
        p = os.path.join(RAW, f"{ok[0]}.csv")
        rows = list(_csv.reader(open(p)))
        print(f"  样例 {ok[0]}: {len(rows)} 行  {rows[-1][0]} → {rows[0][0]}")
    by = {}
    for s, sec in sectors.items():
        by[sec] = by.get(sec, 0) + 1
    print("  行业分布: " + "  ".join(f"{k} {v}" for k, v in sorted(by.items(), key=lambda x: -x[1])))
    if fail:
        print(f"  失败样例: {fail[:5]}")
    print("\n下一步：python3 engine.py   （engine.py 会读 sectors.json 自动组建分行业篮子）")


if __name__ == "__main__":
    main()
