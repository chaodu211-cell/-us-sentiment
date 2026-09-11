#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标普500全成分股 10 年日线拉取 —— 仅用标准库，无需 pip install。

数据源：
    en.wikipedia.org        成分股名单 + GICS 行业
    stockanalysis.com/api   10 年日线（免 key，返回含拆股/分红复权价）
    www.multpl.com          标普500 TTM 每股收益（月频，用于 ERP）

用法：
    python3 fetch_sp500.py --check      # 连通性自检
    python3 fetch_sp500.py              # 全量拉取
    python3 fetch_sp500.py --limit 30   # 只拉前 30 只（快速验证）

输出：raw/<TICKER>.csv（date,adjclose,close,volume，新到旧）、sectors.json
"""
import argparse, json, os, re, ssl, sys, threading, time
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
WIKI = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HIST = "https://stockanalysis.com/api/symbol/{kind}/{sym}/history?range={rng}&period=Day"

GICS_CN = {
    "Information Technology": "信息技术", "Communication Services": "通信服务",
    "Consumer Discretionary": "可选消费", "Financials": "金融", "Health Care": "医疗保健",
    "Industrials": "工业", "Consumer Staples": "日常消费", "Energy": "能源",
    "Utilities": "公用事业", "Real Estate": "房地产", "Materials": "原材料",
}
# 行业ETF（行业成交额集中度）、SPY/QQQ（指数展示/ERP）、
# 杠杆ETF多空两篮（杠杆资金多空比）——15 只必须全部每日刷新，
# 少拉任何一只，该分项就会用陈旧数据继续算，且平滑窗口会把缺口悄悄补上。
LEV_LONG  = ["TQQQ", "UPRO", "SPXL", "SSO", "QLD", "TNA", "SOXL", "FAS", "TECL", "UDOW"]
LEV_SHORT = ["SQQQ", "SPXS", "SDS", "TZA", "SOXS"]
ETFS = (["XLK", "XLC", "XLY", "XLF", "XLV", "XLI", "XLP", "XLE", "XLU", "XLRE", "XLB",
         "SPY", "QQQ"] + LEV_LONG + LEV_SHORT)

CTX = ssl.create_default_context()
_lock = threading.Lock()
_last = [0.0]
MIN_GAP = 0.06          # 全局最小请求间隔，避免给对方站点造成压力


def _pace():
    with _lock:
        wait = MIN_GAP - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


def get(url, timeout=40, retries=4, pace=True):
    last = None
    for i in range(retries):
        if pace:
            _pace()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (401, 403, 404):
                raise
            time.sleep(1.0 + 1.5 * i)
        except Exception as e:
            last = e
            time.sleep(1.0 + 1.5 * i)
    raise last


def constituents():
    """Wikipedia 成分股表 → {代码: 中文行业}"""
    html = get(WIKI, pace=False)
    m = re.search(r'id="constituents"[^>]*>\s*<tbody[^>]*>(.*?)</tbody>', html, re.S)
    if not m:
        raise SystemExit("Wikipedia 表格结构变化，无法解析")
    out = {}
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", m.group(1), re.S):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.S)
        if len(cells) < 4:
            continue
        def clean(x):
            x = re.sub(r"<[^>]+>", "", x)
            return re.sub(r"&amp;", "&", x).replace("&#160;", " ").strip()
        sym, sector = clean(cells[0]), clean(cells[2])
        if re.fullmatch(r"[A-Z][A-Z.\-]{0,6}", sym):
            out[sym.replace(".", "-")] = GICS_CN.get(sector, sector)
    return out


def fetch_one(sym, is_etf, rng):
    """→ (sym, rows, err)；rows = [[date, adjclose, close, volume], ...] 新到旧
    stockanalysis.com 对双类别股（BRK-B、BF-B…）用点号而非连字符，两种写法都试。"""
    err = None
    syms = [sym] if "-" not in sym else [sym, sym.replace("-", ".")]
    for s_ in syms:
        for kind in (["e", "s"] if is_etf else ["s", "e"]):
            try:
                j = json.loads(get(HIST.format(kind=kind, sym=s_.lower(), rng=rng)))
                arr = j.get("data") or []
                if not arr:
                    err = "empty"
                    continue
                rows = []
                for r in arr:
                    try:
                        d = str(r["t"])[:10]
                        c, a, v = float(r["c"]), float(r.get("a", r["c"])), float(r["v"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    rows.append([d, f"{a:g}", f"{c:g}", f"{v:.0f}"])
                rows.sort(reverse=True)
                if rows:
                    return sym, rows, None
                err = "no valid rows"
            except urllib.error.HTTPError as e:
                err = f"HTTP {e.code}"
            except Exception as e:
                err = str(e)[:70]
    return sym, None, err


# ---------- 增量更新 ----------
# 全量口径每只票要下 10 年（约 253KB）×531 只 ≈ 134MB，在跨境链路上就是十分钟。
# 增量只下最近 INC_RANGE（约 13KB），流量降到 1/20；代价是必须自己处理复权重算：
# 拆股和分红会让数据源把**整段历史**的复权价按比例改写，此时本地旧行与新行不在同一
# 刻度上，直接拼接会在接缝处造出一根凭空的涨跌幅。故每次都比对重叠区间，一旦对不上
# 就对这只票退回全量重拉——每天真正需要重拉的只有当天除权除息的那几十只。
INC_RANGE = "6M"        # 数据源只认 10Y/5Y/1Y/6M/3M，其余值会被当成 1Y
INC_TOL = 5e-4          # 重叠区间复权价的相对容差；超过即判定发生了复权重算
INC_MIN_ROWS = 200      # 本地文件太短（新上市/上次没拉全）就别增量了，直接全量
FULL_EVERY_DAYS = 30    # 距上次全量超过这么多天，自动强制全量一次，防止误差长期累积


def _read_local(sym):
    """读本地 raw/<SYM>.csv → [[date, adjclose, close, volume], ...] 新到旧；没有则 None"""
    p = os.path.join(RAW, f"{sym}.csv")
    if not os.path.exists(p):
        return None
    rows = []
    with open(p) as fh:
        for line in fh:
            parts = line.strip().split(",")
            if len(parts) == 4 and parts[0][:4].isdigit():
                rows.append(parts)
    return rows or None


def merge_rows(local, fresh):
    """把新拉的短区间并回本地长历史。

    → (合并后的行, 是否需要全量重拉)
    重叠日期的复权价对不上 = 数据源改写了历史刻度，本地旧行作废，必须全量重拉。
    """
    lo = {r[0]: r for r in local}
    overlap = 0
    for r in fresh:
        old_r = lo.get(r[0])
        if old_r is None:
            continue
        overlap += 1
        try:
            a_new, a_old = float(r[1]), float(old_r[1])
        except ValueError:
            continue
        if a_old > 0 and abs(a_new - a_old) / a_old > INC_TOL:
            return None, True                      # 复权刻度变了
    if overlap == 0:
        return None, True                          # 完全没重叠：本地太旧，短区间接不上
    lo.update({r[0]: r for r in fresh})
    return sorted(lo.values(), reverse=True), False


def fetch_one_incremental(sym, is_etf):
    """先试增量；本地缺失/太短/复权刻度变了，就退回全量。→ (sym, rows, err, 是否走了全量)"""
    local = _read_local(sym)
    if local is None or len(local) < INC_MIN_ROWS:
        s, rows, err = fetch_one(sym, is_etf, "10Y")
        return s, rows, err, True
    s, fresh, err = fetch_one(sym, is_etf, INC_RANGE)
    if not fresh:
        return sym, None, err, False
    merged, need_full = merge_rows(local, fresh)
    if need_full:
        s, rows, err = fetch_one(sym, is_etf, "10Y")
        return s, rows, err, True
    return sym, merged, None, False


def fetch_eps():
    """multpl 标普500 TTM 每股收益（月频）"""
    html = get("https://www.multpl.com/s-p-500-earnings/table/by-month", pace=False)
    rows = []
    MON = {m: i + 1 for i, m in enumerate(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
    for mo, dd, yy, val in re.findall(
            r"<td[^>]*>\s*([A-Z][a-z]{2})\s+(\d{1,2}),\s*(\d{4})\s*</td>\s*"
            r"<td[^>]*>(?:\s|&#x2002;|&nbsp;)*(-?[\d.]+)", html):
        rows.append([f"{yy}-{MON[mo]:02d}-{int(dd):02d}", val])
    rows.sort(reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--range", default="10Y", help="1Y / 5Y / 10Y")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--incremental", action="store_true",
                    help=f"只下最近 {INC_RANGE} 并合并进本地历史；发生复权重算的个股自动退回全量")
    a = ap.parse_args()

    if a.check:
        print("连通性自检：")
        for nm, u in [("en.wikipedia.org", WIKI),
                      ("stockanalysis.com", HIST.format(kind="s", sym="aapl", rng="1Y")),
                      ("www.multpl.com", "https://www.multpl.com/s-p-500-earnings/table/by-month")]:
            try:
                t = get(u, timeout=20, retries=1, pace=False)
                print(f"  ✅ {nm}  ({len(t)} 字节)")
            except Exception as e:
                print(f"  ❌ {nm}  {e}")
        return

    os.makedirs(RAW, exist_ok=True)
    print("① 取标普500成分股名单…")
    uni = constituents()
    print(f"   {len(uni)} 只，行业 {len(set(uni.values()))} 个")
    if a.limit:
        uni = dict(list(uni.items())[:a.limit])

    targets = [(s, False) for s in uni] + [(s, True) for s in ETFS if s not in uni]

    # 距上次全量太久就强制全量一次：增量每次只校验重叠区间，长期跑下去总有边角情况
    # （长时间停机、数据源补历史、个别票停牌）积累不到，定期整体重下一次最省心。
    state_p = os.path.join(RAW, "_fetch_state.json")
    inc = a.incremental
    if inc:
        try:
            last = json.load(open(state_p))["last_full"]
            age = (time.time() - time.mktime(time.strptime(last, "%Y-%m-%d"))) / 86400
            if age > FULL_EVERY_DAYS:
                print(f"   距上次全量 {age:.0f} 天（> {FULL_EVERY_DAYS}），本次强制全量")
                inc = False
        except Exception:
            print("   没有全量记录，本次先走全量")
            inc = False

    mode = f"增量 {INC_RANGE}（自动补全量）" if inc else f"全量 {a.range}"
    print(f"② 拉取 {len(targets)} 个标的 × {mode} 日线，并发 {a.workers}…")
    ok, fail, t0 = [], [], time.time()
    fulls = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        if inc:
            futs = {ex.submit(fetch_one_incremental, s, e): s for s, e in targets}
        else:
            futs = {ex.submit(fetch_one, s, e, a.range): s for s, e in targets}
        for n, f in enumerate(as_completed(futs), 1):
            res = f.result()
            if inc:
                sym, rows, err, was_full = res
                fulls += was_full
            else:
                sym, rows, err = res
            if rows:
                with open(os.path.join(RAW, f"{sym}.csv"), "w") as fh:
                    fh.write("\n".join(",".join(r) for r in rows) + "\n")
                ok.append((sym, len(rows)))
            else:
                fail.append((sym, err))
            if n % 50 == 0 or n == len(targets):
                extra = f"  其中全量重拉 {fulls}" if inc else ""
                print(f"   {n}/{len(targets)}  成功 {len(ok)}  失败 {len(fail)}{extra}  {time.time()-t0:.0f}s")
    if not inc and not a.limit:
        json.dump({"last_full": time.strftime("%Y-%m-%d")}, open(state_p, "w"))

    okset = {s for s, _ in ok}
    sectors = {s: v for s, v in uni.items() if s in okset}
    json.dump(sectors, open(os.path.join(BASE, "sectors.json"), "w"), ensure_ascii=False, indent=1)

    print("③ 取标普500 TTM 每股收益…")
    try:
        eps = fetch_eps()
        with open(os.path.join(RAW, "_sp500_eps.csv"), "w") as fh:
            fh.write("\n".join(",".join(r) for r in eps) + "\n")
        print(f"   {len(eps)} 个月  {eps[-1][0]} → {eps[0][0]}")
    except Exception as e:
        print(f"   ! 失败（{e}），沿用已有 _sp500_eps.csv")

    lens = sorted(n for _, n in ok)
    tag = f"（增量，其中 {fulls} 只因复权重算或本地过短走了全量）" if inc else "（全量）"
    print(f"\n完成{tag}：{len(ok)} 成功 / {len(fail)} 失败，用时 {time.time()-t0:.0f}s")
    print(f"  行数中位 {lens[len(lens)//2] if lens else 0}，最少 {lens[0] if lens else 0}")
    by = {}
    for v in sectors.values():
        by[v] = by.get(v, 0) + 1
    print("  行业分布: " + "  ".join(f"{k}{v}" for k, v in sorted(by.items(), key=lambda x: -x[1])))
    if fail:
        print(f"  失败: {[s for s, _ in fail][:12]}{' …' if len(fail) > 12 else ''}")
    print("\n下一步：python3 engine.py")


if __name__ == "__main__":
    main()
