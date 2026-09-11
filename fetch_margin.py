#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINRA 月频保证金统计 → raw/_margin.csv（研报 Fig 4 的口径），仅用标准库。

研报用 NYSE/FINRA 的**净借款**（融资借方余额 − 现金账户贷方 − 融资账户贷方）
除以标普500市值，作为散户杠杆的主尺。FINRA 把这张表免费公开在：

    https://www.finra.org/investors/insights/investing/margin-statistics

⚠️ 必读：**这个脚本的在线下载地址没有在开发环境里验证过**（当时会话的出口网络封了
   finra.org，连不上）。表本身是免费的，但 FINRA 换过几次文件路径，所以：
     · 在线模式是"尽力而为"：抓落地页 → 找页面上的 xlsx/csv 链接 → 下载解析；
     · **失败不会写坏数据**：解析结果先过量级与单调性校验，不合格直接拒绝写文件；
     · 永远有一条一定能用的退路：手工从上面那个页面下载文件，然后
           python3 fetch_margin.py --file ~/Downloads/margin-statistics.xlsx
       本地解析同一套代码，不碰网络。

用法：
    python3 fetch_margin.py --check            # 只测连通性，不写文件
    python3 fetch_margin.py                    # 在线抓取并写 raw/_margin.csv
    python3 fetch_margin.py --file <xlsx|csv>  # 解析手工下载的文件（离线，推荐）
    python3 fetch_margin.py --url <URL>        # 指定下载地址

输出 raw/_margin.csv：`日期,融资借方余额,现金账户贷方余额,融资账户贷方余额`，新到旧。
engine 侧是**可选**的：文件不在，页面上少一行读数，温度与全部预警不受任何影响。
月频数据 + 次月第 3~4 周才发布，跑一次能管一个月，不必进每日流水线。
"""
import argparse, csv, io, os, re, ssl, sys, urllib.request, zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
OUT = os.path.join(RAW, "_margin.csv")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")
CTX = ssl.create_default_context()
LANDING = "https://www.finra.org/investors/insights/investing/margin-statistics"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

# 校验用的量级：融资借方余额近十年在 4000 亿 ~ 1.5 万亿美元之间。换算成十亿美元后
# 落在 [200, 3000] 之外，说明解析到的根本不是这张表（或者列认错了），拒绝写入。
DEBIT_MIN_BN, DEBIT_MAX_BN = 200.0, 3000.0
MIN_ROWS = 24                  # 少于两年的数据不值得写：分位算不出来


def get(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read()


# ---------- 解析 ----------
def read_xlsx(blob):
    """极简 xlsx 读表：只取第一张工作表的单元格文本。避免引入 openpyxl 依赖
    （本仓库的约定是 pandas/numpy/scipy 之外不再加东西）。"""
    z = zipfile.ZipFile(io.BytesIO(blob))
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(f"{NS}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{NS}t")))
    names = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n))
    if not names:
        raise ValueError("xlsx 里没有工作表")
    rows = []
    for row in ET.fromstring(z.read(names[0])).iter(f"{NS}row"):
        vals = []
        for c in row.findall(f"{NS}c"):
            t, v = c.get("t"), c.find(f"{NS}v")
            if t == "inlineStr":
                is_ = c.find(f"{NS}is")
                txt = "".join(x.text or "" for x in is_.iter(f"{NS}t")) if is_ is not None else ""
            else:
                txt = "" if v is None or v.text is None else v.text
                if t == "s" and txt.isdigit() and int(txt) < len(shared):
                    txt = shared[int(txt)]
            vals.append(txt.strip())
        rows.append(vals)
    return rows


def read_csv_bytes(blob):
    return [r for r in csv.reader(io.StringIO(blob.decode("utf-8-sig", "replace")))]


def as_date(x):
    """FINRA 的月份列见过好几种写法：2026-01、Jan-26、January 2026、1/31/2026，
    以及 xlsx 里的日期序列号。"""
    x = str(x).strip().strip('"')
    if not x:
        return None
    if re.fullmatch(r"\d{5}(\.\d+)?", x):       # Excel 日期序列号（1899-12-30 起）
        return datetime(1899, 12, 30) + timedelta(days=float(x))
    for f in ("%Y-%m-%d", "%m/%d/%Y", "%Y-%m", "%b-%y", "%B-%y", "%b %Y", "%B %Y", "%b-%Y", "%B-%Y"):
        try:
            return datetime.strptime(x, f)
        except ValueError:
            pass
    return None


def as_num(x):
    x = str(x).replace("$", "").replace(",", "").replace('"', "").strip()
    if x in ("", "-", "—", "N/A"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def month_end(d):
    nxt = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return nxt - timedelta(days=1)


def pick_columns(rows):
    """找表头，确定三列的位置。认不出表头就退回"第一个数值列=借方，其后两列=贷方"。"""
    want = (("debit",), ("credit", "cash"), ("credit", "margin"))
    for r in rows[:12]:
        low = [str(c).lower() for c in r]
        hit = []
        for keys in want:
            idx = [i for i, c in enumerate(low) if all(k in c for k in keys)]
            hit.append(idx[0] if idx else None)
        if hit[0] is not None:
            return hit
    return [None, None, None]


def parse_records(rows):
    """→ [(date, debit, credit_cash, credit_margin), ...]，按日期升序。"""
    cols = pick_columns(rows)
    out = []
    for r in rows:
        if not r:
            continue
        d = as_date(r[0])
        if d is None:
            continue
        nums_at = lambda i: (as_num(r[i]) if i is not None and i < len(r) else None)
        if cols[0] is not None:
            vals = [nums_at(i) for i in cols]
        else:                                    # 退路：按出现顺序取前三个数值列
            seq = [as_num(c) for c in r[1:]]
            seq = [v for v in seq if v is not None]
            vals = (seq + [None, None, None])[:3]
        if vals[0] is None:
            continue
        out.append((month_end(d), vals[0], vals[1], vals[2]))
    out.sort(key=lambda x: x[0])
    # 同月多行（表里偶有小计行）只留最后一条
    dedup = {}
    for rec in out:
        dedup[rec[0]] = rec
    return [dedup[k] for k in sorted(dedup)]


def scale_to_bn(recs):
    """统一换算成十亿美元。原表有时以美元、有时以百万美元给数。"""
    med = sorted(v for _, v, _, _ in recs)[len(recs) // 2]
    f = 1e-9 if med > 1e10 else (1e-3 if med > 1e4 else 1.0)
    return [(d, a * f, None if b is None else b * f, None if c is None else c * f) for d, a, b, c in recs]


def validate(recs):
    if len(recs) < MIN_ROWS:
        return False, f"只解析出 {len(recs)} 行（至少要 {MIN_ROWS} 行）"
    med = sorted(v for _, v, _, _ in recs)[len(recs) // 2]
    if not (DEBIT_MIN_BN <= med <= DEBIT_MAX_BN):
        return False, f"融资借方余额中位数 {med:,.0f} 十亿美元，不在 [{DEBIT_MIN_BN:.0f}, {DEBIT_MAX_BN:.0f}] 内，多半是列认错了"
    span = (recs[-1][0] - recs[0][0]).days / 365.25
    if span < 1.5:
        return False, f"时间跨度只有 {span:.1f} 年"
    if (datetime.now() - recs[-1][0]).days > 200:
        return False, f"最新一期是 {recs[-1][0]:%Y-%m}，太旧了"
    return True, f"{len(recs)} 行，{recs[0][0]:%Y-%m} ~ {recs[-1][0]:%Y-%m}，最新净借款 " \
                 f"{recs[-1][1] - sum(v for v in recs[-1][2:] if v):,.0f} 十亿美元"


def write(recs):
    os.makedirs(RAW, exist_ok=True)
    with open(OUT, "w") as f:
        for d, a, b, c in reversed(recs):        # 新到旧，与 raw/ 下其他文件一致
            f.write(f"{d:%Y-%m-%d},{a:.4f},{'' if b is None else f'{b:.4f}'},"
                    f"{'' if c is None else f'{c:.4f}'}\n")
    return OUT


def parse_blob(blob, name=""):
    rows = read_xlsx(blob) if (blob[:2] == b"PK" or name.endswith((".xlsx", ".xlsm"))) else read_csv_bytes(blob)
    return parse_records(rows)


def find_link(html):
    """从落地页里找数据文件链接。FINRA 换过路径，所以认扩展名而不是写死 URL。"""
    cands = re.findall(r'href="([^"]+\.(?:xlsx|csv))"', html, re.I)
    hit = [u for u in cands if "margin" in u.lower()] or cands
    return ["https://www.finra.org" + u if u.startswith("/") else u for u in hit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只测连通性与解析，不写文件")
    ap.add_argument("--file", help="解析手工下载的 xlsx/csv（离线）")
    ap.add_argument("--url", help="指定数据文件地址")
    a = ap.parse_args()

    recs = None
    if a.file:
        with open(a.file, "rb") as f:
            recs = parse_blob(f.read(), a.file.lower())
        src = a.file
    else:
        urls = [a.url] if a.url else []
        if not urls:
            try:
                urls = find_link(get(LANDING).decode("utf-8", "replace"))
                print(f"落地页上找到 {len(urls)} 个候选文件链接")
            except Exception as e:
                print(f"❌ 打不开落地页：{e}")
                print(f"   退路：手工下载 {LANDING} 上的表，然后 "
                      f"python3 {os.path.basename(__file__)} --file <下载到的文件>")
                return 1
        src = None
        for u in urls[:6]:
            try:
                blob = get(u)
                got = parse_blob(blob, u.lower())
                if got:
                    recs, src = got, u
                    break
                print(f"  · {u} 解析不出数据行")
            except Exception as e:
                print(f"  · {u} 取不到：{str(e)[:80]}")
        if recs is None:
            print("❌ 所有候选链接都没拿到可用数据。")
            print(f"   退路：手工下载 {LANDING} 上的表，然后 "
                  f"python3 {os.path.basename(__file__)} --file <下载到的文件>")
            return 1

    recs = scale_to_bn(recs)
    ok, why = validate(recs)
    print(("✅ " if ok else "❌ ") + why + f"（来源 {src}）")
    if not ok:
        print("   没有写文件——宁可没有这一项，也不要一条错的杠杆曲线。")
        return 1
    if a.check:
        print("   --check 模式，未写文件。")
        return 0
    print(f"已写入 {write(recs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
