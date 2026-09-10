#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增量刷新 VIX（CBOE CDN）与10年期美债收益率（美国财政部官网），
均为免 key 公开源，纯标准库。写入 raw/_vix.csv、raw/_dgs10.csv（date,value，新到旧，去重）。
"""
import csv, os, ssl, sys, urllib.request
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(BASE, "raw")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
CTX = ssl.create_default_context()


def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout, context=CTX) as r:
        return r.read().decode("utf-8", "replace")


def merge(path, new_rows):
    """new_rows: [(date_str YYYY-MM-DD, value_str), ...]；与已有文件按日期去重合并，新到旧排序。"""
    old = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    old.append(tuple(line.split(",", 1)))
    merged = {d: v for d, v in old}
    added = 0
    for d, v in new_rows:
        if d not in merged:
            added += 1
        merged[d] = v
    rows = sorted(merged.items(), key=lambda x: x[0], reverse=True)
    with open(path, "w") as f:
        f.write("\n".join(f"{d},{v}" for d, v in rows) + "\n")
    return added, len(rows), rows[0][0] if rows else None


def refresh_vix():
    txt = get("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv")
    rows = []
    r = csv.reader(txt.strip().splitlines())
    next(r)  # header: DATE,OPEN,HIGH,LOW,CLOSE
    for row in r:
        if len(row) < 5:
            continue
        try:
            d = datetime.strptime(row[0], "%m/%d/%Y").strftime("%Y-%m-%d")
            v = float(row[4])
        except ValueError:
            continue
        rows.append((d, f"{v:g}"))
    return merge(os.path.join(RAW, "_vix.csv"), rows)


BACKFILL_YEARS = 11   # 覆盖 10 年量价历史 + ERP 的 252 日分位预热

# 财政部两张表：名义收益率曲线、实际（TIPS）收益率曲线。列名不同，其余格式一致。
CURVES = {
    "nominal": ("daily_treasury_yield_curve",      "10 Yr", "_dgs10.csv",  "10Y美债"),
    "real":    ("daily_treasury_real_yield_curve",  "10 YR", "_dfii10.csv", "10Y实际利率"),
}


def _curve_year(year, kind):
    """取财政部某一年的 10 年期收益率日频序列 → [(YYYY-MM-DD, value), ...]"""
    typ, col, _, _ = CURVES[kind]
    url = (f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
           f"daily-treasury-rates.csv/{year}/all?type={typ}&field_tdr_date_value={year}")
    txt = get(url)
    r = csv.reader(txt.strip().splitlines())
    header = next(r)
    try:
        idx = header.index(col)
    except ValueError:
        raise SystemExit(f"财政部CSV列结构变化，未找到 {col} 列：{header}")
    rows = []
    for row in r:
        if len(row) <= idx or not row[0]:
            continue
        try:
            d = datetime.strptime(row[0], "%m/%d/%Y").strftime("%Y-%m-%d")
            v = float(row[idx])
        except ValueError:
            continue
        rows.append((d, f"{v:g}"))
    return rows


def refresh_curve(kind):
    """当年始终重拉；历史年份缺失时自动回补。

    财政部的 CSV 按年分文件，只拉当年在「已有历史」的机器上够用，但在全新容器里
    raw/ 是空的——ERP / 实际利率会只剩当年那一百多天，252 日滚动分位永远凑不满，
    综合温度整条序列为空。故此处按年检查覆盖度，缺哪年补哪年（已齐备则只拉当年）。
    """
    path = os.path.join(RAW, CURVES[kind][2])
    have = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    have[line.split(",", 1)[0][:4]] = have.get(line.split(",", 1)[0][:4], 0) + 1
    cur = datetime.utcnow().year
    todo = [y for y in range(cur - BACKFILL_YEARS, cur) if have.get(str(y), 0) < 200] + [cur]
    if len(todo) > 1:
        print(f"       {CURVES[kind][3]} 回补历史年份 {todo[0]}–{todo[-2]}（本地缺失）…")
    rows = []
    for y in todo:
        try:
            rows += _curve_year(y, kind)
        except Exception as e:
            if y == cur:
                raise
            print(f"       ! {y} 年回补失败（{str(e)[:60]}），继续")
    return merge(path, rows)


def main():
    a_vix = refresh_vix()
    print(f"VIX    新增 {a_vix[0]} 行，共 {a_vix[1]} 行，最新 {a_vix[2]}")
    for kind in ("nominal", "real"):
        try:
            a = refresh_curve(kind)
            print(f"{CURVES[kind][3]:<7} 新增 {a[0]} 行，共 {a[1]} 行，最新 {a[2]}")
        except Exception as e:
            # 实际利率只用于蓝点的宏观闸门，取不到不该让整条流水线失败
            if kind == "nominal":
                raise
            print(f"{CURVES[kind][3]:<7} ! 取数失败（{str(e)[:70]}），蓝点闸门本次退回不启用")


if __name__ == "__main__":
    main()
