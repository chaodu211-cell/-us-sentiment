# -*- coding: utf-8 -*-
"""单独生成样本外那段的页面：dashboard_oos.html。

用途：标定窗口是 2017-10 起，之前的历史是真样本外。把那段单独画出来，
和主页面并排看，能直接用眼睛判断信号落点合不合理——表格里的均值看不出
"红点是标在顶部还是标在半山腰"。

做法是**让 engine 以截止日当天重算一遍**，而不是把现成的 data.json 切一刀。
差别很重要：切 data.json 只改得动曲线，页头的当前温度、六个分项读数、
杠杆面板、宏观状态行全是 2026 年的值，配一张 2017 年的图，那是错的页面。
以截止日重算出来的是"2017-09-30 那天打开网页会看到的样子"。

无前视：所有分位都是滚动窗口（252 日 / 3 年 / 5 年），t 日的值只用 ≤t 的数据，
所以截断输入等价于时光倒流，不需要额外处理。

用法：
    python3 make_oos_page.py --raw raw_long                 # 默认 → 2017-09-30
    python3 make_oos_page.py --raw raw_long --to 2010-12-31 # 只看金融危机那段
    open dashboard_oos.html

不碰 data.json 与 dashboard.html，日常流水线不受影响。
"""
import argparse, json, os, sys
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw_long", help="原始数据目录（拼接后的长面板）")
    ap.add_argument("--to", default="2017-09-30",
                    help="截止日。默认 2017-09-30——标定窗口从 2017-10 起，"
                         "这之前的都是样本外")
    ap.add_argument("--out", default="dashboard_oos.html")
    ap.add_argument("--data-out", default="data_oos.json")
    a = ap.parse_args()

    raw = os.path.abspath(os.path.join(BASE, a.raw)) if not os.path.isabs(a.raw) else a.raw
    if not os.path.isdir(raw):
        sys.exit(f"目录不存在：{raw}")
    cutoff = pd.Timestamp(a.to)

    import engine as E
    E.RAW = raw
    E.OUT = os.path.join(BASE, a.data_out)

    # 唯一的插入点：build_indicators 之后立刻截断，下游（合成、预警、面板、
    # 页头标量）全部落在截断后的帧上，自然保持一致。
    orig = E.build_indicators

    def truncated():
        rawdf, meta, spy, lev_info = orig()
        rawdf = rawdf[rawdf.index <= cutoff]
        if len(rawdf) == 0:
            sys.exit(f"{a.to} 之前没有数据——raw 目录的历史不够长？")
        if spy is not None:
            spy = spy[spy.index <= cutoff]
        return rawdf, meta, spy, lev_info

    E.build_indicators = truncated
    E.main()

    d = json.load(open(E.OUT, encoding="utf-8"))
    tpl = open(os.path.join(BASE, "dash_tpl.html"), encoding="utf-8").read()
    out = tpl.replace("__DATA__", json.dumps(d, ensure_ascii=False))
    open(os.path.join(BASE, a.out), "w", encoding="utf-8").write(out)

    dates = d["series"]["dates"]
    print(f"\n写入 {a.out}（{len(out)} 字节）")
    print(f"覆盖 {dates[0]} ~ {dates[-1]}（{len(dates)} 个交易日）")
    print(f"截止日读数：红点温度 {d.get('temperature_sell')}  蓝点温度 {d.get('temperature')}")
    print("\n窗口内的信号（天数 / 段数）：")
    for k, v in d["alerts"]["flags"].items():
        idx = [i for i, x in enumerate(v) if x]
        segs = []
        for i in idx:
            if segs and i == segs[-1][-1] + 1:
                segs[-1].append(i)
            else:
                segs.append([i])
        first = dates[idx[0]] if idx else "-"
        last = dates[idx[-1]] if idx else "-"
        print(f"  {k:10s} {len(idx):4d} 天 / {len(segs):3d} 段   {first} ~ {last}")


if __name__ == "__main__":
    main()
