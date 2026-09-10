#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
每日刷新流水线：标普500量价+EPS → VIX/10Y美债 → 计算引擎 → 渲染 dashboard.html。
最后打印本次「新触发」的预警（供上层判断是否需要提醒用户）。

用法：
    python3 daily_refresh.py           全量：每只票重下 10 年（约 134MB），慢但绝对干净
    python3 daily_refresh.py --fast    增量：只下最近半年再并回本地（约 7MB），日常用这个

增量会自己校验复权刻度，发生拆股/分红重算的个股当次自动退回全量；
距上次全量超过 30 天也会自动整体全量一次。所以「一直用 --fast」是安全的。
"""
import json, os, subprocess, sys, time

BASE = os.path.dirname(os.path.abspath(__file__))


def run(cmd, label):
    print(f"\n=== {label} ===")
    t0 = time.time()
    r = subprocess.run([sys.executable] + cmd, cwd=BASE, capture_output=True, text=True)
    print(r.stdout[-3000:])
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"{label} 失败（退出码 {r.returncode}），已终止流水线")
    print(f"({time.time()-t0:.0f}s)")


def main():
    fast = "--fast" in sys.argv
    run(["fetch_sp500.py"] + (["--incremental"] if fast else []),
        "① 拉取标普500量价 + EPS" + ("（增量）" if fast else "（全量）"))
    run(["fetch_vix_dgs10.py"], "② 刷新 VIX / 10年期美债")
    run(["engine.py"], "③ 重新计算情绪温度")
    run(["build.py"], "④ 渲染 dashboard.html")

    print("\n=== ⑤ 检查新触发预警 ===")
    with open(os.path.join(BASE, "data.json"), encoding="utf-8") as f:
        d = json.load(f)
    al = d["alerts"]
    as_of = d["as_of"]

    state_p = os.path.join(BASE, "_pipeline_state.json")
    prev_as_of = None
    if os.path.exists(state_p):
        try:
            prev_as_of = json.load(open(state_p, encoding="utf-8")).get("last_as_of")
        except Exception:
            prev_as_of = None
    is_new_trading_day = (prev_as_of != as_of)

    newly = []
    if is_new_trading_day:
        for rule in al["rules"]:
            k = rule["key"]
            flags = al["flags"][k]
            on_today = bool(flags[-1]) if flags else False
            on_yday = bool(flags[-2]) if len(flags) > 1 else False
            if on_today and not on_yday:
                newly.append(rule)
    json.dump({"last_as_of": as_of}, open(state_p, "w", encoding="utf-8"))

    print(f"截至 {as_of}：综合温度 {d['temperature']}（{d['regime']}）"
          + ("" if is_new_trading_day else "  [与上次运行同一交易日，非首次处理]"))
    if not is_new_trading_day:
        print("NEWLY_TRIGGERED:none")
        print("  本交易日已处理过（大概率是周末/节假日重跑，或当日已刷新过），跳过重复提醒判定")
    elif newly:
        print("NEWLY_TRIGGERED:" + ",".join(r["key"] for r in newly))
        for r in newly:
            print(f"  ⚠️ 新触发 {r['name']}：{r['desc']}")
    else:
        print("NEWLY_TRIGGERED:none")
        print("  无新触发预警（各类预警最新交易日均未由「未触发」转为「触发」）")


if __name__ == "__main__":
    main()
