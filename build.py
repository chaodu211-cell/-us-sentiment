#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 data.json 注入 dash_tpl.html，生成 dashboard.html。"""
import json, os
BASE = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(BASE, "dash_tpl.html"), encoding="utf-8").read()
data = open(os.path.join(BASE, "data.json"), encoding="utf-8").read()
json.loads(data)  # 校验合法性
out = tpl.replace("__DATA__", data)
open(os.path.join(BASE, "dashboard.html"), "w", encoding="utf-8").write(out)
print(f"写入 dashboard.html（{len(out)} 字节）")
