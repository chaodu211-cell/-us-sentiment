# 美股情绪温度计 — 离线运行包

这个项目**不依赖 Claude、不依赖任何大模型、不需要 API key**。
整条流水线是纯 Python + pandas/numpy，数据全部来自公开免费端点。
只要你的机器能跑 Python、能上网，它就能一直跑下去。

---

## 一、五分钟跑起来

```bash
# 1. 装依赖（只有三个）
pip install pandas numpy scipy

# 2. 跑一次完整流程
python3 daily_refresh.py          # 首次用这个（全量）
python3 daily_refresh.py --fast   # 以后日常用这个（增量，快 10 倍以上）

# 3. 用浏览器打开生成的 dashboard.html
```

**全量 vs 增量**：全量每只票重下 10 年日线，531 只 ≈ 134MB，跨境链路上要十分钟；
增量只下最近半年再并回本地，约 7MB，通常几十秒。

增量不是"少算一点"——它算出来的结果和全量**逐项完全一致**（实测温度、原始口径、
六个分项全部相同）。它自己处理复权：拆股/分红会让数据源改写整段历史的复权价，
此时本地旧行与新行不在同一刻度上，硬拼会在接缝处造出一根凭空的涨跌幅；
所以每次都比对重叠区间，对不上的个股当次自动退回全量重拉（每天通常只有几只）。
另外距上次全量超过 30 天会自动强制全量一次。**所以一直用 `--fast` 是安全的。**

`daily_refresh.py` 依次做五件事：拉标普500量价与EPS → 刷新VIX与国债收益率 →
重算温度与预警 → 渲染 dashboard.html → 检查是否有新触发的预警。
全新机器上首次运行约 2–3 分钟（要下载十年历史），之后每天约 30–60 秒。

`dashboard.html` 是**完全自包含的单文件**：数据直接嵌在里面，没有外部
JS/CSS 依赖（只有一个 Google Fonts 链接，断网时自动退回系统字体）。
双击就能看，可以随便拷贝、发邮件、放U盘。

---

## 二、文件清单

| 文件 | 作用 |
|---|---|
| `fetch_sp500.py` | 抓标普500成分股 + 行业ETF + 15只杠杆ETF 的十年日频量价，以及标普500的TTM每股收益 |
| `fetch_vix_dgs10.py` | 抓 VIX、10年期名义国债收益率、10年期TIPS实际利率 |
| `engine.py` | **核心**：六个指标 → 252日滚动分位 → 方向修正 → 等权合成温度 → 六条预警规则 |
| `build.py` | 把 `data.json` 注入 `dash_tpl.html`，输出 `dashboard.html` |
| `dash_tpl.html` | 网页模板（图表、图例、方法说明、已知局限都在这里） |
| `daily_refresh.py` | 串起上面五步的编排脚本 |
| `test_engine.py` | 独立复核：分位数口径对 scipy、无前视、预警互斥等 |
| `raw/` | 十年原始数据（约43MB，540个CSV）。**删了也没关系**，脚本会自动重新下载 |
| `sectors.json` `lev_etfs.json` | 行业分类与杠杆ETF清单，可自行增删 |

改参数只改 `engine.py` 顶部的常量区：窗口长度、平滑天数、各条预警的阈值与确认天数、
实际利率闸门的两个门槛、熊市反弹的均线与温度门槛，都在那里，每个都有注释说明取值理由。
改完跑 `python3 test_engine.py` 确认没破坏基本性质。

---

## 三、数据来源（都是免费公开，无需注册）

| 数据 | 来源 |
|---|---|
| 标普500成分股名单 | Wikipedia |
| 个股 / ETF 十年日频量价 | stockanalysis.com |
| 标普500 TTM 每股收益 | multpl.com（月频） |
| VIX | CBOE 官方 CDN |
| 10年期名义国债、10年期TIPS实际利率 | 美国财政部官网（按年分文件的CSV） |

**如果某个源哪天挂了**：每个抓取函数都是独立的，换源只需改那一个函数的 URL 和解析逻辑，
`engine.py` 完全不用动——它只认 `raw/<TICKER>.csv` 这个格式（`日期,收盘,成交量`，新到旧）。
常见替代：Yahoo Finance（`yfinance` 库）、Stooq、FRED（VIX 用 `VIXCLS`、10年期用 `DGS10`、
实际利率用 `DFII10`）。仓库里还留了一个 `fetch_sp500_yahoo.py` 作为备份实现。

---

## 四、让它每天自动更新

### 方案 A：自己的电脑 / 服务器（最简单）

macOS / Linux，`crontab -e` 加一行（每天美东早八点，按你所在时区换算）：

```
0 8 * * 1-5 cd /path/to/us-sentiment && /usr/bin/python3 daily_refresh.py >> run.log 2>&1
```

Windows 用「任务计划程序」建一个每日任务，操作填 `python.exe`，参数填
`daily_refresh.py`，起始位置填项目目录。

电脑关机那天就不更新，但下次开机跑一次会自动补上（脚本按交易日判重，不会重复报警）。

### 方案 B：GitHub Actions + GitHub Pages（免费、不用自己开机）

把项目推到一个私有仓库，加 `.github/workflows/daily.yml`：

```yaml
name: daily
on:
  schedule: [{cron: "0 12 * * 1-5"}]   # UTC 12:00 = 美东早8点（夏令时）
  workflow_dispatch:
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install pandas numpy scipy
      - run: python3 daily_refresh.py
      - run: |
          git config user.name  github-actions
          git config user.email github-actions@github.com
          git add -A && git commit -m "daily $(date -u +%F)" || exit 0
          git push
```

再在仓库设置里开启 GitHub Pages（指向根目录），`dashboard.html` 就有了一个固定网址。
免费额度对这个用量绰绰有余。**注意** `raw/` 有43MB，要么提交进仓库（可以，GitHub 单仓库
限制远大于此），要么在 workflow 里加 `actions/cache` 缓存它。

### 邮件/推送提醒

`daily_refresh.py` 的标准输出最后一行固定是
`NEWLY_TRIGGERED:none` 或 `NEWLY_TRIGGERED:hot,cold,cold_soft,hot_bear,crowd,narrow`。
任何脚本都能抓这一行来决定要不要发通知，比如：

```bash
OUT=$(python3 daily_refresh.py)
echo "$OUT" | grep -q "NEWLY_TRIGGERED:none" || echo "$OUT" | mail -s "情绪温度计有新预警" you@example.com
```

GitHub Actions 里可以接 issue、Slack webhook、或者 Server酱之类的微信推送。

---

## 五、六条预警规则速查

| 记号 | 名称 | 条件 |
|---|---|---|
| ● 红点 | 红点预警 | 综合温度 > 80，连续3日 |
| ○ 空心红 | 熊市反弹预警 | 跌破下行的200日均线 且 温度 > 60，连续3日 |
| ● 蓝点 | 蓝点预警 | 快口径温度 < 20 且 VIX ≥ 30，当日触发 |
| ○ 空心蓝 | 空心蓝点 | 同上，但处于贴现率重估状态（10年期TIPS < 0.5% 且6个月上行 > 0.25pp，带滞回） |
| ▢ 黑框 | 黑框预警 | TOP2行业成交额占比 > 90分位 且 杠杆多空比 > 80分位，连续3日 |
| ▬ 橙线 | 橙线预警 | 窄幅逼空评分 > 80，连续3日（纳指走势线本身染成橙色） |

**实心 = 标准信号，空心 = 需要打折的信号。**
每条规则的取值理由、回测数据和已知局限，都写在网页的「已知局限」区块和 `engine.py` 的注释里。
那些注释是这套东西最值钱的部分——它们记录了什么试过、什么没用、为什么。别删。

---

## 六、一句提醒

这是个观察工具，不是交易系统。所有回测都是同一段历史上做出来的，
样本只有 2017 年至今、九次蓝点、一次真正的加息周期。
网页上「已知局限」那一栏不是免责声明，是使用说明——用之前先读一遍。
