#!/bin/bash
# 双击我：下载最新行情 → 重算温度 → 打开网页
#
# 从 GitHub 下载整个文件夹（Code → Download ZIP）解压后，双击此文件即可。
# 首次运行需要几分钟（本地没有历史数据，要把 531 只标的的十年日线下下来），
# 之后每次只要几十秒。
#
# 想强制重下十年历史（怀疑数据有问题时），改双击「完整更新并打开.command」。

# 双击启动时，Terminal 的工作目录是家目录而不是脚本所在目录，必须先切过来，
# 否则 python3 daily_refresh.py 会报 "No such file or directory"。
cd "$(dirname "$0")" || { echo "无法进入脚本所在目录"; exit 1; }

# 出错时把原因留在屏幕上——否则 Terminal 窗口一闪而过，什么也看不见。
die() {
  echo
  echo "────────────────────────────────────────"
  echo "❌ $*"
  echo "────────────────────────────────────────"
  echo
  read -n 1 -s -r -p "按任意键关闭此窗口…"
  echo
  exit 1
}

echo "════════════════════════════════════════"
echo "  美股情绪温度计 · 更新并打开"
echo "  目录：$(pwd)"
echo "════════════════════════════════════════"
echo

# ---------- 0. 解除 macOS 隔离标记 ----------
# 从网上下载的 ZIP 解压出来的文件都带 com.apple.quarantine，Gatekeeper 会拦。
# 你能看到这行说明本脚本已经被放行了（多半是右键→打开），顺手把同目录其余文件
# 的标记一并清掉，省得以后再被拦。失败也无所谓，不影响主流程。
xattr -dr com.apple.quarantine . 2>/dev/null || true

# ---------- 1. 找 python3 ----------
PY="$(command -v python3 2>/dev/null)"
[ -n "$PY" ] || die "找不到 python3。
   macOS 自带的命令行工具里就有，装一下即可：
       xcode-select --install
   装完重新双击本文件。"
echo "① Python：$PY（$("$PY" -V 2>&1)）"

# ---------- 2. 检查依赖 ----------
# 整条流水线只依赖这三个包（见 README 第一节）。
if "$PY" -c 'import pandas, numpy, scipy' 2>/dev/null; then
  echo "② 依赖：pandas / numpy / scipy 已就绪"
else
  echo "② 依赖缺失，正在安装 pandas / numpy / scipy…（首次可能要一两分钟）"
  # --user 装到用户目录，不碰系统 Python；较新的 pip 会拒绝装进"外部管理"的环境，
  # 那种情况下退回普通安装再试一次。
  if ! "$PY" -m pip install --user --quiet pandas numpy scipy 2>/dev/null \
     && ! "$PY" -m pip install --quiet pandas numpy scipy 2>/dev/null; then
    die "依赖安装失败。请手动在终端里跑：
       $PY -m pip install --user pandas numpy scipy
   装完再双击本文件。"
  fi
  "$PY" -c 'import pandas, numpy, scipy' 2>/dev/null \
    || die "依赖装完了但仍然导入失败，可能装到了另一个 Python 里。
   手动确认：$PY -c 'import pandas'"
  echo "   安装完成"
fi

# ---------- 3. 首次运行提示 ----------
# raw/ 不在 git 仓库里（43MB、每天重写，提交进去会让历史无限膨胀），
# 所以从 GitHub 下载下来的文件夹里没有它，第一次必须整体下载。
if [ ! -d raw ] || [ -z "$(ls -A raw 2>/dev/null)" ]; then
  echo
  echo "③ ⚠️  首次运行：本地还没有历史数据"
  echo "      需要下载 531 只标的 × 十年日线，约 40MB。"
  echo "      视网络情况 3–10 分钟，跨境链路可能更久。请耐心等待，别关窗口。"
  echo "      以后每次只下增量，几十秒就好。"
else
  echo "③ 本地已有历史数据（raw/ 共 $(ls raw | wc -l | tr -d ' ') 个文件）"
fi

# ---------- 4. 跑流水线 ----------
echo
echo "④ 开始更新……"
echo "────────────────────────────────────────"
if [ "$SENTIMENT_FULL" = "1" ]; then
  MODE=""            # 全量：每只票重下十年
  echo "   模式：全量（强制重下十年历史）"
else
  MODE="--fast"      # 增量：只下最近半年再并回本地；缺文件的票会自动走全量
  echo "   模式：增量（缺数据的标的会自动补全量）"
fi
echo

# tee 到 run.log 便于出问题时回看；run.log 已在 .gitignore 里，不会进仓库。
set -o pipefail
"$PY" daily_refresh.py $MODE 2>&1 | tee run.log
STATUS=$?
set +o pipefail

echo "────────────────────────────────────────"
[ $STATUS -eq 0 ] || die "更新失败（退出码 $STATUS）。
   上面的日志里有具体原因，也存在同目录的 run.log 里。
   常见情况：网络不通或数据源临时不可用——过一会儿再试通常就好了。"

# ---------- 5. 打开网页 ----------
[ -f dashboard.html ] || die "流水线跑完了，但没有生成 dashboard.html。
   请把 run.log 的内容发出来排查。"

echo
echo "⑤ 完成，正在打开网页…"
open dashboard.html 2>/dev/null || die "网页生成好了但打不开。
   手动双击同目录下的 dashboard.html 即可。"

echo
echo "✅ 全部完成。此窗口可以关掉了。"
echo
# 成功时也停一下，让你能看清最后那几行读数（温度、减仓温度、有无新预警）。
read -n 1 -s -r -t 20 -p "按任意键关闭（20 秒后自动关闭）…"
echo
