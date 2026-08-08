"""从已有的 method_comparison.json 重新生成对比图（无需重新训练）。

用法（在项目根目录）：
    python plot/logs/regenerate_plots.py

前提：先把服务器训练后生成的 DL_model/logs/method_comparison.json 拷回本地。
"""

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from plot import plot_method_comparison, plot_paper_figures

SUMMARY = os.path.join(_ROOT, "DL_model", "logs", "method_comparison.json")
LOG_ROOT = os.path.join(_ROOT, "DL_model", "logs")


def main():
    if not os.path.exists(SUMMARY):
        print(f"找不到 {SUMMARY}\n请先把服务器上的 DL_model/logs/method_comparison.json 拷回本地。")
        return
    with open(SUMMARY, encoding="utf-8") as f:
        summaries = json.load(f)
    plot_method_comparison(summaries, os.path.join(LOG_ROOT, "method_comparison.png"))
    plot_paper_figures(summaries, LOG_ROOT)
    print(f"完成：已生成对比图到 {LOG_ROOT}/")


if __name__ == "__main__":
    main()
