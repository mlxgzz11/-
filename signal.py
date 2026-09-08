import os
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from tqdm import tqdm as text_tqdm

import akshare as ak
import pandas as pd
import numpy as np

# ========== 配置 ==========
HISTORY_FILE = Path("history_breadth.csv")
UP_THRESHOLD = 0.04
NORM_WINDOW = 20
BASE_WINDOW = 120
SIGNAL_QUANTILE = 0.80

# 从 GitHub Secrets 读取邮箱配置
SMTP_HOST = "smtp.qq.com"
SMTP_PORT = 465
MAIL_USER = os.environ.get("MAIL_USER")      # 你的QQ邮箱
MAIL_PASS = os.environ.get("MAIL_PASS")      # 16位授权码
MAIL_TO   = os.environ.get("MAIL_TO")        # 收件邮箱（可以和发件相同）

def send_email(subject: str, body: str):
    if not all([MAIL_USER, MAIL_PASS, MAIL_TO]):
        print("邮箱配置缺失，跳过发信")
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = MAIL_USER
    msg["To"] = MAIL_TO
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
        server.login(MAIL_USER, MAIL_PASS)
        server.send_message(msg)
    print("邮件已发送")

def main():
    now = datetime.now()
    print(f"开始运行：{now}")

    # 1. 取全市场行情（临时屏蔽 tqdm.notebook）
    try:
        with patch("akshare.stock.stock_zh_a_sina.get_tqdm", return_value=text_tqdm):
            quotes = ak.stock_zh_a_spot()
    except Exception as e:
        body = f"取数失败：{e}\n时间：{now}"
        send_email("【上涨广度】取数失败", body)
        raise

    print(f"取到股票数：{len(quotes)}")

    # 2. 简单计算今日上涨广度（注意：这里用「最新价 / 昨收」近似，后续可再优化成真正的昨日14:55参考价）
    quotes = quotes.copy()
    quotes["最新价"] = pd.to_numeric(quotes["最新价"], errors="coerce")
    quotes["昨收"] = pd.to_numeric(quotes["昨收"], errors="coerce")
    valid = quotes["最新价"].gt(0) & quotes["昨收"].gt(0)
    ret = quotes.loc[valid, "最新价"] / quotes.loc[valid, "昨收"] - 1
    n_valid = valid.sum()
    n_up4 = (ret > UP_THRESHOLD).sum()
    breadth = (n_up4 + 0.5) / (n_valid + 1) if n_valid > 0 else np.nan

    # 3. 读取历史并计算信号
    if HISTORY_FILE.exists():
        hist = pd.read_csv(HISTORY_FILE, parse_dates=["date"])
    else:
        hist = pd.DataFrame(columns=["date", "breadth", "n_valid", "n_up4"])

    today_str = now.strftime("%Y-%m-%d")
    # 避免同一天重复写入
    hist = hist[hist["date"].dt.strftime("%Y-%m-%d") != today_str]

    new_row = pd.DataFrame([{
        "date": today_str,
        "breadth": breadth,
        "n_valid": n_valid,
        "n_up4": n_up4
    }])
    hist = pd.concat([hist, new_row], ignore_index=True)
    hist = hist.sort_values("date").reset_index(drop=True)
    hist.to_csv(HISTORY_FILE, index=False)

    # 计算得分和门槛（窗口不足时标记）
    hist["normalizer"] = hist["breadth"].shift(1).rolling(NORM_WINDOW, min_periods=NORM_WINDOW).median()
    hist["score"] = hist["breadth"] / hist["normalizer"]
    hist["threshold"] = hist["score"].shift(1).rolling(BASE_WINDOW, min_periods=BASE_WINDOW).quantile(SIGNAL_QUANTILE)

    last = hist.iloc[-1]
    score = last["score"]
    threshold = last["threshold"]
    signal = bool(score >= threshold) if pd.notna(threshold) else False
    window_ok = pd.notna(threshold)

    # 4. 组装邮件
    status = "数据窗口充足，信号有效" if window_ok else "历史窗口不足（需要约140个交易日），仅观察，勿直接交易"
    signal_text = "【建议开仓】" if signal and window_ok else "【空仓 / 观察】"

    body = f"""
上涨广度择时信号（GitHub Actions 测试版）
========================================
运行时间：{now}
采样股票数：{len(quotes)}
有效股票数：{n_valid}
上涨>4%家数：{n_up4}
原始上涨占比：{breadth:.4f}
标准化得分：{score:.4f}
动态门槛：{threshold if pd.notna(threshold) else 'N/A'}
最终信号：{signal_text}
数据状态：{status}

注意：
1. 当前使用「最新价/昨收」近似涨幅，与定稿14:55逻辑不完全一致。
2. 历史窗口未满前，请勿把信号当正式交易依据。
3. 本邮件仅供观察，实际交易请手动确认。
"""

    subject = f"上涨广度 {today_str} {signal_text}"
    send_email(subject, body)
    print(body)
    print("完成")

if __name__ == "__main__":
    main()
