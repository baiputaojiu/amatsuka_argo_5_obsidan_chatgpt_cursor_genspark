"""
日経平均 / TOPIX を用いたレバレッジ戦略比較シミュレーション（Streamlit）

4戦略（現物1倍・現物2倍・ブル2倍・ベア2倍ショート）のパフォーマンスを比較・可視化する。
"""

from pathlib import Path
from datetime import date, datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf

from strategy import (
    TICKERS,
    fetch_data,
    compute_strategies,
    compute_summary,
    compute_divergence,
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"


def create_chart(
    raw_df: pd.DataFrame,
    result: pd.DataFrame,
    divergence: pd.DataFrame,
    ticker_label: str,
) -> plt.Figure:
    """
    3段構成のチャート（戦略比較 / 乖離幅 / ロウソク足）を作成する。
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False, height_ratios=[1.5, 0.8, 1.2])
    fig.subplots_adjust(hspace=0.35)

    # 上段: 4戦略の累積リターン
    ax1 = axes[0]
    ax1.plot(result["Date"], result["spot_1x"], label="現物1倍", linewidth=1.5)
    ax1.plot(result["Date"], result["spot_2x"], label="現物2倍", linewidth=1.5)
    ax1.plot(result["Date"], result["bull_2x"], label="ブル2倍ロング", linewidth=1.5)
    ax1.plot(result["Date"], result["bear_2x_short"], label="ベア2倍ショート", linewidth=1.5)
    ax1.set_ylabel("累積リターン（初日=1）")
    ax1.set_title(f"{ticker_label} レバレッジ戦略比較")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    # 中段: 乖離幅
    ax2 = axes[1]
    ax2.plot(divergence["Date"], divergence["bull_excess"], label="ブル2倍 - 現物2倍 (%)", linewidth=1.2)
    ax2.plot(divergence["Date"], divergence["bear_excess"], label="ベア2倍ショート - 現物2倍 (%)", linewidth=1.2)
    ax2.set_ylabel("乖離 (%)")
    ax2.set_title("現物2倍に対する乖離幅")
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # 下段: ロウソク足
    ax3 = axes[2]
    ohlc = raw_df.copy()
    if "Open" in ohlc.columns and "High" in ohlc.columns and "Low" in ohlc.columns:
        ohlc = ohlc.set_index("Date")
        ohlc = ohlc[["Open", "High", "Low", "Close"]].dropna()
        ohlc.index = pd.DatetimeIndex(ohlc.index)
        mpf.plot(
            ohlc,
            type="candle",
            ax=ax3,
            style="charles",
        )
    else:
        ax3.plot(raw_df["Date"], raw_df["Close"], color="steelblue")
        ax3.set_ylabel("終値")
        ax3.set_title(f"{ticker_label} 終値")
    ax3.set_xlabel("日付")

    plt.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="レバレッジ戦略シミュレーション", layout="wide")
    st.title("日経平均 / TOPIX レバレッジ戦略比較")

    with st.sidebar:
        st.header("設定")
        ticker = st.radio(
            "指数",
            options=list(TICKERS.keys()),
            format_func=lambda k: f"{k} ({TICKERS[k]})",
            index=0,
        )
        default_end = date.today()
        default_start = date(2024, 1, 1)
        start_date = st.date_input("開始日", value=default_start)
        end_date = st.date_input("終了日", value=default_end)

        if st.button("データ取得＆計算"):
            st.session_state["run_calc"] = True
        else:
            st.session_state.setdefault("run_calc", False)

    if st.session_state.get("run_calc"):
        with st.spinner("データを取得しています..."):
            raw_df = fetch_data(ticker, start_date, end_date)

        if raw_df is None or len(raw_df) == 0:
            st.error("データの取得に失敗しました。期間やティッカーを確認してください。")
            return

        result = compute_strategies(raw_df)
        summary = compute_summary(result)
        divergence = compute_divergence(result)

        ticker_label = TICKERS[ticker]
        st.success(f"{ticker_label} ({ticker}): {len(raw_df)} 日分のデータを取得しました。")

        # サマリテーブル
        st.subheader("サマリ")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # チャート
        st.subheader("チャート")
        fig = create_chart(raw_df, result, divergence, ticker_label)
        st.pyplot(fig)
        plt.close(fig)

        # 保存機能
        st.session_state["summary_df"] = summary
        st.session_state["fig_data"] = (raw_df, result, divergence, ticker_label)
        st.session_state["calc_done"] = True

    if st.session_state.get("calc_done"):
        st.divider()
        if st.button("グラフ・CSV保存"):
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")

            # PNG 保存
            raw_df, result, divergence, ticker_label = st.session_state["fig_data"]
            fig = create_chart(raw_df, result, divergence, ticker_label)
            png_path = OUTPUT_DIR / f"leverage_strategy_{ts}.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            st.success(f"グラフを保存しました: {png_path}")

            # CSV 保存
            summary_df = st.session_state["summary_df"]
            csv_path = OUTPUT_DIR / f"leverage_strategy_summary_{ts}.csv"
            summary_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            st.success(f"サマリを保存しました: {csv_path}")


if __name__ == "__main__":
    main()
