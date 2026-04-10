"""
yfinance で複数銘柄の取得可否をテストする。
実行: python test_yfinance_tickers.py
"""
from datetime import date, timedelta

def test_ticker(ticker: str, label: str, years: int = 2) -> tuple[bool, str, int]:
    """1銘柄を download と Ticker().history の両方で試し、成功・件数を返す。"""
    import yfinance as yf
    end_d = date.today()
    start_d = end_d - timedelta(days=years * 365)
    start_str = start_d.isoformat()
    end_str = end_d.isoformat()
    results = []

    # 1) yf.download (session なし)
    try:
        df = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=True, threads=False)
        ok = df is not None and hasattr(df, "empty") and not df.empty
        n = len(df) if df is not None and hasattr(df, "__len__") else 0
        results.append(("download(sessionなし)", ok, n, None))
    except Exception as e:
        results.append(("download(sessionなし)", False, 0, str(e)[:80]))

    # 2) yf.download (curl_cffi session あり)
    try:
        from curl_cffi import requests as curl_requests
        session = curl_requests.Session(impersonate="chrome")
        df = yf.download(ticker, start=start_str, end=end_str, progress=False, auto_adjust=True, threads=False, session=session)
        ok = df is not None and hasattr(df, "empty") and not df.empty
        n = len(df) if df is not None and hasattr(df, "__len__") else 0
        results.append(("download(session=curl_cffi)", ok, n, None))
    except Exception as e:
        results.append(("download(session=curl_cffi)", False, 0, str(e)[:80]))

    # 3) Ticker().history() (session なし)
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=start_str, end=end_str, auto_adjust=True)
        ok = df is not None and hasattr(df, "empty") and not df.empty
        n = len(df) if df is not None and hasattr(df, "__len__") else 0
        results.append(("Ticker().history(sessionなし)", ok, n, None))
    except Exception as e:
        results.append(("Ticker().history(sessionなし)", False, 0, str(e)[:80]))

    # 4) Ticker().history() (curl_cffi session あり)
    try:
        from curl_cffi import requests as curl_requests
        session = curl_requests.Session(impersonate="chrome")
        t = yf.Ticker(ticker, session=session)
        df = t.history(start=start_str, end=end_str, auto_adjust=True)
        ok = df is not None and hasattr(df, "empty") and not df.empty
        n = len(df) if df is not None and hasattr(df, "__len__") else 0
        results.append(("Ticker().history(session=curl_cffi)", ok, n, None))
    except Exception as e:
        results.append(("Ticker().history(session=curl_cffi)", False, 0, str(e)[:80]))

    # いずれか1つでも成功していれば True、件数は最大を返す
    any_ok = any(r[1] for r in results)
    max_n = max(r[2] for r in results)
    detail = "; ".join(f"{r[0]}: {'OK' if r[1] else 'NG'} (n={r[2]})" + (f" err={r[3]}" if r[3] else "") for r in results)
    return any_ok, detail, max_n


def main():
    tickers = [
        ("AAPL", "米国株 Apple"),
        ("MSFT", "米国株 Microsoft"),
        ("^GSPC", "S&P500"),
        ("7203.T", "東証 トヨタ"),
        ("^N225", "日経平均"),
    ]
    print("yfinance 銘柄取得テスト（過去2年）\n")
    for symbol, label in tickers:
        ok, detail, n = test_ticker(symbol, label)
        status = "OK" if ok else "NG"
        print(f"[{status}] {symbol} ({label}) 件数={n}")
        print(f"     {detail}\n")
    print("以上。")


if __name__ == "__main__":
    main()
