"""
Yahoo Finance에서 국내 주식 데이터를 수집하여
PostgreSQL stock_prices 테이블에 적재하는 스크립트.

현재 설정:
- 일봉(interval='1d')
- 최근 5년(period='5y')
"""

import pandas as pd
import yfinance as yf
from sqlalchemy import text

from db_utils import get_engine


STOCKS = [
    {"code": "005930", "name": "삼성전자", "ticker": "005930.KS"},
    {"code": "035420", "name": "네이버", "ticker": "035420.KS"},
    {"code": "005380", "name": "현대차", "ticker": "005380.KS"},
]


def download_daily_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Yahoo Finance에서 일봉 데이터 다운로드.

    Args:
        ticker: 예) "005930.KS"
        period: 예) "5y" (최근 5년)
    """
    print(f"\n[ticker={ticker}] 일봉 데이터 다운로드 ({period})")
    df = yf.download(
        ticker,
        interval="1d",
        period=period,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        print("  ⚠️  다운로드된 데이터가 없습니다.")
        return df

    df = df.reset_index()  # DatetimeIndex → 컬럼

    # 시간 컬럼 이름은 버전에 따라 Datetime/Date/index 등이 될 수 있으므로 보정
    time_col = None
    for cand in ["Datetime", "Date", "datetime", "date", "index"]:
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        raise ValueError(f"시간 컬럼을 찾을 수 없습니다. columns={df.columns}")

    rename_map = {
        time_col: "datetime",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df.rename(columns=rename_map, inplace=True)

    # 컬럼 정리
    df = df[["datetime", "open", "high", "low", "close", "volume"]]
    return df


def save_to_db(engine, df: pd.DataFrame, stock_code: str, stock_name: str):
    """
    stock_prices 테이블에 데이터 삽입.
    """
    if df.empty:
        return

    print(f"  DB 적재 중... (rows={len(df):,})")
    # datetime 타입 보정
    if not pd.api.types.is_datetime64_any_dtype(df["datetime"]):
        df["datetime"] = pd.to_datetime(df["datetime"])

    # 레코드 리스트 생성
    # 필요한 컬럼만 순서를 고정해서 사용
    subset = df[["datetime", "open", "high", "low", "close", "volume"]]
    records = []
    for dt_val, o, h, l, c, v in subset.itertuples(index=False, name=None):
        records.append(
            {
                "stock_code": stock_code,
                "stock_name": stock_name,
                "datetime": dt_val,
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": int(v),
            }
        )

    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO stock_prices
                    (stock_code, stock_name, datetime, open, high, low, close, volume)
                VALUES
                    (:stock_code, :stock_name, :datetime, :open, :high, :low, :close, :volume)
                """
            ),
            records,
        )
    print("  ✅ DB 적재 완료")


def main():
    engine = get_engine()
    print("PostgreSQL 엔진 준비 완료.")

    for s in STOCKS:
        code = s["code"]
        name = s["name"]
        ticker = s["ticker"]

        print("\n" + "=" * 60)
        print(f"{name} ({code}) - Yahoo Finance 수집 시작")
        print("=" * 60)

        df = download_daily_data(ticker, period="5y")
        if df.empty:
            continue

        save_to_db(engine, df, code, name)

        print(f"{name} ({code}) - 완료")

    print("\n모든 종목 처리 완료.")


if __name__ == "__main__":
    main()


