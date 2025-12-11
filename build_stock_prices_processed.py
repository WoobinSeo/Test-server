"""
`stock_prices` 테이블의 일봉 데이터를 기반으로
기술적 지표를 계산하여 `stock_prices_processed` 테이블에 적재하는 스크립트.

용도:
  - `/indicator` 엔드포인트에서 사용할 DB 데이터를 채우기 위함.

실행 예시 (venv 활성화 후):
  python build_stock_prices_processed.py
"""

from __future__ import annotations

from typing import List, Dict, Any

import pandas as pd
from sqlalchemy import text

from db_utils import get_engine, load_config_stocks
from technical_indicators import TechnicalIndicators


def load_daily_from_db(engine, stock_code: str) -> pd.DataFrame:
    """stock_prices 테이블에서 해당 종목 일봉 OHLCV 로드."""
    query = text(
        """
        SELECT stock_code, stock_name, datetime,
               open, high, low, close, volume
        FROM stock_prices
        WHERE stock_code = :code
        ORDER BY datetime ASC
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql_query(query, conn, params={"code": stock_code})
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    TechnicalIndicators를 사용해 모든 지표를 계산하고,
    DB 스키마에 맞게 컬럼명을 정리한다.
    """
    if df.empty:
        return df

    # 기술적 지표 계산
    df = TechnicalIndicators.add_all_indicators(df)

    # NaN 제거 (초기 구간 등)
    df = df.dropna().reset_index(drop=True)

    # TechnicalIndicators에서 사용하는 컬럼명 → DB 컬럼명 매핑
    rename_map = {
        "MA_5": "ma_5",
        "MA_10": "ma_10",
        "MA_20": "ma_20",
        "MA_60": "ma_60",
        "EMA_12": "ema_12",
        "EMA_26": "ema_26",
        "RSI": "rsi",
        "MACD": "macd",
        "MACD_Signal": "macd_signal",
        "MACD_Hist": "macd_hist",
        "BB_Upper": "bb_upper",
        "BB_Middle": "bb_middle",
        "BB_Lower": "bb_lower",
        "BB_Width": "bb_width",
        "BB_PctB": "bb_pctb",
        "Stoch_K": "stoch_k",
        "Stoch_D": "stoch_d",
        "ATR": "atr",
        "Volume_MA_5": "volume_ma_5",
        "Volume_MA_20": "volume_ma_20",
        "Volume_Ratio": "volume_ratio",
        "OBV": "obv",
        "Return": "return_1d",
        "Log_Return": "log_return",
        "Return_5d": "return_5d",
        "Return_10d": "return_10d",
        "Return_20d": "return_20d",
        "HL_Ratio": "hl_ratio",
        "CO_Ratio": "co_ratio",
    }
    df = df.rename(columns=rename_map)

    return df


def save_processed_to_db(engine, df: pd.DataFrame, stock_code: str, stock_name: str) -> None:
    """
    stock_prices_processed 테이블에 데이터 삽입.

    - 기존 해당 종목 데이터는 삭제 후 다시 적재 (id 중복/중복 row 방지)
    """
    if df.empty:
        print(f"  ⚠️  {stock_name} ({stock_code}) 전처리 결과가 비어 있습니다. 건너뜁니다.")
        return

    print(f"  DB 적재 중... (rows={len(df):,})")

    required_cols: List[str] = [
        "stock_code",
        "stock_name",
        "datetime",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma_5",
        "ma_10",
        "ma_20",
        "ma_60",
        "ema_12",
        "ema_26",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "bb_width",
        "bb_pctb",
        "rsi",
        "stoch_k",
        "stoch_d",
        "atr",
        "volume_ma_5",
        "volume_ma_20",
        "volume_ratio",
        "obv",
        "return_1d",
        "log_return",
        "return_5d",
        "return_10d",
        "return_20d",
        "hl_ratio",
        "co_ratio",
    ]

    # 누락 컬럼은 None으로 채워서 DB 스키마와 정렬
    for col in required_cols:
        if col not in df.columns:
            df[col] = None

    subset = df[required_cols].copy()

    # stock_code / stock_name은 인자로 받은 값으로 통일
    subset["stock_code"] = stock_code
    subset["stock_name"] = stock_name

    # 레코드 리스트 생성
    records: List[Dict[str, Any]] = []
    for row in subset.itertuples(index=False, name=None):
        record = {col: val for col, val in zip(required_cols, row)}
        records.append(record)

    with engine.begin() as conn:
        # 기존 데이터 삭제
        conn.execute(
            text("DELETE FROM stock_prices_processed WHERE stock_code = :code"),
            {"code": stock_code},
        )
        # 새 데이터 삽입
        conn.execute(
            text(
                """
                INSERT INTO stock_prices_processed (
                    stock_code,
                    stock_name,
                    datetime,
                    open,
                    high,
                    low,
                    close,
                    volume,
                    ma_5,
                    ma_10,
                    ma_20,
                    ma_60,
                    ema_12,
                    ema_26,
                    macd,
                    macd_signal,
                    macd_hist,
                    bb_upper,
                    bb_middle,
                    bb_lower,
                    bb_width,
                    bb_pctb,
                    rsi,
                    stoch_k,
                    stoch_d,
                    atr,
                    volume_ma_5,
                    volume_ma_20,
                    volume_ratio,
                    obv,
                    return_1d,
                    log_return,
                    return_5d,
                    return_10d,
                    return_20d,
                    hl_ratio,
                    co_ratio
                )
                VALUES (
                    :stock_code,
                    :stock_name,
                    :datetime,
                    :open,
                    :high,
                    :low,
                    :close,
                    :volume,
                    :ma_5,
                    :ma_10,
                    :ma_20,
                    :ma_60,
                    :ema_12,
                    :ema_26,
                    :macd,
                    :macd_signal,
                    :macd_hist,
                    :bb_upper,
                    :bb_middle,
                    :bb_lower,
                    :bb_width,
                    :bb_pctb,
                    :rsi,
                    :stoch_k,
                    :stoch_d,
                    :atr,
                    :volume_ma_5,
                    :volume_ma_20,
                    :volume_ratio,
                    :obv,
                    :return_1d,
                    :log_return,
                    :return_5d,
                    :return_10d,
                    :return_20d,
                    :hl_ratio,
                    :co_ratio
                )
                """
            ),
            records,
        )

    print("  ✅ DB 적재 완료")


def main():
    print(
        """
============================================================
  stock_prices → stock_prices_processed 지표 적재
============================================================
"""
    )
    engine = get_engine()
    stocks = load_config_stocks()

    results: Dict[str, bool] = {}
    for s in stocks:
        code = s["code"]
        name = s["name"]

        print("\n" + "=" * 60)
        print(f"{name} ({code}) 처리 시작")
        print("=" * 60)

        try:
            df = load_daily_from_db(engine, code)
            if df.empty:
                print(f"  ⚠️  DB에 {code} 데이터가 없습니다. 건너뜁니다.")
                results[name] = False
                continue

            df_ind = compute_indicators(df)
            save_processed_to_db(engine, df_ind, code, name)
            results[name] = True
        except Exception as e:
            print(f"[ERROR] {name} 처리 중 오류: {e}")
            results[name] = False

    print("\n요약:")
    for name, ok in results.items():
        print(f"  {name}: {'성공' if ok else '실패'}")


if __name__ == "__main__":
    main()


