"""
일봉 기준 상승/하락/유지 분류용 전처리 스크립트.

파이프라인:
 1) PostgreSQL stock_prices 에서 일봉 데이터 로드
 2) TechnicalIndicators 로 37개 지표 계산
 3) 다음날 종가 기준으로 UP / HOLD / DOWN 레이블 생성
 4) 시간 순으로 70/15/15 (train/val/test) 분리
 5) MinMaxScaler 로 특성 정규화 (train 기준)
 6) data/daily_classification 폴더에 CSV 및 스케일러 저장
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import text
import pickle

from technical_indicators import TechnicalIndicators
from db_utils import get_engine, load_config_stocks


def load_daily_from_db(engine, stock_code: str) -> pd.DataFrame:
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
    if df.empty:
        print(f"[WARN] DB에 {stock_code} 일봉 데이터 없음")
    return df


def add_labels_daily(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    """
    다음날 종가 기준 수익률로 UP/HOLD/DOWN 레이블 생성.

    threshold=0.01 → +1% 이상: UP, -1% 이하: DOWN, 그 외 HOLD
    """
    df = df.copy()
    df["future_close"] = df["close"].shift(-1)
    df["ret_1d"] = (df["future_close"] - df["close"]) / df["close"]

    def classify(x):
        if pd.isna(x):
            return None
        if x > threshold:
            return "UP"
        elif x < -threshold:
            return "DOWN"
        else:
            return "HOLD"

    df["target"] = df["ret_1d"].apply(classify)

    # 마지막 행(drop)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    print("\n  타겟 분포:")
    vc = df["target"].value_counts()
    total = len(df)
    for lab in ["UP", "HOLD", "DOWN"]:
        if lab in vc.index:
            c = vc[lab]
            print(f"    {lab}: {c:,}개 ({c/total*100:.1f}%)")

    df = df.drop(columns=["future_close", "ret_1d"])
    return df


def preprocess_stock_daily(
    engine,
    stock_code: str,
    stock_name: str,
    threshold: float = 0.01,
    output_dir: str = "data/daily_classification",
):
    print(f"\n{'='*60}")
    print(f"{stock_name} ({stock_code}) 일봉 분류 전처리")
    print(f"{'='*60}")
    print(f"설정: 다음날 종가 기준 ±{threshold*100:.1f}% → UP/DOWN 나머지 HOLD")

    df = load_daily_from_db(engine, stock_code)
    if df.empty:
        return False

    print(f"\n1. 원본 일봉: {len(df):,}개")

    # 기술적 지표 추가
    print("\n2. 기술적 지표 계산...")
    df = TechnicalIndicators.add_all_indicators(df)

    # NaN 제거
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    removed = before - len(df)
    print(f"\n3. 결측치 제거: {removed}개 삭제, 최종 {len(df):,}개")

    # 레이블 생성
    print("\n4. 레이블 생성...")
    df = add_labels_daily(df, threshold=threshold)
    print(f"  레이블 생성 후: {len(df):,}개")

    # 특성/타겟 분리
    print("\n5. 특성/타겟 분리 및 데이터 분리(70/15/15)...")
    exclude_cols = ["datetime", "stock_code", "stock_name", "target"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].values
    # 직접 인코딩: DOWN=0, HOLD=1, UP=2
    label_map = {"DOWN": 0, "HOLD": 1, "UP": 2}
    y = df["target"].map(label_map).astype(int).values

    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    X_train, X_val, X_test = (
        X[:train_end],
        X[train_end:val_end],
        X[val_end:],
    )
    y_train, y_val, y_test = (
        y[:train_end],
        y[train_end:val_end],
        y[val_end:],
    )

    print(f"  학습: {len(X_train):,}개")
    print(f"  검증: {len(X_val):,}개")
    print(f"  테스트: {len(X_test):,}개")

    # 정규화
    print("\n6. MinMax 정규화 (train 기준)...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 저장
    print("\n7. CSV 및 스케일러 저장...")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_split(X_split, y_split, split_name: str):
        df_split = pd.DataFrame(X_split, columns=feature_cols)
        df_split["target"] = y_split
        # datetime은 레퍼런스로 넣어두면 좋다 (시퀀스 구성 등)
        if "datetime" in df.columns:
            if split_name == "train":
                idx_slice = slice(0, len(df_split))
            elif split_name == "val":
                idx_slice = slice(train_end, train_end + len(df_split))
            else:
                idx_slice = slice(val_end, val_end + len(df_split))
            df_split["datetime"] = df["datetime"].iloc[idx_slice].values
        fname = out_dir / f"{stock_name}_{split_name}_daily_class.csv"
        df_split.to_csv(fname, index=False, encoding="utf-8-sig")
        print(f"  - {fname}")

    save_split(X_train_scaled, y_train, "train")
    save_split(X_val_scaled, y_val, "val")
    save_split(X_test_scaled, y_test, "test")

    # 스케일러 저장
    scaler_path = out_dir / f"{stock_name}_daily_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump({"scaler": scaler, "features": feature_cols}, f)
    print(f"  - 스케일러 저장: {scaler_path}")

    print(f"\n{stock_name} 전처리 완료!")
    return True


def main():
    print(
        """
    ============================================================
      일봉 상승/하락/유지 분류용 전처리
    ============================================================
    """
    )
    engine = get_engine()
    stocks = load_config_stocks()

    results = {}
    for s in stocks:
        code = s["code"]
        name = s["name"]
        try:
            ok = preprocess_stock_daily(engine, code, name, threshold=0.01)
            results[name] = ok
        except Exception as e:
            print(f"\n[ERROR] {name} 처리 중 오류: {e}")
            results[name] = False

    print("\n요약:")
    for name, ok in results.items():
        print(f"  {name}: {'성공' if ok else '실패'}")


if __name__ == "__main__":
    main()


