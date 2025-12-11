"""
기술적 지표 계산 모듈
"""
import pandas as pd
import numpy as np
from typing import Tuple


class TechnicalIndicators:
    """주식 기술적 지표 계산"""
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, periods: list = [5, 10, 20, 60]) -> pd.DataFrame:
        """
        이동평균선 추가
        
        Args:
            df: 주가 데이터
            periods: 이동평균 기간 리스트
            
        Returns:
            이동평균이 추가된 DataFrame
        """
        df = df.copy()
        
        for period in periods:
            df[f'MA_{period}'] = df['close'].rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def add_exponential_moving_averages(df: pd.DataFrame, periods: list = [12, 26]) -> pd.DataFrame:
        """
        지수이동평균선 추가
        
        Args:
            df: 주가 데이터
            periods: EMA 기간 리스트
            
        Returns:
            EMA가 추가된 DataFrame
        """
        df = df.copy()
        
        for period in periods:
            df[f'EMA_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        return df
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        RSI (Relative Strength Index) 추가
        
        Args:
            df: 주가 데이터
            period: RSI 계산 기간
            
        Returns:
            RSI가 추가된 DataFrame
        """
        df = df.copy()
        
        # 가격 변화
        delta = df['close'].diff()
        
        # 상승/하락 분리
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        # RS 계산
        rs = gain / loss
        
        # RSI 계산
        df['RSI'] = 100 - (100 / (1 + rs))
        
        return df
    
    @staticmethod
    def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        MACD (Moving Average Convergence Divergence) 추가
        
        Args:
            df: 주가 데이터
            fast: 빠른 EMA 기간
            slow: 느린 EMA 기간
            signal: 시그널 라인 기간
            
        Returns:
            MACD가 추가된 DataFrame
        """
        df = df.copy()
        
        # EMA 계산
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        # MACD 라인
        df['MACD'] = ema_fast - ema_slow
        
        # 시그널 라인
        df['MACD_Signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
        
        # MACD 히스토그램
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        
        return df
    
    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
        """
        볼린저 밴드 추가
        
        Args:
            df: 주가 데이터
            period: 이동평균 기간
            std_dev: 표준편차 배수
            
        Returns:
            볼린저 밴드가 추가된 DataFrame
        """
        df = df.copy()
        
        # 중간 밴드 (이동평균)
        df['BB_Middle'] = df['close'].rolling(window=period).mean()
        
        # 표준편차
        std = df['close'].rolling(window=period).std()
        
        # 상단/하단 밴드
        df['BB_Upper'] = df['BB_Middle'] + (std * std_dev)
        df['BB_Lower'] = df['BB_Middle'] - (std * std_dev)
        
        # 밴드 폭 (Bandwidth)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
        
        # %B (Price position within bands)
        df['BB_PctB'] = (df['close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df
    
    @staticmethod
    def add_stochastic(df: pd.DataFrame, period: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """
        스토캐스틱 오실레이터 추가
        
        Args:
            df: 주가 데이터
            period: 기간
            smooth_k: %K 스무딩
            smooth_d: %D 스무딩
            
        Returns:
            스토캐스틱이 추가된 DataFrame
        """
        df = df.copy()
        
        # 최저가/최고가
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        
        # %K 계산
        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['Stoch_K'] = stoch_k.rolling(window=smooth_k).mean()
        
        # %D 계산
        df['Stoch_D'] = df['Stoch_K'].rolling(window=smooth_d).mean()
        
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        ATR (Average True Range) 추가
        
        Args:
            df: 주가 데이터
            period: ATR 계산 기간
            
        Returns:
            ATR이 추가된 DataFrame
        """
        df = df.copy()
        
        # True Range 계산
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        
        # ATR 계산
        df['ATR'] = true_range.rolling(window=period).mean()
        
        return df
    
    @staticmethod
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        거래량 지표 추가
        
        Args:
            df: 주가 데이터
            
        Returns:
            거래량 지표가 추가된 DataFrame
        """
        df = df.copy()
        
        # 거래량 이동평균
        df['Volume_MA_5'] = df['volume'].rolling(window=5).mean()
        df['Volume_MA_20'] = df['volume'].rolling(window=20).mean()
        
        # 거래량 비율
        df['Volume_Ratio'] = df['volume'] / df['Volume_MA_20']
        
        # OBV (On-Balance Volume)
        df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        return df
    
    @staticmethod
    def add_price_change(df: pd.DataFrame) -> pd.DataFrame:
        """
        가격 변화율 추가
        
        Args:
            df: 주가 데이터
            
        Returns:
            가격 변화율이 추가된 DataFrame
        """
        df = df.copy()
        
        # 수익률 (Return)
        df['Return'] = df['close'].pct_change()
        
        # 로그 수익률
        df['Log_Return'] = np.log(df['close'] / df['close'].shift(1))
        
        # N일 수익률
        for period in [5, 10, 20]:
            df[f'Return_{period}d'] = df['close'].pct_change(periods=period)
        
        # 고가-저가 비율
        df['HL_Ratio'] = (df['high'] - df['low']) / df['close']
        
        # 종가-시가 비율
        df['CO_Ratio'] = (df['close'] - df['open']) / df['open']
        
        return df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        모든 기술적 지표 추가
        
        Args:
            df: 주가 데이터
            
        Returns:
            모든 지표가 추가된 DataFrame
        """
        df = df.copy()
        
        print("  기술적 지표 계산 중...")
        
        # 이동평균
        df = TechnicalIndicators.add_moving_averages(df)
        df = TechnicalIndicators.add_exponential_moving_averages(df)
        
        # 모멘텀 지표
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_stochastic(df)
        
        # 변동성 지표
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_atr(df)
        
        # 거래량 지표
        df = TechnicalIndicators.add_volume_indicators(df)
        
        # 가격 변화율
        df = TechnicalIndicators.add_price_change(df)
        
        print(f"  총 {len(df.columns)}개 특성 생성 완료")
        
        return df


