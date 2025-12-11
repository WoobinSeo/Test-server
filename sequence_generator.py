"""
시퀀스 데이터 생성 모듈
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
from pathlib import Path


class SequenceGenerator:
    """시계열 데이터를 LSTM 입력용 시퀀스로 변환"""
    
    def __init__(self, sequence_length: int = 60, prediction_horizon: int = 1):
        """
        Args:
            sequence_length: 입력 시퀀스 길이 (과거 몇 개의 데이터를 볼 것인가)
            prediction_horizon: 예측 기간 (몇 스텝 앞을 예측할 것인가)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
    
    def create_sequences(
        self,
        data: np.ndarray,
        target_col_idx: int = 3  # close price의 인덱스
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        시퀀스 데이터 생성
        
        Args:
            data: 입력 데이터 (samples, features)
            target_col_idx: 예측할 타겟 컬럼의 인덱스 (기본: close)
            
        Returns:
            X (입력 시퀀스), y (타겟)
        """
        X, y = [], []
        
        for i in range(len(data) - self.sequence_length - self.prediction_horizon + 1):
            # 입력: sequence_length 개의 과거 데이터
            X.append(data[i:(i + self.sequence_length)])
            
            # 타겟: prediction_horizon 스텝 후의 종가
            target_idx = i + self.sequence_length + self.prediction_horizon - 1
            y.append(data[target_idx, target_col_idx])
        
        return np.array(X), np.array(y)

    def create_sequences_with_labels(
        self,
        data: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
<<<<<<< Updated upstream
        분류용 시퀀스 데이터 생성

        Args:
            data: 입력 데이터 (samples, features)
            labels: 각 시점의 타겟 레이블 (samples,)

        Returns:
            X: (num_sequences, sequence_length, features)
            y: (num_sequences,)
        """
        X, y = [], []

        if len(data) != len(labels):
            raise ValueError("data와 labels의 길이가 다릅니다.")

=======
        분류용 시퀀스 데이터 생성.

        Args:
            data: (N, F) 형태의 특성 행렬
            labels: (N,) 형태의 레이블 벡터

        Returns:
            X: (M, sequence_length, F)
            y: (M,)  (각 시퀀스의 마지막 시점 레이블)
        """
        if len(data) != len(labels):
            raise ValueError("data와 labels의 길이가 다릅니다.")

        X, y = [], []
>>>>>>> Stashed changes
        max_start = len(data) - self.sequence_length + 1

        for i in range(max_start):
            X.append(data[i : i + self.sequence_length])
<<<<<<< Updated upstream
            # 시퀀스의 마지막 시점 레이블 사용
=======
>>>>>>> Stashed changes
            y.append(labels[i + self.sequence_length - 1])

        return np.array(X), np.array(y)
    
    def prepare_data_from_csv(
        self,
        filepath: str,
        feature_columns: List[str] = None,
        target_column: str = 'close'
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        CSV 파일에서 시퀀스 데이터 생성
        
        Args:
            filepath: CSV 파일 경로
            feature_columns: 사용할 특성 컬럼 (None이면 모두 사용)
            target_column: 타겟 컬럼명
            
        Returns:
            X, y, feature_names
        """
        df = pd.read_csv(filepath)
        
        # 메타데이터 컬럼 제외
        exclude_cols = ['datetime', 'stock_code', 'stock_name']
        
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col not in exclude_cols]
        
        # 데이터 추출
        data = df[feature_columns].values
        
        # 타겟 컬럼 인덱스 찾기
        target_col_idx = feature_columns.index(target_column)
        
        # 시퀀스 생성
        X, y = self.create_sequences(data, target_col_idx)
        
        return X, y, feature_columns
    
    def prepare_datasets(
        self,
        train_file: str,
        val_file: str,
        test_file: str,
        feature_columns: List[str] = None,
        target_column: str = 'close'
    ) -> Tuple[Tuple[np.ndarray, np.ndarray], 
               Tuple[np.ndarray, np.ndarray], 
               Tuple[np.ndarray, np.ndarray],
               List[str]]:
        """
        학습/검증/테스트 데이터셋 준비
        
        Returns:
            (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names
        """
        print(f"\n시퀀스 데이터 생성 중...")
        print(f"  시퀀스 길이: {self.sequence_length}")
        print(f"  예측 기간: {self.prediction_horizon} 스텝 앞")
        
        # 학습 데이터
        X_train, y_train, feature_names = self.prepare_data_from_csv(
            train_file, feature_columns, target_column
        )
        print(f"\n  학습 데이터: {X_train.shape}")
        
        # 검증 데이터
        X_val, y_val, _ = self.prepare_data_from_csv(
            val_file, feature_columns, target_column
        )
        print(f"  검증 데이터: {X_val.shape}")
        
        # 테스트 데이터
        X_test, y_test, _ = self.prepare_data_from_csv(
            test_file, feature_columns, target_column
        )
        print(f"  테스트 데이터: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names







