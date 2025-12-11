"""
단일 종목용 강화학습 트레이딩 환경 (Gymnasium 스타일).

- 상태(state): 최근 N 스텝의 피처 + 현재 포지션/잔고 정보
- 행동(action): [-1, 1] 연속값 → 목표 포지션 비율 (−1=풀숏, 0=현금, 1=풀롱)
- 보상(reward): 한 스텝 수익률 (포지션 반영) - 단순 수수료/슬리피지 패널티
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
import pandas as pd


@dataclass
class TradingEnvConfig:
    csv_path: str
    window_size: int = 60
    initial_cash: float = 1_000_000.0
    max_position: float = 1.0  # 포트폴리오 비중 한도 (절대값)
    transaction_cost: float = 0.0005  # 0.05% 왕복 비용 근사
    reward_scale: float = 1.0  # 보상 스케일 (필요 시 조정)


class SingleStockTradingEnv(gym.Env):
    """
    단일 종목 트레이딩 환경.

    관측:
      - shape: (window_size, n_features + 2)
        - 기존 피처들
        - 추가: [현재 포지션 비율, 누적 수익률]
    행동:
      - Box(low=-1, high=1, shape=(1,))  → 목표 포지션 비율
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: TradingEnvConfig):
        super().__init__()
        self.config = config

        df = pd.read_csv(self.config.csv_path)
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df.sort_values("datetime", inplace=True)
            df.reset_index(drop=True, inplace=True)

        # 타겟/라벨 컬럼은 제외하고 순수 피처만 사용
        exclude_cols = ["datetime", "target", "stock_code", "stock_name"]
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        self.prices = df["close"].values if "close" in df.columns else df[self.feature_cols[0]].values
        self.features = df[self.feature_cols].values.astype(np.float32)

        assert len(self.features) > self.config.window_size + 1, "데이터 길이가 너무 짧습니다."

        self.n_steps = len(self.features)

        # Gym 공간 정의
        obs_dim = len(self.feature_cols) + 2  # 피처 + 포지션 + 누적수익률
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.config.window_size, obs_dim), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # 내부 상태
        self._current_step: int = 0
        self._position: float = 0.0  # 현재 포지션 비율 [-1,1]
        self._cash: float = self.config.initial_cash
        self._equity: float = self.config.initial_cash
        self._equity_start: float = self.config.initial_cash

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._current_step = self.config.window_size
        self._position = 0.0
        self._cash = self.config.initial_cash
        self._equity = self.config.initial_cash
        self._equity_start = self.config.initial_cash

        obs = self._get_observation()
        info = {"equity": self._equity, "position": self._position}
        return obs, info

    def step(self, action: np.ndarray):
        # 행동을 [-max_position, max_position]으로 클리핑
        target_pos = float(np.clip(action[0], -self.config.max_position, self.config.max_position))

        prev_price = float(self.prices[self._current_step - 1])
        curr_price = float(self.prices[self._current_step])

        # 포지션 조정에 따른 거래 비용 반영 (단순화)
        pos_change = abs(target_pos - self._position)
        trade_cost = self._equity * pos_change * self.config.transaction_cost

        # 포지션 업데이트
        self._position = target_pos

        # 가격 변화율 (0 가격 방지)
        if prev_price <= 0.0:
            price_return = 0.0
        else:
            price_return = (curr_price - prev_price) / prev_price

        prev_equity = self._equity
        pnl = prev_equity * self._position * price_return - trade_cost

        self._equity = max(prev_equity + pnl, 1e-6)  # 음수/0 방지
        step_return = (self._equity - prev_equity) / max(prev_equity, 1e-6)

        # 한 스텝 보상: 포트폴리오 수익률 변화
        reward = self.config.reward_scale * step_return

        # 다음 스텝으로 이동
        self._current_step += 1
        terminated = self._current_step >= self.n_steps - 1
        truncated = False

        obs = self._get_observation()
        info = {
            "equity": self._equity,
            "position": self._position,
            "price_return": price_return,
            "step_return": step_return,
        }

        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ #
    # 헬퍼
    # ------------------------------------------------------------------ #
    def _get_observation(self) -> np.ndarray:
        start = self._current_step - self.config.window_size
        end = self._current_step
        feats = self.features[start:end]  # (window, n_features)

        position_col = np.full((self.config.window_size, 1), self._position, dtype=np.float32)
        cum_return = (self._equity / self._equity_start) - 1.0
        equity_col = np.full((self.config.window_size, 1), cum_return, dtype=np.float32)

        obs = np.concatenate([feats, position_col, equity_col], axis=1)
        return obs.astype(np.float32)

    def render(self):
        print(f"step={self._current_step}, equity={self._equity:.2f}, pos={self._position:.2f}")


