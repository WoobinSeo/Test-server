"""
KIS(한국투자증권) 국내주식 주문/잔고 전용 브로커 모듈.

- .env 에 설정된 모의투자/실전 계좌로
  - 현금 매수/매도 주문
  - 잔고 조회
를 수행하는 래퍼입니다.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv


@dataclass
class KISConfig:
    """KIS 계좌/환경 설정 (.env 기반)."""

    app_key: str
    app_secret: str
    account_no: str
    account_code: str
    real_mode: bool = False

    tr_id_order_cash_buy: str = ""
    tr_id_order_cash_sell: str = ""
    tr_id_inquire_balance: str = ""

    @classmethod
    def from_env(cls) -> "KISConfig":
        load_dotenv()

        app_key = os.getenv("KIS_APP_KEY", "")
        app_secret = os.getenv("KIS_APP_SECRET", "")
        account_no = os.getenv("KIS_ACCOUNT_NO", "")
        account_code = os.getenv("KIS_ACCOUNT_CODE", "")
        real_mode = os.getenv("KIS_REAL_MODE", "False").lower() == "true"

        tr_buy = os.getenv("KIS_TR_ID_ORDER_CASH_BUY", "")
        tr_sell = os.getenv("KIS_TR_ID_ORDER_CASH_SELL", "")
        tr_bal = os.getenv("KIS_TR_ID_INQUIRE_BALANCE", "")

        if not app_key or not app_secret:
            raise ValueError("KIS_APP_KEY / KIS_APP_SECRET 이 .env 에 설정되어야 합니다.")
        if not account_no or not account_code:
            raise ValueError("KIS_ACCOUNT_NO / KIS_ACCOUNT_CODE 가 .env 에 설정되어야 합니다.")

        return cls(
            app_key=app_key,
            app_secret=app_secret,
            account_no=account_no,
            account_code=account_code,
            real_mode=real_mode,
            tr_id_order_cash_buy=tr_buy,
            tr_id_order_cash_sell=tr_sell,
            tr_id_inquire_balance=tr_bal,
        )


class KISBroker:
    """KIS 국내주식 주문/잔고 조회 래퍼."""

    def __init__(self, config: Optional[KISConfig] = None):
        self.config = config or KISConfig.from_env()
        self.base_url = (
            "https://openapi.koreainvestment.com:9443"
            if self.config.real_mode
            else "https://openapivts.koreainvestment.com:29443"
        )

        self._access_token: Optional[str] = None
        self._token_expired_at: Optional[datetime] = None

    # ------------------------------------------------------------------ #
    # 내부 유틸: 토큰 / 헤더
    # ------------------------------------------------------------------ #
    def _get_access_token(self) -> str:
        """접근 토큰 발급/캐시."""
        if self._access_token and self._token_expired_at:
            if datetime.now() < self._token_expired_at:
                return self._access_token

        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        data = {
            "grant_type": "client_credentials",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
        }

        resp = requests.post(url, headers=headers, data=json.dumps(data))
        if resp.status_code != 200:
            raise RuntimeError(f"KIS 토큰 발급 실패: {resp.status_code} {resp.text}")

        js = resp.json()
        access_token = js.get("access_token")
        if not access_token:
            raise RuntimeError(f"KIS 토큰 응답에 access_token 이 없습니다: {js}")

        self._access_token = access_token
        # 24시간 유효 → 23시간 후 만료로 취급
        self._token_expired_at = datetime.now() + timedelta(hours=23)
        return access_token

    def _headers(self, tr_id: str) -> Dict[str, str]:
        """KIS REST 호출용 공통 헤더."""
        return {
            "content-type": "application/json; charset=utf-8",
            "authorization": f"Bearer {self._get_access_token()}",
            "appkey": self.config.app_key,
            "appsecret": self.config.app_secret,
            "tr_id": tr_id,
        }

    # ------------------------------------------------------------------ #
    # 주문
    # ------------------------------------------------------------------ #
    def place_cash_order(
        self,
        side: str,
        stock_code: str,
        quantity: int,
        price: int = 0,
        ord_dvsn: str = "01",
        tr_id_override: Optional[str] = None,
        account_no_override: Optional[str] = None,
        account_code_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """현금 매수/매도 공통 함수."""
        if side not in {"BUY", "SELL"}:
            raise ValueError("side 는 'BUY' 또는 'SELL' 이어야 합니다.")
        if quantity <= 0:
            raise ValueError("quantity 는 1 이상이어야 합니다.")

        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"

        if tr_id_override:
            tr_id = tr_id_override
        else:
            tr_id = (
                self.config.tr_id_order_cash_buy
                if side == "BUY"
                else self.config.tr_id_order_cash_sell
            )

        if not tr_id:
            raise ValueError(
                "주문 TR ID 가 설정되지 않았습니다. "
                "KIS_TR_ID_ORDER_CASH_BUY / KIS_TR_ID_ORDER_CASH_SELL 환경변수를 확인하세요."
            )

        headers = self._headers(tr_id)

        body = {
            "CANO": account_no_override or self.config.account_no,
            "ACNT_PRDT_CD": account_code_override or self.config.account_code,
            "PDNO": stock_code,
            "ORD_DVSN": ord_dvsn,
            "ORD_QTY": str(quantity),
            "ORD_UNPR": str(price),
        }

        resp = requests.post(url, headers=headers, data=json.dumps(body))
        try:
            js = resp.json()
        except Exception:
            js = {"raw": resp.text}

        if resp.status_code != 200 or js.get("rt_cd") not in (None, "0"):
            raise RuntimeError(f"KIS 주문 실패: status={resp.status_code}, body={js}")

        return js

    def buy_market(
        self,
        stock_code: str,
        quantity: int,
        tr_id_override: Optional[str] = None,
        account_no_override: Optional[str] = None,
        account_code_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """시장가 매수 주문."""
        return self.place_cash_order(
            side="BUY",
            stock_code=stock_code,
            quantity=quantity,
            price=0,
            ord_dvsn="03",
            tr_id_override=tr_id_override,
            account_no_override=account_no_override,
            account_code_override=account_code_override,
        )

    def sell_market(
        self,
        stock_code: str,
        quantity: int,
        tr_id_override: Optional[str] = None,
        account_no_override: Optional[str] = None,
        account_code_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """시장가 매도 주문."""
        return self.place_cash_order(
            side="SELL",
            stock_code=stock_code,
            quantity=quantity,
            price=0,
            ord_dvsn="03",
            tr_id_override=tr_id_override,
            account_no_override=account_no_override,
            account_code_override=account_code_override,
        )

    # ------------------------------------------------------------------ #
    # 잔고 조회
    # ------------------------------------------------------------------ #
    def get_balance(
        self,
        tr_id_override: Optional[str] = None,
        account_no_override: Optional[str] = None,
        account_code_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        계좌 잔고/보유 주식 조회.

        KIS 문서의 샘플 파라미터를 기본값으로 사용한다.
        """
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"

        if tr_id_override:
            tr_id = tr_id_override
        else:
            tr_id = self.config.tr_id_inquire_balance

        if not tr_id:
            raise ValueError(
                "잔고 조회 TR ID 가 설정되지 않았습니다. "
                "KIS_TR_ID_INQUIRE_BALANCE 환경변수를 확인하세요."
            )

        headers = self._headers(tr_id)

        # KIS 예제 기준 기본 파라미터들
        params = {
            "CANO": account_no_override or self.config.account_no,       # 계좌번호 앞 8자리
            "ACNT_PRDT_CD": account_code_override or self.config.account_code,  # 상품코드 2자리
            "AFHR_FLPR_YN": "N",   # 시간외 단일가 여부
            "OFL_YN": "N",         # 오프라인 여부
            "INQR_DVSN": "01",     # 조회구분
            "UNPR_DVSN": "01",     # 단가구분
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "00",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": "",
        }

        resp = requests.get(url, headers=headers, params=params)
        try:
            js = resp.json()
        except Exception:
            js = {"raw": resp.text}

        if resp.status_code != 200 or js.get("rt_cd") not in (None, "0"):
            raise RuntimeError(f"KIS 잔고 조회 실패: status={resp.status_code}, body={js}")

        return js



