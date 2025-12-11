"""
단순 트레이딩 백엔드 API (FastAPI).

- KIS 모의/실계좌에 시장가 주문을 넣는 엔드포인트
- 계좌 잔고/보유 종목 조회 엔드포인트

강화학습(SAC) 모델은 별도 프로세스에서 신호를 계산하고,
이 API에 주문 요청을 보내는 구조를 가정한다.
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

import bcrypt
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from jose import jwt
from pydantic import BaseModel, Field

from kis_broker import KISBroker, KISConfig
from database import (
    DatabaseManager,
    TradeOrder,
    AccountSnapshot,
    RiskSetting,
    AutoTradeRun,
    StockPrice,
    StockPriceProcessed,
    User,
)


app = FastAPI(title="StuckAI Trading API", version="0.1.0")

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-me")

# React 프론트엔드(예: Vite dev 서버) 연동을 위한 CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 싱글톤 인스턴스 (토큰/DB 재사용)
_db_manager: Optional[DatabaseManager] = None
_broker: Optional[KISBroker] = None


def get_db() -> DatabaseManager:
    """전역 DB 매니저 (연결 및 테이블 생성 포함)."""
    global _db_manager
    if _db_manager is None:
        mgr = DatabaseManager()
        mgr.connect()
        mgr.create_tables()
        _db_manager = mgr
    return _db_manager


def get_broker() -> KISBroker:
    """
    KIS 브로커 인스턴스.
    - .env 의 KIS_* 설정을 사용한다.
    - 한 프로세스당 한 번만 생성하여 토큰을 재사용한다.
    """
    global _broker
    if _broker is None:
        cfg = KISConfig.from_env()
        _broker = KISBroker(cfg)
    return _broker


class MarketOrderRequest(BaseModel):
    stock_code: str = Field(..., description="6자리 종목코드 (예: '005930')")
    quantity: int = Field(..., gt=0, description="주문 수량 (양수)")
    side: str = Field(..., description="'BUY' 또는 'SELL'")


class BalanceResponse(BaseModel):
    raw: dict


# ---------------------------------------------------------------------------
# 리스크 한도 설정 (기본값: .env + DB risk_settings)
# ---------------------------------------------------------------------------

DEFAULT_MAX_POSITION_SHARES = int(os.getenv("RISK_MAX_POSITION_SHARES", "10"))
DEFAULT_MAX_WEIGHT_PCT = float(os.getenv("RISK_MAX_WEIGHT_PCT", "0.5"))  # 0~1
DEFAULT_MAX_DAILY_BUY_AMOUNT = float(os.getenv("RISK_MAX_DAILY_BUY_AMOUNT", "0"))  # 0이면 비활성


def check_risk_limit(broker: KISBroker, stock_code: str, side: str, quantity: int):
    """
    간단한 리스크 체크:
      - 매수:
          * 보유수량 + 주문수량 <= MAX_POSITION_SHARES
          * 매수 후 해당 종목 평가금액 비중이 총자산의 50%를 넘지 않도록 제한
      - 매도:
          * 보유수량/매도가능수량 이상으로 팔 수 없음
    """
    bal = broker.get_balance()
    raw = bal if isinstance(bal, dict) else {}
    holdings = raw.get("output1") or []
    summary_list = raw.get("output2") or []
    summary = summary_list[0] if summary_list else {}

    # DB에서 리스크 설정 조회 (종목별 우선, 없으면 "ALL", 다시 없으면 기본값)
    db = get_db()
    session = db.get_session()
    try:
        setting = (
            session.query(RiskSetting)
            .filter(RiskSetting.active.is_(True))
            .filter(RiskSetting.stock_code.in_([stock_code, "ALL"]))
            .order_by(RiskSetting.stock_code.desc())
            .first()
        )
    finally:
        session.close()

    max_shares = DEFAULT_MAX_POSITION_SHARES
    max_weight_pct = DEFAULT_MAX_WEIGHT_PCT
    max_daily_buy_amount = DEFAULT_MAX_DAILY_BUY_AMOUNT
    if setting:
        if setting.max_position_shares is not None:
            max_shares = setting.max_position_shares
        if setting.max_weight_pct is not None:
            max_weight_pct = setting.max_weight_pct
        if setting.max_daily_buy_amount is not None:
            max_daily_buy_amount = setting.max_daily_buy_amount

    current_qty = 0.0
    sellable = 0.0
    current_eval = 0.0
    current_price = None

    total_eval = 0.0
    for h in holdings:
        try:
            ev = float(h.get("evlu_amt") or 0)
        except (TypeError, ValueError):
            ev = 0.0
        total_eval += ev

        if h.get("pdno") == stock_code:
            try:
                current_qty = float(h.get("hldg_qty") or 0)
            except (TypeError, ValueError):
                current_qty = 0.0
            try:
                sellable = float(h.get("ord_psbl_qty") or 0)
            except (TypeError, ValueError):
                sellable = current_qty
            current_eval = ev
            try:
                current_price = float(h.get("prpr") or 0)
            except (TypeError, ValueError):
                current_price = None

    cash_raw = summary.get("dnca_tot_amt") or summary.get("nass_amt") or 0
    try:
        cash = float(cash_raw)
    except (TypeError, ValueError):
        cash = 0.0

    total_value = total_eval + cash

    if side == "BUY":
        # 일일 최대 매수 금액 한도 (옵션)
        if max_daily_buy_amount and max_daily_buy_amount > 0:
            db = get_db()
            session2 = db.get_session()
            try:
                today = datetime.utcnow().date()
                start = datetime.combine(today, datetime.min.time())
                # 오늘 체결된 BUY 주문 금액 합산
                orders = (
                    session2.query(TradeOrder)
                    .filter(
                        TradeOrder.created_at >= start,
                        TradeOrder.side == "BUY",
                        TradeOrder.status == "OK",
                    )
                    .all()
                )
                spent = 0.0
                for o in orders:
                    try:
                        spent += float(o.order_amount or 0)
                    except (TypeError, ValueError):
                        continue

                est_price = current_price or 0.0
                est_amount = est_price * quantity if est_price > 0 else 0.0

                if est_amount > 0 and spent + est_amount > max_daily_buy_amount + 1e-6:
                    remain = max(0.0, max_daily_buy_amount - spent)
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"리스크 한도 초과: 오늘 남은 매수 가능 금액 {int(remain):,}원을 초과합니다. "
                            f"(설정 한도 {int(max_daily_buy_amount):,}원)"
                        ),
                    )
            finally:
                session2.close()

        # 수량 한도
        if current_qty + quantity > max_shares:
            raise HTTPException(
                status_code=400,
                detail=f"리스크 한도 초과: {stock_code} 최대 보유 수량은 {max_shares}주 입니다. (현재 {int(current_qty)}주 보유)",
            )

        # 비중 한도 (50%)
        if total_value > 0:
            # 현재가 확보: 없으면 평균단가로 근사, 그래도 없으면 비중 체크 스킵
            if (current_price is None or current_price <= 0) and current_qty > 0:
                current_price = current_eval / max(current_qty, 1.0)

            if current_price and current_price > 0:
                projected_stock_value = current_eval + current_price * quantity
                projected_total_value = total_value - current_eval + projected_stock_value
                if projected_total_value > 0:
                    weight = projected_stock_value / projected_total_value
                if weight > max_weight_pct + 1e-9:
                        raise HTTPException(
                            status_code=400,
                        detail=(
                            f"리스크 한도 초과: {stock_code} 매수 시 예상 비중이 {weight*100:.1f}%로, "
                            f"종목별 최대 비중 {max_weight_pct*100:.0f}%를 초과합니다."
                        ),
                        )
    else:  # SELL
        if quantity > sellable:
            raise HTTPException(
                status_code=400,
                detail=f"리스크 한도 초과: 보유/매도가능 수량({int(sellable)}주) 이상은 매도할 수 없습니다.",
            )


class PerformanceSnapshot(BaseModel):
    timestamp: datetime
    total_value: float
    cash: float
    total_buy_amount: float
    total_eval_amount: float
    total_pnl: float


class PerformanceSummary(BaseModel):
    start_value: float
    end_value: float
    total_return_pct: float
    max_drawdown_pct: float
    pnl_sum: float


class PerformanceResponse(BaseModel):
    summary: PerformanceSummary
    snapshots: List[PerformanceSnapshot]


class OrderHistoryItem(BaseModel):
    created_at: datetime
    stock_code: str
    stock_name: Optional[str]
    side: str
    quantity: int
    order_price: Optional[float]
    order_amount: Optional[float]
    status: str


class RiskSettingIn(BaseModel):
    max_position_shares: Optional[int] = None
    max_weight_pct: Optional[float] = None
    max_daily_buy_amount: Optional[float] = None
    active: Optional[bool] = True


class RiskSettingOut(BaseModel):
    stock_code: str
    max_position_shares: Optional[int]
    max_weight_pct: Optional[float]
    max_daily_buy_amount: Optional[float]
    active: bool
    created_at: datetime
    updated_at: Optional[datetime]


class AutoTradeRunResult(BaseModel):
    returncode: int
    stdout: str
    stderr: str


class AutoTradeRunItem(BaseModel):
    id: int
    created_at: datetime
    returncode: int


class SignupRequest(BaseModel):
    username: str
    password: str
    name: str


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    token: str
    name: str


@app.get("/signup-page", response_class=HTMLResponse)
def signup_page():
    """
    단독 회원가입 페이지.
    - 백엔드 /signup API 를 호출한다.
    """
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <title>StuckAI 회원가입</title>
  <style>
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #1f2937, #020617);
      color: #e5e7eb;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    header {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      padding: 10px 18px;
      box-sizing: border-box;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 14px;
      background: rgba(15,23,42,0.95);
      border-bottom: 1px solid #1f2937;
      backdrop-filter: blur(12px);
      z-index: 10;
    }
    .nav-right {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }
    .nav-btn {
      background: #020617;
      color: #e5e7eb;
      border-radius: 999px;
      border: 1px solid #374151;
      padding: 4px 9px;
      font-size: 12px;
      cursor: pointer;
    }
    .container {
      width: 100%;
      max-width: 420px;
      padding: 72px 24px 24px;
      box-sizing: border-box;
    }
    .card {
      background: rgba(15,23,42,0.96);
      border-radius: 18px;
      border: 1px solid rgba(55,65,81,0.9);
      box-shadow: 0 24px 80px rgba(15,23,42,0.95);
      padding: 22px 24px 20px;
      backdrop-filter: blur(18px);
    }
    .title {
      font-size: 20px;
      font-weight: 600;
      margin-bottom: 4px;
    }
    .subtitle {
      font-size: 13px;
      color: #9ca3af;
      margin-bottom: 16px;
    }
    label {
      font-size: 12px;
      color: #9ca3af;
      display: block;
      margin-bottom: 3px;
    }
    input {
      width: 100%;
      box-sizing: border-box;
      background: #020617;
      border-radius: 10px;
      border: 1px solid #374151;
      padding: 7px 9px;
      color: #e5e7eb;
      font-size: 13px;
      outline: none;
      transition: border-color 0.15s, box-shadow 0.15s, background 0.15s;
    }
    input:focus {
      border-color: #60a5fa;
      box-shadow: 0 0 0 1px rgba(37,99,235,0.7);
      background: #020617;
    }
    button {
      border: none;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      color: #020617;
      background: linear-gradient(to right, #4ade80, #22c55e);
      box-shadow: 0 12px 25px rgba(34,197,94,0.45);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      margin-top: 8px;
      width: 100%;
    }
    button:disabled {
      opacity: 0.7;
      cursor: default;
      box-shadow: none;
    }
    .status {
      min-height: 18px;
      font-size: 12px;
      margin-top: 6px;
    }
    .status.ok {
      color: #4ade80;
    }
    .status.err {
      color: #f97373;
    }
    .footer {
      margin-top: 14px;
      font-size: 12px;
      color: #9ca3af;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .link {
      color: #60a5fa;
      cursor: pointer;
      text-decoration: underline;
      text-underline-offset: 2px;
    }
  </style>
</head>
<body>
  <header>
    <div style="font-weight:600;">stuckAI</div>
    <div class="nav-right" id="nav-loggedin-signup" style="display:none;">
      <button class="nav-btn" onclick="window.location.href='/'">홈</button>
      <button class="nav-btn" onclick="window.location.href='/dashboard'">마이페이지</button>
      <button class="nav-btn" onclick="logoutAndGoLogin()">로그아웃</button>
    </div>
  </header>
  <div class="container">
    <div class="card">
      <div class="title">회원가입</div>
      <div class="subtitle">StuckAI 트레이딩 대시보드 이용을 위한 계정을 생성합니다.</div>
      <form id="signup-form">
        <div style="margin-bottom:10px;">
          <label for="signup-username">아이디 (username)</label>
          <input id="signup-username" autocomplete="off" required />
        </div>
        <div style="margin-bottom:10px;">
          <label for="signup-name">이름</label>
          <input id="signup-name" autocomplete="off" required />
        </div>
        <div>
          <label for="signup-password">비밀번호</label>
          <input id="signup-password" type="password" required />
        </div>
        <button type="submit" id="signup-btn">회원가입 완료</button>
        <div class="status" id="signup-status"></div>
      </form>
      <div class="footer">
        <span>이미 계정이 있으신가요?</span>
        <span class="link" onclick="window.location.href='/login-page'">로그인 페이지로 이동</span>
      </div>
    </div>
  </div>

  <script>
    function logoutAndGoLogin() {
      try {
        window.localStorage.removeItem("stuckai_token");
        window.localStorage.removeItem("stuckai_name");
      } catch (e) {}
      window.location.href = "/login-page";
    }

    (function initNav() {
      try {
        const token = window.localStorage.getItem("stuckai_token");
        const nav = document.getElementById("nav-loggedin-signup");
        if (!nav) return;
        if (token) {
          nav.style.display = "flex";
        } else {
          nav.style.display = "none";
        }
      } catch (e) {}
    })();

    const signupForm = document.getElementById("signup-form");
    const signupBtn = document.getElementById("signup-btn");
    const signupStatus = document.getElementById("signup-status");

    signupForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const username = document.getElementById("signup-username").value.trim();
      const name = document.getElementById("signup-name").value.trim();
      const password = document.getElementById("signup-password").value;

      if (!username || !name || !password) {
        signupStatus.textContent = "모든 필드를 입력하세요.";
        signupStatus.className = "status err";
        return;
      }

      signupBtn.disabled = true;
      signupStatus.textContent = "회원가입 요청 중...";
      signupStatus.className = "status";

      try {
        const res = await fetch("/signup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, name, password }),
        });
        const data = await res.json().catch(() => ({}));
        if (res.ok) {
          signupStatus.textContent = "회원가입 성공! 로그인 페이지로 이동합니다.";
          signupStatus.className = "status ok";
          setTimeout(() => {
            window.location.href = "/login-page";
          }, 800);
        } else {
          const msg = data && data.detail ? data.detail : "알 수 없는 오류";
          signupStatus.textContent = "회원가입 실패: " + msg;
          signupStatus.className = "status err";
        }
      } catch (e2) {
        signupStatus.textContent = "요청 에러: " + e2;
        signupStatus.className = "status err";
      } finally {
        signupBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
    """


@app.get("/login-page", response_class=HTMLResponse)
def login_page():
    """
    단독 로그인 페이지.
    - 백엔드 /login, /me API 를 사용.
    """
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <title>StuckAI 로그인</title>
  <style>
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #1f2937, #020617);
      color: #e5e7eb;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    header {
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      padding: 10px 18px;
      box-sizing: border-box;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 14px;
      background: rgba(15,23,42,0.95);
      border-bottom: 1px solid #1f2937;
      backdrop-filter: blur(12px);
      z-index: 10;
    }
    .nav-right {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 12px;
    }
    .nav-btn {
      background: #020617;
      color: #e5e7eb;
      border-radius: 999px;
      border: 1px solid #374151;
      padding: 4px 9px;
      font-size: 12px;
      cursor: pointer;
    }
    .container {
      width: 100%;
      max-width: 420px;
      padding: 72px 24px 24px;
      box-sizing: border-box;
    }
    .card {
      background: rgba(15,23,42,0.96);
      border-radius: 18px;
      border: 1px solid rgba(55,65,81,0.9);
      box-shadow: 0 24px 80px rgba(15,23,42,0.95);
      padding: 22px 24px 18px;
      backdrop-filter: blur(18px);
    }
    .title {
      font-size: 20px;
      font-weight: 600;
      margin-bottom: 4px;
    }
    .subtitle {
      font-size: 13px;
      color: #9ca3af;
      margin-bottom: 16px;
    }
    label {
      font-size: 12px;
      color: #9ca3af;
      display: block;
      margin-bottom: 3px;
    }
    input {
      width: 100%;
      box-sizing: border-box;
      background: #020617;
      border-radius: 10px;
      border: 1px solid #374151;
      padding: 7px 9px;
      color: #e5e7eb;
      font-size: 13px;
      outline: none;
      transition: border-color 0.15s, box-shadow 0.15s, background 0.15s;
    }
    input:focus {
      border-color: #60a5fa;
      box-shadow: 0 0 0 1px rgba(37,99,235,0.7);
      background: #020617;
    }
    button {
      border: none;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      color: #020617;
      background: linear-gradient(to right, #4ade80, #22c55e);
      box-shadow: 0 12px 25px rgba(34,197,94,0.45);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      margin-top: 8px;
      width: 100%;
    }
    button.secondary {
      background: #111827;
      color: #e5e7eb;
      box-shadow: none;
      border-radius: 10px;
      padding: 6px 10px;
      font-size: 12px;
      border: 1px solid #374151;
      width: auto;
    }
    button:disabled {
      opacity: 0.7;
      cursor: default;
      box-shadow: none;
    }
    .status {
      min-height: 18px;
      font-size: 12px;
      margin-top: 6px;
    }
    .status.ok {
      color: #4ade80;
    }
    .status.err {
      color: #f97373;
    }
    .footer {
      margin-top: 14px;
      font-size: 12px;
      color: #9ca3af;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .link {
      color: #60a5fa;
      cursor: pointer;
      text-decoration: underline;
      text-underline-offset: 2px;
    }
    .small {
      font-size: 11px;
      color: #6b7280;
      margin-top: 4px;
    }
  </style>
</head>
<body>
  <header>
    <div style="font-weight:600;">stuckAI</div>
    <div class="nav-right" id="nav-loggedin-login" style="display:none;">
      <button class="nav-btn" onclick="window.location.href='/'">홈</button>
      <button class="nav-btn" onclick="window.location.href='/dashboard'">마이페이지</button>
      <button class="nav-btn" onclick="logoutAndGoLogin()">로그아웃</button>
    </div>
  </header>
  <div class="container">
    <div class="card">
      <div class="title">로그인</div>
      <div class="subtitle">가입한 계정으로 로그인하여 트레이딩 대시보드에 접근합니다.</div>
      <form id="login-form">
        <div style="margin-bottom:10px;">
          <label for="login-username">아이디 (username)</label>
          <input id="login-username" autocomplete="username" required />
        </div>
        <div>
          <label for="login-password">비밀번호</label>
          <input id="login-password" type="password" autocomplete="current-password" required />
        </div>
        <button type="submit" id="login-btn">로그인</button>
        <div class="status" id="login-status"></div>
      </form>
      <div class="small" id="login-user-info">현재 로그인: 없음</div>
      <div class="footer">
        <span class="link" onclick="window.location.href='/signup-page'">아직 계정이 없으신가요? 회원가입</span>
        <button class="secondary" id="btn-go-home">홈페이지</button>
      </div>
    </div>
  </div>

  <script>
    function logoutAndGoLogin() {
      try {
        window.localStorage.removeItem("stuckai_token");
        window.localStorage.removeItem("stuckai_name");
      } catch (e) {}
      window.location.href = "/login-page";
    }

    function getToken() {
      try {
        return window.localStorage.getItem("stuckai_token") || "";
      } catch (e) {
        return "";
      }
    }

    function setToken(token, name) {
      try {
        window.localStorage.setItem("stuckai_token", token);
        window.localStorage.setItem("stuckai_name", name || "");
      } catch (e) {
        console.warn("토큰 저장 실패:", e);
      }
    }

    function updateUserInfoUI() {
      const name = window.localStorage.getItem("stuckai_name");
      const info = document.getElementById("login-user-info");
       const nav = document.getElementById("nav-loggedin-login");
      if (name) {
        info.textContent = "현재 로그인: " + name;
        if (nav) nav.style.display = "flex";
      } else {
        info.textContent = "현재 로그인: 없음";
        if (nav) nav.style.display = "none";
      }
    }

    const loginForm = document.getElementById("login-form");
    const loginBtn = document.getElementById("login-btn");
    const loginStatus = document.getElementById("login-status");
    const btnGoHome = document.getElementById("btn-go-home");

    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const username = document.getElementById("login-username").value.trim();
      const password = document.getElementById("login-password").value;

      if (!username || !password) {
        loginStatus.textContent = "아이디와 비밀번호를 입력하세요.";
        loginStatus.className = "status err";
        return;
      }

      loginBtn.disabled = true;
      loginStatus.textContent = "로그인 요청 중...";
      loginStatus.className = "status";

      try {
        const res = await fetch("/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        const data = await res.json().catch(() => ({}));
        if (res.ok) {
          const token = data.token;
          const name = data.name;
          setToken(token, name);
          updateUserInfoUI();
          loginStatus.textContent = "로그인 성공! 홈페이지로 이동합니다.";
          loginStatus.className = "status ok";
          setTimeout(() => {
            window.location.href = "/";
          }, 800);
        } else {
          const msg = data && data.detail ? data.detail : "알 수 없는 오류";
          loginStatus.textContent = "로그인 실패: " + msg;
          loginStatus.className = "status err";
        }
      } catch (e2) {
        loginStatus.textContent = "요청 에러: " + e2;
        loginStatus.className = "status err";
      } finally {
        loginBtn.disabled = false;
      }
    });

    btnGoHome.addEventListener("click", () => {
      window.location.href = "/";
    });

    updateUserInfoUI();
    if (getToken()) {
      loginStatus.textContent = "저장된 토큰이 있습니다. 바로 로그인 확인이 가능합니다.";
      loginStatus.className = "status ok";
    }
  </script>
</body>
</html>
    """


@app.get("/auth", response_class=HTMLResponse)
def auth_page():
    """
    아주 간단한 회원가입/로그인 프론트엔드 페이지.
    - 브라우저에서 http://localhost:8000/auth 접속
    - /signup, /login, /me 엔드포인트를 사용
    """
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <title>StuckAI Auth</title>
  <style>
    body {
      margin: 0;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #1f2937, #020617);
      color: #e5e7eb;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .container {
      width: 100%;
      max-width: 900px;
      padding: 24px;
      box-sizing: border-box;
    }
    .card {
      background: rgba(15,23,42,0.95);
      border-radius: 18px;
      border: 1px solid rgba(55,65,81,0.9);
      box-shadow: 0 24px 80px rgba(15,23,42,0.9);
      padding: 24px 28px;
      backdrop-filter: blur(18px);
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
    }
    .title {
      font-size: 22px;
      font-weight: 600;
    }
    .chip {
      font-size: 11px;
      padding: 2px 10px;
      border-radius: 999px;
      border: 1px solid rgba(59,130,246,0.6);
      background: rgba(37,99,235,0.15);
      color: #60a5fa;
    }
    .subtitle {
      font-size: 13px;
      color: #9ca3af;
      margin-bottom: 20px;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 20px;
    }
    @media (max-width: 768px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
    .panel-title {
      font-size: 16px;
      font-weight: 500;
      margin-bottom: 10px;
    }
    form {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    label {
      font-size: 12px;
      color: #9ca3af;
      display: block;
      margin-bottom: 3px;
    }
    input {
      width: 100%;
      box-sizing: border-box;
      background: #020617;
      border-radius: 10px;
      border: 1px solid #374151;
      padding: 7px 9px;
      color: #e5e7eb;
      font-size: 13px;
      outline: none;
      transition: border-color 0.15s, box-shadow 0.15s, background 0.15s;
    }
    input:focus {
      border-color: #60a5fa;
      box-shadow: 0 0 0 1px rgba(37,99,235,0.7);
      background: #020617;
    }
    button {
      border: none;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      color: #020617;
      background: linear-gradient(to right, #4ade80, #22c55e);
      box-shadow: 0 12px 25px rgba(34,197,94,0.45);
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      margin-top: 4px;
    }
    button.secondary {
      background: #111827;
      color: #e5e7eb;
      box-shadow: none;
      border-radius: 10px;
      padding: 6px 10px;
      font-size: 12px;
      border: 1px solid #374151;
    }
    button:disabled {
      opacity: 0.7;
      cursor: default;
      box-shadow: none;
    }
    .status {
      min-height: 18px;
      font-size: 12px;
      margin-top: 4px;
    }
    .status.ok {
      color: #4ade80;
    }
    .status.err {
      color: #f97373;
    }
    .token-box {
      margin-top: 8px;
      font-size: 11px;
      color: #9ca3af;
      background: #020617;
      border-radius: 10px;
      border: 1px solid #111827;
      padding: 8px 10px;
      max-height: 80px;
      overflow: auto;
    }
    .hint {
      font-size: 11px;
      color: #6b7280;
      margin-top: 4px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="card">
      <div class="header">
        <div>
          <div class="title">StuckAI 회원 시스템</div>
          <div class="subtitle">백엔드 FastAPI의 <code>/signup</code>, <code>/login</code>, <code>/me</code> 를 직접 호출하는 간단한 로그인/회원가입 화면입니다.</div>
        </div>
        <span class="chip">로컬 전용 · 데모</span>
      </div>

      <div class="grid">
        <section>
          <div class="panel-title">회원가입</div>
          <form id="signup-form">
            <div>
              <label for="signup-username">아이디 (username)</label>
              <input id="signup-username" autocomplete="off" required />
            </div>
            <div>
              <label for="signup-name">이름</label>
              <input id="signup-name" autocomplete="off" required />
            </div>
            <div>
              <label for="signup-password">비밀번호</label>
              <input id="signup-password" type="password" required />
            </div>
            <button type="submit" id="signup-btn">회원가입</button>
            <div class="status" id="signup-status"></div>
          </form>
        </section>

        <section>
          <div class="panel-title">로그인</div>
          <form id="login-form">
            <div>
              <label for="login-username">아이디 (username)</label>
              <input id="login-username" autocomplete="username" required />
            </div>
            <div>
              <label for="login-password">비밀번호</label>
              <input id="login-password" type="password" autocomplete="current-password" required />
            </div>
            <button type="submit" id="login-btn">로그인</button>
            <div class="status" id="login-status"></div>
          </form>

          <div style="margin-top: 14px; display:flex; align-items:center; justify-content:space-between; gap:8px;">
            <div style="font-size: 12px;">
              <div id="login-user-info">현재 로그인: 없음</div>
              <div class="hint">로그인에 성공하면 JWT 토큰이 브라우저 localStorage 에 저장됩니다.</div>
            </div>
            <div style="display:flex; flex-direction:column; gap:6px; align-items:flex-end;">
              <button class="secondary" id="btn-check-me">/me 로 로그인 확인</button>
              <button class="secondary" id="btn-logout">로그아웃</button>
            </div>
          </div>
          <div class="token-box" id="token-box">토큰 정보가 여기에 표시됩니다.</div>
        </section>
      </div>
    </div>
  </div>

  <script>
    function getToken() {
      try {
        return window.localStorage.getItem("stuckai_token") || "";
      } catch (e) {
        return "";
      }
    }

    function setToken(token, name) {
      try {
        window.localStorage.setItem("stuckai_token", token);
        window.localStorage.setItem("stuckai_name", name || "");
      } catch (e) {
        console.warn("토큰 저장 실패:", e);
      }
    }

    function clearToken() {
      try {
        window.localStorage.removeItem("stuckai_token");
        window.localStorage.removeItem("stuckai_name");
      } catch (e) {
        console.warn("토큰 삭제 실패:", e);
      }
    }

    function updateUserInfoUI() {
      const name = window.localStorage.getItem("stuckai_name");
      const info = document.getElementById("login-user-info");
      if (name) {
        info.textContent = "현재 로그인: " + name;
      } else {
        info.textContent = "현재 로그인: 없음";
      }
    }

    const signupForm = document.getElementById("signup-form");
    const signupBtn = document.getElementById("signup-btn");
    const signupStatus = document.getElementById("signup-status");

    const loginForm = document.getElementById("login-form");
    const loginBtn = document.getElementById("login-btn");
    const loginStatus = document.getElementById("login-status");
    const tokenBox = document.getElementById("token-box");
    const btnCheckMe = document.getElementById("btn-check-me");
    const btnLogout = document.getElementById("btn-logout");

    signupForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const username = document.getElementById("signup-username").value.trim();
      const name = document.getElementById("signup-name").value.trim();
      const password = document.getElementById("signup-password").value;

      if (!username || !name || !password) {
        signupStatus.textContent = "모든 필드를 입력하세요.";
        signupStatus.className = "status err";
        return;
      }

      signupBtn.disabled = true;
      signupStatus.textContent = "회원가입 요청 중...";
      signupStatus.className = "status";

      try {
        const res = await fetch("/signup", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, name, password }),
        });
        const data = await res.json().catch(() => ({}));
        if (res.ok) {
          signupStatus.textContent = "회원가입 성공! 이제 로그인 해보세요.";
          signupStatus.className = "status ok";
        } else {
          const msg = data && data.detail ? data.detail : "알 수 없는 오류";
          signupStatus.textContent = "회원가입 실패: " + msg;
          signupStatus.className = "status err";
        }
      } catch (e2) {
        signupStatus.textContent = "요청 에러: " + e2;
        signupStatus.className = "status err";
      } finally {
        signupBtn.disabled = false;
      }
    });

    loginForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const username = document.getElementById("login-username").value.trim();
      const password = document.getElementById("login-password").value;

      if (!username || !password) {
        loginStatus.textContent = "아이디와 비밀번호를 입력하세요.";
        loginStatus.className = "status err";
        return;
      }

      loginBtn.disabled = true;
      loginStatus.textContent = "로그인 요청 중...";
      loginStatus.className = "status";

      try {
        const res = await fetch("/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        const data = await res.json().catch(() => ({}));
        if (res.ok) {
          const token = data.token;
          const name = data.name;
          setToken(token, name);
          updateUserInfoUI();
          loginStatus.textContent = "로그인 성공!";
          loginStatus.className = "status ok";
          tokenBox.textContent = token ? token : "토큰이 없습니다.";
        } else {
          const msg = data && data.detail ? data.detail : "알 수 없는 오류";
          loginStatus.textContent = "로그인 실패: " + msg;
          loginStatus.className = "status err";
        }
      } catch (e2) {
        loginStatus.textContent = "요청 에러: " + e2;
        loginStatus.className = "status err";
      } finally {
        loginBtn.disabled = false;
      }
    });

    btnCheckMe.addEventListener("click", async () => {
      const token = getToken();
      if (!token) {
        tokenBox.textContent = "저장된 토큰이 없습니다. 먼저 로그인하세요.";
        return;
      }
      btnCheckMe.disabled = true;
      tokenBox.textContent = "/me 요청 중...";
      try {
        const res = await fetch("/me?token=" + encodeURIComponent(token));
        const data = await res.json().catch(() => ({}));
        if (res.ok) {
          tokenBox.textContent = "토큰 유효 ✅\\n" + JSON.stringify(data, null, 2);
        } else {
          const msg = data && data.detail ? data.detail : "알 수 없는 오류";
          tokenBox.textContent = "토큰 오류 ❌: " + msg;
        }
      } catch (e2) {
        tokenBox.textContent = "요청 에러: " + e2;
      } finally {
        btnCheckMe.disabled = false;
      }
    });

    btnLogout.addEventListener("click", () => {
      clearToken();
      updateUserInfoUI();
      tokenBox.textContent = "로그아웃 완료. 토큰이 삭제되었습니다.";
      loginStatus.textContent = "";
    });

    // 초기 UI 상태
    updateUserInfoUI();
    const saved = getToken();
    if (saved) {
      tokenBox.textContent = saved;
    }
  </script>
</body>
</html>
    """


@app.post("/signup")
def signup(req: SignupRequest):
    """
    React `Signup` 페이지용 회원가입.
    - username 중복 시 400 에러.
    """
    db = get_db()
    session = db.get_session()
    try:
        # 중복 체크
        exists = session.query(User).filter(User.username == req.username).first()
        if exists:
            raise HTTPException(status_code=400, detail="이미 존재하는 사용자")

        hashed = bcrypt.hashpw(req.password.encode("utf-8"), bcrypt.gensalt())
        user = User(username=req.username, password_hash=hashed, name=req.name)
        session.add(user)
        session.commit()
    except HTTPException:
        session.rollback()
        raise
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"회원가입 실패: {e}")
    finally:
        session.close()

    return {"message": "회원가입 성공"}


@app.post("/login", response_model=TokenResponse)
def login(req: LoginRequest):
    """
    React `Login` 페이지용 로그인.
    - 성공 시 JWT 토큰과 이름 반환.
    """
    db = get_db()
    session = db.get_session()
    try:
        user = session.query(User).filter(User.username == req.username).first()
    finally:
        session.close()

    if not user:
        raise HTTPException(status_code=400, detail="아이디 또는 비밀번호 오류")

    stored_hash = user.password_hash
    if isinstance(stored_hash, memoryview):
        stored_hash = stored_hash.tobytes()

    if not bcrypt.checkpw(req.password.encode("utf-8"), stored_hash):
        raise HTTPException(status_code=400, detail="아이디 또는 비밀번호 오류")

    payload = {
        "username": user.username,
        "name": user.name,
        "exp": datetime.utcnow() + timedelta(hours=3),
    }
    token = jwt.encode(payload, SECRET_KEY)

    return TokenResponse(token=token, name=user.name)


@app.get("/me")
def me(token: str):
    """
    로그인 유지 확인용 간단 엔드포인트.
    - 쿼리 파라미터 ?token=... 으로 토큰 전달 가정.
    - 필요 시 Authorization 헤더 방식으로 확장 가능.
    """
    try:
        data = jwt.decode(token, SECRET_KEY)
        return {"username": data.get("username"), "name": data.get("name")}
    except Exception:
        raise HTTPException(status_code=401, detail="토큰 만료 또는 오류")


# ---------------------------------------------------------------------------
# React 프론트엔드 호환용 간단 계정/트레이드/지표 API
#   - baseURL: http://localhost:8000
#   - /account/info, /account/history
#   - /trade/buy, /trade/sell, /trade/auto
#   - /chart, /indicator, /history
# ---------------------------------------------------------------------------


@app.get("/account/info")
def api_account_info():
    """React `Account` 페이지용: 계좌 요약 + 보유 종목."""
    broker = get_broker()
    try:
        bal = broker.get_balance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KIS 잔고 조회 실패: {e}")

    raw = bal if isinstance(bal, dict) else {}
    holdings = raw.get("output1") or []
    summary_list = raw.get("output2") or []
    summary = summary_list[0] if summary_list else {}

    total_eval = 0.0
    for h in holdings:
        try:
            total_eval += float(h.get("evlu_amt") or 0)
        except (TypeError, ValueError):
            continue

    cash_raw = summary.get("dnca_tot_amt") or summary.get("nass_amt") or 0
    try:
        cash = float(cash_raw)
    except (TypeError, ValueError):
        cash = 0.0

    balance = total_eval + cash

    return {
        "balance": balance,
        "cash": cash,
        "holdings": holdings,
    }


@app.get("/account/history")
def api_account_history(limit: int = Query(50, ge=1, le=500)):
    """React `Account` 페이지용: 최근 주문/거래 내역."""
    db = get_db()
    session = db.get_session()
    try:
        rows = (
            session.query(TradeOrder)
            .order_by(TradeOrder.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()

    history = []
    for r in rows:
        history.append(
            {
                "created_at": r.created_at,
                "stock_code": r.stock_code,
                "stock_name": r.stock_name,
                "side": r.side,
                "quantity": r.quantity,
                "order_price": r.order_price,
                "order_amount": r.order_amount,
                "status": r.status,
            }
        )

    return {"history": history}


class TradeAmountRequest(BaseModel):
    stock_code: str
    amount: float = Field(..., gt=0, description="원화 기준 투자 금액")


def _place_market_order_internal(stock_code: str, side: str, quantity: int):
    """기존 /orders/market 로직을 재사용하기 위한 내부 헬퍼."""
    broker = get_broker()
    db = get_db()

    side_up = side.upper()
    if side_up not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="side 는 'BUY' 또는 'SELL' 이어야 합니다.")

    # 리스크 한도 체크
    check_risk_limit(broker, stock_code=stock_code, side=side_up, quantity=quantity)

    try:
        if side_up == "BUY":
            res = broker.buy_market(stock_code=stock_code, quantity=quantity)
        else:
            res = broker.sell_market(stock_code=stock_code, quantity=quantity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KIS 주문 실패: {e}")

    # 주문 로그 저장
    session = db.get_session()
    try:
        output = res.get("output") if isinstance(res, dict) else None
        stock_name = output.get("PDNAME") if isinstance(output, dict) else None
        order_price = None
        order_amount = None
        if isinstance(output, dict):
            try:
                order_price = float(output.get("ORD_UNPR") or 0)
                qty = float(output.get("ORD_QTY") or quantity)
                order_amount = order_price * qty
            except Exception:
                pass
        order = TradeOrder(
            stock_code=stock_code,
            stock_name=stock_name,
            side=side_up,
            quantity=quantity,
            order_price=order_price,
            order_amount=order_amount,
            status="OK" if isinstance(res, dict) and res.get("rt_cd") in (None, "0") else "ERROR",
            raw_response=json.dumps(res, ensure_ascii=False),
        )
        session.add(order)
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()

    return res


def _infer_quantity_from_amount(stock_code: str, amount: float) -> int:
    """
    금액(원화) 기준 주문에서 수량을 추정.
    - 최근 StockPrice.close 를 조회하여 amount / close 로 수량 계산.
    """
    db = get_db()
    session = db.get_session()
    try:
        row = (
            session.query(StockPrice)
            .filter(StockPrice.stock_code == stock_code)
            .order_by(StockPrice.datetime.desc())
            .first()
        )
    finally:
        session.close()

    if not row:
        raise HTTPException(status_code=400, detail=f"종목 {stock_code} 에 대한 가격 데이터가 없습니다.")

    if not row.close or row.close <= 0:
        raise HTTPException(status_code=400, detail=f"종목 {stock_code} 의 종가 정보가 유효하지 않습니다.")

    qty = int(amount / row.close)
    if qty <= 0:
        raise HTTPException(
            status_code=400,
            detail=f"금액 {int(amount):,}원 으로는 {stock_code} 1주도 매수할 수 없습니다. (종가 약 {row.close:,.0f}원)",
        )
    return qty


@app.get("/chart")
def api_chart(stock_code: str = Query(...), limit: int = Query(200, ge=10, le=1000)):
    """
    React `Chart` 페이지용 캔들 데이터.

    - StockPrice 테이블에서 OHLCV 조회
    """
    db = get_db()
    session = db.get_session()
    try:
        rows = (
            session.query(StockPrice)
            .filter(StockPrice.stock_code == stock_code)
            .order_by(StockPrice.datetime.asc())
            .all()
        )
    finally:
        session.close()

    if not rows:
        raise HTTPException(status_code=404, detail="차트 데이터가 없습니다.")

    candles = []
    for r in rows[-limit:]:
        candles.append(
            {
                "datetime": r.datetime,
                "open": r.open,
                "high": r.high,
                "low": r.low,
                "close": r.close,
                "volume": r.volume,
            }
        )

    return {"stock_code": stock_code, "candles": candles}


@app.get("/indicator")
def api_indicator(stock_code: str = Query(...)):
    """
    React `Indicators` 페이지용 기술적 지표.

    - StockPriceProcessed 테이블의 최신 한 줄을 사용.
    """
    db = get_db()
    session = db.get_session()
    try:
        row = (
            session.query(StockPriceProcessed)
            .filter(StockPriceProcessed.stock_code == stock_code)
            .order_by(StockPriceProcessed.datetime.desc())
            .first()
        )
    finally:
        session.close()

    if not row:
        raise HTTPException(status_code=404, detail="지표 데이터가 없습니다.")

    def safe(v):
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "stock_code": stock_code,
        "MA20": safe(row.ma_20),
        "MA60": safe(row.ma_60),
        "RSI": safe(row.rsi),
        "MACD": safe(row.macd),
        "Signal": safe(row.macd_signal),
        "Histogram": safe(row.macd_hist),
        "Upper": safe(row.bb_upper),
        "Lower": safe(row.bb_lower),
        "%K": safe(row.stoch_k),
        "%D": safe(row.stoch_d),
        "ATR": safe(row.atr),
    }


@app.get("/history")
def api_history(limit: int = Query(50, ge=1, le=500)):
    """
    React `History` 페이지용 예측/거래 히스토리.

    - 현재는 trade_orders 테이블 기반.
    """
    db = get_db()
    session = db.get_session()
    try:
        rows = (
            session.query(TradeOrder)
            .order_by(TradeOrder.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()

    history = []
    for r in rows:
        history.append(
            {
                "created_at": r.created_at,
                "stock_code": r.stock_code,
                "stock_name": r.stock_name,
                "side": r.side,
                "quantity": r.quantity,
                "order_price": r.order_price,
                "order_amount": r.order_amount,
                "status": r.status,
            }
        )

    return {"history": history}


@app.post("/trade/buy")
def api_trade_buy(req: TradeAmountRequest):
    """React `buyStock`용: 금액 기준 매수 API."""
    qty = _infer_quantity_from_amount(req.stock_code, req.amount)
    res = _place_market_order_internal(stock_code=req.stock_code, side="BUY", quantity=qty)
    return {"status": "ok", "quantity": qty, "response": res}


@app.post("/trade/sell")
def api_trade_sell(req: TradeAmountRequest):
    """React `sellStock`용: 금액 기준 매도 API (보유 수량 한도 내)."""
    qty = _infer_quantity_from_amount(req.stock_code, req.amount)
    res = _place_market_order_internal(stock_code=req.stock_code, side="SELL", quantity=qty)
    return {"status": "ok", "quantity": qty, "response": res}


@app.post("/trade/auto")
def api_trade_auto(payload: dict):
    """
    React `autoTrade` 버튼용.
    - 현재는 전체 자동매매 스크립트(auto_trader.py)를 1회 실행.
    - payload 내 stock_code 는 로깅 수준에서만 사용.
    """
    stock_code = payload.get("stock_code")

    script_path = Path(__file__).parent / "auto_trader.py"
    if not script_path.exists():
        raise HTTPException(status_code=500, detail=f"auto_trader 스크립트를 찾을 수 없습니다: {script_path}")

    try:
        proc = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        raise HTTPException(status_code=500, detail=f"auto_trader 실행 시간 초과: {e}")

    msg = "자동 투자 실행 완료"
    if stock_code:
        msg += f" (요청 종목: {stock_code})"

    return {
        "message": msg,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    """
    메인 홈페이지.
    - 회원가입 / 로그인 / 트레이딩 대시보드로 이동 버튼 제공
    """
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <title>StuckAI Home</title>
  <style>
    body { margin: 0; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
           background: radial-gradient(circle at top, #1f2937, #020617); color: #e5e7eb;
           min-height: 100vh; display: flex; align-items: center; justify-content: center; }
    .wrap { width: 100%; max-width: 960px; padding: 24px; box-sizing: border-box; }
    .card { background: rgba(15,23,42,0.96); border-radius: 18px; border: 1px solid rgba(55,65,81,0.9);
            box-shadow: 0 24px 80px rgba(15,23,42,0.95); padding: 22px 26px 22px; backdrop-filter: blur(18px); }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }
    .title { font-size: 22px; font-weight: 600; }
    .chip { font-size: 11px; padding: 2px 10px; border-radius: 999px; background: rgba(34,197,94,0.15);
            color: #4ade80; border: 1px solid rgba(34,197,94,0.4); }
    .subtitle { font-size: 13px; color: #9ca3af; margin-bottom: 12px; }
    .hero { display:grid; grid-template-columns: minmax(0,1.5fr) minmax(0,1fr); gap:22px; margin-bottom:22px; align-items:center; }
    @media (max-width: 880px) { .hero { grid-template-columns: 1fr; } }
    .hero-title { font-size:26px; font-weight:650; margin-bottom:6px; }
    .hero-sub { font-size:13px; color:#9ca3af; margin-bottom:10px; }
    .hero-tags { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:10px; }
    .pill { font-size:11px; padding:2px 9px; border-radius:999px; border:1px solid #374151; background:#020617; color:#e5e7eb; }
    .hero-metrics { display:flex; gap:14px; font-size:12px; color:#9ca3af; margin-top:4px; }
    .metric-label { color:#6b7280; font-size:11px; }
    .metric-value { font-size:14px; font-weight:600; color:#e5e7eb; }
    .hero-chart-wrap { background:#020617; border-radius:14px; border:1px solid #1f2937; padding:10px 12px 12px; box-shadow:0 14px 35px rgba(15,23,42,0.9); }
    .hero-chart-title { font-size:12px; color:#9ca3af; margin-bottom:6px; display:flex; justify-content:space-between; align-items:center; }
    .dot { width:7px; height:7px; border-radius:999px; background:#4ade80; margin-right:4px; }
    .dot-wrap { display:flex; align-items:center; gap:4px; font-size:11px; color:#6b7280; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 16px; }
    @media (max-width: 880px) { .grid { grid-template-columns: repeat(1, minmax(0,1fr)); } }
    .panel { background:#020617; border-radius: 14px; border:1px solid #1f2937; padding:14px 15px 14px; }
    .panel h3 { margin:0 0 6px 0; font-size:15px; }
    .panel p { margin:0 0 10px 0; font-size:12px; color:#9ca3af; }
    button { border:none; border-radius:999px; padding:7px 11px; font-size:13px; font-weight:500;
             cursor:pointer; display:inline-flex; align-items:center; justify-content:center; gap:6px; }
    .btn-main { background:linear-gradient(to right,#4ade80,#22c55e); color:#020617; box-shadow:0 12px 25px rgba(34,197,94,0.45); }
    .btn-outline { background:#020617; color:#e5e7eb; border:1px solid #374151; border-radius:10px; font-size:12px; padding:6px 10px; }
    .hint { font-size:11px; color:#6b7280; margin-top:8px; }
    .user { font-size:12px; color:#9ca3af; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="header">
        <div>
          <div class="title">stuckAI</div>
          <div class="subtitle">SAC 강화학습 + KIS OpenAPI 기반 자동 매매 데모 서비스입니다.</div>
        </div>
        <div style="text-align:right; display:flex; flex-direction:column; align-items:flex-end; gap:6px;">
          <div id="nav-loggedin-home" style="display:none; gap:8px; margin-bottom:2px;">
            <button class="btn-outline" onclick="window.location.href='/'">홈</button>
            <button class="btn-outline" onclick="window.location.href='/dashboard'">마이페이지</button>
            <button class="btn-outline" onclick="logoutAndGoLogin()">로그아웃</button>
          </div>
          <div class="chip">로컬 개발용</div>
          <div id="auth-buttons-home" style="display:flex; gap:8px; margin-top:4px;">
            <button class="btn-outline" onclick="window.location.href='/login-page'">로그인</button>
            <button class="btn-main" onclick="window.location.href='/signup-page'">회원가입</button>
          </div>
          <div id="user-info" class="user" style="margin-top:4px;">현재 로그인: 없음</div>
        </div>
      </div>

      <div class="hero">
        <div>
          <div class="hero-title">강화학습이 스스로 학습한 주식 자동매매 엔진</div>
          <div class="hero-sub">
            삼성전자 · 네이버 · 현대차 3종목에 대해 Soft Actor-Critic 기반으로 학습한 RL 에이전트가
            매일 포지션을 결정하고, KIS OpenAPI를 통해 모의계좌에 주문을 집행합니다.
          </div>
          <div class="hero-tags">
            <span class="pill">Reinforcement Learning · SAC</span>
            <span class="pill">KIS OpenAPI 연동</span>
            <span class="pill">자동 일별 리밸런싱</span>
            <span class="pill">리스크 한도 관리</span>
          </div>
          <div class="hero-metrics">
            <div>
              <div class="metric-label">Backtest 누적 수익률 (예시)</div>
              <div class="metric-value">+38.4%</div>
            </div>
            <div>
              <div class="metric-label">최대 낙폭 관리</div>
              <div class="metric-value">-12.7%</div>
            </div>
            <div>
              <div class="metric-label">운영 종목 수</div>
              <div class="metric-value">3개</div>
            </div>
          </div>
        </div>
        <div class="hero-chart-wrap">
          <div class="hero-chart-title">
            <span>샘플 운용 곡선 (시뮬레이션)</span>
            <div class="dot-wrap"><span class="dot"></span><span>전략 순자산</span></div>
          </div>
          <canvas id="hero-chart" width="360" height="180"></canvas>
        </div>
      </div>

      <div class="grid">
        <section class="panel">
          <h3>01. 회원가입</h3>
          <p>계정을 먼저 만들어야 로그인 후 대시보드에 접근할 수 있습니다.</p>
          <button class="btn-main" onclick="window.location.href='/signup-page'">회원가입 페이지로 이동</button>
        </section>

        <section class="panel">
          <h3>02. 로그인</h3>
          <p>로그인에 성공하면 브라우저에 JWT 토큰이 저장되고, 이름이 상단에 표시됩니다.</p>
          <button class="btn-main" onclick="window.location.href='/login-page'">로그인 페이지로 이동</button>
        </section>

        <section class="panel">
          <h3>03. 트레이딩 대시보드</h3>
          <p>계좌 잔고, 보유 종목, 주문, 리스크 설정 등을 확인하고 제어합니다.</p>
          <button class="btn-main" onclick="window.location.href='/dashboard'">대시보드 열기</button>
          <div class="hint">로그인하지 않아도 열리지만, 상단 로그인 상태는 로컬 토큰 기준으로 표시됩니다.</div>
        </section>
      </div>
    </div>
  </div>

  <script>
    function logoutAndGoLogin() {
      try {
        window.localStorage.removeItem("stuckai_token");
        window.localStorage.removeItem("stuckai_name");
      } catch (e) {}
      window.location.href = "/login-page";
    }

    function updateUserInfo() {
      try {
        const name = window.localStorage.getItem("stuckai_name");
        const el = document.getElementById("user-info");
        const nav = document.getElementById("nav-loggedin-home");
        const auth = document.getElementById("auth-buttons-home");
        if (el) {
          if (name) {
            el.textContent = "현재 로그인: " + name;
          } else {
            el.textContent = "현재 로그인: 없음";
          }
        }
        if (nav && auth) {
          if (name) {
            nav.style.display = "flex";
            auth.style.display = "none";
          } else {
            nav.style.display = "none";
            auth.style.display = "flex";
          }
        }
      } catch (e) {}
    }
    updateUserInfo();

    // 간단한 샘플 그래프 그리기 (더미 데이터 기반)
    (function drawHeroChart() {
      const canvas = document.getElementById("hero-chart");
      if (!canvas || !canvas.getContext) return;
      const ctx = canvas.getContext("2d");
      const w = canvas.width;
      const h = canvas.height;

      // 배경
      ctx.fillStyle = "#020617";
      ctx.fillRect(0, 0, w, h);

      // 축선
      ctx.strokeStyle = "#1f2937";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(32, 12);
      ctx.lineTo(32, h - 18);
      ctx.lineTo(w - 8, h - 18);
      ctx.stroke();

      // 더미 순자산 데이터 (0~1 구간)
      const points = [0.12, 0.18, 0.15, 0.23, 0.28, 0.32, 0.29, 0.37, 0.41, 0.38, 0.44, 0.48];
      const n = points.length;

      // 라인
      ctx.strokeStyle = "#4ade80";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = 32 + (w - 48) * (i / (n - 1));
        const y = (h - 26) - (h - 40) * points[i];
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // 그라데이션 영역
      const grad = ctx.createLinearGradient(0, 20, 0, h - 18);
      grad.addColorStop(0, "rgba(74,222,128,0.32)");
      grad.addColorStop(1, "rgba(15,23,42,0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      for (let i = 0; i < n; i++) {
        const x = 32 + (w - 48) * (i / (n - 1));
        const y = (h - 26) - (h - 40) * points[i];
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.lineTo(32 + (w - 48), h - 18);
      ctx.lineTo(32, h - 18);
      ctx.closePath();
      ctx.fill();
    })();
  </script>
</body>
</html>
    """


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """
    기존 트레이딩 대시보드 (잔고 조회 + 시장가 주문).
    
    - 브라우저에서 http://localhost:8000/dashboard 로 접속
    """
    return """
<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <title>StuckAI Trading Dashboard</title>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; padding: 0; background-color: #0f172a; color: #e5e7eb; }
    header { padding: 16px 24px; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between; align-items: center; }
    .title { font-size: 20px; font-weight: 600; }
    .chip { font-size: 12px; padding: 2px 8px; border-radius: 999px; background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.4); margin-left: 8px; }
    main { padding: 24px; display: grid; grid-template-columns: 2fr 1.5fr; gap: 24px; }
    .card { background-color: #020617; border-radius: 12px; border: 1px solid #1f2937; padding: 16px 18px; box-shadow: 0 10px 30px rgba(15,23,42,0.7); }
    .card h2 { font-size: 16px; margin: 0 0 8px 0; display: flex; align-items: center; justify-content: space-between; }
    .subtitle { font-size: 12px; color: #9ca3af; margin-bottom: 8px; }
    button { background: linear-gradient(to right, #4ade80, #22c55e); color: #020617; border: none; padding: 6px 12px; border-radius: 8px; font-size: 13px; cursor: pointer; font-weight: 500; }
    button:disabled { opacity: 0.6; cursor: default; }
    input, select { background-color: #020617; border-radius: 8px; border: 1px solid #374151; padding: 6px 8px; color: #e5e7eb; font-size: 13px; width: 100%; box-sizing: border-box; }
    label { font-size: 12px; color: #9ca3af; margin-bottom: 4px; display: block; }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin-top: 8px; }
    pre { background-color: #020617; border-radius: 8px; padding: 8px 10px; font-size: 11px; max-height: 360px; overflow: auto; border: 1px solid #111827; }
    .tag { font-size: 11px; padding: 2px 6px; border-radius: 999px; background: #111827; color: #9ca3af; border: 1px solid #1f2937; }
    .row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .small { font-size: 11px; color: #6b7280; }
    .status { font-size: 12px; margin-top: 6px; min-height: 18px; }
    .status.ok { color: #4ade80; }
    .status.err { color: #f97373; }
  </style>
</head>
<body>
  <header>
    <div class="title">
      StuckAI Trading Dashboard
      <span class="chip">SAC + KIS Demo</span>
    </div>
    <div class="row">
      <span class="small">백엔드: FastAPI · 브로커: KIS</span>
      <div class="row" id="nav-loggedin-dashboard" style="gap:8px;">
        <button class="small" style="background:#111827; color:#e5e7eb; border-radius:999px; border:1px solid #374151; padding:4px 8px; cursor:pointer;" onclick="window.location.href='/'">
          홈
        </button>
        <button class="small" style="background:#111827; color:#e5e7eb; border-radius:999px; border:1px solid #374151; padding:4px 8px; cursor:pointer;" onclick="window.location.href='/dashboard'">
          마이페이지
        </button>
        <button class="small" style="background:#7f1d1d; color:#fee2e2; border-radius:999px; border:1px solid #b91c1c; padding:4px 8px; cursor:pointer;" onclick="logoutAndGoLogin()">
          로그아웃
        </button>
      </div>
    </div>
  </header>
  <main>
    <section class="card">
      <h2>
        계좌 잔고 / 포지션
        <button id="btn-balance">잔고 조회</button>
      </h2>
      <div class="subtitle">KIS OpenAPI에서 현재 잔고/보유 종목을 조회해 요약 테이블로 보여줍니다.</div>
      <div id="balance-summary" class="small" style="margin-bottom:8px;"></div>
      <div id="balance-table"></div>
      <details style="margin-top:10px;">
        <summary class="small">원본 JSON 보기</summary>
        <pre id="balance-output" style="margin-top:6px;">{ 잔고 정보를 불러오려면 상단의 "잔고 조회" 버튼을 누르세요 }</pre>
      </details>
    </section>

    <section class="card">
      <h2>
        성과 요약 (최근 30일)
        <button id="btn-refresh-performance">새로고침</button>
      </h2>
      <div class="subtitle">일별 계좌 총자산과 손익을 기반으로 성과를 요약해 보여줍니다.</div>
      <div id="perf-summary" class="small" style="margin-bottom:8px;"></div>
      <div id="perf-table"></div>
    </section>

    <section class="card">
      <h2>
        시장가 주문 테스트
        <span class="tag">POST /orders/market</span>
      </h2>
      <div class="subtitle">강화학습/전략 엔진이 결정한 주문을 이 엔드포인트로 전달해 실제 체결을 시도합니다.</div>
      <div class="grid">
        <div>
          <label for="stock-code">종목코드</label>
          <input id="stock-code" placeholder="예: 005930" />
        </div>
        <div>
          <label for="quantity">수량</label>
          <input id="quantity" type="number" min="1" step="1" value="1" />
        </div>
        <div>
          <label for="side">방향</label>
          <select id="side">
            <option value="BUY">BUY (매수)</option>
            <option value="SELL">SELL (매도)</option>
          </select>
        </div>
      </div>
      <div class="row" style="margin-top: 12px;">
        <button id="btn-order">시장가 주문 전송</button>
        <span class="small">주의: .env 의 KIS_* 설정에 따라 실제 모의/실계좌 주문이 발생할 수 있습니다.</span>
      </div>
      <div id="order-status" class="status"></div>
      <pre id="order-output">{ 주문 응답이 여기에 표시됩니다 }</pre>
    </section>

    <section class="card">
      <h2>
        거래 내역
        <button id="btn-refresh-orders">새로고침</button>
      </h2>
      <div class="subtitle">최근 자동/수동 주문 기록을 확인할 수 있습니다.</div>
      <div class="row" style="margin-bottom:8px%;">
        <div class="small">종목코드로 필터링 (예: 005930)</div>
        <input id="orders-symbol" placeholder="전체" style="max-width:120px;" />
      </div>
      <div id="orders-table"></div>
    </section>

    <section class="card">
      <h2>
        리스크 설정
        <button id="btn-refresh-risk">새로고침</button>
      </h2>
      <div class="subtitle">종목별 최대 보유 수량, 비중 한도 등을 설정합니다.</div>
      <div class="small" style="margin-bottom:8px;">
        - 'ALL' 설정은 공통 기본값으로 사용되며, 종목별 설정이 있으면 그것이 우선합니다.
      </div>
      <div class="grid">
        <div>
          <label for="risk-stock">종목코드 / ALL</label>
          <input id="risk-stock" placeholder="예: 005930 또는 ALL" />
        </div>
        <div>
          <label for="risk-max-shares">최대 보유 수량</label>
          <input id="risk-max-shares" type="number" min="1" step="1" placeholder="비우면 기본값 유지" />
        </div>
        <div>
          <label for="risk-max-weight">최대 비중 (%)</label>
          <input id="risk-max-weight" type="number" min="0" max="100" step="1" placeholder="예: 50" />
        </div>
      </div>
      <div class="grid" style="margin-top:8px;">
        <div>
          <label for="risk-max-daily">일간 최대 매수금액 (원)</label>
          <input id="risk-max-daily" type="number" min="0" step="10000" placeholder="옵션" />
        </div>
        <div>
          <label for="risk-active">활성 여부</label>
          <select id="risk-active">
            <option value="true">활성</option>
            <option value="false">비활성</option>
          </select>
        </div>
        <div>
          <label for="risk-api-key">API Key (보안)</label>
          <input id="risk-api-key" type="password" placeholder="PUT 시 X-API-Key로 사용" />
        </div>
      </div>
      <div class="row" style="margin-top:12px;">
        <button id="btn-save-risk">설정 저장</button>
        <span class="small">주의: 저장 시 API Key 가 필요합니다.</span>
      </div>
      <div id="risk-status" class="status"></div>
      <div id="risk-table" style="margin-top:8px;"></div>
    </section>
  </main>

  <script>
    // --- 간단 로그인 체크: 토큰 없거나 /me 실패 시 로그인 페이지로 이동 ---
    (async function guardDashboard() {
      try {
        const token = window.localStorage.getItem("stuckai_token");
        if (!token) {
          alert("대시보드를 보려면 먼저 로그인 해주세요.");
          window.location.href = "/login-page";
          return;
        }
        // 선택적으로 /me 호출로 토큰 유효성 확인
        const res = await fetch("/me?token=" + encodeURIComponent(token));
        if (!res.ok) {
          window.localStorage.removeItem("stuckai_token");
          window.localStorage.removeItem("stuckai_name");
          alert("로그인 정보가 만료되었습니다. 다시 로그인 해주세요.");
          window.location.href = "/login-page";
        }
      } catch (e) {
        console.warn("대시보드 가드 오류:", e);
      }
    })();

    function logoutAndGoLogin() {
      try {
        window.localStorage.removeItem("stuckai_token");
        window.localStorage.removeItem("stuckai_name");
      } catch (e) {}
      window.location.href = "/login-page";
    }

    async function fetchJson(url, options) {
      const res = await fetch(url, options);
      const text = await res.text();
      try {
        return { ok: res.ok, status: res.status, json: JSON.parse(text) };
      } catch (e) {
        return { ok: res.ok, status: res.status, json: { raw: text } };
      }
    }

    const btnBalance = document.getElementById("btn-balance");
    const balanceOut = document.getElementById("balance-output");
    const balanceSummary = document.getElementById("balance-summary");
    const balanceTable = document.getElementById("balance-table");
    const btnOrder = document.getElementById("btn-order");
    const orderOut = document.getElementById("order-output");
    const orderStatus = document.getElementById("order-status");
    const btnPerf = document.getElementById("btn-refresh-performance");
    const perfSummary = document.getElementById("perf-summary");
    const perfTable = document.getElementById("perf-table");
    const btnOrders = document.getElementById("btn-refresh-orders");
    const ordersTable = document.getElementById("orders-table");
    const ordersSymbol = document.getElementById("orders-symbol");
    const btnRiskRefresh = document.getElementById("btn-refresh-risk");
    const btnRiskSave = document.getElementById("btn-save-risk");
    const riskStock = document.getElementById("risk-stock");
    const riskMaxShares = document.getElementById("risk-max-shares");
    const riskMaxWeight = document.getElementById("risk-max-weight");
    const riskMaxDaily = document.getElementById("risk-max-daily");
    const riskActive = document.getElementById("risk-active");
    const riskApiKey = document.getElementById("risk-api-key");
    const riskStatus = document.getElementById("risk-status");
    const riskTable = document.getElementById("risk-table");

    function renderBalanceNice(raw) {
      balanceTable.innerHTML = "";
      balanceSummary.textContent = "";

      if (!raw || typeof raw !== "object") {
        balanceTable.innerHTML = "<div class='small'>잔고 데이터를 해석할 수 없습니다.</div>";
        return;
      }

      const holdings = Array.isArray(raw.output1) ? raw.output1 : [];
      const summaryArr = Array.isArray(raw.output2) ? raw.output2 : [];
      const summary = summaryArr[0] || {};

      // 항상 보여주고 싶은 주요 종목들 (보유가 없어도 0으로 표시)
      const coreStocks = [
        { code: "005930", name: "삼성전자" },
        { code: "035420", name: "네이버" },
        { code: "005380", name: "현대차" },
      ];
      if (Array.isArray(holdings)) {
        for (const core of coreStocks) {
          const exists = holdings.some((h) => h.pdno === core.code);
          if (!exists) {
            holdings.push({
              pdno: core.code,
              prdt_name: core.name,
              hldg_qty: "0",
              ord_psbl_qty: "0",
              pchs_avg_pric: "-",
              evlu_pfls_amt: "0",
            });
          }
        }
      }

      // 요약 영역: 총 보유수량, 총 매입금액, 평가금액, 손익
      let totalQty = 0;
      let totalBuyAmt = 0;
      let totalEvalAmt = 0;
      let totalPnl = 0;
      for (const h of holdings) {
        const q = parseFloat(h.hldg_qty || "0");
        const buyAmt = parseFloat(h.pchs_amt || "0");
        const evalAmt = parseFloat(h.evlu_amt || "0");
        const pnl = parseFloat(h.evlu_pfls_amt || "0");
        if (!Number.isNaN(q)) totalQty += q;
        if (!Number.isNaN(buyAmt)) totalBuyAmt += buyAmt;
        if (!Number.isNaN(evalAmt)) totalEvalAmt += evalAmt;
        if (!Number.isNaN(pnl)) totalPnl += pnl;
      }

      const cash = summary.dnca_tot_amt || summary.nass_amt || null;
      const parts = [];
      if (!Number.isNaN(totalQty) && totalQty > 0) {
        parts.push(`총 보유수량: ${totalQty}주`);
      }
      if (!Number.isNaN(totalBuyAmt) && totalBuyAmt !== 0) {
        parts.push(`총 매입금액: ${totalBuyAmt.toLocaleString()}원`);
      }
      if (!Number.isNaN(totalEvalAmt) && totalEvalAmt !== 0) {
        parts.push(`평가금액: ${totalEvalAmt.toLocaleString()}원`);
      }
      if (!Number.isNaN(totalPnl) && totalPnl !== 0) {
        const sign = totalPnl >= 0 ? "+" : "";
        parts.push(`평가손익: ${sign}${totalPnl.toLocaleString()}원`);
      }
      if (cash != null) {
        const cashNum = Number(cash);
        if (!Number.isNaN(cashNum)) {
          parts.push(`예수금: ${cashNum.toLocaleString()}원`);
        }
      }

      if (parts.length) {
        balanceSummary.textContent = parts.join(" · ");
      }

      // 보유 종목 테이블
      const columns = [
        { key: "pdno", label: "종목코드" },
        { key: "prdt_name", label: "종목명" },
        { key: "hldg_qty", label: "보유수량" },
        { key: "ord_psbl_qty", label: "매도가능" },
        { key: "pchs_avg_pric", label: "평균매입가" },
        { key: "evlu_pfls_amt", label: "평가손익" },
      ];

      let html = "<table style='width:100%; border-collapse:collapse; font-size:12px;'>";
      html += "<thead><tr>";
      for (const col of columns) {
        html += `<th style="text-align:left; padding:4px 6px; border-bottom:1px solid #1f2937; color:#9ca3af;">${col.label}</th>`;
      }
      html += "</tr></thead><tbody>";

      for (const row of holdings) {
        html += "<tr>";
        for (const col of columns) {
          let v = row[col.key] != null ? row[col.key] : "";
          // 거래가 없어서 값이 비어 있을 때도 보유수량/매도가능은 0으로 표시
          if (col.key === "hldg_qty" || col.key === "ord_psbl_qty") {
            const n = Number(v || 0);
            v = Number.isNaN(n) ? "0" : String(n);
          }
          html += `<td style="padding:4px 6px; border-bottom:1px solid #111827;">${v}</td>`;
        }
        html += "</tr>";
      }
      html += "</tbody></table>";

      balanceTable.innerHTML = html;
    }

    btnBalance.addEventListener("click", async () => {
      btnBalance.disabled = true;
      balanceOut.textContent = "불러오는 중...";
      balanceTable.innerHTML = "";
      balanceSummary.textContent = "";
      try {
        const res = await fetchJson("/accounts/balance");
        const raw = res.json && res.json.raw ? res.json.raw : res.json;
        balanceOut.textContent = JSON.stringify(raw, null, 2);
        if (res.ok) {
          renderBalanceNice(raw);
        }
      } catch (e) {
        balanceOut.textContent = "에러: " + e;
      } finally {
        btnBalance.disabled = false;
      }
    });

    btnPerf.addEventListener("click", async () => {
      btnPerf.disabled = true;
      perfSummary.textContent = "로딩 중...";
      perfTable.innerHTML = "";
      try {
        const res = await fetchJson("/metrics/performance?days=30");
        if (!res.ok) {
          perfSummary.textContent = "성능 조회 실패: " + (res.json && res.json.detail ? res.json.detail : "오류");
          return;
        }
        const data = res.json;
        const s = data.summary || {};
        const snaps = data.snapshots || [];

        perfSummary.textContent =
          `시작자산: ${Math.round(s.start_value || 0).toLocaleString()}원 · ` +
          `현재자산: ${Math.round(s.end_value || 0).toLocaleString()}원 · ` +
          `누적수익률: ${(s.total_return_pct || 0).toFixed(2)}% · ` +
          `최대낙폭: ${(s.max_drawdown_pct || 0).toFixed(2)}% · ` +
          `누적손익: ${((s.pnl_sum || 0) >= 0 ? "+" : "") + Math.round(s.pnl_sum || 0).toLocaleString()}원`;

        if (!snaps.length) {
          perfTable.innerHTML = "<div class='small'>스냅샷 데이터가 없습니다. 먼저 잔고 조회를 실행해 주세요.</div>";
          return;
        }

        let html = "<table style='width:100%; border-collapse:collapse; font-size:12px;'>";
        html += "<thead><tr>";
        const cols = ["시각", "총자산", "예수금", "총매입", "평가금액", "총손익"];
        for (const c of cols) {
          html += `<th style="text-align:left; padding:4px 6px; border-bottom:1px solid #1f2937; color:#9ca3af;">${c}</th>`;
        }
        html += "</tr></thead><tbody>";
        for (const row of snaps.slice().reverse()) {
          const dt = new Date(row.timestamp);
          const ts = dt.toLocaleString();
          const tv = Math.round(row.total_value || 0).toLocaleString();
          const cash = Math.round(row.cash || 0).toLocaleString();
          const tb = Math.round(row.total_buy_amount || 0).toLocaleString();
          const te = Math.round(row.total_eval_amount || 0).toLocaleString();
          const pnl = Math.round(row.total_pnl || 0);
          const pnlStr = (pnl >= 0 ? "+" : "") + pnl.toLocaleString();
          html += `<tr>
            <td style="padding:4px 6px; border-bottom:1px solid  #111827;">${ts}</td>
            <td style="padding:4px 6px; border-bottom:1px solid  #111827;">${tv}</td>
            <td style="padding:4px 6px; border-bottom:1px solid  #111827;">${cash}</td>
            <td style="padding:4px 6px; border-bottom:1px solid  #111827;">${tb}</td>
            <td style="padding:4px 6px; border-bottom:1px solid  #111827;">${te}</td>
            <td style="padding:4px 6px; border-bottom:1px solid  #111827;">${pnlStr}</td>
          </tr>`;
        }
        html += "</tbody></table>";
        perfTable.innerHTML = html;
      } catch (e) {
        perfSummary.textContent = "에러: " + e;
      } finally {
        btnPerf.disabled = false;
      }
    });

    btnOrders.addEventListener("click", async () => {
      btnOrders.disabled = true;
      ordersTable.innerHTML = "<div class='small'>로딩 중...</div>";
      try {
        const symbol = (ordersSymbol.value || "").trim();
        let url = "/orders/history?limit=100";
        if (symbol) {
          url += "&stock_code=" + encodeURIComponent(symbol);
        }
        const res = await fetchJson(url);
        if (!res.ok) {
          ordersTable.innerHTML = "<div class='small'>주문 내역 조회 실패: " + (res.json && res.json.detail ? res.json.detail : "오류") + "</div>";
          return;
        }
        const rows = Array.isArray(res.json) ? res.json : [];
        if (!rows.length) {
          ordersTable.innerHTML = "<div class='small'>표시할 주문 내역이 없습니다.</div>";
          return;
        }
        let html = "<table style='width:100%; border-collapse:collapse; font-size:12px;'>";
        html += "<thead><tr>";
        const cols = ["시간", "종목코드", "종목명", "방향", "수량", "가격", "금액", "상태"];
        for (const c of cols) {
          html += `<th style="text-align:left; padding:4px 6px; border-bottom:1px solid #1f2937; color:#9ca3af;">${c}</th>`;
        }
        html += "</tr></thead><tbody>";
        for (const o of rows) {
          const dt = new Date(o.created_at);
          const ts = dt.toLocaleString();
          const price = o.order_price != null ? o.order_price.toLocaleString() : "-";
          const amt = o.order_amount != null ? o.order_amount.toLocaleString() : "-";
          html += `<tr>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${ts}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${o.stock_code}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${o.stock_name || ""}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${o.side}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${o.quantity}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${price}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${amt}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${o.status}</td>
          </tr>`;
        }
        html += "</tbody></table>";
        ordersTable.innerHTML = html;
      } catch (e) {
        ordersTable.innerHTML = "<div class='small'>에러: " + e + "</div>";
      } finally {
        btnOrders.disabled = false;
      }
    });

    async function loadRiskSettings() {
      riskTable.innerHTML = "<div class='small'>로딩 중...</div>";
      try {
        const res = await fetchJson("/settings/risk");
        if (!res.ok) {
          riskTable.innerHTML = "<div class='small'>리스크 설정 조회 실패: " + (res.json && res.json.detail ? res.json.detail : "오류") + "</div>";
          return;
        }
        const rows = Array.isArray(res.json) ? res.json : [];
        if (!rows.length) {
          riskTable.innerHTML = "<div class='small'>설정된 리스크 규칙이 없습니다.</div>";
          return;
        }
        let html = "<table style='width:100%; border-collapse:collapse; font-size:12px;'>";
        html += "<thead><tr>";
        const cols = ["종목코드", "최대수량", "최대비중(%)", "일간매수한도", "활성", "생성", "수정"];
        for (const c of cols) {
          html += `<th style="text-align:left; padding:4px 6px; border-bottom:1px solid #1f2937; color:#9ca3af;">${c}</th>`;
        }
        html += "</tr></thead><tbody>";
        for (const r of rows) {
          const w = r.max_weight_pct != null ? (r.max_weight_pct * 100).toFixed(0) : "-";
          const daily = r.max_daily_buy_amount != null ? Math.round(r.max_daily_buy_amount).toLocaleString() : "-";
          html += `<tr>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${r.stock_code}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${r.max_position_shares ?? "-"}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${w}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${daily}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${r.active ? "ON" : "OFF"}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${r.created_at}</td>
            <td style="padding:4px 6px; border-bottom:1px solid #111827;">${r.updated_at || ""}</td>
          </tr>`;
        }
        html += "</tbody></table>";
        riskTable.innerHTML = html;
      } catch (e) {
        riskTable.innerHTML = "<div class='small'>에러: " + e + "</div>";
      }
    }

    btnRiskRefresh.addEventListener("click", loadRiskSettings);

    btnRiskSave.addEventListener("click", async () => {
      const code = (riskStock.value || "").trim();
      if (!code) {
        riskStatus.textContent = "종목코드 또는 ALL 을 입력하세요.";
        riskStatus.className = "status err";
        return;
      }
      const body = {};
      if (riskMaxShares.value) body.max_position_shares = Number(riskMaxShares.value);
      if (riskMaxWeight.value) body.max_weight_pct = Number(riskMaxWeight.value) / 100.0;
      if (riskMaxDaily.value) body.max_daily_buy_amount = Number(riskMaxDaily.value);
      body.active = riskActive.value === "true";

      const apiKey = (riskApiKey.value || "").trim();
      const headers = { "Content-Type": "application/json" };
      if (apiKey) headers["X-API-Key"] = apiKey;

      btnRiskSave.disabled = true;
      riskStatus.textContent = "저장 중...";
      riskStatus.className = "status";
      try {
        const res = await fetchJson(`/settings/risk/${encodeURIComponent(code)}`, {
          method: "PUT",
          headers,
          body: JSON.stringify(body),
        });
        if (!res.ok) {
          riskStatus.textContent = "저장 실패: " + (res.json && res.json.detail ? res.json.detail : "오류");
          riskStatus.className = "status err";
          return;
        }
        riskStatus.textContent = "저장 완료";
        riskStatus.className = "status ok";
        await loadRiskSettings();
      } catch (e) {
        riskStatus.textContent = "에러: " + e;
        riskStatus.className = "status err";
      } finally {
        btnRiskSave.disabled = false;
      }
    });

    btnOrder.addEventListener("click", async () => {
      const code = (document.getElementById("stock-code").value || "").trim();
      const qty = parseInt(document.getElementById("quantity").value || "0", 10);
      const side = document.getElementById("side").value;

      if (!code) {
        orderStatus.textContent = "종목코드를 입력하세요.";
        orderStatus.className = "status err";
        return;
      }
      if (!qty || qty <= 0) {
        orderStatus.textContent = "1 이상 수량을 입력하세요.";
        orderStatus.className = "status err";
        return;
      }

      btnOrder.disabled = true;
      orderStatus.textContent = "주문 전송 중...";
      orderStatus.className = "status";
      orderOut.textContent = "";

      try {
        const res = await fetchJson("/orders/market", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ stock_code: code, quantity: qty, side })
        });
        orderOut.textContent = JSON.stringify(res.json, null, 2);
        if (res.ok) {
          orderStatus.textContent = "주문 성공 (status " + res.status + ")";
          orderStatus.className = "status ok";
        } else {
          orderStatus.textContent = "주문 실패 (status " + res.status + ")";
          orderStatus.className = "status err";
        }
      } catch (e) {
        orderStatus.textContent = "요청 에러: " + e;
        orderStatus.className = "status err";
      } finally {
        btnOrder.disabled = false;
      }
    });
  </script>
</body>
</html>
    """


@app.post("/orders/market")
def place_market_order(
    req: MarketOrderRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """
    단순 시장가 주문 엔드포인트.

    강화학습/전략 엔진은:
      1) 어떤 종목을 얼마만큼 매수/매도할지 결정하고
      2) 이 엔드포인트에 POST를 보내 실제 주문을 실행한다.

    주문 결과는 `trade_orders` 테이블에 로그로 저장된다.
    """
    # 간단한 API 키 인증 (옵션)
    expected_key = os.getenv("API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="유효하지 않은 API Key 입니다.")

    broker = get_broker()
    db = get_db()

    side = req.side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(status_code=400, detail="side 는 'BUY' 또는 'SELL' 이어야 합니다.")

    # 리스크 한도 체크
    check_risk_limit(broker, stock_code=req.stock_code, side=side, quantity=req.quantity)

    try:
        if side == "BUY":
            res = broker.buy_market(stock_code=req.stock_code, quantity=req.quantity)
        else:
            res = broker.sell_market(stock_code=req.stock_code, quantity=req.quantity)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KIS 주문 실패: {e}")

    # 주문 로그 저장
    session = db.get_session()
    try:
        output = res.get("output") if isinstance(res, dict) else None
        stock_name = output.get("PDNAME") if isinstance(output, dict) else None
        order_price = None
        order_amount = None
        if isinstance(output, dict):
            try:
                order_price = float(output.get("ORD_UNPR") or 0)
                qty = float(output.get("ORD_QTY") or req.quantity)
                order_amount = order_price * qty
            except Exception:
                pass
        order = TradeOrder(
            stock_code=req.stock_code,
            stock_name=stock_name,
            side=side,
            quantity=req.quantity,
            order_price=order_price,
            order_amount=order_amount,
            status="OK" if isinstance(res, dict) and res.get("rt_cd") in (None, "0") else "ERROR",
            raw_response=json.dumps(res, ensure_ascii=False),
        )
        session.add(order)
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()

    return {"status": "ok", "response": res}


@app.get("/accounts/balance", response_model=BalanceResponse)
def get_account_balance():
    """
    KIS 계좌 잔고/보유 종목 조회.

    조회 결과는 `account_snapshots` 테이블에 요약 형태로 저장된다.
    """
    broker = get_broker()
    db = get_db()
    try:
        bal = broker.get_balance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"KIS 잔고 조회 실패: {e}")

    # 스냅샷 저장
    raw = bal if isinstance(bal, dict) else {}
    holdings = raw.get("output1") or []
    summary_list = raw.get("output2") or []
    summary = summary_list[0] if summary_list else {}

    total_buy = 0.0
    total_eval = 0.0
    total_pnl = 0.0
    for h in holdings:
        try:
            buy_amt = float(h.get("pchs_amt") or 0)
            eval_amt = float(h.get("evlu_amt") or 0)
            pnl = float(h.get("evlu_pfls_amt") or 0)
        except (TypeError, ValueError):
            continue
        total_buy += buy_amt
        total_eval += eval_amt
        total_pnl += pnl

    cash_raw = summary.get("dnca_tot_amt") or summary.get("nass_amt") or 0
    try:
        cash = float(cash_raw)
    except (TypeError, ValueError):
        cash = 0.0

    total_value = total_eval + cash

    session = db.get_session()
    try:
        snap = AccountSnapshot(
            total_value=total_value,
            cash=cash,
            total_buy_amount=total_buy,
            total_eval_amount=total_eval,
            total_pnl=total_pnl,
            raw_response=json.dumps(raw, ensure_ascii=False),
        )
        session.add(snap)
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()

    return BalanceResponse(raw=bal)


@app.get("/metrics/performance", response_model=PerformanceResponse)
def get_performance(days: int = Query(30, ge=1, le=365)):
    """
    최근 N일간의 계좌 성과 요약 및 스냅샷을 반환.

    - days: 최근 N일 (기본 30일)
    """
    db = get_db()
    session = db.get_session()
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)
        rows = (
            session.query(AccountSnapshot)
            .filter(AccountSnapshot.created_at >= cutoff)
            .order_by(AccountSnapshot.created_at.asc())
            .all()
        )
    finally:
        session.close()

    if not rows:
        return PerformanceResponse(
            summary=PerformanceSummary(
                start_value=0.0, end_value=0.0, total_return_pct=0.0, max_drawdown_pct=0.0, pnl_sum=0.0
            ),
            snapshots=[],
        )

    snaps: List[PerformanceSnapshot] = []
    equity: List[float] = []
    for r in rows:
        snaps.append(
            PerformanceSnapshot(
                timestamp=r.created_at,
                total_value=r.total_value or 0.0,
                cash=r.cash or 0.0,
                total_buy_amount=r.total_buy_amount or 0.0,
                total_eval_amount=r.total_eval_amount or 0.0,
                total_pnl=r.total_pnl or 0.0,
            )
        )
        equity.append(r.total_value or 0.0)

    start_val = equity[0]
    end_val = equity[-1]
    total_return_pct = ((end_val - start_val) / start_val * 100.0) if start_val not in (0, None) else 0.0

    # 최대 낙폭 계산
    max_peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > max_peak:
            max_peak = v
        if max_peak > 0:
            dd = (max_peak - v) / max_peak * 100.0
            if dd > max_dd:
                max_dd = dd

    pnl_sum = sum((s.total_pnl for s in snaps))

    summary = PerformanceSummary(
        start_value=start_val,
        end_value=end_val,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_dd,
        pnl_sum=pnl_sum,
    )

    return PerformanceResponse(summary=summary, snapshots=snaps)


@app.get("/orders/history", response_model=List[OrderHistoryItem])
def get_order_history(
    stock_code: Optional[str] = Query(default=None, description="필터링할 종목코드 (예: 005930)"),
    limit: int = Query(100, ge=1, le=1000),
):
    """최근 주문 내역 조회. stock_code 로 필터링 가능."""
    db = get_db()
    session = db.get_session()
    try:
        q = session.query(TradeOrder).order_by(TradeOrder.created_at.desc())
        if stock_code:
            q = q.filter(TradeOrder.stock_code == stock_code)
        rows = q.limit(limit).all()
    finally:
        session.close()

    result: List[OrderHistoryItem] = []
    for r in rows:
        result.append(
            OrderHistoryItem(
                created_at=r.created_at,
                stock_code=r.stock_code,
                stock_name=r.stock_name,
                side=r.side,
                quantity=r.quantity,
                order_price=r.order_price,
                order_amount=r.order_amount,
                status=r.status,
            )
        )
    return result


@app.post("/auto-trade/run-once", response_model=AutoTradeRunResult)
def run_auto_trade_once(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """
    일일 자동 매매 스크립트를 1회 실행합니다.

    내부적으로 `python auto_trader.py`를 서브프로세스로 실행합니다.
    실행 로그는 stdout/stderr 로 반환됩니다.
    """
    # API Key 검증
    expected_key = os.getenv("API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="유효하지 않은 API Key 입니다.")

    script_path = Path(__file__).parent / "auto_trader.py"
    if not script_path.exists():
        raise HTTPException(status_code=500, detail=f"auto_trader 스크립트를 찾을 수 없습니다: {script_path}")

    try:
        proc = subprocess.run(
            ["python", str(script_path)],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        raise HTTPException(status_code=500, detail=f"auto_trader 실행 시간 초과: {e}")

    # 실행 결과를 DB에 기록
    db = get_db()
    session = db.get_session()
    try:
        run = AutoTradeRun(
            returncode=proc.returncode,
            stdout=proc.stdout[-2000:],  # 로그가 너무 길 경우 끝부분만 저장
            stderr=proc.stderr[-2000:],
        )
        session.add(run)
        session.commit()
    except Exception:
        session.rollback()
    finally:
        session.close()

    return AutoTradeRunResult(returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)


@app.get("/auto-trade/status", response_model=List[AutoTradeRunItem])
def get_auto_trade_status(limit: int = Query(5, ge=1, le=50)):
    """
    최근 자동매매 실행 이력을 반환합니다.

    - returncode == 0 이면 정상 종료, 그 외는 오류.
    """
    db = get_db()
    session = db.get_session()
    try:
        rows = (
            session.query(AutoTradeRun)
            .order_by(AutoTradeRun.created_at.desc())
            .limit(limit)
            .all()
        )
    finally:
        session.close()

    result: List[AutoTradeRunItem] = []
    for r in rows:
        result.append(
            AutoTradeRunItem(
                id=r.id,
                created_at=r.created_at,
                returncode=r.returncode,
            )
        )
    return result


@app.get("/settings/risk", response_model=List[RiskSettingOut])
def list_risk_settings(stock_code: Optional[str] = Query(default=None, description="필터링할 종목코드 (예: 005930 또는 ALL)")):
    """
    현재 저장된 리스크/포지션 한도 설정 목록 조회.

    - stock_code 를 지정하면 해당 종목(또는 'ALL')만 반환
    """
    db = get_db()
    session = db.get_session()
    try:
        q = session.query(RiskSetting)
        if stock_code:
            q = q.filter(RiskSetting.stock_code == stock_code)
        rows = q.order_by(RiskSetting.stock_code.asc()).all()
    finally:
        session.close()

    result: List[RiskSettingOut] = []
    for r in rows:
        result.append(
            RiskSettingOut(
                stock_code=r.stock_code,
                max_position_shares=r.max_position_shares,
                max_weight_pct=r.max_weight_pct,
                max_daily_buy_amount=r.max_daily_buy_amount,
                active=bool(r.active),
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
        )
    return result


@app.put("/settings/risk/{stock_code}", response_model=RiskSettingOut)
def upsert_risk_setting(
    stock_code: str,
    body: RiskSettingIn,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """
    특정 종목(또는 'ALL')에 대한 리스크 한도 설정을 생성/수정한다.

    - stock_code: '005930', '035420', 'ALL' 등
    - body 에서 지정된 필드만 갱신 (나머지는 유지)
    """
    expected_key = os.getenv("API_KEY")
    if expected_key and x_api_key != expected_key:
        raise HTTPException(status_code=401, detail="유효하지 않은 API Key 입니다.")

    db = get_db()
    session = db.get_session()
    try:
        setting = session.query(RiskSetting).filter(RiskSetting.stock_code == stock_code).first()
        if setting is None:
            setting = RiskSetting(stock_code=stock_code)

        if body.max_position_shares is not None:
            setting.max_position_shares = body.max_position_shares
        if body.max_weight_pct is not None:
            setting.max_weight_pct = body.max_weight_pct
        if body.max_daily_buy_amount is not None:
            setting.max_daily_buy_amount = body.max_daily_buy_amount
        if body.active is not None:
            setting.active = body.active

        session.add(setting)
        session.commit()
        session.refresh(setting)
    except Exception as e:
        session.rollback()
        raise HTTPException(status_code=500, detail=f"리스크 설정 저장 실패: {e}")
    finally:
        session.close()

    return RiskSettingOut(
        stock_code=setting.stock_code,
        max_position_shares=setting.max_position_shares,
        max_weight_pct=setting.max_weight_pct,
        max_daily_buy_amount=setting.max_daily_buy_amount,
        active=bool(setting.active),
        created_at=setting.created_at,
        updated_at=setting.updated_at,
    )
