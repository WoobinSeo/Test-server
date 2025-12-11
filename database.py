"""
PostgreSQL 데이터베이스 스키마 및 연결 관리
"""
import os
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    BigInteger,
    Text,
    Boolean,
    LargeBinary,
    ForeignKey,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()


class StockPrice(Base):
    """주식 가격 테이블"""
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(50), nullable=False)
    datetime = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    
    def __repr__(self):
        return f"<StockPrice(stock_name='{self.stock_name}', datetime='{self.datetime}', close={self.close})>"


class StockPriceProcessed(Base):
    """전처리된 주식 가격 테이블 (기술적 지표 포함)"""
    __tablename__ = 'stock_prices_processed'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(10), nullable=False, index=True)
    stock_name = Column(String(50), nullable=False)
    datetime = Column(DateTime, nullable=False, index=True)
    
    # OHLCV
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(BigInteger, nullable=False)
    
    # 이동평균
    ma_5 = Column(Float)
    ma_10 = Column(Float)
    ma_20 = Column(Float)
    ma_60 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    
    # MACD
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    
    # 볼린저 밴드
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)
    bb_pctb = Column(Float)
    
    # RSI & Stochastic
    rsi = Column(Float)
    stoch_k = Column(Float)
    stoch_d = Column(Float)
    
    # ATR
    atr = Column(Float)
    
    # 거래량 지표
    volume_ma_5 = Column(Float)
    volume_ma_20 = Column(Float)
    volume_ratio = Column(Float)
    obv = Column(Float)
    
    # 수익률
    return_1d = Column(Float)
    log_return = Column(Float)
    return_5d = Column(Float)
    return_10d = Column(Float)
    return_20d = Column(Float)
    hl_ratio = Column(Float)
    co_ratio = Column(Float)
    
    def __repr__(self):
        return f"<StockPriceProcessed(stock_name='{self.stock_name}', datetime='{self.datetime}')>"


class User(Base):
    """웹 프론트엔드용 로그인 사용자 테이블"""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(LargeBinary, nullable=False)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, server_default=func.now(), index=True)


class UserBrokerConfig(Base):
    """유저별 KIS 계좌 설정 (계좌번호, 상품코드 등)"""

    __tablename__ = "user_broker_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    account_no = Column(String(32), nullable=False)  # 예: "12345678"
    account_code = Column(String(8), nullable=False)  # 예: "01"
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class TradeOrder(Base):
    """KIS 주문 로그"""

    __tablename__ = "trade_orders"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    stock_code = Column(String(16), nullable=False, index=True)
    stock_name = Column(String(100))
    side = Column(String(4), nullable=False)  # BUY / SELL
    quantity = Column(Integer, nullable=False)
    order_price = Column(Float)
    order_amount = Column(Float)
    status = Column(String(20), nullable=False)  # REQUESTED / FILLED / REJECTED
    raw_response = Column(Text)  # KIS 응답 전체 JSON 문자열


class AccountSnapshot(Base):
    """계좌 스냅샷 (잔고/평가금액/손익 요약)"""

    __tablename__ = "account_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    total_value = Column(Float)  # 총 평가금액
    cash = Column(Float)  # 예수금
    total_buy_amount = Column(Float)  # 총 매입금액
    total_eval_amount = Column(Float)  # 총 평가금액 (주식 부분)
    total_pnl = Column(Float)  # 총 평가손익
    raw_response = Column(Text)  # 원본 KIS JSON


class RiskSetting(Base):
    """종목/글로벌 리스크 및 포지션 한도 설정"""

    __tablename__ = "risk_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    stock_code = Column(String(16), unique=True, index=True)  # 종목코드, "ALL" 은 전체 공통 설정
    max_position_shares = Column(Integer, nullable=True)  # 종목별 최대 보유 수량
    max_weight_pct = Column(Float, nullable=True)  # 종목 비중 상한 (0.0~1.0)
    max_daily_buy_amount = Column(Float, nullable=True)  # 일간 최대 순매수 금액
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AutoTradeRun(Base):
    """자동매매 실행 로그"""

    __tablename__ = "auto_trade_runs"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    returncode = Column(Integer, nullable=False)
    stdout = Column(Text)
    stderr = Column(Text)


class DatabaseManager:
    """데이터베이스 연결 및 관리"""
    
    def __init__(self):
        """PostgreSQL 연결 초기화"""
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = os.getenv('DB_PORT', '5432')
        self.database = os.getenv('DB_NAME', 'stock_ai')
        self.user = os.getenv('DB_USER', 'postgres')
        self.password = os.getenv('DB_PASSWORD', '')
        
        # 연결 문자열
        self.connection_string = f'postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}'
        
        # 엔진 생성
        self.engine = None
        self.Session = None
        
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.engine = create_engine(self.connection_string, echo=False)
            self.Session = sessionmaker(bind=self.engine)
            print(f"✅ PostgreSQL 연결 성공: {self.database}")
            return True
        except Exception as e:
            print(f"❌ PostgreSQL 연결 실패: {e}")
            return False
    
    def create_tables(self):
        """테이블 생성"""
        try:
            Base.metadata.create_all(self.engine)
            print("✅ 테이블 생성 완료")
            return True
        except Exception as e:
            print(f"❌ 테이블 생성 실패: {e}")
            return False
    
    def drop_tables(self):
        """테이블 삭제 (주의: 모든 데이터 삭제)"""
        try:
            Base.metadata.drop_all(self.engine)
            print("⚠️  모든 테이블 삭제 완료")
            return True
        except Exception as e:
            print(f"❌ 테이블 삭제 실패: {e}")
            return False
    
    def get_session(self):
        """세션 반환"""
        if self.Session is None:
            self.connect()
        return self.Session()
    
    def close(self):
        """연결 종료"""
        if self.engine:
            self.engine.dispose()
            print("✅ 데이터베이스 연결 종료")



