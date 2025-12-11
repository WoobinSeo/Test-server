"""
PostgreSQL 테이블 생성 스크립트
"""
from database import DatabaseManager
import sys


def main():
    """테이블 생성"""
    print("""
    ============================================================
            PostgreSQL 테이블 생성
    ============================================================
    """)
    
    # DB 연결
    db = DatabaseManager()
    
    print("\n데이터베이스 연결 중...")
    if not db.connect():
        print("\n❌ 데이터베이스 연결 실패")
        print("\n다음을 확인하세요:")
        print("1. PostgreSQL 서버가 실행 중인지")
        print("2. .env 파일에 올바른 DB 정보가 입력되어 있는지")
        print("3. 데이터베이스 'stock_ai'가 생성되어 있는지")
        print("\nPostgreSQL에서 데이터베이스 생성:")
        print("  CREATE DATABASE stock_ai;")
        return 1
    
    print("\n생성할 테이블:")
    print("  1. stock_prices - 원본 주가 데이터 (일봉 OHLCV)")
    print("  2. stock_prices_processed - 전처리 데이터 (기술적 지표 포함)")
    
    response = input("\n테이블을 생성하시겠습니까? (y/n): ")
    
    if response.lower() != 'y':
        print("❌ 취소됨")
        return 0
    
    # 테이블 생성
    print("\n테이블 생성 중...")
    if db.create_tables():
        print("\n✅ 테이블 생성 완료!")
        print("\n생성된 테이블:")
        print("  - stock_prices (id, stock_code, stock_name, datetime, open, high, low, close, volume)")
        print("  - stock_prices_processed (위 컬럼 + 37개 기술적 지표)")
        
        print("\n다음 단계:")
        print("  python import_to_db.py  # 데이터 삽입")
    else:
        print("\n❌ 테이블 생성 실패")
        return 1
    
    db.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



