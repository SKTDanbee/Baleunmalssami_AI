from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError

# 환경 변수 또는 직접 DB_URL 설정
DB_url = "mysql+pymysql://newuser:ansim1234%21@ansim.mysql.database.azure.com:3306/ansim_keypad"

try:
    # SQLAlchemy 엔진 생성
    engine = create_engine(DB_url)
    
    # 연결 테스트: 엔진을 사용하여 데이터베이스에 연결해 봅니다
    with engine.connect() as connection:
        result = connection.execute("SELECT 1")
        print("데이터베이스 연결 성공:", result.fetchone())
except OperationalError as e:
    print("데이터베이스 연결 실패:", e)
except Exception as e:
    print("기타 오류 발생:", e)