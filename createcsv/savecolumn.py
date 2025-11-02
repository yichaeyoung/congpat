import pandas as pd
import sqlite3

# ① CSV 파일 경로
csv_path = '/home/jihoney/workdir/assist_workdir/coddyddld_workspace/hackathon2025/filtered/hehe.csv'


# ② SQLite DB 경로
sqlite_db = '/home/jihoney/workdir/assist_workdir/coddyddld_workspace/Hackathon.db'

# ③ 테이블 이름 지정 (파일명 기준 자동 생성 가능)
table_name = 'hehe'

# ④ CSV 불러오기
df = pd.read_csv(csv_path)

# ⑤ SQLite 연결
conn = sqlite3.connect(sqlite_db)

# ⑥ SQLite에 저장 (기존 테이블 덮어쓰기)
df.to_sql(table_name, conn, if_exists='replace', index=False)

# ⑦ 연결 종료
conn.close()

print(f"✅ CSV 파일 '{csv_path}' 이(가) SQLite DB '{sqlite_db}' 에 '{table_name}' 테이블로 저장 완료!")
