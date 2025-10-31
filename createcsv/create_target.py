import pandas as pd
import sqlite3

# ① CSV 파일 경로
target_csv = '../filtered/d_icd_diagnoses.csv'

# ② SQLite DB 경로
sqlite_db = '/home/jihoney/workdir/assist_workdir/coddyddld_workspace/Hackathon.db'

# ③ CSV 파일 불러오기
target_df = pd.read_csv(target_csv)

# ④ SQLite 연결
conn = sqlite3.connect(sqlite_db)

# ⑤ 임시 테이블로 등록 (SQLite의 임시 메모리 테이블로 사용)
target_df.to_sql('temp_target', conn, if_exists='replace', index=False)

print("✅ d_icd_diagnoses.csv sql 저장 완료!")