import pandas as pd
import sqlite3

# ② SQLite DB 경로
sqlite_db = '/home/jihoney/workdir/assist_workdir/coddyddld_workspace/Hackathon.db'

# ④ SQLite 연결
conn = sqlite3.connect(sqlite_db)

# ⑥ JOIN 쿼리 (예: icustays 테이블에서 해당 subject_id/hadm_id 매칭되는 항목 조회)
query = """
SELECT distinct a.subject_id, a.hadm_id, id.stay_id, he.long_title, he.seq_num
FROM admissions AS a
LEFT JOIN icustay_detail AS id
    ON a.hadm_id = id.hadm_id
LEFT JOIN hehe AS he
    ON a.hadm_id = he.hadm_id
WHERE he.seq_num = 1
limit 100;
"""

# ⑦ 결과 가져오기
result_df = pd.read_sql_query(query, conn)

# ⑧ 결과 저장
result_df.to_csv('../filtered/extract2.csv', index=False)
print("✅ extract2.csv 파일 저장 완료!")
