# congpat

## SW시스템설계및개발II "콩콩팥팥"입니다!

# 1. 주제설명

# 2. 용어설명

# 3. 폴더 구조

```plaintext
sw/
├── createcsv/              # CSV 파일 생성 디렉토리
│   ├── create.py           # CSV 파일 데이터 SQL에 저장
│   ├── extract2.py         # SQL 파일 filtering 하여 CSV 파일로 저장
│   └── savecolumn.py       # CSV 파일 데이터 SQL에 저장
│
├── filtered/                     # 전처리 및 통합된 병원 데이터 CSV 파일 저장 디렉토리
│   ├── admissions.csv            # 입원 정보 (입원/퇴원일시, 병실, 사망 여부 등 HOSP 데이터셋)
│   ├── d_icd_diagnoses.csv       # ICD 진단 코드 마스터 (코드별 진단명 및 설명)
│   ├── d_icd_diagnoses_2.csv     # ICD 진단 코드 매핑 확장본 (추가 라벨링/UTI 관련 필터링 버전)
│   ├── d_labitems.csv            # 검사 항목 정의 테이블 (항목명, 단위, 정상범위 등)
│   ├── d_labitems_final.csv      # 학습용으로 정제된 검사 항목 테이블 (중복 및 불필요 변수 제거)
│   ├── labevents.csv             # 환자별 검사 결과 이벤트 데이터 (검사 시각, 측정값, 항목코드 등)
│   ├── patients.csv              # UTI 모델 학습 대상 환자 목록 (선정된 환자의 기본 조건만 포함)
│   ├── patients2.csv             # 위 환자의 전체 세부 정보 (연령, 성별, 병력, 인종 등 상세 메타데이터) *실제 사용 데이터
│   ├── summarized.csv            # UTI 환자 모델 학습용 통합 데이터셋 (입원+검사+진단 정보 통합본)
│   └── transfers.csv             # 병동/ICU 이동 이력 데이터 (입퇴실 시각, 병동 위치, 이동 순서 등)
│
├── mimic-iv-3.1/                    # 공공데이터 기본데이터셋
│
├── model/                                  
│   ├── highperf_best_exp_final.pt          # 학습 완료 모델
│   └── summarized_with_readmit30_test.csv  # test 데이터셋 환자 군
│
├── sql/
│   └── mimic_iv_full_sqlite.sql      # 공공데이터 저장 SQL
│
├── web/
│   └── test_v5.py           # gradio 기반 웹 제작 프로토타입 (여기만 제작하면 됨)
│
└── README.md                # 프로젝트 개요 및 사용법
```
