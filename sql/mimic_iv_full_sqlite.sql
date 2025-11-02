-- Active: 1753619233545@@127.0.0.1@3306
-- Active: 1753619233545@@127.0.0.1@3306
-- 1. vscode의 extensions에서 sqlite 설치
-- 2. +버튼 눌러서 connect 하기
-- 3. sql active
-- %% 경로 조정 필수 %%

CREATE TABLE admissions (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  admittime TEXT NULL,
  dischtime TEXT NULL,
  deathtime TEXT NULL,
  admission_type TEXT NULL,
  admit_provider_id TEXT NULL,
  admission_location TEXT NULL,
  discharge_location TEXT NULL,
  insurance TEXT NULL,
  language TEXT NULL,
  marital_status TEXT NULL,
  race TEXT NULL,
  edregtime TEXT NULL,
  edouttime TEXT NULL,
  hospital_expire_flag INTEGER NULL
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/hosp/admissions.csv' admissions

CREATE TABLE patients (
  subject_id INTEGER NOT NULL,
  gender TEXT NULL,
  anchor_age INTEGER NULL,
  anchor_year INTEGER NULL,
  anchor_year_group TEXT NULL,
  dod TEXT NULL
);
-- LOAD DATA INFILE removed for SQLite

.mode csv
.import '../mimic-iv-3.1/hosp/patients.csv' patients

CREATE TABLE procedures_icd (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  seq_num INTEGER NOT NULL,
  chartdate TEXT NULL,
  icd_code TEXT NULL,
  icd_version INTEGER NULL
);
-- LOAD DATA INFILE removed for SQLite

.mode csv
.import '../mimic-iv-3.1/hosp/procedures_icd.csv' procedures_icd

CREATE TABLE labevents (
  labevent_id INTEGER NOT NULL,
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NULL,
  specimen_id INTEGER NULL,
  itemid INTEGER NULL,
  order_provider_id TEXT NULL,
  charttime TEXT NULL,
  storetime TEXT NULL,
  value TEXT NULL,
  valuenum REAL NULL,
  valueuom TEXT NULL,
  ref_range_lower REAL NULL,
  ref_range_upper REAL NULL,
  flag TEXT NULL,
  priority TEXT NULL,
  comments TEXT NULL,
  PRIMARY KEY (labevent_id)
);
-- LOAD DATA INFILE removed for SQLite

.mode csv
.import '../mimic-iv-3.1/hosp/labevents.csv' labevents

CREATE TABLE d_labitems (
  itemid INTEGER NOT NULL,
  label TEXT NULL,
  fluid TEXT NULL,
  category TEXT NULL
);
-- LOAD DATA INFILE removed for SQLite

.mode csv
.import '../mimic-iv-3.1/hosp/d_labitems.csv' d_labitems


CREATE TABLE d_icd_procedures (
  icd_code TEXT NOT NULL,
  icd_version INTEGER NOT NULL,
  long_title TEXT NULL
);

.mode csv
.import '../mimic-iv-3.1/hosp/d_icd_procedures.csv' d_icd_procedures

CREATE TABLE d_icd_diagnoses (
  icd_code TEXT NOT NULL,
  icd_version INTEGER NOT NULL,
  long_title TEXT NULL
);

.mode csv
.import '../mimic-iv-3.1/hosp/d_icd_diagnoses.csv' d_icd_diagnoses

CREATE TABLE drgcodes (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  drg_type TEXT NULL,
  drg_code TEXT NULL,
  description TEXT NULL,
  drg_severity INTEGER NULL,
  drg_mortality INTEGER NULL
);

.mode csv
.import '../mimic-iv-3.1/hosp/drgcodes.csv' drgcodes

CREATE TABLE transfers (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NULL,
  transfer_id INTEGER NOT NULL,
  eventtype TEXT NULL,
  careunit TEXT NULL,
  intime TEXT NULL,
  outtime TEXT NULL
);
-- LOAD DATA INFILE removed for SQLite

.mode csv
.import '../mimic-iv-3.1/hosp/transfers.csv' transfers

CREATE TABLE d_hcpcs (
  code TEXT NOT NULL,
  category TEXT NULL,
  long_description TEXT NULL,
  short_description TEXT NULL,
  PRIMARY KEY (code)
);
-- LOAD DATA INFILE removed for SQLite

-- .mode csv
-- .import '../mimic-iv-3.1/hosp/d_hcpcs.csv' d_hcpcs

CREATE TABLE emar (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NULL,
  emar_id  TEXT NOT NULL,
  emar_seq INTEGER NOT NULL,
  poe_id TEXT NULL,
  pharmacy_id INTEGER NULL,
  enter_provider_id TEXT NULL,
  charttime TEXT NULL,
  medication TEXT NULL,
  event_txt TEXT NULL,
  scheduletime TEXT NULL,
  storetime TEXT NULL
);

.mode csv
.import '../mimic-iv-3.1/hosp/emar.csv' emar

CREATE TABLE emar_detail (
  subject_id INTEGER NOT NULL,
  emar_id TEXT NOT NULL,
  emar_seq INTEGER NOT NULL,
  parent_field_ordinal INTEGER NULL,
  administration_type TEXT NULL,
  pharmacy_id INTEGER NULL,
  barcode_type TEXT NULL,
  reason_for_no_barcode TEXT NULL,
  complete_dose_not_given TEXT NULL,
  dose_due TEXT NULL,
  dose_due_unit TEXT NULL,
  dose_given TEXT NULL,
  dose_given_unit TEXT NULL,
  will_remainder_of_dose_be_given TEXT NULL,
  product_amount_given TEXT NULL,
  product_unit TEXT NULL,
  product_code TEXT NULL,
  product_description TEXT NULL,
  product_description_other TEXT NULL,
  prior_infusion_rate TEXT NULL,
  infusion_rate TEXT NULL,
  infusion_rate_adjustment TEXT NULL,
  infusion_rate_adjustment_amount TEXT NULL,
  infusion_rate_unit TEXT NULL,
  route TEXT NULL,
  infusion_complete TEXT NULL,
  completion_interval TEXT NULL,
  new_iv_bag_hung TEXT NULL,
  continued_infusion_in_other_location TEXT NULL,
  restart_interval TEXT NULL,
  side TEXT NULL,
  site TEXT NULL,
  non_formulary_visual_verification TEXT NULL
);
-- LOAD DATA INFILE removed for SQLite

.mode csv
.import '../mimic-iv-3.1/hosp/emar_detail.csv' emar_detail

CREATE TABLE hcpcsevents (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  chartdate TEXT NULL,
  hcpcs_cd TEXT NULL,
  seq_num INTEGER NULL,
  short_description TEXT NULL
);
-- LOAD DATA INFILE removed for SQLite

.mode csv
.import '../mimic-iv-3.1/hosp/hcpcsevents.csv' hcpcsevents

CREATE TABLE microbiologyevents (
  microevent_id INTEGER NOT NULL,
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NULL,
  micro_specimen_id INTEGER NULL,
  order_provider_id TEXT NULL,
  chartdate TEXT NULL,
  charttime TEXT NULL,
  spec_itemid INTEGER NULL,
  spec_type_desc TEXT NULL,
  test_seq INTEGER NULL,
  storedate TEXT NULL,
  storetime TEXT NULL,
  test_itemid INTEGER NULL,
  test_name TEXT NULL,
  org_itemid INTEGER NULL,
  org_name TEXT NULL,
  isolate_num INTEGER NULL,
  quantity TEXT NULL,
  ab_itemid INTEGER NULL,
  ab_name TEXT NULL,
  dilution_text TEXT NULL,
  dilution_comparison TEXT NULL,
  dilution_value REAL NULL,
  interpretation TEXT NULL,
  comments TEXT NULL,
  PRIMARY KEY (microevent_id)
);
-- LOAD DATA INFILE removed for SQLite
-- .mode csv
-- .import '../mimic-iv-3.1/hosp/microbiologyevents.csv' microbiologyevents


CREATE TABLE omr (
  subject_id INTEGER NOT NULL,
  chartdate TEXT NOT NULL,
  seq_num INTEGER NOT NULL,
  result_name TEXT NULL,
  result_value TEXT NULL
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/hosp/omr.csv' omr


CREATE TABLE pharmacy (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  pharmacy_id INTEGER NOT NULL,
  poe_id TEXT NULL,
  starttime TEXT NULL,
  stoptime TEXT NULL,
  medication TEXT NULL,
  proc_type TEXT NULL,
  status TEXT NULL,
  entertime TEXT NULL,
  verifiedtime TEXT NULL,
  route TEXT NULL,
  frequency TEXT NULL,
  disp_sched TEXT NULL,
  infusion_type TEXT NULL,
  sliding_scale TEXT NULL,
  lockout_interval TEXT NULL,
  basal_rate REAL NULL,
  one_hr_max REAL NULL,
  doses_per_24_hrs REAL NULL,
  duration REAL NULL,
  duration_interval TEXT NULL,
  expiration_value REAL NULL,
  expiration_unit TEXT NULL,
  expirationdate TEXT NULL,
  dispensation TEXT NULL,
  fill_quantity TEXT NULL,
  PRIMARY KEY (pharmacy_id)
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/hosp/pharmacy.csv' pharmacy


CREATE TABLE poe (
  poe_id TEXT NOT NULL,
  poe_seq INTEGER NOT NULL,
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  ordertime TEXT NULL,
  order_type TEXT NULL,
  order_subtype TEXT NULL,
  transaction_type TEXT NULL,
  discontinue_of_poe_id TEXT NULL,
  discontinued_by_poe_id TEXT NULL,
  order_provider_id TEXT NULL,
  order_status TEXT NULL
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/hosp/poe.csv' poe


CREATE TABLE poe_detail (
  poe_id TEXT NOT NULL,
  poe_seq INTEGER NOT NULL,
  subject_id INTEGER NOT NULL,
  field_name TEXT NULL,
  field_value TEXT NULL
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/hosp/poe_detail.csv' poe_detail

CREATE TABLE diagnoses_icd (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  seq_num INTEGER NOT NULL,
  icd_code TEXT NULL,
  icd_version INTEGER NULL
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/hosp/diagnoses_icd.csv' diagnoses_icd

CREATE TABLE prescriptions (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  pharmacy_id INTEGER NULL,
  poe_id TEXT NULL,
  poe_seq INTEGER NULL,
  order_provider_id TEXT NULL,
  starttime TEXT NULL,
  stoptime TEXT NULL,
  drug_type TEXT NULL,
  drug TEXT NULL,
  formulary_drug_cd TEXT NULL,
  gsn TEXT NULL,
  ndc TEXT NULL,
  prod_strength TEXT NULL,
  form_rx TEXT NULL,
  dose_val_rx TEXT NULL,
  dose_unit_rx TEXT NULL,
  form_val_disp TEXT NULL,
  form_unit_disp TEXT NULL,
  doses_per_24_hrs REAL NULL,
  route TEXT NULL
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/hosp/prescriptions.csv' prescriptions


CREATE TABLE provider (
  provider_id TEXT NOT NULL,
  PRIMARY KEY (provider_id)
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/hosp/provider.csv' provider


CREATE TABLE services (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  transfertime TEXT NULL,
  prev_service TEXT NULL,
  curr_service TEXT NULL
);
.mode csv
.import '../mimic-iv-3.1/hosp/services.csv' services

CREATE TABLE datetimeevents (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  stay_id INTEGER NOT NULL,
  caregiver_id INTEGER NOT NULL,
  charttime TEXT NOT NULL,
  storetime TEXT NULL,
  itemid INTEGER NOT NULL,
  value TEXT NULL,
  valueuom TEXT NULL,
  warning INTEGER NULL
);
.mode csv
.import '../mimic-iv-3.1/icu/datetimeevents.csv' datetimeevents

CREATE TABLE chartevents (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NULL,
  stay_id INTEGER NULL,
  caregiver_id INTEGER NULL,
  charttime TEXT NOT NULL,
  storetime TEXT NULL,
  itemid INTEGER NOT NULL,
  value TEXT null,
  valuenum double null,
  valueuom TEXT null,
  warning INTEGER null
);
-- LOAD DATA INFILE removed for SQLite

.mode csv
.import '../mimic-iv-3.1/icu/chartevents.csv' chartevents

CREATE TABLE caregiver (
  caregiver_id INTEGER NOT NULL
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/icu/caregiver.csv' caregiver


CREATE TABLE d_items (
  itemid INTEGER NOT NULL,
  label TEXT NULL,
  abbreviation TEXT NULL,
  linksto TEXT NULL,
  category TEXT NULL,
  unitname TEXT NULL,
  param_type TEXT NULL,
  lownormalvalue REAL NULL,
  highnormalvalue REAL NULL
);
.mode csv
.import '../mimic-iv-3.1/icu/d_items.csv' d_items


CREATE TABLE icustays (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  stay_id INTEGER NOT NULL,
  first_careunit TEXT NULL,
  last_careunit TEXT NULL,
  intime TEXT NULL,
  outtime TEXT NULL,
  los REAL NULL
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/icu/icustays.csv' icustays


CREATE TABLE ingredientevents (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  stay_id INTEGER NOT NULL,
  caregiver_id INTEGER NULL,
  starttime TEXT NULL,
  endtime TEXT NULL,
  storetime TEXT NULL,
  itemid INTEGER NOT NULL,
  amount REAL NULL,
  amountuom TEXT NULL,
  rate REAL NULL,
  rateuom TEXT NULL,
  orderid TEXT NULL,
  linkorderid INTEGER NULL,
  statusdescription TEXT NULL,
  originalamount REAL NULL,
  originalrate TEXT NULL
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/icu/ingredientevents.csv' ingredientevents


CREATE TABLE inputevents (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  stay_id INTEGER NOT NULL,
  caregiver_id INTEGER NULL,
  starttime TEXT NULL,
  endtime TEXT NULL,
  storetime TEXT NULL,
  itemid INTEGER NOT NULL,
  amount REAL NULL,
  amountuom TEXT NULL,
  rate REAL NULL,
  rateuom TEXT NULL,
  orderid TEXT NULL,
  linkorderid INTEGER NULL,
  ordercategoryname TEXT NULL,
  secondaryordercategoryname TEXT NULL,
  ordercomponenttypedescription TEXT NULL,
  ordercategorydescription TEXT NULL,
  patientweight REAL NULL,
  totalamount REAL NULL,
  totalamountuom TEXT NULL,
  isopenbag INTEGER NULL,
  continueinnextdept INTEGER NULL,
  statusdescription TEXT NULL,
  originalamount REAL NULL,
  originalrate TEXT NULL
);
-- LOAD DATA INFILE removed for SQLite
.mode csv
.import '../mimic-iv-3.1/icu/inputevents.csv' inputevents


CREATE TABLE outputevents (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  stay_id INTEGER NOT NULL,
  caregiver_id INTEGER NULL,
  charttime TEXT NOT NULL,
  storetime TEXT NULL,
  itemid INTEGER NOT NULL,
  value TEXT NULL,
  valueuom TEXT NULL
);
.mode csv
.import '../mimic-iv-3.1/icu/outputevents.csv' outputevents


CREATE TABLE procedureevents (
  subject_id INTEGER NOT NULL,
  hadm_id INTEGER NOT NULL,
  stay_id INTEGER NOT NULL,
  caregiver_id INTEGER NULL,
  starttime TEXT NULL,
  endtime TEXT NULL,
  storetime TEXT NULL,
  itemid INTEGER NOT NULL,
  value TEXT NULL,
  valueuom TEXT NULL,
  location TEXT NULL,
  locationcategory TEXT NULL,
  orderid TEXT NULL,
  linkorderid INTEGER NULL,
  ordercategoryname TEXT NULL,
  ordercategorydescription TEXT NULL,
  patientweight REAL NULL,
  isopenbag INTEGER NULL,
  continueinnextdept INTEGER NULL,
  statusdescription TEXT NULL,
  originalamount REAL NULL,
  originalrate TEXT NULL
);
.mode csv
.import '../mimic-iv-3.1/icu/procedureevents.csv' procedureevents

