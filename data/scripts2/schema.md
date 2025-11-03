
# table formats

The patients table
Table source: CareVue and Metavision ICU databases.

Table purpose: Defines each SUBJECT_ID in the database, i.e. defines a single patient.

Number of rows: 46,520

Links to:

ADMISSIONS on SUBJECT_ID
ICUSTAYS on SUBJECT_ID
Important considerations
DOB has been shifted for patients older than 89. The median age for the patients whose date of birth was shifted is 91.4.
Table columns
Name	Postgres data type
ROW_ID	INT
SUBJECT_ID	INT
GENDER	VARCHAR(5)
DOB	TIMESTAMP(0)
DOD	TIMESTAMP(0)
DOD_HOSP	TIMESTAMP(0)
DOD_SSN	TIMESTAMP(0)
EXPIRE_FLAG	VARCHAR(5)



The admissions table
Table source: Hospital database.

Table purpose: Define a patient’s hospital admission, HADM_ID.

Number of rows: 58976

Links to:

PATIENTS on SUBJECT_ID
Brief summary
The ADMISSIONS table gives information regarding a patient’s admission to the hospital. Since each unique hospital visit for a patient is assigned a unique HADM_ID, the ADMISSIONS table can be considered as a definition table for HADM_ID. Information available includes timing information for admission and discharge, demographic information, the source of the admission, and so on.

Important considerations
The data is sourced from the admission, discharge and transfer database from the hospital (often referred to as ‘ADT’ data).
Organ donor accounts are sometimes created for patients who died in the hospital. These are distinct hospital admissions with very short, sometimes negative lengths of stay. Furthermore, their DEATHTIME is frequently the same as the earlier patient admission’s DEATHTIME.
All text data, except for that in the INSURANCE column, is stored in upper case.
Table columns
Name	Postgres data type
ROW_ID	INT
SUBJECT_ID	INT
HADM_ID	INT
ADMITTIME	TIMESTAMP(0)
DISCHTIME	TIMESTAMP(0)
DEATHTIME	TIMESTAMP(0)
ADMISSION_TYPE	VARCHAR(50)
ADMISSION_LOCATION	VARCHAR(50)
DISCHARGE_LOCATION	VARCHAR(50)
INSURANCE	VARCHAR(255)
LANGUAGE	VARCHAR(10)
RELIGION	VARCHAR(50)
MARITAL_STATUS	VARCHAR(50)
ETHNICITY	VARCHAR(200)
EDREGTIME	TIMESTAMP(0)
EDOUTTIME	TIMESTAMP(0)
DIAGNOSIS	VARCHAR(300)
HOSPITAL_EXPIRE_FLAG	TINYINT
HAS_CHARTEVENTS_DATA	TINYINT


The diagnoses_icd table
Table source: Hospital database.

Table purpose: Contains ICD diagnoses for patients, most notably ICD-9 diagnoses.

Number of rows: 651,047

Links to:

PATIENTS on SUBJECT_ID
ADMISSIONS on HADM_ID
D_ICD_DIAGNOSES on ICD9_CODE
Important considerations
The ICD codes are generated for billing purposes at the end of the hospital stay.
All ICD codes in MIMIC-III are ICD-9 based. The Beth Israel Deaconess Medical Center will begin using ICD-10 codes in 2015.
The code field for the ICD-9-CM Principal and Other Diagnosis Codes is six characters in length, with the decimal point implied between the third and fourth digit for all diagnosis codes other than the V codes. The decimal is implied for V codes between the second and third digit.
Table columns
Name	PostgreSQL data type	Modifiers
ROW_ID	INT	not null
SUBJECT_ID	INT	not null
HADM_ID	INT	not null
SEQ_NUM	INT	
ICD9_CODE	VARCHAR(10)	


The labevents table
Table source: Hospital database.

Table purpose: Contains all laboratory measurements for a given patient, including out patient data.

Number of rows: 27,854,055

Links to:

PATIENTS on SUBJECT_ID
ADMISSIONS on HADM_ID
D_LABITEMS on ITEMID
Brief summary
The LABEVENTS data contains information regarding laboratory based measurements. The process for acquiring a lab measurement is as follows: first, a member of the clinical staff acquires a fluid from a site in the patient’s body (e.g. blood from an arterial line, urine from a catheter, etc). Next, the fluid is bar coded to associate it with the patient and timestamped to record the time of the fluid acquisition. The lab analyses the data and returns a result within 4-12 hours.

Important considerations
Note that the time associated with this result is the time of the fluid acquisition, not the time that the values were made available to the clinical staff.
The labevents table contains both in-hospital laboratory measurements and out of hospital laboratory measurements from clinics which the patient has visited (since the patient is not “in” a hospital when visiting a clinic, these patients often referred to as “outpatients” and the data is often called “outpatient” data). Laboratory measurements for outpatients do not have a HADM_ID.
In MIMIC-III v1.0, there is a subset of patients for which the outpatient lab data is not available. They can be identified by checking for patients whose data always has an HADM_ID.
In MIMIC-III v1.0, there is a subset of patients for which text laboratory data is missing. This primarily affects the blood gas type recorded with blood gases.
Some items are duplicated between the labevents and chartevents tables. In cases where there is disagreement between measurements, labevents should be taken as the ground truth.
Table columns
Name	Postgres data type
ROW_ID	INT
SUBJECT_ID	INT
HADM_ID	INT
ITEMID	INT
CHARTTIME	TIMESTAMP(0)
VALUE	VARCHAR(200)
VALUENUM	DOUBLE PRECISION
VALUEUOM	VARCHAR(20)
FLAG	VARCHAR(20)


The d_items table
Table source: CareVue and Metavision ICU databases.

Table purpose: Definition table for all items in the ICU databases.

Number of rows: 12,487

Links to:

CHARTEVENTS on ITEMID
DATETIMEEVENTS on ITEMID
INPUTEVENTS_CV on ITEMID
INPUTEVENTS_MV on ITEMID
MICROBIOLOGYEVENTS on SPEC_ITEMID, ORG_ITEMID, or AB_ITEMID (for example, use d_items.ITEMID = microbiologyevents.SPEC_ITEMID)
OUTPUTEVENTS on ITEMID
PROCEDUREEVENTS_MV on ITEMID


 d_icd_diagnoses table
Table source: Online sources.

Table purpose: Definition table for ICD diagnoses.

Number of rows: 14,567

Links to:

DIAGNOSES_ICD ON ICD9_CODE
Brief summary
This table defines International Classification of Diseases Version 9 (ICD-9) codes for diagnoses. These codes are assigned at the end of the patient’s stay and are used by the hospital to bill for care provided.

Table columns
Name	Postgres data type
ROW_ID	INT
ICD9_CODE	VARCHAR(10)
SHORT_TITLE	VARCHAR(50)
LONG_TITLE	VARCHAR(300)

The d_labitems table
Table source: Hospital database.

Table purpose: Definition table for all laboratory measurements.

Number of rows: 753

Links to:

LABEVENTS on ITEMID
Important considerations
The ITEMID from MIMIC-III v1.0 does not match the ITEMID from MIMIC-II v2.6. If a mapping between the two is necessary, please contact the guardians of the database.
Many of the LOINC codes were assigned during a project to standardize the ontology of lab measurements in the MIMIC database. Consequently, the codes were assigned post-hoc, and may not be present for every lab measurement. We welcome improvements to the present codes or assignment of LOINC codes to unmapped data elements from the community.
Table columns
Name	Postgres data type
ROW_ID	INT
ITEMID	INT
LABEL	VARCHAR(100)
FLUID	VARCHAR(100)
CATEGORY	VARCHAR(100)
LOINC_CODE	VARCHAR(100)