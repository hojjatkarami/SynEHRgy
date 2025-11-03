
in mimic_path:

ADMISSIONS.csv            DRGCODES.csv             INPUTEVENTS_CV.csv.gz        PROCEDUREEVENTS_MV.csv       lab_short_pre_proc_train.csv
ADMISSIONS.csv.gz         DRGCODES.csv.gz          INPUTEVENTS_MV.csv           PROCEDURES_ICD.csv           lab_short_pre_proc_val.csv
Admissions_processed.csv  D_CPT.csv                INPUTEVENTS_MV.csv.gz        README.md                    lab_short_tensor.csv
CALLOUT.csv               D_CPT.csv.gz             INPUTS_processed.csv         SERVICES.csv                 lab_short_tensor_train.csv
CALLOUT.csv.gz            D_ICD_DIAGNOSES.csv      LABEVENTS.csv                SHA256SUMS.txt               lab_short_tensor_train_HARD.csv
CAREGIVERS.csv            D_ICD_DIAGNOSES.csv.gz   LABEVENTS.csv.gz             TRANSFERS.csv                lab_short_tensor_val.csv
CAREGIVERS.csv.gz         D_ICD_PROCEDURES.csv     LAB_processed.csv            checksum_md5_unzipped.txt    lab_short_tensor_val_HARD.csv
CHARTEVENTS.csv           D_ICD_PROCEDURES.csv.gz  LICENSE.txt                  checksum_md5_zipped.txt      lung
CHARTEVENTS.csv.gz        D_ITEMS.csv              MICROBIOLOGYEVENTS.csv       death_tag_tensor.csv         note_test.pkl
CPTEVENTS.csv             D_ITEMS.csv.gz           NOTEEVENTS.csv               death_tags.csv               note_train.pkl
CPTEVENTS.csv.gz          D_LABITEMS.csv           OUTPUTEVENTS.csv             help.sh                      note_validate.pkl
DATETIMEEVENTS.csv        D_LABITEMS.csv.gz        OUTPUTS_processed.csv        lab_covariates_val.csv
DATETIMEEVENTS.csv.gz     ICUSTAYS.csv             PATIENTS.csv                 lab_events_short.csv
DIAGNOSES_ICD.csv         ICUSTAYS.csv.gz          PRESCRIPTIONS.csv            lab_short_pre_proc.csv
DIAGNOSES_ICD.csv.gz      INPUTEVENTS_CV.csv       PRESCRIPTIONS_processed.csv  lab_short_pre_proc_test.csv

# selected tables
icd_diag: DIAGNOSES_ICD
labs: labevents
covars: patients



# expected output
a hugging face dataset
where each element is a patient with the following features:

problems refer to icd diagnoses
labs refer to lab events
covariates refer to age/gender from patients.csv

features = Features(
    {
        "subject_id": Value("int64"),
        "hadm_ids": Sequence(Value("int64")),
        "table": Sequence(
            ClassLabel(
                num_classes=4,
                names=["covariates", "labs", "problems"],
            )
        ),
        "reced_dt": Sequence(Value("timestamp[us]")), # None for covariates
        "concept_uid": Sequence(Value("int64")), # should be icd_code , itemid for covariate name (age, gender)
        "value_float": Sequence(Value("float32")), # None for problems
    }
)



# create_vocab.py

create a new script create_vocab.py

read the HF dataset

output1: dataframe with the following columns

type,
concept_uid,
concept_name (retrieve from d_icd_diagnoses and d_items ) 
counts,

output 2: a dictionary

key: concept_uids that have value_float
values: {'uniform_quantiles': 'non_uniform_quantiles':}

save in pickle format



