# Raw data: IBM Telco Customer Churn

| Item | Value |
|---|---|
| Dataset | IBM sample data set "Telco Customer Churn" |
| File | `data/raw/telco_churn.csv` |
| Download URL | https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv |
| SHA-256: | 16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91 |
| Size | 970,457 bytes |
| Rows | 7,043 (plus one header line) |
| Columns | 21 |
| Churn rate | 26.54 percent (1,869 customers churned) |
| How to get it | `scripts/download_data.sh` downloads it and verifies the checksum and row count |

The ingest stage refuses to run if the checksum above does not match the file.
The same data is also known as "blastchar/telco-customer-churn" on Kaggle.

## What the data is and is not

* One row per customer of a (US) telecom company: demographics, the services the customer
  has, the contract, billing information, and whether the customer left.
* There is no timestamp. For de-duplication every real row gets the nominal event time
  `2018-01-01T00:00:00Z`; feedback rows are always later, so they win.
* "Service usage" is only the list of subscribed services plus tenure. There are no call
  minutes, data volumes or support-ticket counts.
* The file is a snapshot, so it cannot show how churn changes over time.

## Known quality issues

* `TotalCharges` is a blank text (" ") for 11 customers; all 11 have tenure 0 (brand new).
  The validate stage sets them to 0.0.
* The classes are imbalanced (about 27 percent churn). The project does not re-weight or
  oversample; it tunes the decision threshold and keeps probabilities calibrated.

## Known biases and fairness note

* `gender` and `senior_citizen` are model inputs. Using them can disadvantage a group.
  Fairness is NOT assessed beyond the per-segment churn rates in `reports/eda_summary.json`.
* The data comes from one company and one period, so it may not describe other markets.

## Synthetic and simulated data

* Synthetic training rows (`source = synthetic`) are noisy copies of real training rows. They
  add volume, not information, and are never used in any test set.
* Simulated feedback batches have synthetic labels (`synthetic_labels: true` in the manifest).
  They prove the retraining loop works mechanically; they are not real outcomes.
