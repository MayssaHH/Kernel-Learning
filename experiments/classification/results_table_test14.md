# Test14 Full Results Table

Paper baselines (Table 2 accuracy, %) plus our three architectures from latest `paper_strict` runs per dataset.

| Dataset | EasyMKL | AverageMKL | CKA | SMKL (Paper) | Our Archi+AllRBF | Our Archi+AllLinear | Our Archi+HalfRBFHalfLinear |
|---|---:|---:|---:|---:|---:|---:|---:|
| iris | **100.0** | **100.0** | 96.7 | **100.0** | 90.000 | 86.667 | 93.333 |
| wine | 97.2 | 97.2 | 91.7 | **100.0** | 91.667 | 97.222 | 91.667 |
| breastcancer | 93.0 | 92.1 | 94.7 | **98.3** | 92.982 | 97.368 | 92.982 |
| ionosphere | 73.2 | 74.6 | 85.9 | 93.0 | **95.775** | 85.915 | 91.549 |
| spambase | 90.4 | 87.6 | 81.0 | 90.9 | **92.725** | 90.554 | 90.554 |
| banknote | **100.0** | **100.0** | 85.1 | **100.0** | 92.364 | 87.273 | 92.727 |
| heart | 85.2 | 85.2 | 86.9 | **93.4** | 73.770 | 85.246 | 80.328 |
| haberman | 61.3 | 62.9 | 66.1 | 67.7 | 80.645 | 79.032 | **82.258** |
| mammographic | 80.8 | 79.3 | 75.1 | 84.5 | 83.420 | 83.420 | **85.492** |
| parkinsons | 82.1 | 82.1 | 74.4 | **89.7** | 82.051 | 66.667 | 79.487 |

## Run UUIDs Used (Our 3 Architectures)

| Dataset | all_rbf UUID | all_linear UUID | half_rbf_half_linear UUID |
|---|---|---|---|
| iris | 15999a3d-cf70-4340-a503-f6ed7fb48c38 | b9f1cd2e-bf9a-475e-86e7-63d0508709b0 | c5061488-74e1-4360-9d19-557165aa52bd |
| wine | e8fb945c-2b28-45d4-9e14-62573e33721a | d9b17391-0d63-43ae-9e10-a41dd0c5beff | c92e48d0-c24f-4e2b-9195-6e38f943b660 |
| breastcancer | 23016234-03ec-408e-b00b-0a552590cc40 | b92458c7-2af3-4676-846b-e3ce340ff36e | 561bf808-4700-439c-9536-daa77ddb1676 |
| ionosphere | a9dc72e3-507b-4dd1-9c59-dbcd2391d5d4 | a28a7659-6361-4c00-9a7a-ac9983c653b5 | eee0fe46-a9c4-490a-be8b-bc3ba49bff6d |
| spambase | 25b057ef-d307-4a71-ae59-d58b1b8479e3 | 7058eb56-adcd-4b25-a2d8-cb84140a8617 | f0c31dec-5fd4-4d3e-8c29-b437ad138579 |
| banknote | 19bbe29b-4464-4bab-9fd2-bc6d45c338e7 | 67851847-0606-4033-9939-673954e7aa67 | 65a907eb-95cc-4779-8424-e7bf647e653f |
| heart | c46df3a6-6911-443f-aec8-a43e4c892d61 | 8fb647c2-2323-40ab-a973-2c6f7f426c00 | 0621741b-3511-4620-903f-bcfa13739528 |
| haberman | eb582308-7bbb-430e-aad2-408ed7eea189 | d22d5595-0a9c-4846-96a9-9ea6bcc3e222 | e4e272d4-dca3-4026-a309-85183072c106 |
| mammographic | bf03dc70-eadb-4b23-a4e3-444065cf4619 | 96ca6081-6ee3-4fbf-bf60-cff062ba4384 | 12f7e2ec-e1d5-42c5-b5bd-60fa2bbbac70 |
| parkinsons | 274b39a3-2665-4d35-8423-be0560158d7c | 756be28f-ee68-4ebc-b6d1-f3d9d6e1e5cb | 3649b945-b067-44e9-92ed-cffd977b97ac |

## Notes

- Bold values indicate the highest accuracy in that dataset row (ties are all bolded).
- Paper numbers come from `experiments/classification/paper_smkl_2025_results.json` (Table 2).
- Our numbers come from latest `*_test14_uci_<dataset>/manifest.json` with `config.protocol = paper_strict`.
- Units are test accuracy percentages.
