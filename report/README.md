# Report

The technical report is written as one Markdown file per section in `report/sections/`. Each section has an owner (see `00_front.md`), and the owner edits only their own files. That way everyone can push straight to `main` without editing the same file.

| File | Section | Owner |
|---|---|---|
| `00_front.md` | Title page, section owners | Dhyan |
| `01_overview.md` | 1. Problem and approach overview | Dhyan |
| `02_eda.md` | 2.2 Dataset understanding and EDA | Dhruva |
| `03_models.md` | 3. Model experimentation and comparison | Dilipan |
| `04_tuning.md` | 4. Bayesian hyperparameter optimisation | Dilipan |
| `05_hmm.md` | 5. HMM and Viterbi route reconstruction | Dhyeya |
| `06_maps.md` | 6. Geographical visualisation (Folium) | Dhruva |
| `07_deployment.md` | 7. API, UI and deployment | Dhyeya |

Screenshots and exported figures go in `report/figures/`. Use one file per figure, prefixed with the section number (e.g. `06_map_trip76.png`).

Other files:
- `results_tables.md`: every result table, auto-generated (`python scripts/make_report_tables.py`).
- `TEAM_GUIDE.md`: ownership, viva questions, demo-day checklist.
- `VIVA_PREP.md`: short answers for the viva topics.
