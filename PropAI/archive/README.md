# Archive Folder

This folder contains deprecated models and documentation that are no longer actively used but are preserved for reference.

## Models (archive/models/)

| Model | Reason for Archive | Last Performance |
|-------|-------------------|------------------|
| model_v16_under.py | Poor performance (40.9%) | Feb 2026 |
| model_v2.py | Superseded by newer versions | - |
| model_v3.py | Superseded by newer versions | - |
| model_v4.py | Superseded by newer versions | - |
| model_v5.py | Superseded by newer versions | - |
| model_v6/ | Superseded by newer versions | - |
| model_v7/ | Superseded by newer versions | - |
| model_v8.py | Superseded by newer versions | - |

## Documentation (archive/documentation/)

| File | Reason for Archive |
|------|-------------------|
| MODEL_V2_SYSTEM.md | Superseded by newer model docs |
| REPORT.md | Old report format |
| REPORT_V2.md | Old report format |
| REPORT_V6.md | Old report format |
| HYBRID_MODEL.md | Integrated into main documentation |
| MODEL_IMPROVEMENTS_SUMMARY.md | Consolidated into 1_model_performance.txt |

## Restoration

If you need to restore any archived item:

```bash
# Restore a model
mv archive/models/model_vX.py src/nba_props/engine/

# Restore documentation
mv archive/documentation/FILE.md documentation/
```

---
Archived: February 4, 2026
