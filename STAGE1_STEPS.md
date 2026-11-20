# Stage 1 — Data Lead runbook (Windows PowerShell)

Do NOT open any CSV in Excel. Run everything from this folder.

## 1. Repo (one person creates it, adds the other three as collaborators)
GitHub → New repository → name `mp299-fuel-sales` → **Public** → no README.
Settings → Collaborators → add all three other members.

## 2. Python env + DVC
```powershell
cd "C:\Users\YASH BANSAL\Downloads\G119_MP299\mp299-fuel-sales"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 3. git + DVC init, track the data
```powershell
git init -b main
git config user.name  "<your GitHub username>"
git config user.email "<email linked to your GitHub>"
dvc init
dvc add data/raw/train.csv data/raw/later.csv
git status          # train.csv / later.csv must NOT be listed; their .dvc files must be
```

## 4. Check it worked
```powershell
$env:PYTHONPATH="src"; python -m mp299.data_checks
pytest -q           # 4 passed
Get-FileHash data/raw/train.csv   # SHA256 must be 7D5E7FB87A385182FC2A2372E6CD4A5F8B48220316B37160299E895CC8E86711
```

## 5. Commit and push in small pieces (spread across members)
```powershell
git add .dvc .dvcignore data/raw/*.dvc data/raw/.gitignore data/raw/holdout_features.csv
git commit -m "DVC init; track train.csv and later.csv"
git add README.md requirements.txt params.yaml .gitignore
git commit -m "README, pinned requirements, params"
git add src tests
git commit -m "Seven data checks in src/mp299 + tests"
git add reports/findings.yaml
git commit -m "findings.yaml: data_issues filled"
git remote add origin https://github.com/<owner>/mp299-fuel-sales.git
git push -u origin main
```
The other members: `git clone`, set their own `git config user.name/email`,
make a commit (e.g. README tweaks, `explore.py`), and push. **Every member must
have commits under their own account.**

## Done when
`data/raw/train.csv.dvc` and `later.csv.dvc` are visible on GitHub and
`reports/findings.yaml` has the data part filled.

## Stage 1 answers (be ready to explain in the viva)
- sentinel: staff = 999, 66 rows → treat as missing, impute median in Stage 2
- stuck sensor: nearby_pumps = 6 from 2026-04-06 to 2026-04-24 (100 rows) → set to missing / impute
- target typos: 12 rows with litres_per_day ≥ 101,170 → divide by 10
- leak, unit mix-up, duplicates, spelling variants: not present
