# README Screenshot Plan

Use these exact filenames so I can wire them into the README fast:

1. `assets/screenshots/demo-metrics.png`
   Capture the terminal after running:
   ```bash
   python run_demo.py
   ```
   Keep these lines visible:
   - precision
   - recall
   - f1
   - roc_auc
   - pr_auc
   - confusion matrix

2. `assets/screenshots/amount-distribution.png`
   Use the generated chart already in:
   `artifacts/amount-distribution.png`

3. `assets/screenshots/hourly-fraud-rate.png`
   Use the generated chart already in:
   `artifacts/hourly-fraud-rate.png`

4. `assets/screenshots/top-flagged-transactions.png`
   Capture the lower section of the terminal output from:
   ```bash
   python run_demo.py
   ```
   Keep the top 10 flagged transactions visible.

## Recommended Order In README

1. Project overview
2. Metrics screenshot
3. Amount distribution chart
4. Hourly fraud-rate chart
5. Top flagged transactions screenshot

## Fastest Workflow

1. Create folder `assets/screenshots`
2. Copy the two chart PNGs from `artifacts/`
3. Take the two terminal screenshots
4. Put all four files in `assets/screenshots`
5. Send me the files and I will place them in the README and finish the push
