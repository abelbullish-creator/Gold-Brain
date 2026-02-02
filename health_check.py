name: Gold Brain Automation

on:
  schedule:
    - cron: '*/15 * * * *'  # Runs every 15 minutes
  workflow_dispatch:       # Enables manual "Run" button

jobs:
  run-logic:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: pip install pandas yfinance supabase requests pytz numpy

      - name: Run Health Check
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
        run: python health_check.py
