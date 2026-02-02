def run_gold_brain():
    # --- INTERNAL HEALTH CHECK ---
    print("🔍 System Check...")
    if not os.environ.get("SUPABASE_URL") or not os.environ.get("SUPABASE_KEY"):
        print("❌ CRITICAL: Environment variables missing!")
        return
    # --- END CHECK ---
    
    supabase = create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))
    # ... rest of your code ...
name: Gold Brain Automation

on:
  schedule:
    - cron: '*/15 * * * *'
  workflow_dispatch:

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

      - name: Run Gold Bot
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
        run: python brain_logic.py
