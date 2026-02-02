import os
from supabase import create_client

def run_health_check():
    print("🚀 Starting Sentinel Health Check...")
    
    # 1. Credentials Check
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    
    if not url or not key:
        print("❌ ERROR: Missing SUPABASE_URL or SUPABASE_KEY.")
        return

    supabase = create_client(url, key)
    
    # 2. READ Test (Sentinel Memory)
    try:
        print("\n--- Testing Database READ ---")
        mem = supabase.table("sentinel_memory").select("*").eq("id", 1).execute()
        if mem.data:
            print(f"✅ READ SUCCESS: Strategy exists (SMA: {mem.data[0].get('sma_len')})")
        else:
            print("⚠️ READ WARNING: Table exists but is empty.")
    except Exception as e:
        print(f"❌ READ FAILED: {e}")

    # 3. WRITE Test (Gold Prices)
    try:
        print("\n--- Testing Database WRITE ---")
        test_data = {
            "price": 0.0,
            "signal": "HEALTH_CHECK",
            "asset": "SYSTEM",
            "notes": "Testing Service Role Bypass"
        }
        res = supabase.table("gold_prices").insert(test_data).execute()
        
        if res.data:
            print("✅ WRITE SUCCESS: Logged test entry.")
            # Cleanup
            supabase.table("gold_prices").delete().eq("signal", "HEALTH_CHECK").execute()
            print("✅ CLEANUP SUCCESS: Test entry removed.")
    except Exception as e:
        print(f"❌ WRITE FAILED: {e}")
        print("💡 TIP: Verify your 'service_role' key is correct in GitHub Secrets.")

if __name__ == "__main__":
    run_health_check()
