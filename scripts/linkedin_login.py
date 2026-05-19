import asyncio
import os
import sys
from pathlib import Path

# Add project root to path so we can import if needed
sys.path.append(str(Path(__file__).parent.parent))

try:
    from linkedin_scraper import BrowserManager, wait_for_manual_login
except ImportError:
    print("Error: linkedin-scraper not installed. Please run 'pip install linkedin-scraper'")
    sys.exit(1)

async def create_session():
    session_path = "linkedin_session.json"
    print(f"--- LinkedIn Login Script ---")
    print(f"This script will open a browser for you to log in to LinkedIn.")
    print(f"Once you log in, the session will be saved to {session_path}")
    print(f"------------------------------")
    
    try:
        async with BrowserManager(headless=False) as browser:
            # Navigate to LinkedIn
            await browser.page.goto("https://www.linkedin.com/login")
            
            print("\n[ACTION REQUIRED] Please log in to LinkedIn in the browser window that just opened.")
            print("The script will wait until you have successfully logged in (up to 5 minutes)...")
            
            # Wait for manual login by checking cookies or URL
            logged_in = False
            for i in range(300):
                await asyncio.sleep(1)
                try:
                    cookies = await browser.page.context.cookies()
                    li_at_cookie = next((c for c in cookies if c['name'] == 'li_at'), None)
                    if li_at_cookie and li_at_cookie['value']:
                        logged_in = True
                        print("\n[SUCCESS] Detected active LinkedIn session cookie!")
                        break
                except Exception:
                    pass
                
                if "linkedin.com/feed" in browser.page.url:
                    logged_in = True
                    break
                    
                if i % 10 == 0:
                    print(".", end="", flush=True)
            
            if not logged_in:
                raise TimeoutError("Manual login timeout. Please try again and complete login faster.")
            
            # Save session
            await browser.save_session(session_path)
            print(f"\n[SUCCESS] Session saved to {session_path}!")
            print("You can now use the LinkedIn Research tool.")
            
    except Exception as e:
        print(f"\n[ERROR] Failed to create session: {e}")

if __name__ == "__main__":
    asyncio.run(create_session())
