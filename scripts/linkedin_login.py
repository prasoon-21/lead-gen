import asyncio
import os
import sys
from pathlib import Path

# Add project root to path so we can import if needed
sys.path.append(str(Path(__file__).parent.parent))

try:
    from linkedin_scraper import BrowserManager
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
            await browser.page.goto("https://www.linkedin.com/login", wait_until="domcontentloaded")
            
            print("\n[ACTION REQUIRED] Please log in to LinkedIn in the browser window that just opened.")
            print("The script will wait until you have successfully logged in (up to 5 minutes)...")
            
            try:
                # Wait for navigation to the feed page, which indicates a successful login.
                # This is more efficient than polling in a loop.
                await browser.page.wait_for_url("**/feed/**", timeout=300_000) # 5 minutes
                print("\n[SUCCESS] Detected successful login.")

            except Exception:
                # Fallback check for the session cookie if URL navigation isn't detected
                print("\nLogin URL change not detected, checking for session cookie as a fallback...")
                cookies = await browser.page.context.cookies()
                li_at_cookie = next((c for c in cookies if c['name'] == 'li_at'), None)
                if not (li_at_cookie and li_at_cookie['value']):
                    raise TimeoutError("Manual login timed out or the session cookie could not be detected. Please try again.")
                print("[SUCCESS] Detected active LinkedIn session cookie!")
            
            # Save session
            await browser.save_session(session_path)
            print(f"\n[SUCCESS] Session saved to {session_path}!")
            print("You can now use the LinkedIn Research tool.")
            
    except Exception as e:
        print(f"\n[ERROR] Failed to create session: {e}")

if __name__ == "__main__":
    asyncio.run(create_session())
