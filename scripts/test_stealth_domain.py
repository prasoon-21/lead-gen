import asyncio
import sys
sys.path.insert(0, ".")
import json

async def run_test_with_domain(domain_val):
    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth

    # Create temporary cookie session file with the specified domain
    session_data = {
      "cookies": [
        {
          "name": "li_at",
          "value": "AQEDAWidpXUFtXPjAAABnj0TdSwAAAGeYR_5LE0AmryJ1PFf-J_MfT47VePdd5D8Lv0dXz6CMnVPQvlF-JJYx35KwxmfBSoL6Jh13Mtorbeu6OUnkSuAFCBR-rEZpRl1Y6TXmiNib0AVPnaNisuTF4sU",
          "domain": domain_val,
          "path": "/",
          "expires": 1810667126.159561,
          "httpOnly": True,
          "secure": True,
          "sameSite": "None"
        },
        {
          "name": "JSESSIONID",
          "value": "ajax:0000000000000000000",
          "domain": domain_val,
          "path": "/",
          "expires": 1810667126.159561,
          "httpOnly": False,
          "secure": True,
          "sameSite": "None"
        }
      ],
      "origins": []
    }
    
    temp_path = f"linkedin_session_{domain_val.strip('.')}.json"
    with open(temp_path, "w") as f:
        json.dump(session_data, f, indent=2)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"]
        )
        ctx = await browser.new_context(
            storage_state=temp_path,
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="en-US",
        )
        page = await ctx.new_page()
        await Stealth().apply_stealth_async(page)
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")

        print(f"Testing with domain {domain_val}...")
        await page.goto("https://www.linkedin.com/in/tomdoran", timeout=30000, wait_until="domcontentloaded")
        
        try:
            await page.wait_for_selector("main h1", timeout=10000)
            print(f"  Success!")
        except Exception:
            print(f"  Failed (main element not found)")
            
        title = await page.title()
        print(f"  Page Title: {title}")
        
        await browser.close()

async def main():
    await run_test_with_domain(".linkedin.com")
    print("-" * 40)
    await run_test_with_domain(".www.linkedin.com")

asyncio.run(main())
