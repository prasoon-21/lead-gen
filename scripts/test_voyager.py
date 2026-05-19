import asyncio
import sys
sys.path.insert(0, ".")
import json

async def test():
    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox", "--disable-dev-shm-usage"]
        )
        ctx = await browser.new_context(
            storage_state="linkedin_session.json",
            viewport={"width": 1920, "height": 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            locale="en-US",
        )
        page = await ctx.new_page()
        await Stealth().apply_stealth_async(page)
        await page.add_init_script("Object.defineProperty(navigator, 'webdriver', { get: () => undefined });")

        # Navigate to linkedin to establish session and get fresh JSESSIONID
        print("Loading LinkedIn feed to get fresh JSESSIONID...")
        await page.goto("https://www.linkedin.com/", timeout=30000, wait_until="domcontentloaded")
        await asyncio.sleep(2)
        print("URL:", page.url)
        print("Title:", await page.title())

        # Get all cookies after the real page load
        cookies = await ctx.cookies()
        cookie_map = {c["name"]: c["value"] for c in cookies}
        
        li_at = cookie_map.get("li_at", "")
        jsessionid = cookie_map.get("JSESSIONID", "")
        
        print(f"\nli_at: {li_at[:30]}...")
        print(f"JSESSIONID: {jsessionid[:40]}")
        print(f"All cookie names: {[c['name'] for c in cookies]}")

        # Save all cookies to session file
        storage = await ctx.storage_state()
        with open("linkedin_session.json", "w") as f:
            json.dump(storage, f, indent=2)
        print("\nSaved full session state to linkedin_session.json")

        # Now test the voyager API with proper CSRF token
        import httpx
        # CSRF token is the JSESSIONID value (without "ajax:" prefix sometimes)
        csrf = jsessionid
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/vnd.linkedin.normalized+json+2.1",
            "Accept-Language": "en-US,en;q=0.9",
            "x-li-lang": "en_US",
            "x-restli-protocol-version": "2.0.0",
            "csrf-token": jsessionid,
            "Cookie": "; ".join(f"{c['name']}={c['value']}" for c in cookies),
            "Referer": "https://www.linkedin.com/",
        }

        print("\nTesting Voyager API with proper CSRF token...")
        async with httpx.AsyncClient(follow_redirects=True, timeout=20) as client:
            resp = await client.get(
                "https://www.linkedin.com/voyager/api/identity/profiles/tomdoran",
                headers=headers
            )
            print(f"Profile API status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"Name: {data.get('firstName', '')} {data.get('lastName', '')}")
                print(f"Headline: {data.get('headline', '')[:100]}")
            else:
                print(f"Response: {resp.text[:200]}")

        await browser.close()

asyncio.run(test())
