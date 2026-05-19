import asyncio
import sys
sys.path.insert(0, ".")

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

        print("Navigating to Tom Doran profile...")
        await page.goto("https://www.linkedin.com/in/tomdoran", timeout=30000, wait_until="domcontentloaded")
        
        # Wait for page to render
        try:
            await page.wait_for_selector("main h1", timeout=15000)
            print("Main element found!")
        except Exception:
            print("Timeout waiting for main element")
            
        await asyncio.sleep(2)
        
        title = await page.title()
        print(f"Page Title: {title}")
        
        # Check name
        try:
            name = await page.locator("main h1").first.text_content(timeout=5000)
            print(f"Name: {name.strip()}")
        except Exception as e:
            print(f"Failed to get name: {e}")
            
        # Check for Contact info link
        try:
            ci = page.get_by_role("link", name="Contact info")
            ci_count = await ci.count()
            print(f"Contact info links found: {ci_count}")
            if ci_count > 0:
                await ci.first.click()
                await asyncio.sleep(2)
                emails = await page.locator("a[href^='mailto:']").all()
                print(f"Mailto links in contact modal: {len(emails)}")
                for a in emails:
                    href = await a.get_attribute("href") or ""
                    print(f"  Email: {href.replace('mailto:', '')}")
        except Exception as e:
            print(f"Contact info check failed: {e}")

        await browser.close()

asyncio.run(test())
