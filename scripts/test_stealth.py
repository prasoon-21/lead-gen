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

        await page.goto("https://www.linkedin.com/in/tomdoran", timeout=30000, wait_until="domcontentloaded")
        await page.wait_for_selector("main", timeout=15000)
        await asyncio.sleep(3)

        # Get name
        name = (await page.locator("main h1").first.text_content(timeout=5000) or "").strip()
        print("Name:", name.encode("ascii", "replace").decode())

        # Get headline - try multiple options
        for sel in [".text-body-medium.break-words", "main .t-16", ".pv-text-details__left-panel .text-body-medium", ".text-body-medium"]:
            try:
                count = await page.locator(sel).count()
                if count > 0:
                    text = (await page.locator(sel).first.text_content(timeout=2000) or "").strip()
                    print(f"Headline via '{sel}': {text.encode('ascii','replace').decode()[:100]}")
                    break
            except Exception:
                pass

        # Scroll down to load more content
        await page.evaluate("window.scrollTo(0, 600)")
        await asyncio.sleep(2)

        # Look for "Contact info" — try different strategies
        print()
        print("=== CONTACT INFO SEARCH ===")
        
        # Method 1: by role
        ci1 = await page.get_by_role("link", name="Contact info").count()
        print(f"By role 'link' name 'Contact info': {ci1}")
        
        # Method 2: by text
        ci2 = await page.get_by_text("Contact info").count()
        print(f"By text 'Contact info': {ci2}")
        
        # Method 3: by partial text
        ci3 = await page.locator("a:has-text('Contact info')").count()
        print(f"By locator a:has-text: {ci3}")
        
        # Method 4: by data attribute
        ci4 = await page.locator("[data-control-name='contact_see_more']").count()
        print(f"By data-control-name: {ci4}")

        # Method 5: search for any link containing "contact"
        all_links = await page.locator("a").all()
        contact_links = []
        for link in all_links:
            try:
                text = (await link.text_content(timeout=500) or "").strip().lower()
                href = await link.get_attribute("href") or ""
                if "contact" in text or "contact" in href.lower():
                    contact_links.append(f"text='{text[:50]}' href='{href[:80]}'")
            except Exception:
                pass
        print(f"Links with 'contact': {contact_links[:5]}")

        # Try scrolling up - Contact info is near the top
        await page.evaluate("window.scrollTo(0, 0)")
        await asyncio.sleep(1)
        ci_after_scroll = await page.get_by_role("link", name="Contact info").count()
        print(f"Contact info after scroll to top: {ci_after_scroll}")

        await browser.close()

asyncio.run(test())
