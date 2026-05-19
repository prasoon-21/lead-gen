import asyncio
import sys
sys.path.insert(0, ".")
import json
import httpx

with open("linkedin_session.json") as f:
    data = json.load(f)

cookies = {c["name"]: c["value"] for c in data.get("cookies", [])}
li_at = cookies.get("li_at", "")

async def test():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    
    async with httpx.AsyncClient(follow_redirects=False, timeout=15) as client:
        # Step 1: Request main page to get fresh server-side cookies
        print("Step 1: Fetching LinkedIn home page to get fresh JSESSIONID...")
        r = await client.get("https://www.linkedin.com", headers=headers)
        
        # Get JSESSIONID from client cookies
        jsessionid = client.cookies.get("JSESSIONID", "")
        print(f"Fresh JSESSIONID from server: {jsessionid}")
        
        # Step 2: Inject user's active li_at cookie
        client.cookies.set("li_at", li_at, domain=".linkedin.com")
        
        # Step 3: Build Voyager headers with csrf-token set to JSESSIONID value
        csrf_token = jsessionid.replace('"', "")
        voyager_headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "application/vnd.linkedin.normalized+json+2.1",
            "Accept-Language": "en-US,en;q=0.9",
            "x-li-lang": "en_US",
            "x-restli-protocol-version": "2.0.0",
            "csrf-token": csrf_token,
            "Referer": "https://www.linkedin.com/",
        }
        
        print("\nStep 4: Querying Voyager API for Tom Doran profile...")
        resp = await client.get(
            "https://www.linkedin.com/voyager/api/identity/profiles/tomdoran",
            headers=voyager_headers
        )
        print(f"Status Code: {resp.status_code}")
        print(f"Headers: {dict(resp.headers)}")
        if resp.status_code == 200:
            print(f"Response: {resp.text[:500]}")
        else:
            print(f"Redirected/Failed: {resp.text[:300]}")

asyncio.run(test())
