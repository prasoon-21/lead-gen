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
    # Use a dummy JSESSIONID
    csrf = "ajax:1234567890123456789"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/vnd.linkedin.normalized+json+2.1",
        "Accept-Language": "en-US,en;q=0.9",
        "x-li-lang": "en_US",
        "x-restli-protocol-version": "2.0.0",
        "csrf-token": csrf,
        "Referer": "https://www.linkedin.com/",
    }
    
    # Pass cookies using the cookies parameter
    client_cookies = {
        "li_at": li_at,
        "JSESSIONID": f'"{csrf}"'
    }
    
    async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
        resp = await client.get("https://www.linkedin.com/voyager/api/me", headers=headers, cookies=client_cookies)
        print("Test 1 (JSESSIONID with quotes):")
        print(f"  Status: {resp.status_code}")
        print(f"  Response: {resp.text[:300]}")
        print()
        
        # Test 2: without quotes in JSESSIONID cookie value
        client_cookies_no_quotes = {
            "li_at": li_at,
            "JSESSIONID": csrf
        }
        resp2 = await client.get("https://www.linkedin.com/voyager/api/me", headers=headers, cookies=client_cookies_no_quotes)
        print("Test 2 (JSESSIONID without quotes):")
        print(f"  Status: {resp2.status_code}")
        print(f"  Response: {resp2.text[:300]}")
        print()

asyncio.run(test())
