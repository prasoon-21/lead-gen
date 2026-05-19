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
    # Let's try different formats of CSRF token and JSESSIONID
    permutations = [
        # 1. Standard dummy
        {"csrf": "ajax:0000000000000000000", "cookie_jsession": '"ajax:0000000000000000000"'},
        # 2. No quotes in JSESSIONID cookie
        {"csrf": "ajax:0000000000000000000", "cookie_jsession": 'ajax:0000000000000000000'},
        # 3. Simple random number
        {"csrf": "ajax:1234567890123456789", "cookie_jsession": '"ajax:1234567890123456789"'},
        # 4. Simple random number no quotes
        {"csrf": "ajax:1234567890123456789", "cookie_jsession": 'ajax:1234567890123456789'},
    ]
    
    async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
        for i, p in enumerate(permutations, 1):
            csrf = p["csrf"]
            jsess = p["cookie_jsession"]
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "application/vnd.linkedin.normalized+json+2.1",
                "Accept-Language": "en-US,en;q=0.9",
                "x-li-lang": "en_US",
                "x-restli-protocol-version": "2.0.0",
                "csrf-token": csrf,
                "Cookie": f"li_at={li_at}; JSESSIONID={jsess}",
                "Referer": "https://www.linkedin.com/",
            }
            
            resp = await client.get("https://www.linkedin.com/voyager/api/me", headers=headers)
            print(f"Permutation {i} (csrf={csrf}, JSESSIONID={jsess}):")
            print(f"  Status: {resp.status_code}")
            print(f"  Response: {resp.text[:200]}")
            print()

asyncio.run(test())
