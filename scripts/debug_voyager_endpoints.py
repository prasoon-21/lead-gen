import asyncio
import sys
sys.path.insert(0, ".")
import json
import httpx

# Load session
with open("linkedin_session.json") as f:
    data = json.load(f)

cookies = {c["name"]: c["value"] for c in data.get("cookies", [])}
li_at = cookies.get("li_at", "")
jsessionid = cookies.get("JSESSIONID", "")

print(f"li_at: {li_at[:20]}...")
print(f"JSESSIONID: {jsessionid}")

async def test():
    # Build standard headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "application/vnd.linkedin.normalized+json+2.1",
        "Accept-Language": "en-US,en;q=0.9",
        "x-li-lang": "en_US",
        "x-restli-protocol-version": "2.0.0",
        "csrf-token": jsessionid.replace('"', ""),
        "Cookie": f"li_at={li_at}; JSESSIONID=\"{jsessionid.replace('\"', '')}\"",
        "Referer": "https://www.linkedin.com/",
    }

    # Let's try to query identity/profiles with multiple variants
    async with httpx.AsyncClient(follow_redirects=True, timeout=10) as client:
        # Try 1: identity/profiles/tomdoran
        print("\nTry 1: voyager/api/identity/profiles/tomdoran")
        resp = await client.get("https://www.linkedin.com/voyager/api/identity/profiles/tomdoran", headers=headers)
        print(f"Status: {resp.status_code}")
        print(f"Response: {resp.text[:300]}")

        # Try 2: Let's see if we can query /voyager/api/me
        print("\nTry 2: voyager/api/me")
        resp2 = await client.get("https://www.linkedin.com/voyager/api/me", headers=headers)
        print(f"Status: {resp2.status_code}")
        print(f"Response: {resp2.text[:300]}")

asyncio.run(test())
