import asyncio
import httpx
import json

with open("linkedin_session.json") as f:
    data = json.load(f)

cookies = {c["name"]: c["value"] for c in data.get("cookies", [])}
li_at = cookies.get("li_at", "")

async def test():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cookie": f"li_at={li_at}",
    }
    
    async with httpx.AsyncClient(follow_redirects=False, timeout=10) as client:
        # Request a simple page like /feed/
        resp = await client.get("https://www.linkedin.com/feed/", headers=headers)
        print(f"Status Code: {resp.status_code}")
        print(f"Location Header: {resp.headers.get('Location', '')}")
        print(f"Set-Cookie Header: {resp.headers.get('Set-Cookie', '')}")

asyncio.run(test())
