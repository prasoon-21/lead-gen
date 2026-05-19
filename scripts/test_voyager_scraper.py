import asyncio
import sys
sys.path.insert(0, ".")

async def test():
    from core.tools.external.linkedin_research import LinkedInResearchTool
    from core.tools.base import ToolContext
    
    ctx = ToolContext(session_id="test", trace_id="test-trace")
    tool = LinkedInResearchTool()
    
    print("Testing Tom Doran HTTP Voyager Scraper...")
    result = await tool.run({"url": "https://www.linkedin.com/in/tomdoran", "type": "person"}, ctx)
    
    print()
    print("===== VOYAGER API RESULT =====")
    for k, v in result.items():
        val_str = str(v).encode("ascii", "replace").decode("ascii")
        print(f"  {k}: {val_str}")

asyncio.run(test())
