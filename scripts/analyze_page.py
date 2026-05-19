import re

with open("scripts/debug_page.html", encoding="utf-8") as f:
    html = f.read()

# Find h1 tags
h1s = re.findall(r"<h1[^>]*>(.*?)</h1>", html, re.DOTALL)
print("=== H1 TAGS ===")
for h in h1s[:5]:
    clean = re.sub(r"<[^>]+>", "", h).strip()
    if clean:
        print("H1:", clean[:150])

# Find class names containing "heading"
print()
print("=== HEADING CLASSES ===")
heading_classes = re.findall(r'class="([^"]*heading[^"]*)"', html)
unique_classes = list(set(heading_classes))[:10]
for c in unique_classes:
    print(" ", c[:100])

# Context around first heading class
print()
print("=== FIRST HEADING CONTEXT ===")
idx = html.find("heading")
if idx > 0:
    snippet = html[max(0, idx-300):idx+500]
    # Strip tags for readability
    clean = re.sub(r"<[^>]+>", " ", snippet)
    clean = re.sub(r"\s+", " ", clean)
    print(clean[:600])

# Look for the name more broadly - search for Tom Doran
print()
print("=== NAME SEARCH ===")
idx2 = html.lower().find("tom doran")
if idx2 > 0:
    snippet2 = html[max(0, idx2-200):idx2+300]
    print(snippet2[:500])
else:
    print("'Tom Doran' not found in page HTML")
    # look for just Tom
    idx3 = html.lower().find("tom ")
    if idx3 > 0:
        print("Found 'Tom' at:", idx3)
        print(html[idx3:idx3+200])
