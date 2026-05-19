import re

def calculate_spam_score(subject: str, body: str) -> dict:
    """
    Calculates a heuristic spam score (0 to 10) for email content.
    Lower score is better (less likely to be spam).
    """
    score = 10.0
    triggers = []

    # 1. Spammy Keywords (Negative impact)
    spam_keywords = [
        r"free", r"winner", r"guaranteed", r"no cost", r"money back",
        r"urgent", r"winner", r"cash", r"prize", r"investment",
        r"act now", r"limited time", r"offer", r"discount", r"save"
    ]
    
    combined_text = (subject + " " + body).lower()
    for kw in spam_keywords:
        matches = re.findall(rf"\b{kw}\b", combined_text)
        if matches:
            count = len(matches)
            deduction = min(0.5 * count, 2.0)
            score -= deduction
            triggers.append(f"Spammy keyword found: '{kw}' ({count}x)")

    # 2. Excessive Capitalization
    if subject.isupper() and len(subject) > 5:
        score -= 2.0
        triggers.append("Subject is all caps")
    
    # 3. Excessive Punctuation
    if "!!!" in subject or "???" in subject:
        score -= 1.0
        triggers.append("Excessive punctuation in subject")

    # 4. Link Density
    links = re.findall(r"http[s]?://", body)
    if len(links) > 3:
        score -= 1.5
        triggers.append(f"Too many links ({len(links)})")
    
    # 5. Short Body
    if len(body.split()) < 20:
        score -= 1.0
        triggers.append("Email body is too short")

    # 6. Unsubscribe Link (Positive impact if present)
    if "unsubscribe" in body.lower() or "opt-out" in body.lower():
        score += 1.0
        triggers.append("Unsubscribe link detected (+1)")

    # Clamp score between 0 and 10
    score = max(0.0, min(10.0, score))
    
    status = "Good"
    if score < 5:
        status = "Spammy"
    elif score < 8:
        status = "Average"

    return {
        "score": round(score, 1),
        "status": status,
        "triggers": triggers
    }
