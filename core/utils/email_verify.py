import smtplib
import socket
import dns.resolver
from email_validator import validate_email, EmailNotValidError

def verify_email_legitimacy(email: str, sender_email: str = "verify@agentic-core.ai") -> dict:
    """
    Verifies if an email address is likely legitimate using:
    1. Syntax validation
    2. DNS MX record lookup
    3. SMTP Handshake (Ping)
    """
    result = {
        "email": email,
        "valid_syntax": False,
        "mx_found": False,
        "smtp_verified": False,
        "status": "invalid",
        "message": ""
    }

    # 1. Syntax Validation
    try:
        valid = validate_email(email, check_deliverability=False)
        email = valid.email
        result["valid_syntax"] = True
    except EmailNotValidError as e:
        result["message"] = str(e)
        return result

    # 2. MX Record Lookup
    domain = email.split('@')[1]
    try:
        records = dns.resolver.resolve(domain, 'MX')
        mx_record = str(records[0].exchange)
        result["mx_found"] = True
    except Exception as e:
        result["message"] = f"No MX records found for {domain}"
        return result

    # 3. SMTP Handshake
    server = None
    try:
        # Get local hostname
        host = socket.gethostname()
        
        # Connect to MX server
        server = smtplib.SMTP(timeout=10)
        server.set_debuglevel(0)
        
        server.connect(mx_record)
        server.helo(host)
        server.mail(sender_email)
        
        # 3a. Check for Catch-all (Optional but recommended)
        # Try a random string first to see if the server accepts anything
        random_email = f"verify_test_{socket.gethostname()[:5]}@{domain}"
        code_rand, _ = server.rcpt(random_email)
        is_catch_all = (code_rand == 250)
        
        # 3b. Check the actual email
        code, message = server.rcpt(email)
        msg_text = message.decode('utf-8', errors='ignore')
        
        if code == 250:
            result["smtp_verified"] = True
            result["status"] = "valid"
            result["message"] = "Email is valid and reachable." if not is_catch_all else "Valid (Catch-all domain)"
        elif code == 550:
            # Outlook/Google often return 550 for reputation blocks
            if "outlook" in mx_record.lower() or "google" in mx_record.lower():
                result["status"] = "risky"
                result["message"] = "Server is protected (Outlook/Google). Mailbox might exist but probe was rejected."
            else:
                result["status"] = "invalid"
                result["message"] = f"Mailbox does not exist (550)."
        else:
            result["status"] = "risky"
            result["message"] = f"Server responded with code {code}: {msg_text}"
            
    except Exception as e:
        result["status"] = "risky"
        result["message"] = f"SMTP Verification failed: {str(e)}"
    finally:
        if server:
            try:
                server.quit()
            except:
                pass

    return result
