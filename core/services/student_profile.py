import os
import json
from typing import Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False


class StudentProfileService:
    
    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets.readonly",
        "https://www.googleapis.com/auth/drive.readonly"
    ]
    
    def __init__(
        self,
        credentials_path: Optional[str] = None,
        spreadsheet_id: Optional[str] = None,
        sheet_name: str = "Sheet1"
    ):
        self.credentials_path = credentials_path or os.getenv("GOOGLE_CREDENTIALS_PATH")
        self.credentials_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        self.spreadsheet_id = spreadsheet_id or os.getenv("GOOGLE_SHEETS_STUDENTS_ID") or os.getenv("STUDENT_SHEET_ID")
        self.sheet_name = sheet_name
        self._client = None
        self._sheet = None
    
    def _init_client(self):
        if not GSPREAD_AVAILABLE:
            print("   ⚠️ gspread not available")
            return False
        
        if self._client is not None:
            return True
        
        try:
            creds = None
            
            if self.credentials_json:
                creds_dict = json.loads(self.credentials_json)
                creds = Credentials.from_service_account_info(
                    creds_dict,
                    scopes=self.SCOPES
                )
            elif self.credentials_path and os.path.exists(self.credentials_path):
                creds = Credentials.from_service_account_file(
                    self.credentials_path,
                    scopes=self.SCOPES
                )
            
            if creds:
                self._client = gspread.authorize(creds)
                spreadsheet = self._client.open_by_key(self.spreadsheet_id)
                self._sheet = spreadsheet.worksheet(self.sheet_name)
                return True
            else:
                print("   ⚠️ No Google credentials found")
                return False
                
        except Exception as e:
            print(f"   ⚠️ StudentProfileService init failed: {e}")
        
        return False
    
    async def get_student_profile(self, student_id: str) -> Dict[str, Any]:
        if not self._init_client() or self._sheet is None:
            return self._default_profile(student_id)
        
        try:
            records = self._sheet.get_all_records()
            
            for record in records:
                if str(record.get("studentId", "")) == str(student_id):
                    return self._parse_profile(record)
            
            return self._default_profile(student_id)
            
        except Exception as e:
            print(f"Error fetching student profile: {e}")
            return self._default_profile(student_id)
    
    def _parse_profile(self, record: Dict[str, Any]) -> Dict[str, Any]:
        raw_class = record.get("Class/Course", "10")
        class_number = str(raw_class).replace("Class_", "").replace("Class", "").strip()
        if not class_number.isdigit():
            class_number = "10"
        
        institute_type = record.get("Institute_Type", "School")
        is_school = institute_type.lower() == "school"
        
        profile = {
            "name": record.get("Name", "Student"),
            "email": record.get("Email"),
            "class": f"Class_{class_number}",
            "board": record.get("Board", "CBSE") if is_school else None,
            "medium": record.get("Medium", "English"),
            "institute_type": institute_type,
            "institute_name": record.get("Institute_Name", "N/A"),
            "learning_preferences": record.get("Learning_Preferences") or record.get("Learning_Prefrences", "Standard learning style"),
            "learning_goal": "conceptual understanding",
            "chapter_name": "N/A",
            "topic": "N/A",
            "student_found": True,
        }
        
        if not is_school:
            profile["course"] = record.get("Class/Course")
        
        return profile
    
    def _default_profile(self, student_id: str) -> Dict[str, Any]:
        return {
            "name": "Student",
            "email": None,
            "class": "Class_10",
            "board": "CBSE",
            "medium": "English",
            "institute_type": "School",
            "institute_name": "N/A",
            "learning_preferences": "Standard learning style",
            "learning_goal": "conceptual understanding",
            "chapter_name": "N/A",
            "topic": "N/A",
            "student_found": False,
        }
