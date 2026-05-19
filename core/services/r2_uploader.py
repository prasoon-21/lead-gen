import os
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

try:
    import boto3
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


class R2Uploader:
    
    def __init__(
        self,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        bucket_name: Optional[str] = None,
        public_url: Optional[str] = None
    ):
        self.access_key_id = access_key_id or os.getenv("R2_ACCESS_KEY_ID")
        self.secret_access_key = secret_access_key or os.getenv("R2_SECRET_ACCESS_KEY")
        self.endpoint = endpoint or os.getenv("R2_ENDPOINT")
        self.bucket_name = bucket_name or os.getenv("R2_BUCKET_NAME", "testing")
        self.public_url = public_url or os.getenv("R2_PUBLIC_URL", "https://pub-4130cc99ac744e9ba63a054df383ead7.r2.dev")
        self._client = None
    
    def _init_client(self):
        if not BOTO3_AVAILABLE:
            print("   ⚠️ boto3 not available for R2 upload")
            return False
        
        if self._client is not None:
            return True
        
        if not self.access_key_id or not self.secret_access_key or not self.endpoint:
            print("   ⚠️ R2 credentials not configured")
            return False
        
        try:
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key_id,
                aws_secret_access_key=self.secret_access_key,
                config=Config(signature_version="s3v4")
            )
            return True
        except Exception as e:
            print(f"   ⚠️ R2 client init failed: {e}")
            return False
    
    async def upload(
        self,
        image_bytes: bytes,
        filename: str,
        content_type: str = "image/png"
    ) -> Optional[str]:
        if not self._init_client():
            return None
        
        if not filename.endswith(".png"):
            filename = f"{filename}.png"
        
        try:
            print(f"   📤 Uploading to R2: {filename}")
            
            self._client.put_object(
                Bucket=self.bucket_name,
                Key=filename,
                Body=image_bytes,
                ContentType=content_type
            )
            
            public_url = f"{self.public_url.rstrip('/')}/{filename}"
            print(f"   ✅ Uploaded: {public_url}")
            
            return public_url
            
        except Exception as e:
            print(f"   ⚠️ R2 upload failed: {e}")
            return None
