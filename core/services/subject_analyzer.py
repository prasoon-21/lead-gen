from typing import Dict, Any, Optional, List
import os

from google import genai
from google.genai import types as genai_types

from core.adapters.base import BaseLLMAdapter, LLMResponse


class SubjectAnalyzer:
    
    LITE_MODEL = "gemini-2.5-flash-lite"
    
    VALID_SUBJECTS = [
        "Mathematics", "Science", "Physics", "Chemistry", "Biology",
        "Social Science", "English", "Hindi",
    ]
    
    SUBJECT_NORMALIZATION = {
        "mathematics": "Mathematics", "math": "Mathematics", "maths": "Mathematics",
        "science": "Science", "sci": "Science",
        "physics": "Physics", "phy": "Physics",
        "chemistry": "Chemistry", "chem": "Chemistry",
        "biology": "Biology", "bio": "Biology",
        "social science": "Social Science", "social": "Social Science",
        "history": "Social Science", "geography": "Social Science", "civics": "Social Science",
        "political science": "Social Science", "economics": "Social Science",
        "english": "English", "eng": "English",
        "hindi": "Hindi",
    }
    
    def __init__(self, adapter: BaseLLMAdapter = None):
        self.adapter = adapter
        self._lite_model = None

        if self.adapter is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key:
                self._lite_client = genai.Client(api_key=api_key)
                self._lite_model_name = self.LITE_MODEL
    
    async def classify_subject(
        self,
        query: str,
        current_subject: Optional[str] = None
    ) -> str:
        heuristic = self._heuristic_subject(query)
        if heuristic:
            return self._normalize_subject(heuristic)

        prompt = f"""Question: "{query}"
Current: {current_subject or "N/A"}

→ Subject? ONE word: Mathematics/Science/Physics/Chemistry/Biology/Social Science/English/Hindi

(NCERT syllabus only. Use Current if it matches.)"""
        
        if self.adapter:
            response = await self.adapter.generate(prompt=prompt, temperature=0.1, max_tokens=32)
            text = response.text
        elif hasattr(self, '_lite_client'):
            config = genai_types.GenerateContentConfig(temperature=0.1, max_output_tokens=10)
            response = self._lite_client.models.generate_content(
                model=self._lite_model_name, contents=prompt, config=config
            )
            text = response.text if response else "Mathematics"
        else:
            text = "Mathematics"
        
        subject = text.strip().split()[0] if text else "Mathematics"
        return self._normalize_subject(subject)

    def _heuristic_subject(self, query: str) -> Optional[str]:
        query_lower = query.lower()
        subject_keywords = [
            (
                "Chemistry",
                [
                    "atom", "atomic", "molecule", "molecular", "electron", "proton",
                    "neutron", "periodic", "chemical", "reaction", "acid", "base",
                    "salt", "ionic", "covalent", "bond", "valence", "isotope",
                    "compound", "mixture",
                ],
            ),
            (
                "Physics",
                [
                    "force", "motion", "velocity", "acceleration", "newton",
                    "current", "voltage", "resistance", "magnet", "magnetic",
                    "energy", "power", "work", "light", "optics", "ray",
                    "lens", "mirror", "gravity",
                ],
            ),
            (
                "Biology",
                [
                    "cell", "organism", "photosynthesis", "respiration", "gene",
                    "dna", "enzyme", "plant", "animal", "ecosystem",
                ],
            ),
            (
                "Mathematics",
                [
                    "equation", "algebra", "geometry", "triangle", "circle",
                    "area", "volume", "matrix", "integral", "derivative",
                    "factor", "ratio", "percent", "probability", "statistics",
                ],
            ),
            (
                "English",
                [
                    "grammar", "essay", "poem", "poetry", "novel",
                    "verb", "noun", "adjective", "tense", "comprehension",
                ],
            ),
            (
                "Hindi",
                [
                    "व्याकरण", "कविता", "कहानी", "संज्ञा", "क्रिया", "समास", "उपसर्ग",
                ],
            ),
            (
                "Social Science",
                [
                    "history", "geography", "civics", "economics", "constitution",
                    "democracy", "map", "latitude", "longitude",
                ],
            ),
        ]

        for subject, keywords in subject_keywords:
            for keyword in keywords:
                if keyword in query_lower:
                    return subject

        return None
    
    async def detect_intent(
        self,
        query: str,
        intents: List[str]
    ) -> str:
        intents_str = "/".join(intents)
        prompt = f"""Query: "{query}"

Classify intent. ONE word only: {intents_str}"""
        
        response = await self.adapter.generate(
            prompt=prompt,
            temperature=0.2,
            max_tokens=20
        )
        
        detected = response.text.strip().split()[0] if response.text else intents[0]
        return detected if detected in intents else intents[0]
    
    async def extract_json(
        self,
        text: str,
        schema_description: str
    ) -> Dict[str, Any]:
        prompt = f"""Extract from text: "{text}"

Schema: {schema_description}

Output valid JSON only."""
        
        return await self.adapter.generate_json(prompt=prompt)
    
    async def classify_query_type(self, query: str) -> str:
        exact_patterns = [
            "exercise", "example", "theorem", "question", "problem",
            "solve", "find", "calculate", "prove"
        ]
        
        query_lower = query.lower()
        for pattern in exact_patterns:
            if pattern in query_lower:
                return "exact"
        
        return "conceptual"
    
    def _normalize_subject(self, subject: str) -> str:
        normalized = self.SUBJECT_NORMALIZATION.get(subject.lower(), subject)
        return normalized if normalized in self.VALID_SUBJECTS else "Mathematics"
