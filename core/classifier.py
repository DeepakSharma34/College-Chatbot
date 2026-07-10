import os
from google import genai

class collegeChunkClassifier:
    def __init__(self):
        """
        Lightweight API-based classifier. 
        Uses Gemini Flash instead of heavy local transformer models to stay under 1GB RAM.
        """
        print(" Loading Lightweight API Classifier...")
      
        self.client = genai.Client()
        self.labels = ["placements", "Hostel", "Fees", "Syllabus", "Courses", "Academics"]

    def predict_category(self, chunk_text: str) -> str:
        try:
     
            prompt = (
                f"Analyze the following student query and classify it into exactly one of these "
                f"categories: {self.labels}.\n"
                f"If the query does not clearly fit any of those specific categories, return 'General'.\n"
                f"Return ONLY the exact category name as text, with no extra punctuation or sentences.\n\n"
                f"Query: {chunk_text}"
            )
            
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt
            )
            
            predicted_label = response.text.strip()
            
         
            for label in self.labels:
                if label.lower() in predicted_label.lower():
                    return label
                    
            return "General"
            
        except Exception as e:
            print(f"Classification runtime error: {e}")
            return "General"