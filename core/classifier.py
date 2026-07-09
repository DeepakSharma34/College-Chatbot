from transformers import pipeline

class collegeChunkClassifier:
    def __init__(self):
        """
        Load a local NLP model capable of reading text and understand topics.
        """

        print(" Loading local Zero-Shot NLP Classifier...")

        self.classifier = pipeline("zero-shot-classification" , model="facebook/bart-large-mnli" , device=-1)
        self.labels = ["placements" ,"Hostel" , "Fees" , "Syllabus" , "Courses" , "Academics"]

    def predict_category(self , chunk_text : str) -> str:
        try:
            prediction = self.classifier(chunk_text , candidate_labels = self.labels)
            highest_label = prediction['labels'][0]
            confidence = prediction['scores'][0]
            
            return highest_label if confidence > 0.35 else "General"
        except Exception:
            return "General"