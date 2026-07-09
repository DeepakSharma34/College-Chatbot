from pydantic import BaseModel, Field, field_validator
from typing import List


class collegeDataPayload(BaseModel):
    text : str = Field (... , min_length=10 , description="The structural text snippet." )
    category : str = Field(... , description="Target Domain Classification")
    source : str = Field(..., description="The explicit URL or filename where this came from.")
    tags : List[str] = Field(default=[] , description="Extra keywords for easier lookups.")

    @field_validator('category')
    @classmethod

    def validate_category(cls , value:str) -> str:
        """ Ensures that every chunk belongs ONLY to our official college groupings. """
        allowed = {"Placements", "Hostel", "Fees", "Syllabus", "Courses", "Academics", "General"}
        if value not in allowed:
            return "General" # Fallback if something gets classified weirdly
        return value