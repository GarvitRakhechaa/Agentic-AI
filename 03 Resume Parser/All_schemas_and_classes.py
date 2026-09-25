from openai import BaseModel


class JobD(BaseModel):
    role:str
    required_skills: list[str]
    preferred_skills: list[str]
    minimum_experience: float | None
    education_requirements: list[str]
    responsiblities: list[str]

jobd_schema = JobD.model_json_schema()


class MatchResult(BaseModel):
    score: float 
    details: dict 

match_schema = MatchResult.model_json_schema()

class Experience(BaseModel):
    company: str | None = None
    role: str | None = None 
    duration: str | None = None 
    description: str | None = None
    skills_used: list[str] = []

class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    education: list[str] = []
    projects: list[str] = []
    certifications: list[str] = []

resume_schema = Resume.model_json_schema()
