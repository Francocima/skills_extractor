from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import spacy
from typing import List, Dict, Optional
import re

# Initialize FastAPI app
app = FastAPI(title="Skills Extractor API",
              description="API to extract technical skills from job descriptions",
              version="1.0.0")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # Fallback to a smaller model if the larger one isn't available
    nlp = spacy.blank("en")
    print("Warning: en_core_web_sm model not found. Using blank model instead.")
    print("Install the model with: python -m spacy download en_core_web_sm")

# Define request models
class JobDescriptionItem(BaseModel):
    job_id: str
    text: str

class JobDescriptionBatch(BaseModel):
    job_descriptions: List[JobDescriptionItem]

# Define response models
class SkillsResult(BaseModel):
    job_id: str
    skills: List[str]
    count: int

class BatchSkillsResponse(BaseModel):
    results: List[SkillsResult]
    total_processed: int

# Technical skills dictionary - can be expanded
TECHNICAL_SKILLS = {
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "php", "swift", "golang", "rust", "kotlin", 
    "scala", "perl", "r", "matlab", "bash", "shell", "sql", "nosql", "html", "css", "sass", "less",
    
    # Frameworks and Libraries
    "django", "flask", "fastapi", "spring", "react", "angular", "vue", "node.js", "express", "laravel", "symfony",
    "ruby on rails", "asp.net", "jquery", "bootstrap", "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", 
    "numpy", "matplotlib", "seaborn", "d3.js", "plotly", "selenium", "pytest", "jest", "mocha", "spacy",
    
    # Databases
    "mysql", "postgresql", "mongodb", "oracle", "sql server", "sqlite", "redis", "cassandra", "couchbase", 
    "dynamodb", "firebase", "neo4j", "elasticsearch", "mariadb",
    
    # DevOps and Infrastructure
    "docker", "kubernetes", "jenkins", "github actions", "gitlab ci", "travis ci", "aws", "azure", "gcp", 
    "terraform", "ansible", "chef", "puppet", "vagrant", "prometheus", "grafana", "elk stack", "nginx", "apache",
    "linux", "unix", "windows server", "vmware", "virtualbox",
    
    # Version Control
    "git", "github", "gitlab", "bitbucket", "svn", "mercurial",
    
    # Methodologies
    "agile", "scrum", "kanban", "waterfall", "tdd", "bdd", "ci/cd", "devops", "microservices", "soa", "rest", 
    "graphql", "soap", "grpc",
    
    # Other Tech Skills
    "machine learning", "deep learning", "artificial intelligence", "data science", "big data", "data analysis",
    "data visualization", "etl", "data mining", "nlp", "computer vision", "blockchain", "iot", "ar/vr",
    "cybersecurity", "network security", "cloud computing", "serverless", "web development", "mobile development",
    "responsive design", "ux/ui", "seo", "api design", "system design", "distributed systems", "parallel computing",
    "web scraping", "data engineering", "backend", "frontend", "full stack", "qa automation"
}

# Function to extract skills using spaCy
def extract_skills(text: str) -> List[str]:
    # Process the text with spaCy
    doc = nlp(text.lower())
    
    # Extract potential skills
    extracted_skills = set()
    
    # Check for single word skills
    for token in doc:
        if token.text.lower() in TECHNICAL_SKILLS and len(token.text) > 1:
            extracted_skills.add(token.text.lower())
    
    # Check for multi-word skills
    for skill in TECHNICAL_SKILLS:
        if " " in skill and skill.lower() in text.lower():
            extracted_skills.add(skill.lower())
    
     # Look for potential skills that are capitalized or in all-caps
    # Only use this if the named entity recognition is available
    if nlp.has_pipe("ner"):
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PRODUCT"] and len(ent.text) > 1 and not ent.text.isdigit():
                skill_candidate = ent.text.lower()
                # Check if any word in the entity is a known skill
                for word in skill_candidate.split():
                    if word in TECHNICAL_SKILLS:
                        extracted_skills.add(skill_candidate)
                        break
    
    # Use noun_chunks only if the parser is available
    if nlp.has_pipe("parser"):
        try:
            # Also check for noun chunks that might be skills
            for chunk in doc.noun_chunks:
                chunk_text = chunk.text.lower()
                # Check if any part of the chunk is a known skill
                for skill in TECHNICAL_SKILLS:
                    if skill in chunk_text and len(skill) > 1:
                        extracted_skills.add(skill)
        except Exception as e:
            print(f"Warning: Error while processing noun chunks: {e}")
    else:
        # Alternative approach if noun_chunks is not available: just check for adjacent tokens
        for i in range(len(doc) - 1):
            bigram = f"{doc[i].text} {doc[i+1].text}".lower()
            if bigram in TECHNICAL_SKILLS:
                extracted_skills.add(bigram)
    
    return sorted(list(extracted_skills))
  
# Single job description endpoint (kept for backward compatibility)
@app.post("/extract-skills", response_model=SkillsResult)
async def extract_skills_from_job_description(job_desc: JobDescriptionItem):
    if not job_desc.text:
        raise HTTPException(status_code=400, detail="Job description text cannot be empty")
    
    skills = extract_skills(job_desc.text)
    
    return {
        "job_id": job_desc.job_id,
        "skills": skills,
        "count": len(skills)
    }

# Batch processing endpoint for multiple job descriptions
@app.post("/extract-skills-batch", response_model=BatchSkillsResponse)
async def extract_skills_from_batch(batch: JobDescriptionBatch):
    if not batch.job_descriptions:
        raise HTTPException(status_code=400, detail="Job descriptions list cannot be empty")
    
    results = []
    
    for job_desc in batch.job_descriptions:
        if not job_desc.text:
            # Skip empty job descriptions
            continue
            
        skills = extract_skills(job_desc.text)
        
        results.append({
            "job_id": job_desc.job_id,
            "skills": skills,
            "count": len(skills)
        })
    
    return {
        "results": results,
        "total_processed": len(results)
    }

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Skills Extractor API",
        "endpoints": {
            "/extract-skills": "Process a single job description",
            "/extract-skills-batch": "Process multiple job descriptions at once"
        }
    }

# If you want to run the app directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("skills_extractor:app", host="0.0.0.0", port=8080)
