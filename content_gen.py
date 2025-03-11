from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field, validator
from typing import Optional
import uvicorn
from crewai import Agent, Task, Crew, Process
import os
from crewai_tools import SerperDevTool
import litellm
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()


app = FastAPI(
    title="Educational Content Generator API",
    description="API for generating educational content adapted to different knowledge levels",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")



os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
os.environ["SERPER_API_KEY"] = SERPER_API_KEY

# Initialize search tools
serper_tool = SerperDevTool()

class GeminiLLM:
    """Wrapper for the Gemini model integration with CrewAI"""
    
    def __init__(self, model_name="gemini/gemini-2.0-flash-lite"):
        self.model_name = model_name
    
    def generate(self, messages):
        response = litellm.completion(
            model=self.model_name,
            messages=messages
        )
        return response.choices[0].message.content
        
    def chat(self, messages):
        return self.generate(messages)

# Initialize Gemini LLM
llm = GeminiLLM()

class SimplifiedContentGenerator:
    """Simplified class with just 2 agents to generate educational content"""
    
    def __init__(self):
        self.agents = self._create_agents()
    
    def _create_agents(self):
        """Initialize a team of two specialized agents"""
        
        # Research agent to gather information from the web
        research_agent = Agent(
            role="Educational Researcher",
            goal="Find comprehensive information on educational topics and prepare research materials",
            backstory="""You are an expert researcher who can find reliable information on any topic, 
                      adapting the depth based on the learner's level. You're skilled at gathering 
                      relevant facts, concepts, and examples.""",
            tools=[serper_tool],
            llm=llm,
            verbose=True
        )
        
        # Content creator to synthesize information
        content_creator = Agent(
            role="Content Developer",
            goal="Create well-structured educational content with appropriate visuals and technical details",
            backstory="""You transform research into well-organized educational materials that match 
                      the user's knowledge level. You're skilled at explaining complex topics clearly,
                      identifying necessary visuals, and integrating technical content appropriately.""",
            llm=llm,
            verbose=True
        )
        
        return {
            "researcher": research_agent,
            "content_creator": content_creator
        }
    
    def generate_content(self, topic, level):
        """Generate educational content related to the given topic adapted to user level"""
        
        # Validate user level
        valid_levels = ["beginner", "intermediate", "advanced"]
        if level.lower() not in valid_levels:
            return f"Error: Level must be one of {valid_levels}"
        
        # Task 1: Research the topic
        research_task = Task(
            description=f"""
            Research the topic: '{topic}' thoroughly with focus on a {level} level understanding.
            Your research should include:
            1. Key concepts and definitions at the appropriate {level} level
            2. Important principles and explanations with suitable depth
            3. Examples and applications relevant to {level} learners
            4. Necessary visual elements (diagrams, charts, images) that would help explain this topic
            5. Technical details, formulas, and equations appropriate for {level} level
            6. Reliable sources and references
            """,
            expected_output="""
            A comprehensive research report containing:
            - Key concepts and definitions suited to the user's level
            - Important principles with appropriate depth
            - Examples tailored to the specified level
            - Practical applications with relevant complexity
            - Recommendations for visual elements
            - Technical content with appropriate complexity
            - References to reliable sources
            """,
            agent=self.agents["researcher"]
        )
        
        # Task 2: Create the educational content
        content_task = Task(
            description=f"""
            Create comprehensive educational content about '{topic}' for a {level} learner based on the research.
            The content should:
            1. Have a clear structure with an introduction, main sections, and conclusion
            2. Use language and explanations appropriate for a {level} learner
            3. Include descriptions of recommended visual elements (diagrams, charts, etc.)
            4. Integrate technical content with the right level of complexity
            5. Provide practical examples and applications
            6. End with a summary and suggestions for further learning
            """,
            expected_output="""
            Complete educational content including:
            - Level-appropriate introduction and overview
            - Core concepts with explanations at correct depth
            - Integrated descriptions of visual elements
            - Technical content with appropriate complexity
            - Practical examples tailored to level
            - Summary and suggestions for further learning
            """,
            agent=self.agents["content_creator"],
            context=[research_task]
        )
        
        # Create and execute the crew workflow
        crew = Crew(
            agents=list(self.agents.values()),
            tasks=[research_task, content_task],
            verbose=True,
            process=Process.sequential
        )
        
        # Execute and return results
        result = crew.kickoff()
        
        # Return the final output
        if hasattr(result, 'raw_output'):
            return result.raw_output
        elif hasattr(result, 'output'):
            return result.output
        else:
            return str(result)

# Create a Pydantic model for request validation
class ContentRequest(BaseModel):
    topic: str = Field(..., description="The educational topic to generate content about", min_length=2)
    level: str = Field(..., description="The user's knowledge level", min_length=5)
    
    @validator('level')
    def check_valid_level(cls, v):
        valid_levels = ["beginner", "intermediate", "advanced"]
        if v.lower() not in valid_levels:
            raise ValueError(f"Level must be one of {valid_levels}")
        return v.lower()

class ContentResponse(BaseModel):
    topic: str
    level: str
    content: str

# Initialize the content generator
content_generator = SimplifiedContentGenerator()

@app.get("/")
async def root():
    return {"message": "Welcome to the Educational Content Generator API", 
            "instructions": "Send a POST request to /generate with a JSON body containing 'topic' and 'level'"}

@app.post("/generate", response_model=ContentResponse)
async def generate_content(request: ContentRequest = Body(...)):
    try:
        content = content_generator.generate_content(request.topic, request.level)
        
        return ContentResponse(
            topic=request.topic,
            level=request.level,
            content=content
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating content: {str(e)}"
        )


uvicorn.run(app, host="0.0.0.0", port=8000)
