from pydantic import BaseModel

class ChatRequest(BaseModel):
    requirements: str

class ChatResponse(BaseModel):
    status: str
    image_url: str
    message: str
