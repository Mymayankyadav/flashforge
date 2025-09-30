from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from google import genai
from google.genai.types import Part
import os
import json
from typing import List, Optional

# Initialize the clients
app = FastAPI(title="Flashcard Generator API")
client= genai.Client(vertexai=True, 
                     project=os.getenv("GCP_PROJECT_ID"), 
                     location = os.getenv("GCP_REGION")
                )
# Pydantic models for request/response bodies
class Flashcard(BaseModel):
    front: str
    back: str

class GenerateFlashcardsRequest(BaseModel):
    prompt: str
    pdf_uri: HttpUrl  # URL to the PDF file
    num_flashcards: Optional[int] = 5

class RefinementRequest(BaseModel):
    chat_history: List[dict]
    prompt: str
    pdf_uri: HttpUrl  # URL to the PDF file

class FlashcardResponse(BaseModel):
    flashcards: List[Flashcard]
    status: str = "success"

class RefineFlashcardResponse(BaseModel):
    flashcard: Flashcard
    status: str = "success"

# Utility function to call the Gemini model with PDF URI
def generate_with_gemini_and_pdf(prompt: str, pdf_uri: str) -> str:
    """
    Sends a prompt and PDF URI to the Gemini model and returns the response
    """
    try:
        # Create file part from PDF URI
        pdf_part = Part.from_uri(
            file_uri=pdf_uri,
            mime_type="application/pdf"
        )
        
        # Generate content with PDF and prompt
        response = client.models.generate_content(
            model='gemini-2.5-pro',
            contents=[pdf_part, prompt],
            config={
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling Gemini API: {str(e)}")

@app.post("/generate-flashcards/", response_model=FlashcardResponse)
async def generate_flashcards(request: GenerateFlashcardsRequest):
    """
    Generates a list of flashcards from a given prompt and PDF URI using Gemini's native PDF support
    """
    try:
        # Construct the enhanced prompt for flashcard generation with JSON format requirement
        flashcard_system_prompt = f"""
        You are a helpful assistant that creates educational flashcards.
        Based on the content of the provided PDF, create {request.num_flashcards} flashcards according to this instruction: {request.prompt}
        
        REQUIREMENTS:
        1. Each flashcard must have a 'front' (question, term, or concept) and a 'back' (answer, definition, or explanation)
        2. Format both front and back using Markdown for better readability
        3. For mathematical expressions, use LaTeX within $..$ symbols (e.g., $E = mc^2$) or \[..\] for display mode
        4. Create comprehensive flashcards that cover key concepts from the PDF
        5. Return ONLY a valid JSON array without any additional text or markdown formatting
        6. Ensure flashcards are accurate and based solely on the PDF content
        
        JSON FORMAT:
        [
            {{"front": "Front content 1", "back": "Back content 1"}},
            {{"front": "Front content 2", "back": "Back content 2"}}
        ]
        
        Ensure the response is parseable JSON and contains exactly {request.num_flashcards} flashcards.
        """
        
        # Get model response with PDF URI
        model_output = generate_with_gemini_and_pdf(flashcard_system_prompt, str(request.pdf_uri))
        
        # Parse the JSON response
        try:
            # Clean the response - remove any markdown code blocks
            cleaned_output = model_output.strip()
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[7:]
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[:-3]
            cleaned_output = cleaned_output.strip()
            
            # Parse JSON
            flashcards_data = json.loads(cleaned_output)
            
            # Validate and create Flashcard objects
            flashcards = []
            for item in flashcards_data:
                if isinstance(item, dict) and 'front' in item and 'back' in item:
                    flashcards.append(Flashcard(
                        front=item['front'].strip(),
                        back=item['back'].strip()
                    ))
            
            if not flashcards:
                raise HTTPException(status_code=500, detail="No valid flashcards generated")
            
            # Limit to requested number
            flashcards = flashcards[:request.num_flashcards]
                
            return FlashcardResponse(flashcards=flashcards)
            
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract JSON from the response
            import re
            json_match = re.search(r'\[.*\]', cleaned_output, re.DOTALL)
            if json_match:
                try:
                    flashcards_data = json.loads(json_match.group())
                    flashcards = []
                    for item in flashcards_data:
                        if isinstance(item, dict) and 'front' in item and 'back' in item:
                            flashcards.append(Flashcard(
                                front=item['front'].strip(),
                                back=item['back'].strip()
                            ))
                    if flashcards:
                        return FlashcardResponse(flashcards=flashcards[:request.num_flashcards])
                except:
                    pass
            
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse model response as JSON: {str(e)}\nModel output: {model_output}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.post("/refine-flashcard/", response_model=RefineFlashcardResponse)
async def refine_flashcard(request: RefinementRequest):
    """
    Refines a specific flashcard based on chat history and the PDF context
    """
    try:
        # Construct chat history context
        chat_context = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
            for msg in request.chat_history[-10:]  # Limit to last 10 messages
        ])
        
        # Construct refinement prompt
        refinement_prompt = f"""
        You are a helpful assistant that refines educational flashcards based on chat history and PDF content.
        
        CHAT HISTORY:
        {chat_context}
        
        REFINEMENT REQUEST:
        {request.prompt}
        
        Based on the PDF content and the conversation history above, refine the flashcard.
        
        REQUIREMENTS:
        1. Return a SINGLE JSON object with 'front' and 'back' keys
        2. Use Markdown formatting in both front and back
        3. For mathematical expressions, use LaTeX within $..$ symbols or \[..\] for display mode
        4. Make the flashcard more accurate, comprehensive, and educational based on the PDF content
        5. Return ONLY the JSON object without any additional text
        
        JSON FORMAT:
        {{"front": "Refined front content", "back": "Refined back content"}}
        """
        
        # Get model response
        model_output = generate_with_gemini_and_pdf(refinement_prompt, str(request.pdf_uri))
        
        # Parse the JSON response
        try:
            # Clean the response
            cleaned_output = model_output.strip()
            if cleaned_output.startswith("```json"):
                cleaned_output = cleaned_output[7:]
            if cleaned_output.endswith("```"):
                cleaned_output = cleaned_output[:-3]
            cleaned_output = cleaned_output.strip()
            
            # Parse JSON
            flashcard_data = json.loads(cleaned_output)
            
            if isinstance(flashcard_data, dict) and 'front' in flashcard_data and 'back' in flashcard_data:
                return RefineFlashcardResponse(
                    flashcard=Flashcard(
                        front=flashcard_data['front'].strip(),
                        back=flashcard_data['back'].strip()
                    )
                )
            else:
                raise HTTPException(status_code=500, detail="Invalid flashcard format in response")
                
        except json.JSONDecodeError as e:
            # Try to extract JSON object from response
            import re
            json_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
            if json_match:
                try:
                    flashcard_data = json.loads(json_match.group())
                    if isinstance(flashcard_data, dict) and 'front' in flashcard_data and 'back' in flashcard_data:
                        return RefineFlashcardResponse(
                            flashcard=Flashcard(
                                front=flashcard_data['front'].strip(),
                                back=flashcard_data['back'].strip()
                            )
                        )
                except:
                    pass
            
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse refinement response as JSON: {str(e)}\nModel output: {model_output}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Flashcard Generator API with Gemini PDF URI Support"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "Flashcard Generator API"}

# Example request models for documentation
@app.get("/examples")
async def get_examples():
    return {
        "generate_flashcards_example": {
            "prompt": "Create flashcards about machine learning concepts",
            "pdf_uri": "https://example.com/document.pdf",
            "num_flashcards": 5
        },
        "refine_flashcard_example": {
            "chat_history": [
                {"role": "user", "content": "Make this flashcard more detailed"},
                {"role": "assistant", "content": "Current flashcard content"}
            ],
            "prompt": "Add mathematical formulas to explain the concept better",
            "pdf_uri": "https://example.com/document.pdf"
        }
    }
