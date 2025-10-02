from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from google import genai
from google.genai.types import Part
import os
import json
from typing import List, Optional, Dict, Any

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
            model='gemini-2.5-flash',
            contents=[pdf_part, prompt],
            config={
                "temperature": 0.2,
                "top_p": 0.7,
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
        print(request)
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
                print(flashcard_data)
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


class TopicNode(BaseModel):
    topic: str
    description: Optional[str] = None
    page_reference: Optional[List[int]] = None
    subtopics: Optional[List['TopicNode']] = None
    depth: Optional[int] = 0

# Make TopicNode forward reference work
TopicNode.update_forward_refs()

class TopicTreeResponse(BaseModel):
    topic_tree: List[TopicNode]
    total_topics: int
    max_depth: int
    status: str = "success"

class GenerateTopicTreeRequest(BaseModel):
    pdf_uri: HttpUrl
    prompt: Optional[str] = "Break down this document into a hierarchical topic structure"
    max_depth: Optional[int] = 4
    include_page_references: Optional[bool] = True

# ... (keep your existing utility functions and endpoints)

@app.post("/generate-topic-tree/", response_model=TopicTreeResponse)
async def generate_topic_tree(request: GenerateTopicTreeRequest):
    """
    Analyzes a PDF and breaks it down into a hierarchical topic tree structure
    """
    try:
        # Construct the topic tree generation prompt
        topic_tree_prompt = f"""
        Analyze the provided PDF document and create a comprehensive hierarchical topic tree.
        
        USER REQUEST: {request.prompt}
        
        REQUIREMENTS:
        1. Create a hierarchical topic tree with up to {request.max_depth} levels of depth
        2. Each topic should have:
           - A clear, concise topic name
           - A brief description of what the topic covers
           - Page references where the topic is discussed {f"(include page numbers)" if request.include_page_references else ""}
        3. The structure should reflect the document's actual organization
        4. Make the tree comprehensive but not overly detailed
        5. Return ONLY valid JSON format - no additional text
        
        JSON FORMAT:
        {{
            "topics": [
                {{
                    "topic": "Main Topic 1",
                    "description": "Brief description of main topic 1",
                    "page_reference": [1, 2, 3],
                    "subtopics": [
                        {{
                            "topic": "Subtopic 1.1",
                            "description": "Description of subtopic 1.1",
                            "page_reference": [2],
                            "subtopics": [
                                // More nested subtopics up to {request.max_depth} levels
                            ]
                        }}
                    ]
                }}
            ]
        }}
        
        Ensure the response is valid JSON and the structure makes educational sense.
        """
        
        # Get model response with PDF URI
        model_output = generate_with_gemini_and_pdf(topic_tree_prompt, str(request.pdf_uri))
        
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
            tree_data = json.loads(cleaned_output)
            
            # Validate and create TopicNode objects
            if 'topics' in tree_data and isinstance(tree_data['topics'], list):
                topic_tree = []
                total_topics = 0
                max_depth = 0
                
                def build_topic_nodes(nodes, current_depth=1):
                    nonlocal total_topics, max_depth
                    topic_list = []
                    
                    for node in nodes:
                        if isinstance(node, dict) and 'topic' in node:
                            # Update max depth
                            max_depth = max(max_depth, current_depth)
                            total_topics += 1
                            
                            # Build subtopics recursively
                            subtopics = []
                            if 'subtopics' in node and isinstance(node['subtopics'], list):
                                subtopics = build_topic_nodes(node['subtopics'], current_depth + 1)
                            
                            # Create TopicNode
                            topic_node = TopicNode(
                                topic=node['topic'].strip(),
                                description=node.get('description', '').strip(),
                                page_reference=node.get('page_reference', []),
                                subtopics=subtopics,
                                depth=current_depth
                            )
                            topic_list.append(topic_node)
                    
                    return topic_list
                
                topic_tree = build_topic_nodes(tree_data['topics'])
                
                if not topic_tree:
                    raise HTTPException(status_code=500, detail="No valid topic tree generated")
                
                return TopicTreeResponse(
                    topic_tree=topic_tree,
                    total_topics=total_topics,
                    max_depth=max_depth
                )
            else:
                raise HTTPException(status_code=500, detail="Invalid topic tree format in response")
                
        except json.JSONDecodeError as e:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', cleaned_output, re.DOTALL)
            if json_match:
                try:
                    tree_data = json.loads(json_match.group())
                    if 'topics' in tree_data:
                        # Re-process with the extracted JSON
                        topic_tree = []
                        total_topics = 0
                        max_depth = 0
                        
                        def build_topic_nodes(nodes, current_depth=1):
                            nonlocal total_topics, max_depth
                            topic_list = []
                            
                            for node in nodes:
                                if isinstance(node, dict) and 'topic' in node:
                                    max_depth = max(max_depth, current_depth)
                                    total_topics += 1
                                    
                                    subtopics = []
                                    if 'subtopics' in node and isinstance(node['subtopics'], list):
                                        subtopics = build_topic_nodes(node['subtopics'], current_depth + 1)
                                    
                                    topic_node = TopicNode(
                                        topic=node['topic'].strip(),
                                        description=node.get('description', '').strip(),
                                        page_reference=node.get('page_reference', []),
                                        subtopics=subtopics,
                                        depth=current_depth
                                    )
                                    topic_list.append(topic_node)
                            
                            return topic_list
                        
                        topic_tree = build_topic_nodes(tree_data['topics'])
                        
                        if topic_tree:
                            return TopicTreeResponse(
                                topic_tree=topic_tree,
                                total_topics=total_topics,
                                max_depth=max_depth
                            )
                except:
                    pass
            
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse topic tree response as JSON: {str(e)}\nModel output: {model_output}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error in topic tree generation: {str(e)}")

# New Pydantic models for text improvement
class TextImprovementRequest(BaseModel):
    pdf_uri: HttpUrl
    text: str
    improvement_goal: Optional[str] = "Make this text more detailed, accurate, and well-formatted using the PDF content"
    context_hint: Optional[str] = None  # Additional context about what user is trying to say
    output_format: Optional[str] = "markdown"  # markdown, html, or plain_text

class TextImprovementResponse(BaseModel):
    original_text: str
    improved_text: str
    confidence_score: Optional[float] = None  # AI's confidence in understanding the intent
    improvements_made: List[str]  # List of specific improvements
    status: str = "success"

#Enhanced text improvement endpoint with PDF context
@app.post("/improve-text/", response_model=TextImprovementResponse)
async def improve_text_with_pdf(request: TextImprovementRequest):
    """
    Takes vague, incomplete, or concise text and improves it using the PDF context.
    The AI understands the user's intent and completes/expands the text appropriately.
    """
    try:
        # Construct the improvement prompt with PDF context
        improvement_prompt = f"""
        You are an expert editor and content enhancer. The user has provided some text that may be:
        - Vague or incomplete
        - Missing context
        - Too concise
        - Lacking proper formatting
        - Grammatically incorrect
        
        YOUR TASK:
        Understand the user's intent from their text and improve it using the PDF document as reference.
        
        USER'S ORIGINAL TEXT: "{request.text}"
        
        IMPROVEMENT GOAL: {request.improvement_goal}
        {f"CONTEXT HINT: {request.context_hint}" if request.context_hint else ""}
        OUTPUT FORMAT: {request.output_format}
        
        REQUIREMENTS:
        1. FIRST, analyze the PDF content to understand the context and subject matter
        2. THEN, interpret what the user is trying to express in their original text
        3. IMPROVE the text by:
           - Completing incomplete thoughts
           - Adding missing context from the PDF
           - Correcting grammatical errors
           - Expanding concise points with relevant details
           - Adding proper formatting ({request.output_format.upper()} format)
           - Including mathematical expressions in LaTeX ($...$) when needed
        4. Maintain the original intent and core message
        5. Make the text more professional, clear, and comprehensive
        
        IMPORTANT: 
        - Use the PDF content to ensure accuracy and relevance
        - If the original text mentions concepts from the PDF, expand on them using the PDF's information
        - If the text is a question, provide a more complete and well-formed question
        - If the text is a statement, make it more detailed and accurate
        
        RETURN FORMAT:
        Provide a JSON object with:
        {{
            "improved_text": "The enhanced and completed text",
            "confidence_score": 0.95,  # Your confidence in understanding user intent (0-1)
            "improvements_made": [
                "Completed incomplete sentence about X",
                "Added mathematical notation",
                "Expanded concept Y with details from PDF",
                "Corrected grammar and structure"
            ]
        }}
        
        Return ONLY the JSON object, no additional text.
        """
        
        # Get model response with PDF URI
        model_output = generate_with_gemini_and_pdf(improvement_prompt, str(request.pdf_uri))
        
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
            improvement_data = json.loads(cleaned_output)
            
            # Validate response structure
            if ('improved_text' in improvement_data and 
                'confidence_score' in improvement_data and 
                'improvements_made' in improvement_data):
                
                # Validate confidence score
                confidence = improvement_data['confidence_score']
                if not (0 <= confidence <= 1):
                    confidence = 0.8  # Default confidence if invalid
                
                return TextImprovementResponse(
                    original_text=request.text,
                    improved_text=improvement_data['improved_text'].strip(),
                    confidence_score=confidence,
                    improvements_made=improvement_data['improvements_made'],
                    status="success"
                )
            else:
                raise HTTPException(status_code=500, detail="Invalid improvement response format")
                
        except json.JSONDecodeError as e:
            # If JSON parsing fails, try to extract the improved text directly
            # and create a basic response
            improved_text = cleaned_output
            if len(improved_text) > len(request.text) + 10:  # Basic check if improvement happened
                return TextImprovementResponse(
                    original_text=request.text,
                    improved_text=improved_text,
                    confidence_score=0.7,
                    improvements_made=["Enhanced text content and formatting"],
                    status="success"
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to parse improvement response: {str(e)}\nModel output: {model_output}"
                )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error in text improvement: {str(e)}")

# Enhanced version of your original format-content endpoint with PDF support
class ContentFormatRequest(BaseModel):
    content: str
    pdf_uri: HttpUrl  # Added PDF URI for context
    format_type: Optional[str] = "markdown"  # markdown, html, or latex

class ContentFormatResponse(BaseModel):
    original_content: str
    formatted_content: str
    improvements: List[str]
    success: bool
    error: Optional[str] = None

def enhance_math_formatting_with_pdf(content: str, pdf_uri: str) -> str:
    """Apply AI-powered math formatting using PDF context"""
    prompt = f"""
    Convert the following educational content to properly formatted markdown with LaTeX math mode.
    Use the PDF document as reference to understand the context and ensure accuracy.
    
    CONTENT TO FORMAT: "{content}"
    
    FORMATTING RULES:
    1. Wrap mathematical variables and expressions in $...$ for inline math mode
    2. Use proper LaTeX commands: \\in, \\ldots, \\cdots, \\rightarrow, \\frac, etc.
    3. Maintain the original meaning and structure
    4. Correct grammar and improve clarity if needed
    5. Use the PDF context to ensure mathematical symbols and concepts are accurate
    6. Return only the full formatted markdown, no additional text or explanations
    7. Don't cut any part of the original content
    
    Example transformation:
    Input: "The following are equivalent: 1.The matrix of T with respect to v1,ldots,vn is upper triangular 2. span(v1,ldots,vn) is invariant under T for each k=1,ldots,n 3. Tvk in span(v1,ldots,vn)"
    Output: "The following are equivalent:
    1. The matrix of $T$ with respect to $v_1,\\ldots,v_n$ is upper triangular
    2. $\\text{span}(v_1,\\ldots,v_n)$ is invariant under $T$ for each $k=1,\\ldots,n$
    3. $Tv_k \\in \\text{span}(v_1,\\ldots,v_n)$"
    
    Now format this content using the PDF as reference:
    """
    
    try:
        pdf_part = Part.from_uri(
            file_uri=pdf_uri,
            mime_type="application/pdf"
        )
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[pdf_part, prompt],
            config={
                "temperature": 0.3,
                "top_p": 0.8,
                "top_k": 40
            }
        )
        return response.text.strip()
    except Exception as e:
        raise Exception(f"AI formatting failed: {e}")

@app.post("/format-content", response_model=ContentFormatResponse)
async def format_content_with_pdf(request: ContentFormatRequest):
    """Transform plain text content to formatted text with LaTeX math using PDF context"""
    try:
        formatted_content = enhance_math_formatting_with_pdf(request.content, str(request.pdf_uri))
        
        # Analyze what improvements were made
        improvements = []
        if len(formatted_content) > len(request.content):
            improvements.append("Expanded and clarified content")
        if "$" in formatted_content:
            improvements.append("Added mathematical notation")
        if "**" in formatted_content or "#" in formatted_content:
            improvements.append("Added markdown formatting")
        
        return ContentFormatResponse(
            original_content=request.content,
            formatted_content=formatted_content,
            improvements=improvements,
            success=True
        )
        
    except Exception as e:
        return ContentFormatResponse(
            original_content=request.content,
            formatted_content=request.content,  # Return original as fallback
            improvements=[],
            success=False,
            error=str(e)
        )


# Update examples endpoint
@app.get("/examples")
async def get_examples():
    return {
        "generate_flashcards_example": {
            "prompt": "Create flashcards about machine learning concepts",
            "pdf_uri": "https://example.com/document.pdf",
            "num_flashcards": 5
        },
        "improve_text_example": {
            "pdf_uri": "https://example.com/math-textbook.pdf",
            "text": "matrix upper triangular equivalent conditions",
            "improvement_goal": "Expand this into a complete mathematical statement",
            "context_hint": "This is about linear algebra and matrix properties"
        },
        "format_content_example": {
            "content": "The derivative of f(x) = x^2 is 2x and the integral is x^3/3",
            "pdf_uri": "https://example.com/calculus-textbook.pdf",
            "format_type": "markdown"
        },
        "batch_improvement_example": {
            "pdf_uri": "https://example.com/document.pdf",
            "texts": [
                "quantum mechanics basics",
                "schrodinger equation important",
                "wave function probability"
            ],
            "improvement_goal": "Make these into complete study notes"
        }
    }
