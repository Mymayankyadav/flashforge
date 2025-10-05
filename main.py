from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from typing import List, Optional
import requests
import PyPDF2
import io
from google import genai
from google.genai import types
import os
import json
import tempfile

app = FastAPI(title="PDF Analysis API", description="Extract topic trees and generate flashcards from PDF URLs")

# Initialize Gemini client
client= genai.Client(vertexai=True, 
                     project=os.getenv("GCP_PROJECT_ID"), 
                     location = os.getenv("GCP_REGION")
                )
# Pydantic Models
class TopicNode(BaseModel):
    title: str
    description: str
    start_page: int
    end_page: int
    subtopics: List["TopicNode"] = []

class TopicTreeRequest(BaseModel):
    pdf_url: HttpUrl
    pages: List[int]
    max_depth: int = 3

class TopicTreeResponse(BaseModel):
    topics: List[TopicNode]
    total_pages_processed: int

class Flashcard(BaseModel):
    front: str
    back: str
    source_pages: List[int]

class FlashcardRequest(BaseModel):
    pdf_url: HttpUrl
    topic_title: str
    topic_description: str
    start_page: int
    end_page: int

class FlashcardResponse(BaseModel):
    topic_title: str
    topic_description: str
    flashcards: List[Flashcard]

class LeafFlashcardRequest(BaseModel):
    pdf_url: HttpUrl
    topic_tree: TopicTreeResponse

class LeafFlashcardResponse(BaseModel):
    leaf_nodes: List[FlashcardResponse]

# Utility Functions
def download_pdf_from_url(pdf_url: str) -> bytes:
    """
    Download PDF from HTTPS URL and return as bytes
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(pdf_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Verify it's a PDF
        content_type = response.headers.get('content-type', '').lower()
        if 'pdf' not in content_type:
            # Check first few bytes for PDF signature
            if not response.content.startswith(b'%PDF'):
                raise HTTPException(status_code=400, detail="URL does not point to a valid PDF file")
        
        return response.content
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF from URL: {str(e)}")

def split_pdf_by_pages(pdf_bytes: bytes, target_pages: List[int]) -> bytes:
    """
    Split PDF to keep only specified pages and return as bytes
    Pages are 1-indexed in API, 0-indexed internally
    """
    try:
        pdf_file = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        total_pages = len(pdf_reader.pages)
        
        # Validate page numbers (convert to 0-indexed for processing)
        valid_pages = []
        for page_num in target_pages:
            if 1 <= page_num <= total_pages:
                valid_pages.append(page_num - 1)  # Convert to 0-indexed
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Page {page_num} is out of range. PDF has {total_pages} pages."
                )
        
        if not valid_pages:
            raise HTTPException(status_code=400, detail="No valid pages found in PDF")
        
        pdf_writer = PyPDF2.PdfWriter()
        
        for page_idx in valid_pages:
            pdf_writer.add_page(pdf_reader.pages[page_idx])
        
        # Create new PDF in memory
        output_buffer = io.BytesIO()
        pdf_writer.write(output_buffer)
        output_buffer.seek(0)
        
        return output_buffer.getvalue()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF splitting failed: {str(e)}")

def find_leaf_nodes(topic_tree: List[TopicNode]) -> List[TopicNode]:
    """
    Recursively find all leaf nodes (nodes with no subtopics)
    """
    leaf_nodes = []
    
    def _traverse(node: TopicNode):
        if not node.subtopics:
            leaf_nodes.append(node)
        else:
            for child in node.subtopics:
                _traverse(child)
    
    for node in topic_tree:
        _traverse(node)
        
    return leaf_nodes

# API Endpoints
@app.post("/generate-topic-tree", response_model=TopicTreeResponse)
async def generate_topic_tree(request: TopicTreeRequest):
    """
    Generate a hierarchical topic tree from specified PDF pages using Gemini
    """
    try:
        # Download PDF from URL
        pdf_bytes = download_pdf_from_url(str(request.pdf_url))
        
        # Split PDF to keep only specified pages
        split_pdf_bytes = split_pdf_by_pages(pdf_bytes, request.pages)
        
        # Generate topic tree using Gemini with PDF bytes
        prompt = f"""
        Analyze the PDF content and generate a hierarchical topic tree with up to {request.max_depth} levels of depth.
        
        For each topic, provide:
        - Title: Clear, descriptive topic name
        - Description: 1-2 sentence explanation of the topic content
        - Start page: First page where this topic appears (using 1-indexed page numbers as in the original PDF)
        - End page: Last page where this topic appears
        - Subtopic: Nested sub-topics for deeper hierarchy
        
        The provided PDF contains pages {request.pages} from the original document. 
        Create a structured, organized topic tree that captures the main ideas and their relationships.
        
        Return your response as a valid JSON array of topic objects with this exact structure:
        [
            {{
                "title": "Topic Name",
                "description": "Topic description",
                "start_page": 1,
                "end_page": 3,
                "subtopics": [
                    {{
                        "title": "Subtopic Name",
                        "description": "Subtopic description", 
                        "start_page": 2,
                        "end_page": 3,
                        "subtopics": []
                    }}
                ]
            }}
        ]
        """
        
        # Use Gemini to analyze the PDF bytes directly
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                types.Part.from_bytes(
                    data=split_pdf_bytes,
                    mime_type='application/pdf'
                ),
                prompt
            ],
            config={
                "temperature": 0.3,
                "max_output_tokens": 4000,
            }
        )
        
        # Parse the response
        try:
            # Extract JSON from response (handle potential markdown code blocks)
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]  # Remove ```json
            if response_text.startswith('```'):
                response_text = response_text[3:]  # Remove ```
            if response_text.endswith('```'):
                response_text = response_text[:-3]  # Remove closing ```
                
            topics_data = json.loads(response_text)
            topics = [TopicNode(**topic) for topic in topics_data]
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse AI response as JSON: {str(e)}\nResponse: {response.text}"
            )
        
        return TopicTreeResponse(
            topics=topics,
            total_pages_processed=len(request.pages)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Topic tree generation failed: {str(e)}")

@app.post("/generate-flashcards", response_model=FlashcardResponse)
async def generate_flashcards(request: FlashcardRequest):
    """
    Generate flashcards from a specific topic using its page range
    """
    try:
        # Download PDF from URL
        pdf_bytes = download_pdf_from_url(str(request.pdf_url))
        
        # Create page range for the topic
        page_range = list(range(request.start_page, request.end_page + 1))
        
        # Split PDF for the specific topic page range
        split_pdf_bytes = split_pdf_by_pages(pdf_bytes, page_range)
        
        # Generate flashcards using Gemini
        prompt = f"""
        Based on the PDF content about "{request.topic_title}" - {request.topic_description}, 
        generate a list of educational flashcards.
        
        For each flashcard, provide:
        - Front: A clear question, term, or concept prompt
        - Back: A detailed explanation, definition, or answer
        - Source pages: The specific page numbers where this information appears (from pages {page_range})
        
        Create flashcards that cover:
        - Key concepts and definitions
        - Important facts and details
        - Conceptual relationships
        - Practical applications
        - Theorem and a proof 

        IMPORTANT FORMATTING INSTRUCTIONS:
        - Use **Markdown** for formatting text (bold, italics, lists, headers)
        - Use LaTeX math formatting for mathematical expressions: $equation$ for inline and $$equation$$ for block math
        - Use code blocks for programming concepts
        - Create clear, well-structured flashcards
                                   
        The number of flashcards should be automatically determined based on the content density and importance.
        
        Return your response as a valid JSON array of flashcard objects with this exact structure:
        [
            {{
                "front": "Question or concept",
                "back": "Detailed answer or explanation",
                "source_pages": [1, 2]
            }}
        ]
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                types.Part.from_bytes(
                    data=split_pdf_bytes,
                    mime_type='application/pdf'
                ),
                prompt
            ],
            config={
                "temperature": 0.7,
                "max_output_tokens": 3000,
            }
        )
        
        # Parse flashcards from response
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            flashcards_data = json.loads(response_text)
            flashcards = [Flashcard(**card) for card in flashcards_data]
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse flashcards from AI response: {str(e)}\nResponse: {response.text}"
            )
        
        return FlashcardResponse(
            topic_title=request.topic_title,
            topic_description=request.topic_description,
            flashcards=flashcards
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flashcard generation failed: {str(e)}")

@app.post("/generate-flashcards-from-leaves", response_model=LeafFlashcardResponse)
async def generate_flashcards_from_leaves(request: LeafFlashcardRequest):
    """
    Generate flashcards for all leaf nodes in a topic tree
    """
    try:
        # Download PDF from URL once
        pdf_bytes = download_pdf_from_url(str(request.pdf_url))
        
        # Find all leaf nodes
        leaf_nodes = find_leaf_nodes(request.topic_tree.topics)
        
        if not leaf_nodes:
            raise HTTPException(status_code=400, detail="No leaf nodes found in the topic tree")
        
        results = []
        
        # Generate flashcards for each leaf node
        for leaf in leaf_nodes:
            flashcard_request = FlashcardRequest(
                pdf_url=request.pdf_url,
                topic_title=leaf.title,
                topic_description=leaf.description,
                start_page=leaf.start_page,
                end_page=leaf.end_page
            )
            
            # Use the internal PDF bytes we already downloaded
            page_range = list(range(leaf.start_page, leaf.end_page + 1))
            split_pdf_bytes = split_pdf_by_pages(pdf_bytes, page_range)
            
            # Generate flashcards (simplified version of the flashcard generation logic)
            prompt = f"""
            Generate educational flashcards for: {leaf.title} - {leaf.description}
            
            For each flashcard provide:
            - Front: Question or concept
            - Back: Detailed explanation  
            - Source pages: Specific page numbers from {page_range}
            
            Return as JSON array of {{"front": "", "back": "", "source_pages": []}}
            """
            
            response = client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=[
                    types.Part.from_bytes(
                        data=split_pdf_bytes,
                        mime_type='application/pdf'
                    ),
                    prompt
                ],
                config={"temperature": 0.7, "max_output_tokens": 3000}
            )
            
            try:
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                    
                flashcards_data = json.loads(response_text)
                flashcards = [Flashcard(**card) for card in flashcards_data]
                
                results.append(FlashcardResponse(
                    topic_title=leaf.title,
                    topic_description=leaf.description,
                    flashcards=flashcards
                ))
            except (json.JSONDecodeError, KeyError) as e:
                # Continue with other leaf nodes even if one fails
                print(f"Failed to parse flashcards for {leaf.title}: {str(e)}")
                continue
        
        return LeafFlashcardResponse(leaf_nodes=results)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Leaf flashcard generation failed: {str(e)}")
        
class CustomFlashcardRequest(BaseModel):
    pdf_url: HttpUrl
    source_pages: List[int]
    custom_prompt: str
    topic_title: str = "Custom Flashcards"
    topic_description: str = "Generated from custom prompt"

class CustomFlashcardResponse(BaseModel):
    topic_title: str
    topic_description: str
    custom_prompt: str
    flashcards: List[Flashcard]
    source_pages: List[int]

@app.post("/generate-custom-flashcards", response_model=CustomFlashcardResponse)
async def generate_custom_flashcards(request: CustomFlashcardRequest):
    """
    Generate flashcards using a custom prompt with LaTeX and Markdown formatting support
    """
    try:
        # Download PDF from URL
        pdf_bytes = download_pdf_from_url(str(request.pdf_url))
        
        # Split PDF for the specified source pages
        split_pdf_bytes = split_pdf_by_pages(pdf_bytes, request.source_pages)
        
        # Enhanced prompt with formatting instructions
        enhanced_prompt = f"""
        CUSTOM PROMPT: {request.custom_prompt}
        
        IMPORTANT FORMATTING INSTRUCTIONS:
        - Use **Markdown** for formatting text (bold, italics, lists, headers)
        - Use LaTeX math formatting for mathematical expressions: $equation$ for inline and $$equation$$ for block math
        - Use code blocks for programming concepts
        - Create clear, well-structured flashcards
        
        For each flashcard, provide:
        - Front: Question, concept, or term (formatted with Markdown/LaTeX)
        - Back: Detailed explanation with proper formatting
        - Source pages: Specific page numbers from {request.source_pages}
        
        Return your response as a valid JSON array of flashcard objects with this exact structure:
        [
            {{
                "front": "**Concept Name** with $mathematical$ notation",
                "back": "Detailed explanation with:\n- Bullet points\n- **Bold text**\n- $E = mc^2$\n- ```code blocks```",
                "source_pages": [1, 2]
            }}
        ]
        
        Ensure all mathematical expressions are properly formatted with LaTeX and text uses Markdown for clarity.
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[
                types.Part.from_bytes(
                    data=split_pdf_bytes,
                    mime_type='application/pdf'
                ),
                enhanced_prompt
            ],
            config={
                "temperature": 0.7,
                "max_output_tokens": 4000,
                "response_mime_type": "application/json",
            }
        )
        
        # Parse flashcards from response
        try:
            response_text = response.text.strip()
            # Clean response text
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            flashcards_data = json.loads(response_text)
            flashcards = [Flashcard(**card) for card in flashcards_data]
            
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse custom flashcards from AI response: {str(e)}\nResponse: {response.text}"
            )
        
        return CustomFlashcardResponse(
            topic_title=request.topic_title,
            topic_description=request.topic_description,
            custom_prompt=request.custom_prompt,
            flashcards=flashcards,
            source_pages=request.source_pages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom flashcard generation failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "PDF Analysis API - Use /generate-topic-tree, /generate-flashcards, and /generate-flashcards-from-leaves endpoints"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "PDF Analysis API"}
