from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict, Any
import requests
import PyPDF2
import io
from google import genai
from google.genai import types
import os
import json
import re

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
    source_pages: List[int]
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
    source_pages: List[int]
    input_prompt: str
    max_cards: int=5

class FlashcardResponse(BaseModel):
    topic_title: str
    topic_description: str
    flashcards: List[Flashcard]

class LeafFlashcardRequest(BaseModel):
    pdf_url: HttpUrl
    topic_tree: TopicTreeResponse

class LeafFlashcardResponse(BaseModel):
    leaf_nodes: List[FlashcardResponse]

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

def map_ai_page_references(ai_page_numbers: List[int], original_source_pages: List[int]) -> List[int]:
    """
    Map AI's page references (based on split PDF) back to original PDF page numbers
    """
    mapped_pages = []
    for ai_page in ai_page_numbers:
        # AI pages are 1-indexed relative to the split PDF
        if 1 <= ai_page <= len(original_source_pages):
            mapped_pages.append(original_source_pages[ai_page - 1])
        else:
            # If AI references a page outside split PDF range, use the closest valid page
            if ai_page < 1:
                mapped_pages.append(original_source_pages[0])
            else:
                mapped_pages.append(original_source_pages[-1])
    return mapped_pages

def parse_json_with_latex(json_string: str) -> List[dict]:
    """
    Parse JSON string that may contain LaTeX with unescaped backslashes
    """
    try:
        # First try direct JSON parsing
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # If that fails, try to fix common LaTeX backslash issues
        try:
            # Replace common LaTeX patterns with properly escaped versions
            fixed_json = json_string
            
            # Fix unescaped backslashes in common LaTeX commands
            latex_patterns = [
                (r'\\frac', r'\\\\frac'),
                (r'\\sqrt', r'\\\\sqrt'),
                (r'\\alpha', r'\\\\alpha'),
                (r'\\beta', r'\\\\beta'),
                (r'\\gamma', r'\\\\gamma'),
                (r'\\delta', r'\\\\delta'),
                (r'\\le', r'\\\\le'),
                (r'\\ge', r'\\\\ge'),
                (r'\\neq', r'\\\\neq'),
                (r'\\approx', r'\\\\approx'),
                (r'\\equiv', r'\\\\equiv'),
                (r'\\pm', r'\\\\pm'),
                (r'\\mp', r'\\\\mp'),
                (r'\\times', r'\\\\times'),
                (r'\\div', r'\\\\div'),
                (r'\\cdot', r'\\\\cdot'),
                (r'\\circ', r'\\\\circ'),
                (r'\\infty', r'\\\\infty'),
                (r'\\partial', r'\\\\partial'),
                (r'\\nabla', r'\\\\nabla'),
                (r'\\forall', r'\\\\forall'),
                (r'\\exists', r'\\\\exists'),
                (r'\\in', r'\\\\in'),
                (r'\\notin', r'\\\\notin'),
                (r'\\subset', r'\\\\subset'),
                (r'\\subseteq', r'\\\\subseteq'),
                (r'\\supset', r'\\\\supset'),
                (r'\\supseteq', r'\\\\supseteq'),
                (r'\\cap', r'\\\\cap'),
                (r'\\cup', r'\\\\cup'),
                (r'\\emptyset', r'\\\\emptyset'),
                (r'\\mathbb', r'\\\\mathbb'),
                (r'\\mathbf', r'\\\\mathbf'),
                (r'\\mathcal', r'\\\\mathcal'),
                (r'\\mathfrak', r'\\\\mathfrak'),
                (r'\\mathscr', r'\\\\mathscr'),
                (r'\\mathrm', r'\\\\mathrm'),
                (r'\\mathsf', r'\\\\mathsf'),
                (r'\\mathtt', r'\\\\mathtt'),
                (r'\\mathit', r'\\\\mathit'),
                (r'\\textbf', r'\\\\textbf'),
                (r'\\textit', r'\\\\textit'),
                (r'\\text', r'\\\\text'),
                (r'\\overline', r'\\\\overline'),
                (r'\\underline', r'\\\\underline'),
                (r'\\widehat', r'\\\\widehat'),
                (r'\\widetilde', r'\\\\widetilde'),
                (r'\\overrightarrow', r'\\\\overrightarrow'),
                (r'\\overleftarrow', r'\\\\overleftarrow'),
                (r'\\dot', r'\\\\dot'),
                (r'\\ddot', r'\\\\ddot'),
                (r'\\bar', r'\\\\bar'),
                (r'\\vec', r'\\\\vec'),
                (r'\\hat', r'\\\\hat'),
                (r'\\tilde', r'\\\\tilde'),
            ]
            
            for pattern, replacement in latex_patterns:
                fixed_json = re.sub(pattern, replacement, fixed_json)
            
            return json.loads(fixed_json)
            
        except json.JSONDecodeError as e2:
            # If fixing doesn't work, try a more aggressive approach
            try:
                # Remove all backslashes and hope for the best
                fixed_json = json_string.replace('\\', '')
                return json.loads(fixed_json)
            except json.JSONDecodeError as e3:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to parse JSON even after fixing LaTeX: {str(e3)}"
                )

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
        
        IMPORTANT PAGE NUMBERING:
        - The provided PDF contains pages {request.pages} from the original document.
        - When citing page numbers, ALWAYS use the original PDF page numbers: {request.pages}
        - The first page of this split PDF corresponds to original page {request.pages[0]}
        - The last page corresponds to original page {request.pages[-1]}
        
        For each topic, provide:
        - Title: Clear, descriptive topic name
        - Description: 1-2 sentence explanation of the topic content
        - Source pages: List of original page numbers where this topic appears (from {request.pages})
        - Subtopic: Nested sub-topics for deeper hierarchy
        - Make sure that topic is covered in the pdf with some details not just casually mentioned
        
        Create a structured, organized topic tree that captures the main ideas and their relationships.
        
        Return your response as a valid JSON array of topic objects with this exact structure:
        [
            {{
                "title": "Topic Name",
                "description": "Topic description",
                "source_pages": [1, 2, 3],
                "subtopics": [
                    {{
                        "title": "Subtopic Name",
                        "description": "Subtopic description", 
                        "source_pages": [2, 3],
                        "subtopics": []
                    }}
                ]
            }}
        ]
        """
        
        # Use Gemini to analyze the PDF bytes directly
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(
                    data=split_pdf_bytes,
                    mime_type='application/pdf'
                ),
                prompt
            ],
            config={
                "temperature": 0.7
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
    Generate flashcards from a specific topic using its source pages
    """
    try:
        # Download PDF from URL
        pdf_bytes = download_pdf_from_url(str(request.pdf_url))
        
        # Split PDF for the specific topic source pages
        split_pdf_bytes = split_pdf_by_pages(pdf_bytes, request.source_pages)
        
        # Generate flashcards using Gemini with explicit JSON formatting instructions
        prompt = f"""
        Based on the PDF content about "{request.topic_title}" - {request.topic_description}, 
        generate a list of educational flashcards.
        this the is the user request: {request.input_prompt} make sure you keep this request in mind. 
        
        IMPORTANT PAGE NUMBERING:
        - The provided PDF contains pages {request.source_pages} from the original document.
        - When citing source pages, ALWAYS use the original PDF page numbers: {request.source_pages}
        - The first page of this split PDF corresponds to original page {request.source_pages[0]}
        - The last page corresponds to original page {request.source_pages[-1]}
        
        IMPORTANT FORMATTING INSTRUCTIONS:
        - Use **Markdown** for text formatting (bold, italics, lists, headers)
        - Use LaTeX math formatting: $equation$ for inline and $$equation$$ for block math
        - Use code blocks for programming concepts with ```language code```
        - Create clear, well-structured flashcards
        
        IMPORTANT JSON FORMATTING:
        - You MUST return valid JSON format
        - Escape all backslashes in LaTeX with double backslashes (e.g., \\\\frac instead of \\frac)
        - Use proper JSON string escaping
        
        For each flashcard, provide:
        - Front: A clear question, term, or concept prompt (use Markdown/LaTeX for formatting)
        - Back: A detailed explanation, definition, or answer (use Markdown/LaTeX for formatting)
        - Source pages: The specific ORIGINAL page numbers where this information appears (from {request.source_pages})
        
        Create flashcards that cover:
        - Key concepts and definitions
        - Important facts and details
        - Conceptual relationships
        - Practical applications
        - Term and its definition
        - Theroem and its proof
        
        The number of flashcards should be automatically determined based on the content density and importance, with maximum numbers of flashcards: {request.max_cards}.
        
        Return your response as a valid JSON array of flashcard objects with this exact structure:
        [
            {{
                "front": "**Concept Name** with $mathematical$ notation",
                "back": "Detailed explanation with:\\n- Bullet points\\n- **Bold text**\\n- $E = mc^2$\\n- ```python\\nprint('code example')\\n```",
                "source_pages": [1, 2]
            }}
        ]
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(
                    data=split_pdf_bytes,
                    mime_type='application/pdf'
                ),
                prompt
            ],
            config={
                "temperature": 0.7,
                "response_mime_type": "application/json",
            }
        )
        
        # Parse flashcards from response using improved parser
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            flashcards_data = parse_json_with_latex(response_text)
            
            # Map AI page references back to original PDF pages
            for flashcard_data in flashcards_data:
                ai_source_pages = flashcard_data.get("source_pages", [])
                flashcard_data["source_pages"] = map_ai_page_references(ai_source_pages, request.source_pages)
            
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
            # Use the internal PDF bytes we already downloaded
            split_pdf_bytes = split_pdf_by_pages(pdf_bytes, leaf.source_pages)
            
            # Generate flashcards with formatting instructions
            prompt = f"""
            Generate educational flashcards for: {leaf.title} - {leaf.description}
            
            IMPORTANT PAGE NUMBERING:
            - The provided PDF contains pages {leaf.source_pages} from the original document.
            - When citing source pages, ALWAYS use the original PDF page numbers: {leaf.source_pages}
            
            IMPORTANT FORMATTING:
            - Use **Markdown** for text formatting
            - Use LaTeX math: $equation$ for inline and $$equation$$ for block
            - Use code blocks for programming concepts
            - Escape LaTeX backslashes with double backslashes
            
            For each flashcard provide:
            - Front: Question or concept (with formatting)
            - Back: Detailed explanation (with formatting)
            - Source pages: Specific ORIGINAL page numbers from {leaf.source_pages}
            
            Return as valid JSON array of {{"front": "", "back": "", "source_pages": []}}
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(
                        data=split_pdf_bytes,
                        mime_type='application/pdf'
                    ),
                    prompt
                ],
                config={
                    "temperature": 0.7,
                    "response_mime_type": "application/json",
                }
            )
            
            try:
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.startswith('```'):
                    response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
                    
                flashcards_data = parse_json_with_latex(response_text)
                
                # Map AI page references back to original PDF pages
                for flashcard_data in flashcards_data:
                    ai_source_pages = flashcard_data.get("source_pages", [])
                    flashcard_data["source_pages"] = map_ai_page_references(ai_source_pages, leaf.source_pages)
                
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
        
        IMPORTANT PAGE NUMBERING:
        - The provided PDF contains pages {request.source_pages} from the original document.
        - When citing source pages, ALWAYS use the original PDF page numbers: {request.source_pages}
        - The first page of this split PDF corresponds to original page {request.source_pages[0]}
        - The last page corresponds to original page {request.source_pages[-1]}
        
        IMPORTANT JSON AND FORMATTING INSTRUCTIONS:
        - You MUST return valid JSON format
        - Escape all backslashes in LaTeX with double backslashes (e.g., \\\\frac instead of \\frac)
        - Use **Markdown** for formatting text (bold, italics, lists, headers)
        - Use LaTeX math formatting for mathematical expressions: $equation$ for inline and $$equation$$ for block math
        - Use code blocks for programming concepts with ```language code```
        - Use proper JSON string escaping
        - Create clear, well-structured flashcards
        
        For each flashcard, provide:
        - Front: Question, concept, or term (formatted with Markdown/LaTeX)
        - Back: Detailed explanation with proper formatting
        - Source pages: Specific ORIGINAL page numbers from {request.source_pages}
        
        Return your response as a valid JSON array of flashcard objects with this exact structure:
        [
            {{
                "front": "**Concept Name** with $mathematical$ notation",
                "back": "Detailed explanation with:\\n- Bullet points\\n- **Bold text**\\n- $E = mc^2$\\n- ```python\\nprint('code example')\\n```",
                "source_pages": [1, 2]
            }}
        ]
        
        Ensure all mathematical expressions are properly formatted with LaTeX and text uses Markdown for clarity.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
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
        
        # Parse flashcards from response using improved parser
        try:
            response_text = response.text.strip()
            # Clean response text
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            flashcards_data = parse_json_with_latex(response_text)
            
            # Map AI page references back to original PDF pages
            for flashcard_data in flashcards_data:
                ai_source_pages = flashcard_data.get("source_pages", [])
                flashcard_data["source_pages"] = map_ai_page_references(ai_source_pages, request.source_pages)
              
            
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
      
class IncompleteFlashcard(BaseModel):
    front: str
    back: str

class RefineFlashcardRequest(BaseModel):
    pdf_url: HttpUrl
    source_pages: List[int]
    incomplete_flashcard: IncompleteFlashcard
    current_topic: str = "General Topic"

class RefineFlashcardResponse(BaseModel):
    original_flashcard: IncompleteFlashcard
    refined_flashcard: Flashcard
    source_pages: List[int]
    interpretation: str

@app.post("/refine-flashcard", response_model=RefineFlashcardResponse)
async def refine_flashcard(request: RefineFlashcardRequest):
    """
    Refine an incomplete flashcard by understanding user intent using the PDF context
    Returns a single refined flashcard
    """
    try:
        # Download PDF from URL
        pdf_bytes = download_pdf_from_url(str(request.pdf_url))
        
        # Split PDF for the specified source pages
        split_pdf_bytes = split_pdf_by_pages(pdf_bytes, request.source_pages)
        
        # Enhanced prompt for understanding user intent and refining a single flashcard
        prompt = f"""
        USER'S INCOMPLETE FLASHCARD:
        Front: "{request.incomplete_flashcard.front}"
        Back: "{request.incomplete_flashcard.back}"
        
        TOPIC CONTEXT: {request.current_topic}
        
        IMPORTANT PAGE NUMBERING:
        - The provided PDF contains pages {request.source_pages} from the original document.
        - When citing source pages, ALWAYS use the original PDF page numbers: {request.source_pages}
        
        TASK:
        1. Analyze the user's incomplete flashcard and understand what they're trying to achieve
        2. Use the PDF content to refine and improve this single flashcard
        3. Enhance both front and back with better clarity, accuracy, and educational value
        4. Maintain the core concept but improve the expression and completeness
        
        REFINEMENT GUIDELINES:
        - If the front is vague, make it more specific and clear
        - If the back is incomplete, expand it with comprehensive information from the PDF
        - Add proper formatting (Markdown for text, LaTeX for math)
        - Ensure the flashcard is self-contained and educational
        - Focus on the key learning objectives from the PDF pages
        
        FORMATTING REQUIREMENTS:
        - Use **Markdown** for text formatting
        - Use LaTeX math: $equation$ for inline and $$equation$$ for block math
        - Use code blocks for programming concepts
        - Escape LaTeX backslashes with double backslashes
        
        OUTPUT STRUCTURE:
        - Provide a brief interpretation of what improvements were made
        - Return exactly ONE refined flashcard with front, back, and source_pages
        
        Return your response as valid JSON with this exact structure:
        {{
            "interpretation": "Brief explanation of improvements made",
            "refined_flashcard": {{
                "front": "Refined and improved front of the flashcard",
                "back": "Refined and improved back with detailed explanation",
                "source_pages": [1, 2]
            }}
        }}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=split_pdf_bytes,
                    mime_type='application/pdf'
                ),
                prompt
            ],
            config={
                "temperature": 0.7,
                "response_mime_type": "application/json",
            }
        )
        
        # Parse the response
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            refinement_data = parse_json_with_latex(response_text)
            
            # Get the single refined flashcard
            refined_data = refinement_data.get("refined_flashcard", {})
            
            # Map AI page references back to original PDF pages
            ai_source_pages = refined_data.get("source_pages", [])
            refined_data["source_pages"] = map_ai_page_references(ai_source_pages, request.source_pages)
            
            refined_flashcard = Flashcard(**refined_data)
            
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse refined flashcard from AI response: {str(e)}\nResponse: {response.text}"
            )
        
        return RefineFlashcardResponse(
            original_flashcard=request.incomplete_flashcard,
            refined_flashcard=refined_flashcard,
            source_pages=request.source_pages,
            interpretation=refinement_data.get("interpretation", "Interpretation not provided")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Flashcard refinement failed: {str(e)}")
        
class FormatImprovementRequest(BaseModel):
    pdf_url: HttpUrl
    source_pages: List[int]
    text_to_format: str
    target_language: str = "mathematics"  # mathematics, programming, physics, general

class FormatImprovementResponse(BaseModel):
    original_text: str
    formatted_text: str
    improvements_made: List[str]
    source_pages: List[int]

@app.post("/improve-formatting", response_model=FormatImprovementResponse)
async def improve_formatting(request: FormatImprovementRequest):
    """
    Improve text formatting by converting to proper LaTeX and Markdown using PDF context
    """
    try:
        # Download PDF from URL
        pdf_bytes = download_pdf_from_url(str(request.pdf_url))
        
        # Split PDF for the specified source pages
        split_pdf_bytes = split_pdf_by_pages(pdf_bytes, request.source_pages)
        
        # Prompt for intelligent formatting improvement
        prompt = f"""
        TEXT TO FORMAT: "{request.text_to_format}"
        TARGET DOMAIN: {request.target_language}
        
        IMPORTANT PAGE NUMBERING:
        - The provided PDF contains pages {request.source_pages} from the original document.
        
        TASK:
        Analyze the provided text and improve its formatting by converting it to proper LaTeX and Markdown.
        Use the PDF content as context to understand the domain and appropriate formatting rules.
        
        FORMATTING RULES:
        
        MATHEMATICS:
        - Variables: x, y, z → $x$, $y$, $z$
        - Sets: for i in V → for $i \\in V$
        - Functions: f(x) → $f(x)$
        - Equations: x = y + z → $x = y + z$
        - Fractions: 1/2 → $\\frac{1}{2}$
        - Greek letters: alpha, beta → $\\alpha$, $\\beta$
        - Operators: sum, product → $\\sum$, $\\prod$
        - Relations: <=, >=, != → $\\le$, $\\ge$, $\\neq$
        
        PROGRAMMING:
        - Code blocks: Use ```language for multi-line code
        - Inline code: Use `code` for short code snippets
        - Keywords: bold important programming concepts
        
        GENERAL:
        - **Bold** for important terms
        - *Italics* for emphasis
        - Lists with - or * for bullet points
        - Headers with ## for sections
        
        IMPROVEMENT STRATEGY:
        1. Identify mathematical expressions and convert to LaTeX
        2. Identify code and format appropriately
        3. Add Markdown formatting for readability
        4. Maintain the original meaning while improving clarity
        5. Use context from the PDF to determine the correct domain-specific formatting
        
        Return your response as valid JSON with this exact structure:
        {{
            "formatted_text": "The fully formatted text with LaTeX and Markdown",
            "improvements_made": [
                "Converted variables to LaTeX",
                "Added code formatting",
                "Applied Markdown structure"
            ]
        }}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=split_pdf_bytes,
                    mime_type='application/pdf'
                ),
                prompt
            ],
            config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            }
        )
        
        # Parse the response
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            formatting_data = parse_json_with_latex(response_text)
            
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse formatting improvement from AI response: {str(e)}\nResponse: {response.text}"
            )
        
        return FormatImprovementResponse(
            original_text=request.text_to_format,
            formatted_text=formatting_data.get("formatted_text", request.text_to_format),
            improvements_made=formatting_data.get("improvements_made", []),
            source_pages=request.source_pages
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Formatting improvement failed: {str(e)}")
        
class QuestionExtractionRequest(BaseModel):
    pdf_url: HttpUrl
    pages: List[int]
    question_types: List[str] = ["multiple_choice", "short_answer", "problem_solving"]  # Types of questions to extract
    include_answers: bool = False  # Whether to extract answers if available

class ExtractedQuestion(BaseModel):
    question_number: Optional[str] = None
    question_text: str
    question_type: str
    source_page: int
    options: List[str] = []  # For multiple choice questions
    answer: Optional[str] = None  # Only if include_answers is True

class QuestionExtractionResponse(BaseModel):
    questions: List[ExtractedQuestion]
    total_questions: int
    source_pages: List[int]
    extraction_summary: str

@app.post("/extract-questions", response_model=QuestionExtractionResponse)
async def extract_questions(request: QuestionExtractionRequest):
    """
    Extract questions from exercises in specified PDF pages with LaTeX formatting
    """
    try:
        # Download PDF from URL
        pdf_bytes = download_pdf_from_url(str(request.pdf_url))
        
        # Split PDF to keep only specified pages
        split_pdf_bytes = split_pdf_by_pages(pdf_bytes, request.pages)
        
        # Build question type description
        question_types_str = ", ".join(request.question_types)
        answer_instruction = "Include answers if they are present in the text." if request.include_answers else "Do not include answers, only extract the questions."
        
        # Prompt for question extraction with LaTeX formatting
        prompt = f"""
        Analyze the PDF content and extract all questions from exercises, problems, and assessments.
        
        IMPORTANT PAGE NUMBERING:
        - The provided PDF contains pages {request.pages} from the original document.
        - When citing source pages, ALWAYS use the original PDF page numbers: {request.pages}
        
        QUESTION TYPES TO EXTRACT: {question_types_str}
        {answer_instruction}
        
        EXTRACTION GUIDELINES:
        1. Look for sections labeled: Exercises, Problems, Questions, Assessment, Review Questions, etc.
        2. Extract both standalone questions and questions within problem sets
        3. Identify multiple choice questions and extract both question and options
        4. For mathematical questions, preserve all mathematical notation using LaTeX
        5. Maintain the original numbering if present
        6. Make sure the question is complete when extracted. 
        
        FORMATTING REQUIREMENTS:
        - Use LaTeX math formatting for all mathematical expressions: $equation$ for inline and $$equation$$ for block math
        - Preserve all mathematical symbols, equations, and notation
        - Keep the original structure and wording as much as possible
        - Format chemical formulas and scientific notation appropriately
        - Escape all backslashes in LaTeX with double backslashes (e.g., \\\\frac instead of \\frac)
        
        QUESTION TYPES:
        - multiple_choice: Questions with options (A, B, C, D, etc.)
        - short_answer: Brief response questions
        - problem_solving: Mathematical problems requiring step-by-step solutions
        - true_false: True/False questions
        - fill_in_blank: Fill in the blank questions
        
        OUTPUT STRUCTURE:
        Return your response as valid JSON with this exact structure:
        {{
            "extraction_summary": "Brief summary of what was extracted",
            "questions": [
                {{
                    "question_number": "1.1",
                    "question_text": "What is the derivative of $f(x) = x^2$?",
                    "question_type": "short_answer",
                    "source_page": 1,
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "answer": "$2x$"
                }},
                {{
                    "question_number": null,
                    "question_text": "Solve the equation: $$x^2 + 2x + 1 = 0$$",
                    "question_type": "problem_solving", 
                    "source_page": 1,
                    "options": [],
                    "answer": null
                }}
            ]
        }}
        
        Note: Only include 'answer' field if explicitly requested and if answers are present in the text.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=split_pdf_bytes,
                    mime_type='application/pdf'
                ),
                prompt
            ],
            config={
                "temperature": 0.2,
                "response_mime_type": "application/json",
            }
        )
        
        # Parse the response
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            extraction_data = parse_json_with_latex(response_text)
            
            # Process extracted questions
            questions_data = extraction_data.get("questions", [])
            questions = []
            
            for q_data in questions_data:
                # Map AI page references back to original PDF pages
                ai_source_page = q_data.get("source_page", 1)
                if 1 <= ai_source_page <= len(request.pages):
                    q_data["source_page"] = request.pages[ai_source_page - 1]
                else:
                    q_data["source_page"] = request.pages[0]  # Fallback to first page
                
                # Remove answer if not requested
                if not request.include_answers:
                    q_data["answer"] = None
                
                questions.append(ExtractedQuestion(**q_data))
            
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse extracted questions from AI response: {str(e)}\nResponse: {response.text}"
            )
        
        return QuestionExtractionResponse(
            questions=questions,
            total_questions=len(questions),
            source_pages=request.pages,
            extraction_summary=extraction_data.get("extraction_summary", "Questions extracted successfully")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question extraction failed: {str(e)}")


class ProofCompletionRequest(BaseModel):
    pdf_url: HttpUrl
    pages: List[int]
    theorem_or_question: str
    proof_guidelines: str
    include_step_by_step: bool = True

class ProofStep(BaseModel):
    step_number: int
    description: str
    mathematical_expression: Optional[str] = None

class ProofCompletionResponse(BaseModel):
    theorem_or_question: str
    proof_guidelines: str
    complete_proof: str
    proof_steps: List[ProofStep]
    source_pages: List[int]
    proof_summary: str

@app.post("/complete-proof", response_model=ProofCompletionResponse)
async def complete_proof(request: ProofCompletionRequest):
    """
    Complete a mathematical proof based on guidelines using PDF context
    """
    try:
        # Download PDF from URL
        pdf_bytes = download_pdf_from_url(str(request.pdf_url))
        
        # Split PDF to keep only specified pages
        split_pdf_bytes = split_pdf_by_pages(pdf_bytes, request.pages)
        
        # Prompt for proof completion with LaTeX formatting
        prompt = f"""
        THEOREM/QUESTION TO PROVE: {request.theorem_or_question}
        PROOF GUIDELINES: {request.proof_guidelines}
    
        TASK:
        Write a complete, rigorous mathematical proof based on the given theorem/question and proof guidelines.
        
        PROOF REQUIREMENTS:
        1. Follow the provided guidelines precisely
        2. Provide a step-by-step logical progression
        3. Include all necessary mathematical justifications
        4. Use proper mathematical notation and terminology
        5. Ensure the proof is self-contained and easy to follow
        
        FORMATTING REQUIREMENTS:
        - Use LaTeX math formatting for all mathematical expressions: $equation$ for inline and $$equation$$ for block math
        - Use **Markdown** for text formatting and structure
        - Clearly label assumptions, definitions, and conclusions
        - Use proper mathematical symbols and notation
        - Escape all backslashes in LaTeX with double backslashes (e.g., \\\\frac instead of \\frac)
        
        PROOF STRUCTURE:
        - Start with clear statement of what is to be proven
        - List any assumptions or given conditions
        - Provide step-by-step logical reasoning
        - Include necessary mathematical expressions and equations
        - End with a clear conclusion
        
        OUTPUT STRUCTURE:
        Return your response as valid JSON with this exact structure:
        {{
            "proof_summary": "Brief summary of the proof approach",
            "complete_proof": "The full proof text with LaTeX and Markdown formatting",
            "proof_steps": [
                {{
                    "step_number": 1,
                    "description": "Description of this proof step",
                    "mathematical_expression": "$$ mathematical expression $$"
                }},
                {{
                    "step_number": 2,
                    "description": "Next step description", 
                    "mathematical_expression": "$inline math$"
                }}
            ]
        }}
        
        The proof should be mathematically rigorous and follow standard proof techniques appropriate for the content. **Keep the proof short**.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                  prompt
            ],
            config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            }
        )
        
        # Parse the response
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            proof_data = parse_json_with_latex(response_text)
            
            # Process proof steps
            proof_steps_data = proof_data.get("proof_steps", [])
            proof_steps = [ProofStep(**step) for step in proof_steps_data]
            
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse proof completion from AI response: {str(e)}\nResponse: {response.text}"
            )
        
        return ProofCompletionResponse(
            theorem_or_question=request.theorem_or_question,
            proof_guidelines=request.proof_guidelines,
            complete_proof=proof_data.get("complete_proof", ""),
            proof_steps=proof_steps,
            source_pages=request.pages,
            proof_summary=proof_data.get("proof_summary", "Proof completed successfully")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proof completion failed: {str(e)}")      
class ProofAnalysisRequest(BaseModel):
    pdf_url: HttpUrl
    pages: List[int]
    proof_to_analyze: str
    theorem_statement: str
    analysis_depth: str = "rigorous"  # quick, detailed, rigorous

class LogicalFlaw(BaseModel):
    flaw_type: str
    description: str
    location_in_proof: str
    severity: str  # minor, moderate, critical
    suggested_correction: Optional[str] = None

class CounterExample(BaseModel):
    description: str
    mathematical_expression: Optional[str] = None
    explanation: str

class ProofAnalysisResponse(BaseModel):
    theorem_statement: str
    proof_provided: str
    is_valid: bool
    overall_assessment: str
    logical_flaws: List[LogicalFlaw]
    counter_examples: List[CounterExample]
    corrected_proof: Optional[str] = None
    source_pages: List[int]
    analysis_methodology: str

@app.post("/analyze-proof", response_model=ProofAnalysisResponse)
async def analyze_proof(request: ProofAnalysisRequest):
    """
    Rigorously analyze a mathematical proof for logical flaws and counterexamples
    """
    try:
        # Download PDF from URL
        pdf_bytes = download_pdf_from_url(str(request.pdf_url))
        
        # Split PDF to keep only specified pages
        split_pdf_bytes = split_pdf_by_pages(pdf_bytes, request.pages)
        
        # Prompt for rigorous proof analysis
        prompt = f"""
        THEOREM STATEMENT: {request.theorem_statement}
        PROOF TO ANALYZE: {request.proof_to_analyze}
        ANALYSIS DEPTH: {request.analysis_depth}
        
        IMPORTANT PAGE NUMBERING:
        - The provided PDF contains pages {request.pages} from the original document.
        
        TASK:
        Conduct a rigorous mathematical analysis of the given proof. Identify any logical flaws, 
        gaps in reasoning, incorrect assumptions, or mathematical errors. Provide counterexamples 
        if the proof is invalid or the theorem is false.
        
        ANALYSIS METHODOLOGY:
        1. Check each logical step for validity
        2. Verify all mathematical assumptions and premises
        3. Test boundary cases and edge cases
        4. Look for circular reasoning or unfounded leaps
        5. Verify the proof structure and conclusion
        
        COMMON FLAW TYPES TO IDENTIFY:
        - Circular reasoning
        - False dichotomy
        - Begging the question
        - Incorrect use of mathematical induction
        - Invalid assumptions
        - Gaps in logical progression
        - Incorrect application of theorems
        - Arithmetic or algebraic errors
        - Quantifier errors (∀ vs ∃)
        - Case analysis omissions
        
        COUNTEREXAMPLE GUIDELINES:
        - Provide explicit counterexamples if the theorem is false
        - Include mathematical expressions and explanations
        - Show why the counterexample violates the theorem conditions
        - Use LaTeX for mathematical notation
        
        FORMATTING REQUIREMENTS:
        - Use LaTeX math formatting: $equation$ for inline and $$equation$$ for block math
        - Use **Markdown** for clear structure and emphasis
        - Escape all backslashes in LaTeX with double backslashes (e.g., \\\\frac instead of \\frac)
        
        OUTPUT STRUCTURE:
        Return your response as valid JSON with this exact structure:
        {{
            "is_valid": false,
            "overall_assessment": "Detailed assessment of proof validity",
            "analysis_methodology": "Description of analysis approach used",
            "logical_flaws": [
                {{
                    "flaw_type": "Circular Reasoning",
                    "description": "Detailed description of the flaw",
                    "location_in_proof": "Step 3, line 2",
                    "severity": "critical",
                    "suggested_correction": "How to fix this flaw"
                }}
            ],
            "counter_examples": [
                {{
                    "description": "Clear description of counterexample",
                    "mathematical_expression": "$$ mathematical expression $$",
                    "explanation": "Why this counterexample invalidates the proof/theorem"
                }}
            ],
            "corrected_proof": "Optional: A corrected version of the proof if flaws can be fixed"
        }}
        
        Be thorough and mathematically rigorous in your analysis. Even subtle flaws should be identified.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(
                    data=split_pdf_bytes,
                    mime_type='application/pdf'
                ),
                prompt
            ],
            config={
                "temperature": 0.1,  # Low temperature for rigorous analysis
                "max_output_tokens": 5000,
                "response_mime_type": "application/json",
            }
        )
        
        # Parse the response
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            analysis_data = parse_json_with_latex(response_text)
            
            # Process logical flaws
            flaws_data = analysis_data.get("logical_flaws", [])
            logical_flaws = [LogicalFlaw(**flaw) for flaw in flaws_data]
            
            # Process counter examples
            counter_examples_data = analysis_data.get("counter_examples", [])
            counter_examples = [CounterExample(**ce) for ce in counter_examples_data]
            
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse proof analysis from AI response: {str(e)}\nResponse: {response.text}"
            )
        
        return ProofAnalysisResponse(
            theorem_statement=request.theorem_statement,
            proof_provided=request.proof_to_analyze,
            is_valid=analysis_data.get("is_valid", False),
            overall_assessment=analysis_data.get("overall_assessment", "Analysis incomplete"),
            logical_flaws=logical_flaws,
            counter_examples=counter_examples,
            corrected_proof=analysis_data.get("corrected_proof"),
            source_pages=request.pages,
            analysis_methodology=analysis_data.get("analysis_methodology", "Rigorous mathematical analysis")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Proof analysis failed: {str(e)}")

from google.cloud import texttospeech
import base64
from pydub import AudioSegment
import tempfile
import io as python_io

# Initialize Text-to-Speech client
tts_client = texttospeech.TextToSpeechClient()

# New Pydantic Models for Multi-Speaker Podcast
class DialogueLine(BaseModel):
    speaker: str  # "host" or "expert"
    text: str
    voice_name: Optional[str] = None  # Make this optional

class PodcastScript(BaseModel):
    title: str
    introduction: str
    dialogues: List[DialogueLine]
    conclusion: str

class MultiSpeakerPodcastRequest(BaseModel):
    pdf_url: HttpUrl
    pages: List[int]
    prompt: str
    host_voice: Optional[str] = "en-US-Chirp-HD-F"  # Male voice
    expert_voice: Optional[str] = "en-US-Chirp-HD-D"  # Female voice
    speaking_rate: Optional[float] = 1.0
    pitch: Optional[float] = 0.0
    audio_encoding: Optional[str] = "MP3"
    include_pauses: Optional[bool] = True
    pause_duration: Optional[float] = 0.5  # seconds

class MultiSpeakerPodcastResponse(BaseModel):
    audio_content: str  # Base64 encoded audio
    script: PodcastScript
    total_pages_processed: int
    voices_used: Dict[str, str]
    audio_format: str
    duration_estimate: float
    segment_count: int

# New Utility Functions for Multi-Speaker Podcast
def generate_dialogue_script(pdf_content: bytes, pages: List[int], prompt: str) -> PodcastScript:
    """
    Generate a podcast script with host-expert dialogue using Gemini
    """
    try:
        enhanced_prompt = f"""
        Create an engaging, educational podcast script with dialogue between a HOST and an EXPERT based on the provided PDF content.
        
        USER REQUEST: {prompt}
        
        IMPORTANT PAGE NUMBERING:
        - The provided PDF contains pages {pages} from the original document.
        - When referring to content locations, ALWAYS mention the original page numbers: {pages}
        
        SCRIPT STRUCTURE:
        1. Title: A catchy title for the podcast
        2. Introduction: Host introduces the topic and expert
        3. Dialogues: Alternating dialogue between Host and Expert
        4. Conclusion: Host summarizes key points and thanks expert
        
        DIALOGUE REQUIREMENTS:
        - Create natural, conversational dialogue
        - Host should ask questions, guide the conversation, and provide context
        - Expert should provide detailed explanations, insights, and technical details
        - Include 8-15 dialogue exchanges (back and forth)
        - Make the dialogue engaging and educational
        - Host should occasionally summarize or clarify complex points
        - Expert should use examples and analogies when appropriate
        - Make sure to not contain latex or anything with escapes. Enunciate the latex symbols.
        
        CHARACTER ROLES:
        - HOST: Curious, engaging, good at asking clarifying questions, maintains flow
        - EXPERT: Knowledgeable, authoritative, but able to explain complex topics simply
        
        RETURN FORMAT:
        You MUST return a valid JSON object with this exact structure:
        {{
            "title": "Podcast Title",
            "introduction": "Host's introduction text here...",
            "dialogues": [
                {{"speaker": "host", "text": "Host dialogue text..."}},
                {{"speaker": "expert", "text": "Expert response text..."}},
                {{"speaker": "host", "text": "Follow up question..."}},
                {{"speaker": "expert", "text": "Detailed explanation..."}}
            ],
            "conclusion": "Host's concluding remarks..."
        }}
        
        IMPORTANT: Do NOT include "voice_name" in the dialogue objects. Only include "speaker" and "text".
        
        Ensure the dialogue flows naturally and covers the key concepts from the PDF.
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=[
                types.Part.from_bytes(
                    data=pdf_content,
                    mime_type='application/pdf'
                ),
                enhanced_prompt
            ],
            config={
                "temperature": 0.8,
                "response_mime_type": "application/json",
            }
        )
        
        # Parse the JSON response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
            
        script_data = json.loads(response_text)
        
        # Validate and create PodcastScript object
        return PodcastScript(**script_data)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dialogue script generation failed: {str(e)}")

def generate_audio_segment(text: str, voice_name: str, speaking_rate: float, pitch: float, audio_encoding: str) -> bytes:
    """
    Generate audio for a single text segment
    """
    synthesis_input = texttospeech.SynthesisInput(text=text)
    
    encoding_map = {
        "MP3": texttospeech.AudioEncoding.MP3,
        "LINEAR16": texttospeech.AudioEncoding.LINEAR16,
        "OGG_OPUS": texttospeech.AudioEncoding.OGG_OPUS
    }
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=encoding_map.get(audio_encoding, texttospeech.AudioEncoding.MP3),
        speaking_rate=speaking_rate,
        pitch=pitch,
        effects_profile_id=["headphone-class-device"]
    )
    
    voice = texttospeech.VoiceSelectionParams(
        language_code=voice_name.split("-")[0] + "-" + voice_name.split("-")[1],
        name=voice_name
    )
    
    response = tts_client.synthesize_speech(
        input=synthesis_input,
        voice=voice,
        audio_config=audio_config
    )
    
    return response.audio_content

def add_pause(audio_segment: AudioSegment, pause_duration: float) -> AudioSegment:
    """
    Add a pause to an audio segment
    """
    pause = AudioSegment.silent(duration=int(pause_duration * 1000))  # pydub works in milliseconds
    return audio_segment + pause

def concatenate_audio_segments(segments: List[bytes], pause_duration: float, audio_encoding: str) -> bytes:
    """
    Concatenate multiple audio segments with pauses between them
    """
    if not segments:
        raise ValueError("No audio segments to concatenate")
    
    # Convert encoding string to pydub format
    encoding_map = {
        "MP3": "mp3",
        "LINEAR16": "wav",
        "OGG_OPUS": "ogg"
    }
    
    format_name = encoding_map.get(audio_encoding, "mp3")
    
    # Load first segment
    combined = AudioSegment.from_file(python_io.BytesIO(segments[0]), format=format_name)
    
    # Add remaining segments with pauses
    for i in range(1, len(segments)):
        # Add pause
        if pause_duration > 0:
            pause = AudioSegment.silent(duration=int(pause_duration * 1000))
            combined = combined + pause
        
        # Add next segment
        next_segment = AudioSegment.from_file(python_io.BytesIO(segments[i]), format=format_name)
        combined = combined + next_segment
    
    # Export combined audio
    output_buffer = python_io.BytesIO()
    combined.export(output_buffer, format=format_name)
    output_buffer.seek(0)
    
    return output_buffer.getvalue()

def generate_multi_speaker_audio(
    script: PodcastScript, 
    host_voice: str, 
    expert_voice: str, 
    speaking_rate: float, 
    pitch: float, 
    audio_encoding: str,
    include_pauses: bool,
    pause_duration: float
) -> tuple[bytes, float, int]:
    """
    Generate multi-speaker audio by creating segments for each dialogue line and concatenating them
    """
    try:
        audio_segments = []
        total_duration = 0
        segment_count = 0
        
        # Generate audio for introduction
        print(f"Generating introduction audio with voice: {host_voice}")
        intro_audio = generate_audio_segment(
            script.introduction, host_voice, speaking_rate, pitch, audio_encoding
        )
        audio_segments.append(intro_audio)
        segment_count += 1
        
        # Estimate duration for introduction (rough calculation)
        intro_word_count = len(script.introduction.split())
        total_duration += (intro_word_count / 150) * 60  # 150 words per minute
        
        if include_pauses and pause_duration > 0:
            pause_audio = AudioSegment.silent(duration=int(pause_duration * 1000))
            # For pause, we'll create a silent segment in the same format
            pause_buffer = python_io.BytesIO()
            pause_audio.export(pause_buffer, format="mp3" if audio_encoding == "MP3" else audio_encoding.lower())
            audio_segments.append(pause_buffer.getvalue())
            total_duration += pause_duration
        
        # Generate audio for each dialogue line
        for i, dialogue in enumerate(script.dialogues):
            # Determine voice based on speaker role, not from dialogue.voice_name
            voice_to_use = host_voice if dialogue.speaker == "host" else expert_voice
            print(f"Generating audio for {dialogue.speaker} line {i+1} with voice: {voice_to_use}")
            
            dialogue_audio = generate_audio_segment(
                dialogue.text, voice_to_use, speaking_rate, pitch, audio_encoding
            )
            audio_segments.append(dialogue_audio)
            segment_count += 1
            
            # Estimate duration for this dialogue
            dialogue_word_count = len(dialogue.text.split())
            total_duration += (dialogue_word_count / 150) * 60
            
            # Add pause between dialogues (but not after the last one)
            if include_pauses and pause_duration > 0 and i < len(script.dialogues) - 1:
                pause_audio = AudioSegment.silent(duration=int(pause_duration * 1000))
                pause_buffer = python_io.BytesIO()
                pause_audio.export(pause_buffer, format="mp3" if audio_encoding == "MP3" else audio_encoding.lower())
                audio_segments.append(pause_buffer.getvalue())
                total_duration += pause_duration
        
        # Add pause before conclusion if needed
        if include_pauses and pause_duration > 0 and script.dialogues:
            pause_audio = AudioSegment.silent(duration=int(pause_duration * 1000))
            pause_buffer = python_io.BytesIO()
            pause_audio.export(pause_buffer, format="mp3" if audio_encoding == "MP3" else audio_encoding.lower())
            audio_segments.append(pause_buffer.getvalue())
            total_duration += pause_duration
        
        # Generate audio for conclusion
        print(f"Generating conclusion audio with voice: {host_voice}")
        conclusion_audio = generate_audio_segment(
            script.conclusion, host_voice, speaking_rate, pitch, audio_encoding
        )
        audio_segments.append(conclusion_audio)
        segment_count += 1
        
        # Estimate duration for conclusion
        conclusion_word_count = len(script.conclusion.split())
        total_duration += (conclusion_word_count / 150) * 60
        
        # Concatenate all audio segments
        print(f"Concatenating {len(audio_segments)} audio segments...")
        final_audio = concatenate_audio_segments(audio_segments, 0, audio_encoding)  # No additional pauses since we already added them
        
        return final_audio, total_duration, segment_count
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-speaker audio generation failed: {str(e)}")

# Enhanced Podcast API Endpoint with Multiple Speakers
@app.post("/generate-multi-speaker-podcast", response_model=MultiSpeakerPodcastResponse)
async def generate_multi_speaker_podcast(request: MultiSpeakerPodcastRequest):
    """
    Generate a multi-speaker podcast (host + expert) with different voices from specified PDF pages
    """
    try:
        # Validate voice parameters
        validate_voice_parameters(request.host_voice, request.speaking_rate, request.pitch)
        validate_voice_parameters(request.expert_voice, request.speaking_rate, request.pitch)
        
        # Download PDF from URL
        pdf_bytes = download_pdf_from_url(str(request.pdf_url))
        
        # Split PDF to keep only specified pages
        split_pdf_bytes = split_pdf_by_pages(pdf_bytes, request.pages)
        
        # Generate dialogue script using Gemini
        script = generate_dialogue_script(split_pdf_bytes, request.pages, request.prompt)
        
        # Generate multi-speaker audio
        final_audio, estimated_duration, segment_count = generate_multi_speaker_audio(
            script=script,
            host_voice=request.host_voice,
            expert_voice=request.expert_voice,
            speaking_rate=request.speaking_rate,
            pitch=request.pitch,
            audio_encoding=request.audio_encoding,
            include_pauses=request.include_pauses,
            pause_duration=request.pause_duration
        )
        
        # Encode final audio to base64
        audio_base64 = base64.b64encode(final_audio).decode('utf-8')
        
        return MultiSpeakerPodcastResponse(
            audio_content=audio_base64,
            script=script,
            total_pages_processed=len(request.pages),
            voices_used={
                "host": request.host_voice,
                "expert": request.expert_voice
            },
            audio_format=request.audio_encoding,
            duration_estimate=estimated_duration,
            segment_count=segment_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Multi-speaker podcast generation failed: {str(e)}")

# Helper function to validate voices
def validate_voice_parameters(voice_name: str, speaking_rate: float, pitch: float) -> None:
    """
    Validate TTS parameters
    """
    # Validate speaking rate (0.25 to 4.0)
    if not 0.25 <= speaking_rate <= 4.0:
        raise HTTPException(
            status_code=400, 
            detail=f"Speaking rate must be between 0.25 and 4.0, got {speaking_rate}"
        )
    
    # Validate pitch (-20.0 to 20.0)
    if not -20.0 <= pitch <= 20.0:
        raise HTTPException(
            status_code=400, 
            detail=f"Pitch must be between -20.0 and 20.0, got {pitch}"
        )
    
    # Basic voice name validation
    if not voice_name or len(voice_name.split("-")) < 3:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice name format. Expected format like 'en-US-Standard-D', got {voice_name}"
        )


# Optional: Enhanced available voices endpoint
@app.get("/available-voices")
async def get_available_voices(language_code: str = "en"):
    """
    Get list of available TTS voices for a language code
    """
    try:
        voices = tts_client.list_voices(language_code=language_code)
        
        available_voices = []
        for voice in voices.voices:
            voice_info = {
                "name": voice.name,
                "language_codes": list(voice.language_codes),
                "ssml_gender": texttospeech.SsmlVoiceGender(voice.ssml_gender).name,
                "natural_sample_rate_hertz": voice.natural_sample_rate_hertz
            }
            available_voices.append(voice_info)
        
        return {
            "language_code": language_code,
            "available_voices": available_voices
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch available voices: {str(e)}")
class ImageOCRRequest(BaseModel):
    image_base64: str
    domain_hint: str = "mathematics"  # mathematics, physics, general, etc.

class ImageOCRResponse(BaseModel):
    raw_text: str
    latex_formatted: str
    markdown_formatted: str
    confidence_notes: List[str]

@app.post("/ocr-image-to-latex", response_model=ImageOCRResponse)
async def ocr_image_to_latex(request: ImageOCRRequest):
    """
    Extract text from base64 image using Gemini OCR and format with LaTeX and Markdown
    Typically used for handwritten math equations and notes
    """
    try:
        # Remove data URL prefix if present
        image_data = request.image_base64
        if ',' in image_data:
            image_data = image_data.split(',', 1)[1]
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        
        # Verify it's a valid image (basic check)
        if len(image_bytes) < 100:  # Arbitrary minimum size
            raise HTTPException(status_code=400, detail="Image data too small to be valid")
        
        # Enhanced prompt for handwritten math OCR and formatting
        prompt = f"""
        Analyze this handwritten image and perform OCR to extract the text content.
        
        DOMAIN CONTEXT: {request.domain_hint}
        
        IMPORTANT INSTRUCTIONS:
        1. First, extract the raw text exactly as it appears in the image
        2. Then create a properly formatted LaTeX version with correct mathematical notation
        3. Finally, create a Markdown version with proper formatting for readability
        
        FOR HANDWRITTEN MATHEMATICS:
        - Convert handwritten equations to proper LaTeX syntax
        - Use $...$ for inline math and $$...$$ for display math
        - Handle fractions: ½ → \\frac{{1}}{{2}}
        - Handle exponents: x² → x^2
        - Handle square roots: √x → \\sqrt{{x}}
        - Handle Greek letters: α, β, γ → \\alpha, \\beta, \\gamma
        - Handle operators: × → \\times, ÷ → \\div, ± → \\pm
        - Handle relations: ≤ → \\leq, ≥ → \\geq, ≠ → \\neq
        - Handle integrals: ∫ → \\int, ∑ → \\sum, ∏ → \\prod
        - Handle arrows: → → \\rightarrow, ← → \\leftarrow
        
        FOR GENERAL TEXT:
        - Use **bold** for headings and important terms
        - Use *italics* for emphasis
        - Use bullet points for lists
        - Preserve the original structure and flow
        
        Return your response as valid JSON with this exact structure:
        {{
            "raw_text": "The raw extracted text exactly as recognized",
            "latex_formatted": "The content formatted with proper LaTeX syntax",
            "markdown_formatted": "The content formatted with Markdown for readability",
            "confidence_notes": [
                "Note about any ambiguous characters",
                "Note about formatting decisions made"
            ]
        }}
        """
        
        # Use Gemini for image analysis and OCR
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp",  # Using flash model for faster OCR
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type='image/jpeg'  # Will work for PNG too as Gemini handles multiple formats
                ),
                prompt
            ],
            config={
                "temperature": 0.1,  # Low temperature for consistent OCR
                "response_mime_type": "application/json",
            }
        )
        
        # Parse the response
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            ocr_data = parse_json_with_latex(response_text)
            
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse OCR response from AI: {str(e)}\nResponse: {response.text}"
            )
        
        return ImageOCRResponse(
            raw_text=ocr_data.get("raw_text", ""),
            latex_formatted=ocr_data.get("latex_formatted", ""),
            markdown_formatted=ocr_data.get("markdown_formatted", ""),
            confidence_notes=ocr_data.get("confidence_notes", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image OCR processing failed: {str(e)}")

class TextFormattingRequest(BaseModel):
    unformatted_text: str
    target_domain: str = "mathematics"  # mathematics, programming, physics, general
    formatting_level: str = "comprehensive"  # minimal, moderate, comprehensive

class TextFormattingResponse(BaseModel):
    original_text: str
    formatted_text: str
    improvements_made: List[str]
    formatting_type: str  # latex, markdown, mixed

@app.post("/format-plain-text", response_model=TextFormattingResponse)
async def format_plain_text(request: TextFormattingRequest):
    """
    Improve formatting of plain unformatted text using LaTeX and Markdown
    Similar to /improve-formatting but without PDF context
    """
    try:
        # Enhanced prompt for text-only formatting
        prompt = f"""
        UNFORMATTED TEXT: "{request.unformatted_text}"
        
        TARGET DOMAIN: {request.target_domain}
        FORMATTING LEVEL: {request.formatting_level}
        
        TASK:
        Convert this unformatted text into properly structured content using LaTeX for mathematics
        and Markdown for general text formatting. The text does not come from a PDF - it's raw
        input that needs intelligent formatting.
        
        DOMAIN-SPECIFIC FORMATTING RULES:
        
        MATHEMATICS/PHYSICS:
        - Variables: x, y, z → $x$, $y$, $z$
        - Sets: i in V → $i \\in V$
        - Functions: f(x) → $f(x)$
        - Equations: E = mc² → $E = mc^2$
        - Fractions: 1/2 → \\frac{{1}}{{2}}, (a+b)/c → \\frac{{a+b}}{{c}}
        - Greek letters: alpha → \\alpha, beta → \\beta, Gamma → \\Gamma
        - Operators: sum, product → \\sum, \\prod
        - Relations: <= → \\leq, >= → \\geq, != → \\neq, ≈ → \\approx
        - Derivatives: df/dx → \\frac{{df}}{{dx}}, ∂f/∂x → \\frac{{\\partial f}}{{\\partial x}}
        - Integrals: ∫ f(x) dx → \\int f(x) dx
        
        PROGRAMMING/COMPUTER SCIENCE:
        - Code blocks: Use ```language ... ``` for multi-line code
        - Inline code: Use `code` for short snippets
        - Variables and functions: Use `monospace` for code elements in text
        - Data structures: Use **bold** for important concepts
        
        GENERAL ACADEMIC TEXT:
        - **Bold** for definitions and key terms
        - *Italics* for emphasis and book titles
        - Use headings with ## for sections
        - Use bullet points (- or *) for lists
        - Use numbered lists for sequences
        - Use blockquotes (> ) for important notes
        
        FORMATTING LEVELS:
        - Minimal: Only convert obvious mathematical expressions
        - Moderate: Add basic structure and clear mathematical formatting
        - Comprehensive: Full formatting with sections, emphasis, and detailed mathematical typesetting
        
        Return your response as valid JSON with this exact structure:
        {{
            "formatted_text": "The fully formatted text with LaTeX and Markdown",
            "improvements_made": [
                "Converted variables to LaTeX",
                "Added section headers",
                "Formatted mathematical equations",
                "Applied code formatting"
            ],
            "formatting_type": "mixed"  # or "latex", "markdown" depending on content
        }}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
            config={
                "temperature": 0.3,
                "response_mime_type": "application/json",
            }
        )
        
        # Parse the response
        try:
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
                
            formatting_data = parse_json_with_latex(response_text)
            
        except (json.JSONDecodeError, KeyError) as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse text formatting response: {str(e)}\nResponse: {response.text}"
            )
        
        return TextFormattingResponse(
            original_text=request.unformatted_text,
            formatted_text=formatting_data.get("formatted_text", request.unformatted_text),
            improvements_made=formatting_data.get("improvements_made", []),
            formatting_type=formatting_data.get("formatting_type", "mixed")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text formatting failed: {str(e)}")
        
@app.get("/")
async def root():
    return {"message": "PDF Analysis API - Use /generate-topic-tree, /generate-flashcards, /generate-flashcards-from-leaves, and /generate-custom-flashcards endpoints"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "PDF Analysis API"}
