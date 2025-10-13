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
        
        The proof should be mathematically rigorous and follow standard proof techniques appropriate for the content.
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

@app.get("/")
async def root():
    return {"message": "PDF Analysis API - Use /generate-topic-tree, /generate-flashcards, /generate-flashcards-from-leaves, and /generate-custom-flashcards endpoints"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "PDF Analysis API"}
