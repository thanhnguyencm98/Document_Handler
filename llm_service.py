import json
import requests
import time
import re
from typing import List, Dict, Any, Union, Optional
import math
import gc

class LLMService:
    """
    Service for interacting with the Ollama models with improved handling of large texts.
    """
    
    def __init__(self, model="llama3.1", api_url="http://localhost:11434", temperature=0.7, max_tokens=8192):
        """
        Initialize the LLM service.
        
        Args:
            model: Name of the Ollama model to use
            api_url: URL of the Ollama API
            temperature: Controls randomness in the output (0.0-1.0)
            max_tokens: Maximum tokens to generate in response
        """
        self.model = model
        self.api_url = api_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Estimate of chars per token for most models (rough approximation)
        self.chars_per_token = 4
    
    def analyze_document(self, text: str) -> Dict[str, Any]:
        """
        Analyze document text to generate comprehensive summary, category, key concepts, and keywords.
        Improved to handle large documents through intelligent chunking and synthesis.
        
        Args:
            text: Document text to analyze
            
        Returns:
            dict: Analysis results containing summary, category, important_points, sections_title, 
                 sections_brief, and keywords
        """
        # Get document length for better handling and reporting
        doc_length = len(text)
        
        # For very large documents, implement multi-stage analysis
        if doc_length > 50000:  # For docs > 50K chars
            return self._analyze_large_document(text)
        
        # For moderately sized documents, use the standard approach with smart sampling
        text_sample = self._smart_sample_text(text, max_length=32000)
        
        # Create the prompt with enhanced instructions
        prompt = f"""
        You are an expert document analyst with expertise in information extraction, summarization, and content analysis. Perform a thorough, multi-dimensional analysis of the following document content, providing exceptional detail and insight.

        Document length is approximately {doc_length} characters. Keep this in mind when determining appropriate level of detail.

        Deliver a comprehensive response including:

        1. SUMMARY (30-50 sentences):
           - Provide a detailed executive summary capturing all major themes, arguments, purposes, and conclusions
           - Highlight nuances, underlying implications, and contextual significance
           - Include critical insights that would be valuable to someone who hasn't read the document
           - Maintain the document's original tone and perspective while presenting information objectively
           - Ensure all key sections of the document are represented proportionally

        2. CATEGORY:
           - Identify the primary category that best describes this document (e.g., Financial, Technical, Legal, Educational, Marketing, etc.)
           - Provide 2-3 subcategories for more precise classification
           - Explain briefly why this categorization is appropriate

        3. IMPORTANT POINTS (15-30 items):
           - Identify critical insights, findings, arguments, and key data points
           - Format as clear, informative bullet points that convey essential information
           - Include both explicit statements and implied conclusions
           - Note any contradictions, limitations, or qualifications in the document
           - Highlight unique or novel perspectives presented
           - Include quantitative data when present
           - Order points by significance, not by order of appearance

        4. STRUCTURAL ANALYSIS:
           - Divide the document into 5-15 logical, coherent sections based on content flow and topic transitions
           - For each section, provide:
               a. A precise, descriptive title that encapsulates the main topic/theme (max 10 words)
               b. A detailed summary (15-30 sentences) explaining key points, arguments, data, and relevance
               c. Include section's relationship to overall document purpose
               d. Note any critical questions raised or answered in this section

        5. KEYWORDS (10-20 items):
           - Identify specific terms, concepts, or phrases central to understanding the document
           - Include both technical/specialized terminology and conceptual themes
           - Rank keywords by relevance and frequency
           - Include any proper nouns, organizations, or named entities

        6. CONTEXTUAL METADATA:
           - Infer the likely audience, purpose, and context of the document
           - Note any temporal references that date the document
           - Identify potential authorship clues (expertise level, perspective, etc)

        Document content:
        {text_sample}

        Respond in JSON format with the following structure:
        {{
            "summary": "comprehensive document summary here with exceptional detail",
            "category": {{
                "primary": "main category",
                "subcategories": ["subcategory1", "subcategory2"],
                "rationale": "brief explanation of categorization"
            }},
            "important_points": [
                "First critical point as a comprehensive bullet point",
                "Second critical point with detailed insight",
                "Additional important points...",
                "Final important point"
            ],
            "sections_title": [
                "First section title",
                "Second section title",
                "Additional section titles...",
                "Final section title"
            ],
            "sections_brief": [
                "First section detailed summary with context and relationship to document purpose",
                "Second section detailed summary",
                "Additional section summaries...",
                "Final section detailed summary"
            ],
            "keywords": ["keyword1", "keyword2", "keyword3", ...],
            "metadata": {{
                "likely_audience": "description of intended audience",
                "document_purpose": "inferred purpose of document",
                "temporal_context": "any time references or dating information",
                "authorship_insights": "clues about authorship"
            }}
        }}

        Ensure your analysis is thorough, insightful, and directly relevant to the document's content. Focus on extracting maximum value for a reader who needs to understand this document deeply. Only return the JSON object, nothing else.
        """
        
        # Make API call to Ollama with improved parameters
        response = self._call_ollama_api(
            prompt,
            temperature=0.2,  # Lower temperature for more consistent analysis
            system_prompt="You are an expert document analyst specialized in extracting detailed information from texts."
        )
        
        # Parse the response to extract JSON
        try:
            # Extract JSON from the response text
            json_str = self._extract_json(response)
            result = json.loads(json_str)
            
            # Handle the new format with nested category
            if isinstance(result.get("category"), dict):
                # Extract category and flatten structure
                category_data = result["category"]
                result["category"] = category_data["primary"]
                
                # Add subcategories if not present
                if "subcategories" not in result:
                    result["subcategories"] = category_data.get("subcategories", [])
                
                # Add category rationale if not present
                if "category_rationale" not in result:
                    result["category_rationale"] = category_data.get("rationale", "")
            
            # Add metadata if missing (for backward compatibility)
            if "metadata" not in result:
                result["metadata"] = {
                    "likely_audience": "",
                    "document_purpose": "",
                    "temporal_context": "",
                    "authorship_insights": ""
                }
            
            # Ensure expected fields are present
            required_fields = ["summary", "category", "important_points", "sections_title", "sections_brief", "keywords"]
            if not all(key in result for key in required_fields):
                missing_fields = [field for field in required_fields if field not in result]
                raise ValueError(f"Response missing required fields: {missing_fields}")
            
            return result
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {response}")
            
            # Return a more helpful default structure if parsing fails
            return {
                "summary": "Failed to generate comprehensive summary. The document was processed, but the analysis failed. Please try again or try with a different model.",
                "category": "Uncategorized",
                "important_points": ["The document was processed, but detailed extraction failed. Please try again."],
                "sections_title": ["Document content"],
                "sections_brief": ["The document content was processed, but section analysis failed."],
                "keywords": ["document"],
                "metadata": {
                    "likely_audience": "Unknown",
                    "document_purpose": "Unknown",
                    "temporal_context": "Unknown",
                    "authorship_insights": "Unknown"
                }
            }
    
    def _analyze_large_document(self, text: str) -> Dict[str, Any]:
        """
        Process large documents by splitting into multiple chunks, analyzing each,
        and synthesizing the results.
        
        Args:
            text: Full document text
            
        Returns:
            dict: Combined analysis results
        """
        # Determine chunking strategy based on document size
        doc_length = len(text)
        chunk_size = 30000  # Size of each chunk in characters
        overlap = 2000      # Overlap between chunks for context continuity
        
        # Calculate number of chunks needed
        num_chunks = max(1, math.ceil((doc_length - overlap) / (chunk_size - overlap)))
        
        # Create chunks with overlap
        chunks = []
        for i in range(num_chunks):
            start = max(0, i * (chunk_size - overlap))
            end = min(doc_length, start + chunk_size)
            chunks.append(text[start:end])
        
        # First pass - analyze each chunk separately
        chunk_results = []
        for i, chunk in enumerate(chunks):
            print(f"Analyzing chunk {i+1}/{len(chunks)}...")
            
            # Create chunk-specific prompt
            chunk_prompt = f"""
            You are analyzing chunk {i+1} of {len(chunks)} from a large document (total length: {doc_length} characters).
            
            Analyze this chunk in detail, providing:
            1. Summary of this chunk's content
            2. Main topics/themes present
            3. Key points found in this chunk
            4. Important terms/keywords
            5. How this chunk likely connects to the rest of the document
            
            Focus on the specific content in this chunk, but note potential connections to previous/future sections.
            
            Document chunk content:
            {chunk}
            
            Respond in JSON format:
            {{
                "chunk_summary": "summary of this specific chunk",
                "chunk_themes": ["theme1", "theme2", ...],
                "chunk_key_points": ["key point 1", "key point 2", ...],
                "chunk_keywords": ["keyword1", "keyword2", ...],
                "structure_analysis": {{"likely_position": "beginning/middle/end", "potential_section_titles": ["title1", "title2", ...]}}
            }}
            
            Be thorough but focused. Only return the JSON object.
            """
            
            # Call API for this chunk
            response = self._call_ollama_api(
                chunk_prompt,
                temperature=0.2,
                system_prompt="You are analyzing a portion of a larger document. Focus on extracting key information from this specific segment."
            )
            
            # Parse and store result
            try:
                json_str = self._extract_json(response)
                chunk_result = json.loads(json_str)
                chunk_results.append(chunk_result)
            except Exception as e:
                print(f"Error parsing chunk {i+1} response: {e}")
                # Add placeholder for failed chunk
                chunk_results.append({
                    "chunk_summary": f"[Analysis failed for chunk {i+1}]",
                    "chunk_themes": ["unknown"],
                    "chunk_key_points": [f"Failed to analyze chunk {i+1}"],
                    "chunk_keywords": ["unknown"],
                    "structure_analysis": {"likely_position": "unknown", "potential_section_titles": [f"Chunk {i+1}"]}
                })
        
        # Second pass - synthesize results from all chunks
        synthesis_prompt = self._create_synthesis_prompt(chunk_results, doc_length)
        
        # Call API for final synthesis
        synthesis_response = self._call_ollama_api(
            synthesis_prompt,
            temperature=0.3,
            system_prompt="You are synthesizing analyses of multiple chunks of a document into a comprehensive whole."
        )
        
        # Parse final result
        try:
            json_str = self._extract_json(synthesis_response)
            final_result = json.loads(json_str)
            
            # Ensure the result has the expected structure
            if isinstance(final_result.get("category"), dict):
                category_data = final_result["category"]
                final_result["category"] = category_data["primary"]
                final_result["subcategories"] = category_data.get("subcategories", [])
                final_result["category_rationale"] = category_data.get("rationale", "")
            
            # Add metadata if missing
            if "metadata" not in final_result:
                final_result["metadata"] = {
                    "likely_audience": "",
                    "document_purpose": "",
                    "temporal_context": "",
                    "authorship_insights": ""
                }
            
            # Add document size metadata
            final_result["metadata"]["document_size"] = f"{doc_length} characters ({len(chunks)} chunks analyzed)"
            
            return final_result
        except Exception as e:
            print(f"Error in final synthesis: {e}")
            print(f"Response was: {synthesis_response}")
            
            # Return a fallback result based on chunk analyses
            return self._create_fallback_synthesis(chunk_results, doc_length)
    
    def _create_synthesis_prompt(self, chunk_results: List[Dict[str, Any]], doc_length: int) -> str:
        """
        Create a prompt for synthesizing results from multiple document chunks.
        
        Args:
            chunk_results: List of analysis results from individual chunks
            doc_length: Full document length in characters
            
        Returns:
            str: Prompt for synthesis
        """
        # Extract key information from all chunks
        all_themes = []
        all_key_points = []
        all_keywords = []
        all_sections = []
        
        for i, result in enumerate(chunk_results):
            # Collect all themes
            themes = result.get("chunk_themes", [])
            all_themes.extend(themes)
            
            # Collect key points
            key_points = result.get("chunk_key_points", [])
            all_key_points.extend(key_points)
            
            # Collect keywords
            keywords = result.get("chunk_keywords", [])
            all_keywords.extend(keywords)
            
            # Collect section titles
            section_titles = result.get("structure_analysis", {}).get("potential_section_titles", [])
            if section_titles:
                position = result.get("structure_analysis", {}).get("likely_position", f"chunk {i+1}")
                all_sections.append(f"From {position} chunk: {', '.join(section_titles)}")
        
        # Create a consolidated summary of each chunk
        chunk_summaries = []
        for i, result in enumerate(chunk_results):
            summary = result.get("chunk_summary", f"[No summary for chunk {i+1}]")
            chunk_summaries.append(f"Chunk {i+1} summary: {summary}")
        
        # Build the synthesis prompt
        synthesis_prompt = f"""
        You are synthesizing analyses of multiple chunks from a document of approximately {doc_length} characters.
        
        Below are the analysis results from {len(chunk_results)} document chunks. Synthesize these into a comprehensive document analysis.
        
        CHUNK SUMMARIES:
        {' '.join(chunk_summaries)}
        
        DOCUMENT THEMES:
        {', '.join(all_themes)}
        
        KEY POINTS:
        {' '.join(all_key_points)}
        
        POTENTIAL SECTION STRUCTURE:
        {' '.join(all_sections)}
        
        PROMINENT KEYWORDS:
        {', '.join(all_keywords)}
        
        Based on these analyses, provide a comprehensive document analysis with the following structure:
        
        1. Create an integrated SUMMARY (30-50 sentences) that captures the full document's scope and message
        2. Determine the primary CATEGORY and SUBCATEGORIES for the document
        3. Compile and prioritize 15-30 IMPORTANT POINTS from across all chunks
        4. Create a coherent STRUCTURAL ANALYSIS with 5-15 logical sections and detailed summaries
        5. Select 10-20 most significant KEYWORDS from across the document
        6. Infer CONTEXTUAL METADATA about audience, purpose, temporal context and authorship
        
        Respond in JSON format with the following structure:
        {{
            "summary": "comprehensive document summary here with exceptional detail",
            "category": {{
                "primary": "main category",
                "subcategories": ["subcategory1", "subcategory2"],
                "rationale": "brief explanation of categorization"
            }},
            "important_points": [
                "First critical point as a comprehensive bullet point",
                "Second critical point with detailed insight",
                "Additional important points...",
                "Final important point"
            ],
            "sections_title": [
                "First section title",
                "Second section title",
                "Additional section titles...",
                "Final section title"
            ],
            "sections_brief": [
                "First section detailed summary with context and relationship to document purpose",
                "Second section detailed summary",
                "Additional section summaries...",
                "Final section detailed summary"
            ],
            "keywords": ["keyword1", "keyword2", "keyword3", ...],
            "metadata": {{
                "likely_audience": "description of intended audience",
                "document_purpose": "inferred purpose of document",
                "temporal_context": "any time references or dating information",
                "authorship_insights": "clues about authorship"
            }}
        }}
        
        Ensure your analysis is coherent, comprehensive, and represents the entire document. Only return the JSON object.
        """
        
        return synthesis_prompt
    
    def _create_fallback_synthesis(self, chunk_results: List[Dict[str, Any]], doc_length: int) -> Dict[str, Any]:
        """
        Create a fallback synthesis if the main synthesis fails.
        
        Args:
            chunk_results: List of analysis results from individual chunks
            doc_length: Full document length in characters
            
        Returns:
            dict: Synthesized analysis results
        """
        # Extract and combine information from chunks
        combined_summary = []
        all_themes = set()
        all_key_points = []
        all_keywords = set()
        section_titles = []
        section_briefs = []
        
        for i, result in enumerate(chunk_results):
            # Add summary
            summary = result.get("chunk_summary", f"Content from chunk {i+1}")
            combined_summary.append(summary)
            
            # Add themes
            themes = result.get("chunk_themes", [])
            all_themes.update(themes)
            
            # Add key points
            key_points = result.get("chunk_key_points", [])
            all_key_points.extend(key_points)
            
            # Add keywords
            keywords = result.get("chunk_keywords", [])
            all_keywords.update(keywords)
            
            # Add section titles and briefs
            potential_sections = result.get("structure_analysis", {}).get("potential_section_titles", [])
            for section in potential_sections:
                section_titles.append(section)
                section_briefs.append(f"Content related to '{section}' from document chunk {i+1}")
        
        # Create final result structure
        return {
            "summary": " ".join(combined_summary),
            "category": "Multiple Categories",
            "subcategories": list(all_themes)[:3],
            "category_rationale": "Based on combined analysis of document chunks",
            "important_points": all_key_points[:30],
            "sections_title": section_titles[:15],
            "sections_brief": section_briefs[:15],
            "keywords": list(all_keywords)[:20],
            "metadata": {
                "likely_audience": "Multiple potential audiences",
                "document_purpose": "Multiple potential purposes",
                "temporal_context": "Various temporal references across document",
                "authorship_insights": "Based on document style and content analysis",
                "document_size": f"{doc_length} characters ({len(chunk_results)} chunks analyzed)",
                "analysis_note": "This is a fallback synthesis due to synthesis processing error"
            }
        }
    
    def answer_question(self, question: str, context: str) -> str:
        """
        Answer a question based on document context with improved handling for large contexts.
        
        Args:
            question: User's question
            context: Document content to use as context
            
        Returns:
            str: Detailed answer to the question based on the context
        """
        context_length = len(context)
        
        # For very large contexts, use a multi-stage approach
        if context_length > 50000:
            return self._answer_question_large_context(question, context)
        
        # For medium to large contexts, use smart context selection
        max_context_length = 32000
        if context_length > max_context_length:
            context = self._smart_context_selection(question, context, max_length=max_context_length)
        
        # Create enhanced prompt with better instructions for detailed answers
        prompt = f"""
        You are an expert document analysis assistant with the ability to extract precise information, identify patterns, and provide comprehensive, well-structured answers.

        I'm going to provide you with content from one or more documents, followed by a question. Your task is to:

        1. Carefully analyze all provided document content
        2. Focus specifically on answering the question with complete accuracy and thoroughness
        3. Provide a comprehensive answer that includes ALL relevant information from the documents
        4. Structure your response logically with paragraphs or sections as appropriate
        5. Include direct quotations from the document when they strengthen your answer
        6. Cite specific sections or parts of the document when referencing information
        7. Note any quantitative data, figures, or statistics that support your answer
        8. Acknowledge when information appears contradictory and explain the nuance
        9. Provide context around your answer when it helps understanding

        DOCUMENT CONTENT:
        {context}

        QUESTION: {question}

        If the answer cannot be definitively determined from the document(s), explicitly state this limitation, then:
        1. Explain what information would be needed to provide a complete answer
        2. Offer your best reasoned inference based on the available information
        3. Clarify why you believe this inference is reasonable given the context
        4. Identify any assumptions you're making

        Format your response as a detailed, clear, and comprehensive answer that directly addresses the question. Include relevant quotations, data points, and specific references when available. Your goal is to provide the most helpful, accurate, and complete response possible based solely on the document content provided.

        ANSWER:
        """
        
        # Make API call to Ollama with enhanced parameters
        response = self._call_ollama_api(
            prompt,
            temperature=0.3,  # Lower temperature for more factual responses
            system_prompt="You are an expert document analyst who provides comprehensive, detailed answers based solely on the document content provided."
        )
        
        return response
    
    def _answer_question_large_context(self, question: str, context: str) -> str:
        """
        Answer a question when the context is very large using a multi-stage approach.
        
        Args:
            question: User's question
            context: Large document content
            
        Returns:
            str: Answer to question based on relevant parts of context
        """
        # Stage 1: Split context into manageable chunks with overlap
        chunks = self._split_into_chunks(context, chunk_size=20000, overlap=1000)
        
        # Stage 2: Find most relevant chunks for the question
        relevant_chunks = self._find_relevant_chunks(question, chunks)
        
        # Stage 3: For each relevant chunk, get a preliminary answer
        preliminary_answers = []
        for i, chunk in enumerate(relevant_chunks):
            chunk_prompt = f"""
            You are analyzing a portion (chunk {i+1} of {len(relevant_chunks)}) of a large document to answer a specific question.
            
            DOCUMENT CHUNK:
            {chunk}
            
            QUESTION: {question}
            
            Based ONLY on the information in this document chunk, provide:
            1. A partial answer to the question based on this chunk's content
            2. Highlight any specific information, quotes, or data points from this chunk that are relevant
            3. Note if this chunk alone is insufficient to fully answer the question
            4. Indicate what additional information might be needed
            
            Focus only on what's present in this chunk, not on general knowledge. Be specific and precise.
            """
            
            response = self._call_ollama_api(
                chunk_prompt,
                temperature=0.2,
                system_prompt="You are analyzing a document chunk to find relevant information for a specific question."
            )
            
            preliminary_answers.append(response)
        
        # Stage 4: Synthesize all preliminary answers into a final comprehensive answer
        synthesis_prompt = f"""
        You are synthesizing information from multiple document chunks to provide a comprehensive answer to a question.
        
        QUESTION: {question}
        
        The following are preliminary answers based on different chunks of the document:
        
        {" ".join([f"CHUNK {i+1} ANALYSIS:\n{answer}\n\n" for i, answer in enumerate(preliminary_answers)])}
        
        Based on all available information across these document chunks, provide a comprehensive, well-structured answer to the original question. Your answer should:
        
        1. Synthesize all relevant information from the different chunks
        2. Resolve any contradictions or inconsistencies between chunks, if present
        3. Provide direct quotes or specific references when appropriate
        4. Acknowledge any limitations in the available information
        5. Be thorough, accurate, and directly address the question
        
        Your goal is to provide the most complete and accurate answer possible based solely on the document content analyzed.
        """
        
        final_answer = self._call_ollama_api(
            synthesis_prompt,
            temperature=0.3,
            system_prompt="You are synthesizing information from multiple document chunks to provide a comprehensive answer."
        )
        
        return final_answer
    
    def _split_into_chunks(self, text: str, chunk_size: int = 20000, overlap: int = 1000) -> List[str]:
        """
        Split large text into overlapping chunks with improved memory efficiency.
        
        Args:
            text: Large text to split
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks in characters
            
        Returns:
            list: List of text chunks
        """
        chunks = []
        text_length = len(text)
        
        # Use generators and iterators instead of creating multiple large string copies
        # Create only the indices first, then extract chunks
        chunk_indices = []
        
        current_pos = 0
        while current_pos < text_length:
            # Determine end position for current chunk
            end_pos = min(current_pos + chunk_size, text_length)
            
            # If not at the end of text and not at a natural break, find a good break point
            if end_pos < text_length:
                # Look for paragraph break near the end position
                paragraph_break = text.rfind("\n\n", current_pos, end_pos)
                if paragraph_break != -1 and paragraph_break > current_pos + (chunk_size / 2):
                    # Found a paragraph break in the second half of the chunk
                    end_pos = paragraph_break + 2  # Include the newlines
                else:
                    # No good paragraph break, try sentence break
                    sentence_break = text.rfind(". ", current_pos, end_pos)
                    if sentence_break != -1 and sentence_break > current_pos + (chunk_size / 2):
                        end_pos = sentence_break + 2  # Include the period and space
            
            # Store indices rather than creating string copies immediately
            chunk_indices.append((current_pos, end_pos))
            
            # Move position for next chunk, with overlap
            current_pos = end_pos - overlap
            
            # Ensure we don't get stuck if no good break points
            if current_pos < text_length and len(chunk_indices) > 1 and current_pos <= (current_pos - chunk_size + overlap):
                current_pos = min(current_pos + chunk_size // 2, text_length)
    
        # Process chunks in batches to reduce memory pressure
        batch_size = 3  # Process a few chunks at a time
        for i in range(0, len(chunk_indices), batch_size):
            batch_indices = chunk_indices[i:i+batch_size]
            for start, end in batch_indices:
                # Create chunk only when needed
                chunk = text[start:end]
                chunks.append(chunk)
                
                # Force garbage collection after each batch to free memory
                gc.collect()
    
        return chunks
    
    def _find_relevant_chunks(self, question: str, chunks: List[str], max_chunks: int = 3) -> List[str]:
        """
        Find the most relevant chunks for answering the question.
        
        Args:
            question: User's question
            chunks: List of document chunks
            max_chunks: Maximum number of chunks to return
            
        Returns:
            list: List of most relevant chunks
        """
        if len(chunks) <= max_chunks:
            return chunks  # Return all chunks if we have fewer than max_chunks
        
        # For each chunk, evaluate its relevance to the question
        chunk_relevance = []
        
        for i, chunk in enumerate(chunks):
            relevance_prompt = f"""
            QUESTION: {question}
            
            DOCUMENT CHUNK:
            {chunk[:5000]}  # Use start of chunk for efficiency
            
            On a scale of 1-10, how relevant is this document chunk to answering the question?
            Consider:
            - Direct mentions of concepts in the question
            - Information that would be needed to provide a complete answer
            - Context that explains terminology or concepts in the question
            
            Respond with a single number from view from 1-10, with 10 being highest relevance.
            """
            
            response = self._call_ollama_api(
                relevance_prompt,
                temperature=0.1,
                system_prompt="You are evaluating the relevance of document chunks to a specific question."
            )
            
            # Extract the relevance score from the response
            score = 5  # Default score
            try:
                # Look for single number in response
                scores = re.findall(r'\b([0-9]|10)\b', response)
                if scores:
                    score = int(scores[0])
            except Exception:
                # If parsing fails, use default
                pass
            
            chunk_relevance.append((i, score, chunk))
        
        # Sort chunks by relevance score (descending)
        chunk_relevance.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top max_chunks chunks
        top_chunks = [chunk for _, _, chunk in chunk_relevance[:max_chunks]]
        
        return top_chunks
    
    def _smart_sample_text(self, text: str, max_length: int = 32000) -> str:
        """
        Intelligently sample text to stay within size limits while preserving key content.
        
        Args:
            text: Original text
            max_length: Maximum length in characters
            
        Returns:
            str: Sampled text
        """
        text_length = len(text)
        
        # If text is already within limits, return it intact
        if text_length <= max_length:
            return text
        
        # For slightly longer texts, use beginning and end with more weight on beginning
        if text_length < max_length * 2:
            # Use 70% beginning, 30% end
            beginning_chars = int(max_length * 0.7)
            ending_chars = max_length - beginning_chars - 50  # 50 chars for the ellipsis part
            
            beginning = text[:beginning_chars]
            ending = text[-ending_chars:] if ending_chars > 0 else ""
            
            return beginning + "\n\n[...content omitted for length...]\n\n" + ending

    def _extract_json(self, text):
        """
        Extract JSON from text that might contain additional content with improved reliability.
        
        Args:
            text: Text that contains JSON
            
        Returns:
            str: Extracted JSON string
        """
        # Pattern to match JSON objects, handling nested structures
        json_pattern = r'(\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}))*\})'
        
        # First try to match the entire JSON object
        match = re.search(json_pattern, text)
        if match:
            potential_json = match.group(0)
            try:
                # Validate it's proper JSON by parsing it
                json.loads(potential_json)
                return potential_json
            except json.JSONDecodeError:
                pass
        
        # If regex approach fails, try bracketing approach
        try:
            # Look for JSON patterns
            start_idx = text.find("{")
            if start_idx == -1:
                raise ValueError("No JSON object found in response")
            
            # Find matching closing brace
            brace_count = 0
            for i in range(start_idx, len(text)):
                if text[i] == "{":
                    brace_count += 1
                elif text[i] == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        # Found matching closing brace
                        json_str = text[start_idx:i+1]
                        # Validate by parsing
                        json.loads(json_str)
                        return json_str
            
            # If we get here, no matching closing brace was found
            raise ValueError("Malformed JSON in response - no closing brace found")
        
        except (ValueError, json.JSONDecodeError) as e:
            # Last resort: try to fix common JSON issues
            try:
                # Replace common issues like unquoted keys
                cleaned_text = re.sub(r'(\w+)(:)', r'"\1"\2', text)
                # Fix single quotes to double quotes
                cleaned_text = cleaned_text.replace("'", '"')
                
                # Find the JSON in the cleaned text
                start_idx = cleaned_text.find("{")
                end_idx = cleaned_text.rfind("}")
                
                if start_idx != -1 and end_idx != -1:
                    json_str = cleaned_text[start_idx:end_idx+1]
                    # Validate by parsing
                    json.loads(json_str)
                    return json_str
            except:
                pass
            
            # If all attempts fail, raise the original error
            raise ValueError(f"Could not extract valid JSON from response: {str(e)}")

    def _call_ollama_api(self, prompt, max_retries=3, retry_delay=2, temperature=None, system_prompt=None):
        """
        Call the Ollama API with retry logic and enhanced parameters.
        
        Args:
            prompt: The prompt to send to the model
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            temperature: Optional temperature override
            system_prompt: Optional system prompt to guide model behavior
            
        Returns:
            str: The model's response text
        """
        headers = {"Content-Type": "application/json"}
        
        # Set default values if not provided
        temp = temperature if temperature is not None else self.temperature
        
        # Prepare the API request data
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temp,
                "num_predict": self.max_tokens,
                "top_p": 0.95,        # Focus on more likely tokens
                "top_k": 40,          # Consider top 40 tokens
                "presence_penalty": 0.2,  # Slightly discourage repetition
                "frequency_penalty": 0.2  # Slightly discourage frequent tokens
            }
        }
        
        # Add system prompt if provided
        if system_prompt:
            data["system"] = system_prompt
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    headers=headers,
                    json=data,
                    timeout=120  # Increased timeout for longer responses
                )
                
                if response.status_code == 200:
                    return response.json().get("response", "")
                
                # If server error, wait and retry
                if response.status_code >= 500:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue
                    
                # Other errors
                response.raise_for_status()
            
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    raise Exception(f"Failed to call Ollama API after {max_retries} attempts: {str(e)}")
        
        return ""