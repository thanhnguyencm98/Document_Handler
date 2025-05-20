import json
import requests
import time

class LLMService:
    """
    Service for interacting with the Ollama Llama 3.1 model.
    """
    
    def __init__(self, model="llama3.1", api_url="http://localhost:11434"):
        """
        Initialize the LLM service.
        
        Args:
            model: Name of the Ollama model to use
            api_url: URL of the Ollama API
        """
        self.model = model
        self.api_url = api_url
    
    def analyze_document(self, text):
        """
        Analyze document text to generate summary, category, key concepts, and keywords.
        
        Args:
            text: Document text to analyze
            
        Returns:
            dict: Analysis results containing summary, category, key_concepts, and keywords
        """
        # Prepare a sample of the document text (first 4000 chars or so to keep prompt size reasonable)
        text_sample = text[:4000] + ("..." if len(text) > 4000 else "")
        
        # Create the prompt
        prompt = f"""
        Analyze the following document content and provide:
        1. A detail summary (10-20 sentences)
        2. A category that best describes this document (e.g., Financial, Technical, Legal, Educational, Marketing, etc.)
        3. A list of 5-20 key concepts in detail or important points from the document, presented as separate bullet points
        4. A list of 5-10 relevant keywords

        Document content:
        {text_sample}

        Respond in JSON format with the following structure:
        {{
            "summary": "document summary here",
            "category": "document category here",
            "key_concepts": [
                "First key concept or important point as a complete bullet point",
                "Second key concept or important point as a complete bullet point",
                "Additional key concepts..."
            ],
            "keywords": ["keyword1", "keyword2", "keyword3", ...]
        }}

        Only return the JSON object, nothing else.
        """
        
        # Make API call to Ollama
        response = self._call_ollama_api(prompt)
        
        # Parse the response to extract JSON
        try:
            # Extract JSON from the response text
            json_str = self._extract_json(response)
            result = json.loads(json_str)
            
            # Ensure expected fields are present
            if not all(key in result for key in ["summary", "category", "key_concepts", "keywords"]):
                raise ValueError("Response missing required fields")
            
            return result
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {response}")
            
            # Return a default structure if parsing fails
            return {
                "summary": "Failed to generate summary. The document was processed, but the analysis failed.",
                "category": "Uncategorized",
                "key_concepts": ["The document was processed, but key concepts extraction failed."],
                "keywords": ["document"]
            }
    
    def _call_ollama_api(self, prompt, max_retries=3, retry_delay=2):
        """
        Call the Ollama API with retry logic.
        
        Args:
            prompt: The prompt to send to the model
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            
        Returns:
            str: The model's response text
        """
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    return response.json().get("response", "")
                
                # If server error, wait and retry
                if response.status_code >= 500:
                    time.sleep(retry_delay)
                    continue
                    
                # Other errors
                response.raise_for_status()
            
            except (requests.RequestException, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise Exception(f"Failed to call Ollama API after {max_retries} attempts: {str(e)}")
        
        return ""
    
    def _extract_json(self, text):
        """
        Extract JSON from text that might contain additional content.
        
        Args:
            text: Text that contains JSON
            
        Returns:
            str: Extracted JSON string
        """
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
                    return text[start_idx:i+1]
        
        # If we get here, no matching closing brace was found
        raise ValueError("Malformed JSON in response")