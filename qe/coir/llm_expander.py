"""
LLM-based Query Expansion module.
Implements HyDE (Hypothetical Document Embeddings), technical enrichment, and chain-of-thought expansion.
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExpansionResult:
    """Result of query expansion."""
    original_query: str
    expanded_query: str
    expansion_terms: List[str]
    method: str
    metadata: Dict[str, Any]


class LLMExpander:
    """
    LLM-based Query Expander using HyDE-style expansion.
    
    Supports multiple expansion strategies:
    - Direct Translation: Simple Indonesian to English
    - HyDE: Generate hypothetical documents
    - Technical Enrichment: Add domain-specific terms
    - Chain-of-Thought: Explain expansion rationale
    """
    
    # Default prompt templates
    HIDE_PROMPT = """Given an Indonesian query, generate a hypothetical English document/code snippet 
that would be relevant to this query. The hypothetical document should contain technical terms 
and code that matches the user's intent.

Indonesian Query: {query}

Generate a hypothetical English document (1-2 sentences):"""

    TECHNICAL_ENRICHMENT_PROMPT = """Given an Indonesian query related to programming/code, 
expand it with relevant English technical terms, library names, and API concepts.

Indonesian Query: {query}

Task: 
1. Translate the query to English
2. Add relevant technical terms (e.g., library names, function names, concepts)
3. Provide the expanded query

Expanded terms (comma-separated):"""

    CHAIN_OF_THOUGHT_PROMPT = """Given an Indonesian query, explain step-by-step what the user is looking for 
and generate an expanded English query with technical terms.

Indonesian Query: {query}

Follow these steps:
1. Identify the intent: What does the user want to accomplish?
2. Translate to English: What English words would they use?
3. Add technical terms: What libraries, functions, or concepts are relevant?

Provide your response in JSON format:
{{
    "intent": "...",
    "translation": "...",
    "technical_terms": ["...", "..."],
    "expanded_query": "..."
}}"""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 512,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the LLM expander.
        
        Args:
            provider: LLM provider ("openai" or "google")
            model: Model name
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate
            api_key: API key (will use env var if not provided)
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client."""
        if self.provider == "openai":
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                logger.info(f"Initialized OpenAI client with model: {self.model}")
            except ImportError:
                logger.warning("OpenAI client not available. Install with: pip install openai")
                
        elif self.provider == "google":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai
                logger.info(f"Initialized Google Gemini client with model: {self.model}")
            except ImportError:
                logger.warning("Google GenerativeAI not available. Install with: pip install google-generativeai")
    
    def expand(
        self,
        query: str,
        method: str = "hyde",
        num_terms: int = 5,
    ) -> ExpansionResult:
        """
        Expand a query using LLM.
        
        Args:
            query: Input query (Indonesian)
            method: Expansion method ("hyde", "technical", "cot")
            num_terms: Number of expansion terms to extract
            
        Returns:
            ExpansionResult with expanded query and terms
        """
        if method == "hyde":
            return self._expand_hyde(query, num_terms)
        elif method == "technical":
            return self._expand_technical(query, num_terms)
        elif method == "cot":
            return self._expand_cot(query, num_terms)
        else:
            raise ValueError(f"Unknown expansion method: {method}")
    
    def _expand_hyde(self, query: str, num_terms: int) -> ExpansionResult:
        """HyDE-style expansion: Generate hypothetical document."""
        prompt = self.HIDE_PROMPT.format(query=query)
        
        response = self._generate(prompt)
        
        # Extract terms from response
        terms = self._extract_terms_from_text(response, num_terms)
        
        # Create expanded query
        expanded_query = f"{query} {response}"
        
        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=terms,
            method="hyde",
            metadata={"hypothetical_doc": response}
        )
    
    def _expand_technical(self, query: str, num_terms: int) -> ExpansionResult:
        """Technical enrichment expansion."""
        prompt = self.TECHNICAL_ENRICHMENT_PROMPT.format(query=query)
        
        response = self._generate(prompt)
        
        # Extract terms
        terms = self._extract_terms_from_text(response, num_terms)
        
        # Create expanded query
        expanded_query = f"{query} {' '.join(terms)}"
        
        return ExpansionResult(
            original_query=query,
            expanded_query=expanded_query,
            expansion_terms=terms,
            method="technical",
            metadata={"raw_response": response}
        )
    
    def _expand_cot(self, query: str, num_terms: int) -> ExpansionResult:
        """Chain-of-thought expansion."""
        prompt = self.CHAIN_OF_THOUGHT_PROMPT.format(query=query)
        
        response = self._generate(prompt)
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                terms = parsed.get("technical_terms", [])[:num_terms]
                expanded_query = parsed.get("expanded_query", query)
                
                return ExpansionResult(
                    original_query=query,
                    expanded_query=expanded_query,
                    expansion_terms=terms,
                    method="cot",
                    metadata=parsed
                )
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse CoT response as JSON")
        
        # Fallback
        terms = self._extract_terms_from_text(response, num_terms)
        
        return ExpansionResult(
            original_query=query,
            expanded_query=f"{query} {' '.join(terms)}",
            expansion_terms=terms,
            method="cot",
            metadata={"raw_response": response}
        )
    
    def _generate(self, prompt: str) -> str:
        """Generate text using LLM."""
        if self.provider == "openai":
            return self._generate_openai(prompt)
        elif self.provider == "google":
            return self._generate_google(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API."""
        if self._client is None:
            return prompt  # Fallback to original
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            return ""
    
    def _generate_google(self, prompt: str) -> str:
        """Generate using Google Gemini API."""
        if self._client is None:
            return prompt  # Fallback to original
        
        try:
            model = self._client.GenerativeModel(self.model)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Google generation failed: {e}")
            return ""
    
    def _extract_terms_from_text(self, text: str, num_terms: int) -> List[str]:
        """Extract expansion terms from text."""
        # Simple extraction: split by common delimiters
        text = text.lower()
        
        # Remove common words and punctuation
        for char in [".", ",", "!", "?", ";", ":", "\n", "\t"]:
            text = text.replace(char, " ")
        
        words = text.split()
        
        # Filter short words and common stopwords
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
            "into", "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once", "here", "there",
            "when", "where", "why", "how", "all", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "just", "and", "but", "if", "or", "because",
            "until", "while", "this", "that", "these", "those", "what", "which",
            "who", "whom", "its", "it", "i", "you", "he", "she", "we", "they",
        }
        
        terms = [w for w in words if len(w) > 2 and w not in stopwords]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        return unique_terms[:num_terms]
    
    def expand_batch(
        self,
        queries: List[str],
        method: str = "hyde",
        num_terms: int = 5,
    ) -> List[ExpansionResult]:
        """
        Expand multiple queries.
        
        Args:
            queries: List of input queries
            method: Expansion method
            num_terms: Number of expansion terms
            
        Returns:
            List of ExpansionResults
        """
        results = []
        for query in queries:
            result = self.expand(query, method, num_terms)
            results.append(result)
        
        return results


class LocalLLMExpander(LLMExpander):
    """
    LLM expander using local models (e.g., via Ollama).
    """
    
    def __init__(
        self,
        model: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        """
        Initialize local LLM expander.
        
        Args:
            model: Local model name
            base_url: Ollama API base URL
            temperature: Generation temperature
            max_tokens: Maximum tokens
        """
        super().__init__(
            provider="local",
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.base_url = base_url
        self._client = None
        self._initialize_local_client()
    
    def _initialize_local_client(self):
        """Initialize local client (Ollama)."""
        try:
            import requests
            self._client = requests
            logger.info(f"Initialized local client with model: {self.model}")
        except ImportError:
            logger.warning("Requests not available")
    
    def _generate(self, prompt: str) -> str:
        """Generate using local Ollama API."""
        if self._client is None:
            return prompt
        
        try:
            response = self._client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
            )
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            logger.error(f"Local generation failed: {e}")
            return ""
    
    def _generate_google(self, prompt: str) -> str:
        raise NotImplementedError("Google not available for LocalLLMExpander")
    
    def _generate_openai(self, prompt: str) -> str:
        raise NotImplementedError("OpenAI not available for LocalLLMExpander")
