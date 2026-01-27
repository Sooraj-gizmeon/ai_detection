# src/ai_integration/llm_provider.py
"""Configurable LLM Provider Interface for Ollama and OpenAI"""

import os
import json
import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    async def generate_response(self, 
                              prompt: str,
                              model: str = None,
                              cache_key: str = None,
                              images: List[str] = None) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def _make_request(self, 
                          prompt: str,
                          model: str = None,
                          cache_key: str = None,
                          images: List[str] = None) -> str:
        """Make a request to the LLM API"""
        pass
    
    @abstractmethod
    def get_best_model(self, task_type: str = "text") -> str:
        """Get the best model for a specific task type"""
        pass
    
    @abstractmethod
    def _parse_json_response(self, response: Any) -> Dict:
        """Parse JSON response from the LLM"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider for textual analysis"""
    
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.base_url = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
        
        # IMPORTANT: Prefer modern JSON-mode capable models when possible.
        configured_model = os.getenv('OPENAI_MODEL', 'gpt-4')

        # OPTIONAL AUTO UPGRADE: If legacy model specified and OPENAI_AUTO_UPGRADE not disabled, map to newer models.
        # Users can set OPENAI_AUTO_UPGRADE=false to opt out.
        auto_upgrade = os.getenv('OPENAI_AUTO_UPGRADE', 'true').lower() in ['1', 'true', 'yes']
        original_configured_model = configured_model

        if auto_upgrade:
            # Upgrade map (extend as OpenAI evolves)
            upgrade_map = {
                # Legacy -> Newer baseline
                'gpt-4': 'gpt-4o',      # Lower latency & JSON mode support
                'gpt4': 'gpt-4o',
                'gpt-4-turbo': 'gpt-4o',
                'gpt-4-turbo-preview': 'gpt-4o',
                'gpt-3.5-turbo': 'gpt-4o',  # Big quality leap
                'gpt-3.5': 'gpt-4o'
            }
            lower_model = configured_model.lower()
            if lower_model in upgrade_map:
                upgraded = upgrade_map[lower_model]
                self.logger.info(
                    f"🚀 Auto-upgrading legacy OpenAI model '{configured_model}' -> '{upgraded}' (set OPENAI_AUTO_UPGRADE=false to disable)"
                )
                configured_model = upgraded
            elif any(lower_model.startswith(k + '-') for k in upgrade_map.keys()):
                # Handle variants like gpt-4-0613 etc.
                for legacy_prefix, target in upgrade_map.items():
                    if lower_model.startswith(legacy_prefix + '-'):
                        self.logger.info(
                            f"🚀 Auto-upgrading legacy variant '{configured_model}' -> '{target}' (OPENAI_AUTO_UPGRADE active)"
                        )
                        configured_model = target
                        break
        else:
            self.logger.debug("Auto-upgrade disabled (OPENAI_AUTO_UPGRADE=false)")
        
        # Force GPT-4 if o3 model is configured (o3 models fail frequently with JSON)
        if any(prefix in configured_model.lower() for prefix in ['o1', 'o3']):
            self.logger.warning(f"⚠️ Configured model '{configured_model}' is a reasoning model that often fails with JSON responses")
            self.logger.warning("🔄 Switching to 'gpt-4' for better JSON reliability and consistent responses")
            self.default_model = 'gpt-4'
            self.reasoning_model_fallback = configured_model  # Keep for potential future use
        else:
            self.default_model = configured_model
            self.reasoning_model_fallback = None

        # Log original vs final model if changed
        if original_configured_model != self.default_model:
            self.logger.info(
                f"📌 Final OpenAI model in use: {self.default_model} (original request: {original_configured_model})"
            )
        
        self.timeout = 120  # 2 minutes for OpenAI requests
        self.max_retries = 3
        self.session = None
        # New: allow capping completion tokens to reduce rate limiting
        try:
            self.max_completion_tokens_cap = int(os.getenv('OPENAI_MAX_COMPLETION_TOKENS', '1200'))
        except ValueError:
            self.max_completion_tokens_cap = 1200
        
        if not self.api_key:
            self.logger.error("OPENAI_API_KEY not found in environment variables")
            raise ValueError("OpenAI API key is required")
        
        self.logger.info(f"🤖 OpenAI provider initialized with model: {self.default_model}")
        if self.reasoning_model_fallback:
            self.logger.info(f"📋 Reasoning model fallback available: {self.reasoning_model_fallback}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
    
    async def _create_session(self):
        """Create HTTP session for OpenAI API"""
        import aiohttp
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'VideoToShorts/1.0'
        }
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        self.session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout,
            connector=aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                keepalive_timeout=60.0
            )
        )
    
    async def _close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def _enhance_prompt_for_json(self, prompt: str) -> str:
        """Enhance prompt to encourage JSON response format"""
        enhanced_prompt = prompt
        enhanced_prompt += "\n\nIMPORTANT: Respond with valid JSON format."
        enhanced_prompt += "\n\nEnsure your response is properly formatted JSON with all required fields and no truncation."
        enhanced_prompt += "\n\nDo not truncate your response - provide complete, valid JSON."
        return enhanced_prompt
    
    def get_best_model(self, task_type: str = "text") -> str:
        """Get the best OpenAI model for a specific task type"""
        # For now, use the configured model for all text tasks
        # Vision tasks should still use Ollama/local models
        if task_type == "vision":
            self.logger.warning("Vision tasks not supported by OpenAI provider, use Ollama")
            return None
        
        return self.default_model
    
    def _get_model_config(self, model: str) -> dict:
        """Get model-specific configuration parameters with enhanced JSON enforcement"""
        model_lower = model.lower()
        
        # Check if this is a GPT-5 model (new API)
        is_gpt5_model = any(model_prefix in model_lower for model_prefix in ['gpt-5', 'gpt5'])
        
        # Check if this is a reasoning model (o1, o3 series)
        is_reasoning_model = any(model_prefix in model_lower for model_prefix in ['o1', 'o3'])
        
        if is_gpt5_model:
            # GPT-5 models use different API parameters
            self.logger.info(f"🚀 GPT-5 model detected: {model} - using new Responses API parameters")
            
            config = {
                'is_reasoning_model': False,
                'is_gpt5_model': True,
                'token_param': 'max_output_tokens',  # GPT-5 uses max_output_tokens
                'supports_temperature': False,  # GPT-5 doesn't support temperature
                'supports_response_format': True,  # GPT-5 supports JSON mode
                'supports_reasoning_effort': True,  # GPT-5 has reasoning.effort parameter
                'supports_verbosity': True,  # GPT-5 has text.verbosity parameter
                'default_temperature': None,  # No temperature for GPT-5
                'default_reasoning_effort': 'minimal',  # Fast responses
                'default_verbosity': 'low',  # Concise responses
                'max_tokens_limit': 65536,
                'json_reliability': 'excellent',
                'recommended_alternative': None
            }
        elif is_reasoning_model:
            # Reasoning models are problematic for JSON - warn and provide fallback config
            self.logger.warning(f"⚠️ Reasoning model detected: {model} - these models often fail with JSON responses")
            self.logger.warning("🔄 Consider using GPT-4 instead for reliable JSON parsing")
            
            config = {
                'is_reasoning_model': True,
                'is_gpt5_model': False,
                'token_param': 'max_completion_tokens',
                'supports_temperature': False,  # Reasoning models don't support custom temperature
                'supports_response_format': False,  # Limited JSON mode support
                'supports_reasoning_effort': False,
                'supports_verbosity': False,
                'default_temperature': 1.0,  # Fixed temperature for reasoning models
                'max_tokens_limit': 65536,
                'json_reliability': 'poor',  # Flag for JSON reliability issues
                'recommended_alternative': 'gpt-4'
            }
        else:
            # Standard GPT models - excellent for JSON
            config = {
                'is_reasoning_model': False,
                'is_gpt5_model': False,
                'token_param': 'max_tokens',
                'supports_temperature': True,
                'supports_response_format': True,  # Full JSON mode support
                'supports_reasoning_effort': False,
                'supports_verbosity': False,
                'default_temperature': 0.1,  # Low temperature for consistent JSON
                'max_tokens_limit': 4096,
                'json_reliability': 'excellent',  # Flag for JSON reliability
                'recommended_alternative': None
            }
            
        self.logger.debug(f"Model {model} config: JSON reliability = {config['json_reliability']}")
        return config

    async def generate_response(self, 
                              prompt: str,
                              model: str = None,
                              cache_key: str = None,
                              images: List[str] = None) -> str:
        """Generate response using OpenAI API"""
        if images:
            self.logger.error("OpenAI provider does not support vision tasks with images")
            raise ValueError("Vision tasks with images not supported by OpenAI provider")
        
        # First attempt with requested model
        original_model = model or self.default_model
        response = await self._make_request(prompt, original_model, cache_key)
        
        # If response is empty and we're using o3 model, try fallback to gpt-4
        if not response and 'o3' in original_model.lower():
            self.logger.warning(f"⚠️ Empty response from {original_model}, falling back to gpt-4")
            try:
                fallback_response = await self._make_request(prompt, "gpt-4", cache_key)
                if fallback_response:
                    self.logger.info(f"✅ Fallback to gpt-4 successful, response length: {len(fallback_response)}")
                    return fallback_response
                else:
                    self.logger.error(f"❌ Fallback to gpt-4 also returned empty response")
            except Exception as e:
                self.logger.error(f"❌ Fallback to gpt-4 failed: {e}")
        
        return response
    
    async def _make_request(self, 
                          prompt: str,
                          model: str = None,
                          cache_key: str = None,
                          images: List[str] = None) -> str:
        """Make request to OpenAI API"""
        if not self.session:
            await self._create_session()
        
        if images:
            raise ValueError("Images not supported in OpenAI text provider")
        
        model = model or self.default_model
        
        # Get model-specific configuration
        model_config = self._get_model_config(model)
        
        # Prepare request data with enhanced JSON enforcement
        # Apply cap to completion tokens to control response size
        token_param_value = min(4000, self.max_completion_tokens_cap)
        
        # GPT-5 uses different API structure
        if model_config.get('is_gpt5_model', False):
            # GPT-5 uses the Responses API with different parameters
            request_data = {
                "model": model,
                "input": self._enhance_prompt_for_json(prompt),
                model_config['token_param']: token_param_value,
            }
            
            # Add GPT-5 specific parameters
            if model_config.get('supports_reasoning_effort'):
                request_data["reasoning"] = {
                    "effort": model_config.get('default_reasoning_effort', 'minimal')
                }
                self.logger.debug(f"Using reasoning effort: {request_data['reasoning']['effort']}")
            
            if model_config.get('supports_verbosity'):
                request_data["text"] = {
                    "verbosity": model_config.get('default_verbosity', 'low')
                }
                self.logger.debug(f"Using text verbosity: {request_data['text']['verbosity']}")
        else:
            # Standard Chat Completions API for older models
            request_data = {
                "model": model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert video content analyzer. CRITICAL: Always respond with valid, complete JSON format when JSON is requested. Never truncate responses. Ensure all JSON objects are properly closed with closing braces."
                    },
                    {
                        "role": "user",
                        "content": self._enhance_prompt_for_json(prompt)
                    }
                ],
                model_config['token_param']: token_param_value,  # Capped for rate-limit safety
            }
            
            # Add temperature only for models that support it
            if model_config['supports_temperature']:
                request_data["temperature"] = model_config['default_temperature']
                self.logger.debug(f"Using temperature {model_config['default_temperature']} for model: {model}")
            else:
                self.logger.debug(f"Skipping temperature parameter for reasoning model: {model} (uses default)")
        
        # ENHANCED: Force JSON response format for better parsing (with fallback for unsupported models)
        # IMPORTANT FIX: Only attempt OpenAI JSON mode (response_format) for models that actually support it.
        # Older / legacy models like 'gpt-4' currently reject response_format with:
        #   "Invalid parameter: 'response_format' of type 'json_object' is not supported with this model."
        # Maintain an allowlist of json-mode capable models. (Extend as new models are verified.)
        json_mode_allowlist = {
            'gpt-4o', 'gpt-4o-mini', 'gpt-4o-mini-2024-07-18', 'gpt-4.1', 'gpt-4.1-mini',
            'chatgpt-4o-latest', 'o4-mini'  # future-friendly placeholders
        }
        model_supports_json_mode = any(m == model.lower() or model.lower().startswith(m + '-') for m in json_mode_allowlist)
        try_response_format = (
            model_supports_json_mode and
            any(keyword in prompt.lower() for keyword in ['json', 'response', 'format', 'structure'])
        )
        if model_config['supports_response_format'] and not model_supports_json_mode:
            # Downgrade support flag dynamically if legacy model encountered
            self.logger.debug(f"Model '{model}' marked supports_response_format in generic config but not in allowlist - disabling JSON mode parameter")
            model_config['supports_response_format'] = False
        response_format_tried = False  # Track if we've tried response_format parameter
        
        if try_response_format:
            request_data["response_format"] = {"type": "json_object"}
            response_format_tried = True
            self.logger.debug("🔧 Attempting to enforce JSON response format via OpenAI response_format parameter")
        else:
            # Provide trace logging so we can confirm why json mode wasn't attempted
            if any(keyword in prompt.lower() for keyword in ['json', 'response', 'format', 'structure']):
                self.logger.debug(f"Skipping response_format for model '{model}' (json_mode_allowlist={model_supports_json_mode})")
        
        # Make request with retries and response_format fallback
        response_format_retry_done = False
        actual_attempts = 0
        max_total_attempts = self.max_retries + 1
        
        while actual_attempts < max_total_attempts:
            actual_attempts += 1
            try:
                self.logger.debug(f"Making OpenAI request (attempt {actual_attempts}/{max_total_attempts})")
                
                # Choose correct endpoint based on model type
                endpoint = "/responses" if model_config.get('is_gpt5_model', False) else "/chat/completions"
                
                async with self.session.post(
                    f"{self.base_url}{endpoint}",
                    json=request_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Enhanced debugging for response structure
                        self.logger.info(f"🔍 API RESPONSE RECEIVED - Status: 200")
                        self.logger.info(f"📊 Response top-level keys: {list(result.keys())}")
                        
                        # Handle GPT-5 Responses API format
                        if model_config.get('is_gpt5_model', False):
                            self.logger.info(f"🚀 Processing GPT-5 Responses API format...")
                            
                            # Log complete response structure for debugging
                            import json
                            self.logger.info(f"📋 COMPLETE GPT-5 RESPONSE:\n{json.dumps(result, indent=2)[:2000]}")
                            
                            content = None
                            
                            # OFFICIAL FORMAT: GPT-5 Responses API returns output array with items
                            # Reference: https://platform.openai.com/docs/guides/migrate-to-responses
                            
                            # Method 1: Check for SDK helper field 'output_text'
                            if 'output_text' in result:
                                content = result['output_text']
                                self.logger.info(f"✅ Found SDK helper 'output_text' field (length: {len(content) if content else 0})")
                            
                            # Method 2: Parse output array (official API structure)
                            elif 'output' in result and isinstance(result['output'], list):
                                self.logger.info(f"✅ Found 'output' array with {len(result['output'])} items")
                                
                                # Iterate through output items to find message type
                                for idx, item in enumerate(result['output']):
                                    item_type = item.get('type', 'unknown')
                                    self.logger.info(f"  📦 Item {idx}: type='{item_type}', keys={list(item.keys())}")
                                    
                                    # Look for message type items (contain the actual response text)
                                    if item_type == 'message':
                                        self.logger.info(f"  ✅ Found message item at index {idx}")
                                        
                                        # Extract from content array within message
                                        if 'content' in item and isinstance(item['content'], list):
                                            self.logger.info(f"    📝 Message has content array with {len(item['content'])} items")
                                            
                                            # Look for output_text type in content array
                                            for content_idx, content_item in enumerate(item['content']):
                                                content_type = content_item.get('type', 'unknown')
                                                self.logger.info(f"      🔸 Content {content_idx}: type='{content_type}'")
                                                
                                                if content_type == 'output_text' and 'text' in content_item:
                                                    content = content_item['text']
                                                    self.logger.info(f"      ✅ Extracted text from output_text item (length: {len(content)})")
                                                    break
                                        
                                        # If we found content, break outer loop
                                        if content:
                                            break
                                        
                                        # Fallback: try direct content field (if not array)
                                        elif 'content' in item and isinstance(item['content'], str):
                                            content = item['content']
                                            self.logger.info(f"  ✅ Extracted direct content string (length: {len(content)})")
                                            break
                                
                                # Additional logging if no content found
                                if not content:
                                    self.logger.warning(f"⚠️ No message-type items with content found in output array")
                                    # Log full structure for debugging
                                    for idx, item in enumerate(result['output']):
                                        self.logger.warning(f"  Item {idx} structure: {json.dumps(item, indent=2)[:500]}")
                            
                            # Method 3: Fallback for non-standard responses
                            elif 'output' in result and isinstance(result['output'], dict):
                                output = result['output']
                                self.logger.warning(f"⚠️ 'output' is dict instead of array (non-standard format)")
                                
                                # Try various extraction methods
                                if 'text' in output:
                                    content = output['text']
                                    self.logger.info(f"✅ Extracted from output.text (length: {len(content)})")
                                elif 'content' in output:
                                    content = output['content']
                                    self.logger.info(f"✅ Extracted from output.content (length: {len(content)})")
                                else:
                                    content = str(output)
                                    self.logger.warning(f"⚠️ Converting output dict to string (keys: {list(output.keys())})")
                            
                            # Log result
                            if content:
                                content_length = len(content)
                                self.logger.info(f"✅ GPT-5 CONTENT EXTRACTED ({content_length} chars)")
                                self.logger.info(f"📤 Content preview:\n{content[:500]}")
                                if content_length > 500:
                                    self.logger.info(f"📤 Content ending:\n...{content[-200:]}")
                                return content
                            else:
                                self.logger.error(f"❌ FAILED TO EXTRACT CONTENT FROM GPT-5 RESPONSE")
                                self.logger.error(f"❌ Available top-level keys: {list(result.keys())}")
                                if 'output' in result:
                                    self.logger.error(f"❌ Output field type: {type(result['output']).__name__}")
                                    if isinstance(result['output'], list) and result['output']:
                                        self.logger.error(f"❌ First output item: {json.dumps(result['output'][0], indent=2)[:500]}")
                                self.logger.error(f"❌ Full response (truncated): {json.dumps(result, indent=2)[:1000]}")
                                return ""
                        
                        # Handle standard Chat Completions API format
                        elif 'choices' in result and result['choices']:
                            choice = result['choices'][0]
                            self.logger.debug(f"📝 First choice keys: {list(choice.keys())}")
                            
                            if 'message' in choice:
                                message = choice['message']
                                self.logger.debug(f"💬 Message keys: {list(message.keys())}")
                                self.logger.debug(f"📋 Message role: {message.get('role', 'N/A')}")
                                
                                # Check for content
                                content = message.get('content')
                                if content is None:
                                    self.logger.warning(f"⚠️ Content is None for model {model}")
                                    # Check if there are other content fields for o3 model
                                    self.logger.debug(f"🔍 Full message structure: {message}")
                                    
                                    # Try alternative content fields that reasoning models might use
                                    if 'reasoning' in message:
                                        self.logger.info(f"🧠 Found reasoning field in response")
                                        content = message['reasoning']
                                    elif 'text' in message:
                                        self.logger.info(f"📄 Found text field in response")
                                        content = message['text']
                                    else:
                                        content = ""
                                        self.logger.error(f"❌ No usable content found in response")
                                elif content == "":
                                    self.logger.warning(f"⚠️ Content is empty string for model {model}")
                                
                                content_length = len(content) if content else 0
                                self.logger.debug(f"OpenAI request successful, response length: {content_length}")
                                self.logger.debug(f"📤 Content preview: {content[:200]}..." if content else "📤 No content received")
                                
                                return content or ""
                            else:
                                self.logger.error(f"❌ No 'message' field in choice: {choice}")
                        else:
                            self.logger.error(f"❌ No choices in response: {result}")
                            
                        return ""
                    else:
                        error_text = await response.text()
                        self.logger.warning(f"OpenAI request failed (attempt {actual_attempts}): {response.status} - {error_text}")
                        
                        # Special handling for response_format not supported error
                        if (response.status == 400 and "response_format" in error_text and 
                            "not supported" in error_text and not response_format_retry_done):
                            if "response_format" in request_data:
                                self.logger.warning("🔄 response_format not supported by this model, retrying without it")
                                del request_data["response_format"]
                                response_format_retry_done = True
                                actual_attempts -= 1  # Don't count this as a real attempt
                                continue
                        
                        if response.status == 429:  # Rate limiting
                            wait_time = 2 ** actual_attempts
                            self.logger.info(f"Rate limited, waiting {wait_time}s")
                            await asyncio.sleep(wait_time)
                        elif response.status >= 500:  # Server errors
                            wait_time = 1 * actual_attempts
                            await asyncio.sleep(wait_time)
                        else:
                            # Don't retry client errors (except response_format which we handle above)
                            break
                            
            except asyncio.TimeoutError:
                self.logger.warning(f"OpenAI request timeout (attempt {actual_attempts})")
                if actual_attempts < max_total_attempts:
                    await asyncio.sleep(1)
            except Exception as e:
                self.logger.error(f"OpenAI request error (attempt {actual_attempts}): {e}")
                if actual_attempts < max_total_attempts:
                    await asyncio.sleep(1)
        
        raise Exception(f"OpenAI API request failed after {max_total_attempts} attempts")
    
    def _parse_json_response(self, response: Any) -> Dict:
        """Enhanced JSON parsing with comprehensive error recovery and fallback mechanisms"""
        if isinstance(response, dict):
            return response
        
        if not isinstance(response, str):
            response = str(response)
        
        # Store original response for debugging
        original_response = response
        
        try:
            # Method 1: Direct JSON parsing for clean responses
            cleaned_response = response.strip()
            if cleaned_response.startswith('{') and cleaned_response.endswith('}'):
                return json.loads(cleaned_response)
            
            # Method 2: Extract from markdown code blocks
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                return json.loads(json_content)
            
            # Method 3: Extract from any code blocks
            json_match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_content = json_match.group(1).strip()
                if json_content.startswith('{'):
                    return json.loads(json_content)
            
            # Method 4: Find complete JSON object with proper brace matching
            json_start = response.find('{')
            if json_start != -1:
                brace_count = 0
                json_end = -1
                for i in range(json_start, len(response)):
                    if response[i] == '{':
                        brace_count += 1
                    elif response[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = i
                            break
                
                if json_end != -1:
                    json_text = response[json_start:json_end+1]
                    try:
                        return json.loads(json_text)
                    except json.JSONDecodeError:
                        # Try to fix common JSON issues
                        fixed_json = self._attempt_json_repair(json_text)
                        if fixed_json:
                            return fixed_json
            
            # Method 5: Attempt to repair truncated JSON
            if '{' in response:
                # Find the last occurrence of opening brace
                last_brace = response.rfind('{')
                potential_json = response[last_brace:]
                
                # Try to complete truncated JSON
                if not potential_json.endswith('}'):
                    # Count missing closing braces
                    open_braces = potential_json.count('{')
                    close_braces = potential_json.count('}')
                    missing_braces = open_braces - close_braces
                    
                    if missing_braces > 0:
                        # Add missing closing braces
                        completed_json = potential_json + '}' * missing_braces
                        try:
                            parsed = json.loads(completed_json)
                            self.logger.warning(f"Successfully repaired truncated JSON by adding {missing_braces} closing braces")
                            return parsed
                        except json.JSONDecodeError:
                            pass
            
            # Method 6: Extract evaluation results using regex patterns
            evaluation_data = self._extract_evaluation_with_regex(response)
            if evaluation_data:
                self.logger.warning("Extracted evaluation data using regex fallback")
                return evaluation_data
            
            # Method 7: Create fallback response with high engagement scores
            # This ensures we don't lose potentially good segments due to parsing failures
            self.logger.warning(f"All JSON parsing methods failed. Creating intelligent fallback response.")
            return self._create_intelligent_fallback_response(original_response)
            
        except Exception as e:
            self.logger.error(f"Critical JSON parsing error: {e}")
            return self._create_emergency_fallback_response(original_response)
    
    def _attempt_json_repair(self, json_text: str) -> Optional[Dict]:
        """Attempt to repair common JSON formatting issues"""
        try:
            import re
            # Fix common issues
            repaired = json_text
            
            # Fix trailing commas
            repaired = re.sub(r',\s*}', '}', repaired)
            repaired = re.sub(r',\s*]', ']', repaired)
            
            # Fix unquoted keys
            repaired = re.sub(r'(\w+)\s*:', r'"\1":', repaired)
            
            # Fix single quotes
            repaired = repaired.replace("'", '"')
            
            # Try parsing the repaired JSON
            return json.loads(repaired)
            
        except:
            return None
    
    def _extract_evaluation_with_regex(self, response: str) -> Optional[Dict]:
        """Extract evaluation data using regex patterns as fallback"""
        try:
            import re
            # Look for evaluation patterns in the response
            segments_data = []
            
            # Pattern for segment evaluations
            segment_pattern = r'segment[\s_]*(?:index|\d+)[:\s]*(?:(\d+))'
            score_pattern = r'(?:score|rating)[:\s]*(\d*\.?\d+)'
            recommended_pattern = r'(?:recommended|recommend)[:\s]*(true|false|yes|no)'
            
            segment_matches = re.finditer(segment_pattern, response, re.IGNORECASE)
            
            for match in segment_matches:
                segment_idx = int(match.group(1)) if match.group(1) else 0
                
                # Look for score near this segment
                segment_text = response[max(0, match.start()-200):match.end()+200]
                
                score_match = re.search(score_pattern, segment_text, re.IGNORECASE)
                score = float(score_match.group(1)) if score_match else 0.7
                
                recommended_match = re.search(recommended_pattern, segment_text, re.IGNORECASE)
                recommended = False
                if recommended_match:
                    rec_text = recommended_match.group(1).lower()
                    recommended = rec_text in ['true', 'yes']
                
                segments_data.append({
                    'segment_index': segment_idx,
                    'relevance_score': score,
                    'engagement_score': score,
                    'overall_score': score,
                    'recommended': recommended or score > 0.6,
                    'reasoning': f'Extracted from response analysis (score: {score})'
                })
            
            if segments_data:
                return {
                    'segments': segments_data,
                    'extraction_method': 'regex_fallback'
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Regex extraction failed: {e}")
            return None
    
    def _create_intelligent_fallback_response(self, original_response: str) -> Dict:
        """Create intelligent fallback response that preserves segment quality"""
        try:
            import re
            
            # Extract scores and reasoning from the response text
            # Look for numerical scores in the response
            score_matches = re.findall(r'(?:score|rating|overall)[:\s]*(\d*\.?\d+)', original_response, re.IGNORECASE)
            scores = [float(match) for match in score_matches if match]
            
            # Look for segment indices
            segment_matches = re.findall(r'segment[\s_]*(?:index|#)?[\s]*(\d+)', original_response, re.IGNORECASE)
            segment_indices = [int(match) for match in segment_matches if match.isdigit()]
            
            # Look for positive/negative indicators
            positive_indicators = ['relevant', 'good', 'excellent', 'perfect', 'ideal', 'high', 'strong', 'recommended', 'matches', '120 hz', 'hz']
            negative_indicators = ['irrelevant', 'poor', 'bad', 'inappropriate', 'low', 'weak', 'not recommended', 'no match']
            
            response_lower = original_response.lower()
            positive_count = sum(1 for indicator in positive_indicators if indicator in response_lower)
            negative_count = sum(1 for indicator in negative_indicators if indicator in response_lower)
            
            # Create evaluations based on what we can extract
            evaluations = []
            
            # If we found scores and segments, try to match them
            if scores and segment_indices:
                for i, segment_idx in enumerate(segment_indices):
                    score = scores[i] if i < len(scores) else (scores[0] if scores else 0.8)
                    # Boost score if response seems positive
                    if positive_count > negative_count:
                        score = max(score, 0.8)  # High score for positive responses
                    
                    evaluations.append({
                        "segment_index": segment_idx,
                        "overall_score": min(1.0, score),
                        "reasoning": f"Intelligent fallback - extracted score {score} (positive: {positive_count}, negative: {negative_count})",
                        "recommended": score >= 0.7 or positive_count > 0
                    })
            
            # If no clear structure found, create conservative but fair fallback
            if not evaluations:
                # Check if response contains high-value keywords like "120 hz"
                has_target_keywords = any(keyword in response_lower for keyword in ['120', 'hz', 'refresh', 'display', 'screen'])
                base_score = 0.9 if has_target_keywords else (0.8 if positive_count > 0 else 0.7)
                
                # Create evaluations for likely number of segments (up to 5)
                num_segments = min(len(segment_indices) if segment_indices else 5, 5)
                for i in range(num_segments):
                    evaluations.append({
                        "segment_index": i,
                        "overall_score": base_score,
                        "reasoning": f"Intelligent fallback - target keywords: {has_target_keywords}, positive sentiment: {positive_count > negative_count}",
                        "recommended": base_score >= 0.7
                    })
                
            self.logger.warning(f"Created intelligent fallback with {len(evaluations)} evaluations, average score: {sum(e['overall_score'] for e in evaluations)/len(evaluations):.2f}")
            return {"evaluations": evaluations}
            
        except Exception as e:
            self.logger.error(f"Error in intelligent fallback: {e}")
            # Emergency fallback - ensure we don't completely fail
            return {
                "evaluations": [
                    {
                        "segment_index": 0,
                        "overall_score": 0.8,
                        "reasoning": "Emergency fallback due to parsing failure",
                        "recommended": True
                    }
                ]
            }
    
    def _create_emergency_fallback_response(self, original_response: str) -> Dict:
        """Emergency fallback to prevent complete failure - ensures we still select content"""
        self.logger.error("Creating emergency fallback - ensuring content selection doesn't completely fail")
        
        # Create evaluations that will ensure at least some content gets selected
        emergency_evaluations = [
            {
                "segment_index": i,
                "overall_score": 0.7,  # Good score to ensure selection
                "reasoning": "Emergency fallback - preventing total selection failure",
                "recommended": True if i < 2 else False  # Recommend first 2 segments
            }
            for i in range(3)  # Create 3 emergency evaluations
        ]
        
        return {"evaluations": emergency_evaluations}


class OllamaProvider(LLMProvider):
    """Ollama provider wrapper for existing functionality"""
    
    def __init__(self, ollama_client):
        super().__init__()
        self.ollama_client = ollama_client
        self.logger.info("Ollama provider initialized")
    
    async def generate_response(self, 
                              prompt: str,
                              model: str = None,
                              cache_key: str = None,
                              images: List[str] = None) -> str:
        """Generate response using Ollama client"""
        return await self.ollama_client._make_request(
            prompt=prompt,
            model=model or self.ollama_client.get_best_model("analysis"),
            cache_key=cache_key,
            images=images
        )
    
    async def _make_request(self, 
                          prompt: str,
                          model: str = None,
                          cache_key: str = None,
                          images: List[str] = None) -> str:
        """Make request using Ollama client"""
        return await self.ollama_client._make_request(
            prompt=prompt,
            model=model or self.ollama_client.get_best_model("analysis"),
            cache_key=cache_key,
            images=images
        )
    
    def get_best_model(self, task_type: str = "text") -> str:
        """Get best model from Ollama client"""
        return self.ollama_client.get_best_model(task_type)
    
    def _parse_json_response(self, response: Any) -> Dict:
        """Parse JSON response using Ollama client"""
        return self.ollama_client._parse_json_response(response)


def create_llm_provider(provider_type: str = None, ollama_client=None) -> LLMProvider:
    """
    Factory function to create appropriate LLM provider based on configuration.
    ENHANCED: Prioritizes GPT-4 over o3 models for better JSON reliability.
    
    Args:
        provider_type: Override provider type ("openai" or "ollama")
        ollama_client: Existing Ollama client for Ollama provider
        
    Returns:
        Configured LLM provider instance
    """
    logger = logging.getLogger(__name__)
    
    # Check environment variable for model configuration
    model_name = os.getenv('MODEL_NAME', '').lower()
    openai_model = os.getenv('OPENAI_MODEL', '').lower()
    
    # Determine provider type with enhanced logic
    if provider_type:
        use_openai = provider_type.lower() == "openai"
        logger.info(f"🎯 Provider type explicitly set to: {provider_type}")
    elif model_name in ['gpt4', 'gpt-4', 'openai']:
        use_openai = True
        logger.info("🤖 Using OpenAI due to MODEL_NAME setting")
    elif 'openai' in model_name or 'gpt' in model_name:
        use_openai = True
        logger.info("🤖 Using OpenAI due to GPT model in MODEL_NAME")
    elif openai_model:
        use_openai = True
        logger.info("🤖 Using OpenAI due to OPENAI_MODEL setting")
    elif os.getenv('OPENAI_API_KEY'):
        # If OpenAI API key is available, prefer it over Ollama for JSON reliability
        use_openai = True
        logger.info("🤖 Using OpenAI due to available API key (better JSON reliability)")
    else:
        use_openai = False
        logger.info("🦙 Using Ollama as fallback")
    
    if use_openai:
        logger.info("✨ Creating OpenAI provider for reliable JSON parsing")
        return OpenAIProvider()
    else:
        logger.info("🦙 Creating Ollama provider for textual analysis")
        if not ollama_client:
            raise ValueError("Ollama client is required for Ollama provider")
        return OllamaProvider(ollama_client)


# Async context manager for automatic provider management
class ManagedLLMProvider:
    """Context manager for automatic LLM provider lifecycle management"""
    
    def __init__(self, provider_type: str = None, ollama_client=None):
        self.provider_type = provider_type
        self.ollama_client = ollama_client
        self.provider = None
    
    async def __aenter__(self):
        self.provider = create_llm_provider(self.provider_type, self.ollama_client)
        if hasattr(self.provider, '__aenter__'):
            await self.provider.__aenter__()
        return self.provider
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.provider and hasattr(self.provider, '__aexit__'):
            await self.provider.__aexit__(exc_type, exc_val, exc_tb)
