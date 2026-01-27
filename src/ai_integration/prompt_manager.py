# src/ai_integration/prompt_manager.py
"""Prompt management for AI integration"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path


class PromptManager:
    """
    Manage AI prompts for different analysis tasks.
    """
    
    def __init__(self, prompts_config_path: str = None):
        """
        Initialize prompt manager.
        
        Args:
            prompts_config_path: Path to prompts configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.prompts_config_path = prompts_config_path
        
        # Load default prompts
        self.prompts = self._load_default_prompts()
        
        # Load custom prompts if config file exists
        if prompts_config_path and Path(prompts_config_path).exists():
            try:
                self._load_custom_prompts(prompts_config_path)
            except Exception as e:
                self.logger.warning(f"Failed to load custom prompts: {e}")
    
    def _load_default_prompts(self) -> Dict:
        """Load default prompt templates."""
        return {
            # Scene Detection Prompts
            'scene_detection': {
                'identify_scenes': """
                Analyze this video transcript to identify distinct scenes and natural breakpoints:
                
                Transcript: {transcript}
                
                Please identify scenes and provide a JSON response:
                {{
                    "scenes": [
                        {{
                            "start_time": 0.0,
                            "end_time": 30.0,
                            "scene_type": "introduction|main_content|conclusion|transition",
                            "content_summary": "brief description",
                            "importance_score": 0.0-1.0,
                            "transition_quality": 0.0-1.0
                        }}
                    ],
                    "natural_breaks": [15.5, 45.2, 78.9],
                    "optimal_segments": [
                        {{
                            "start_time": 0.0,
                            "end_time": 30.0,
                            "segment_score": 0.0-1.0,
                            "reason": "why this makes a good segment"
                        }}
                    ]
                }}
                """,
                
                'analyze_transitions': """
                Analyze the transitions between segments in this transcript:
                
                Transcript: {transcript}
                
                Identify smooth transition points for cutting:
                {{
                    "transition_points": [
                        {{
                            "timestamp": 0.0,
                            "transition_type": "topic_change|speaker_change|pause|natural_break",
                            "smoothness_score": 0.0-1.0,
                            "cutting_potential": 0.0-1.0
                        }}
                    ],
                    "problematic_cuts": [
                        {{
                            "timestamp": 0.0,
                            "reason": "why this would be a bad cut"
                        }}
                    ]
                }}
                """
            },
            
            # Content Analysis Prompts
            'content_analysis': {
                'topic_classification': """
                Classify the content and topics discussed in this transcript:
                
                Transcript: {transcript}
                
                Provide classification in JSON format:
                {{
                    "primary_topic": "main topic",
                    "secondary_topics": ["topic1", "topic2"],
                    "content_category": "educational|entertainment|lifestyle|business|tech|other",
                    "expertise_level": "beginner|intermediate|advanced",
                    "target_audience": "general|professionals|students|creators",
                    "key_concepts": ["concept1", "concept2"],
                    "actionable_insights": ["insight1", "insight2"]
                }}
                """,
                
                'emotional_analysis': """
                Analyze the emotional content and tone of this transcript:
                
                Transcript: {transcript}
                
                Provide emotional analysis:
                {{
                    "overall_sentiment": "positive|negative|neutral|mixed",
                    "emotional_intensity": 0.0-1.0,
                    "emotion_timeline": [
                        {{
                            "timestamp": 0.0,
                            "emotion": "joy|excitement|surprise|concern|etc",
                            "intensity": 0.0-1.0
                        }}
                    ],
                    "engagement_emotions": ["emotions that increase engagement"],
                    "emotional_peaks": [
                        {{
                            "timestamp": 0.0,
                            "emotion": "emotion_name",
                            "reason": "what caused this emotional peak"
                        }}
                    ]
                }}
                """,
                
                'action_detection': """
                Identify action-oriented moments and demonstrations in this transcript:
                
                Transcript: {transcript}
                
                Find action moments:
                {{
                    "action_moments": [
                        {{
                            "timestamp": 0.0,
                            "action_type": "demonstration|instruction|example|movement",
                            "description": "what action is happening",
                            "visual_importance": 0.0-1.0,
                            "instructional_value": 0.0-1.0
                        }}
                    ],
                    "demonstration_segments": [
                        {{
                            "start_time": 0.0,
                            "end_time": 0.0,
                            "demo_type": "how_to|example|comparison|before_after",
                            "complexity": "simple|moderate|complex"
                        }}
                    ]
                }}
                """
            },
            
            # Smart Zoom Prompts
            'smart_zoom': {
                'framing_strategy': """
                Based on this transcript, recommend framing strategies for vertical video:
                
                Transcript: {transcript}
                
                Recommend framing in JSON:
                {{
                    "segments": [
                        {{
                            "start_time": 0.0,
                            "end_time": 30.0,
                            "framing_strategy": "close_up|medium_shot|wide_shot|dynamic",
                            "zoom_level": 1.0-3.0,
                            "focus_area": "face|upper_body|full_body|object|text",
                            "reasoning": "why this framing works best",
                            "importance": 0.0-1.0
                        }}
                    ],
                    "key_focus_moments": [
                        {{
                            "timestamp": 0.0,
                            "focus_reason": "important point|demonstration|emotion",
                            "recommended_zoom": 1.0-3.0
                        }}
                    ]
                }}
                """,
                
                'subject_priority': """
                Analyze subject importance and priority for framing:
                
                Transcript: {transcript}
                
                Determine subject priorities:
                {{
                    "primary_subjects": [
                        {{
                            "subject_type": "speaker|demonstrator|object|text",
                            "importance": 0.0-1.0,
                            "screen_time_priority": 0.0-1.0,
                            "focus_periods": [
                                {{"start": 0.0, "end": 30.0, "reason": "why focus here"}}
                            ]
                        }}
                    ],
                    "multi_subject_scenes": [
                        {{
                            "start_time": 0.0,
                            "end_time": 30.0,
                            "subjects": ["subject1", "subject2"],
                            "framing_recommendation": "how to frame multiple subjects"
                        }}
                    ]
                }}
                """,
                
                'zoom_transitions': """
                Suggest optimal zoom transitions based on content flow:
                
                Transcript: {transcript}
                
                Plan zoom transitions:
                {{
                    "zoom_plan": [
                        {{
                            "timestamp": 0.0,
                            "zoom_action": "zoom_in|zoom_out|hold|pan",
                            "target_zoom": 1.0-3.0,
                            "transition_speed": "slow|medium|fast",
                            "trigger_reason": "emphasis|subject_change|content_shift"
                        }}
                    ],
                    "smooth_transitions": [
                        {{
                            "from_time": 0.0,
                            "to_time": 5.0,
                            "transition_type": "gradual_zoom|cut|pan",
                            "smoothness_score": 0.0-1.0
                        }}
                    ]
                }}
                """
            },
            
            # Engagement Analysis Prompts
            'engagement': {
                'hook_analysis': """
                Analyze the opening hooks and attention-grabbing elements:
                
                Transcript: {transcript}
                
                Evaluate hooks:
                {{
                    "opening_hook": {{
                        "text": "actual hook text",
                        "effectiveness": 0.0-1.0,
                        "hook_type": "question|statement|surprise|promise|controversy",
                        "improvement_suggestions": ["suggestion1", "suggestion2"]
                    }},
                    "hook_moments": [
                        {{
                            "timestamp": 0.0,
                            "hook_text": "text that hooks attention",
                            "hook_strength": 0.0-1.0,
                            "hook_type": "curiosity|surprise|value|emotional"
                        }}
                    ],
                    "attention_retention": [
                        {{
                            "timestamp": 0.0,
                            "retention_factor": "question|reveal|climax|insight",
                            "retention_strength": 0.0-1.0
                        }}
                    ]
                }}
                """,
                
                'viral_potential': """
                Assess viral potential and shareability:
                
                Transcript: {transcript}
                
                Analyze viral potential:
                {{
                    "viral_score": 0.0-1.0,
                    "viral_elements": [
                        {{
                            "element": "surprise|humor|controversy|relatability|education",
                            "strength": 0.0-1.0,
                            "timestamp": 0.0
                        }}
                    ],
                    "shareability_factors": ["factor1", "factor2"],
                    "trending_potential": {{
                        "tiktok": 0.0-1.0,
                        "youtube_shorts": 0.0-1.0,
                        "instagram_reels": 0.0-1.0
                    }},
                    "viral_moments": [
                        {{
                            "timestamp": 0.0,
                            "moment_type": "climax|punchline|revelation|transformation",
                            "viral_potential": 0.0-1.0
                        }}
                    ]
                }}
                """,
                
                'audience_engagement': """
                Predict audience engagement and response:
                
                Transcript: {transcript}
                
                Predict engagement:
                {{
                    "engagement_prediction": {{
                        "overall_engagement": 0.0-1.0,
                        "comment_likelihood": 0.0-1.0,
                        "share_likelihood": 0.0-1.0,
                        "completion_rate": 0.0-1.0
                    }},
                    "engagement_triggers": [
                        {{
                            "timestamp": 0.0,
                            "trigger": "question|call_to_action|controversial_point|relatable_moment",
                            "engagement_type": "comment|share|like|save",
                            "strength": 0.0-1.0
                        }}
                    ],
                    "drop_off_risks": [
                        {{
                            "timestamp": 0.0,
                            "risk_factor": "slow_pace|complex_content|off_topic|low_energy",
                            "risk_level": 0.0-1.0
                        }}
                    ]
                }}
                """
            },
            
            # Quality Assessment Prompts
            'quality': {
                'content_quality': """
                Assess overall content quality and production value:
                
                Transcript: {transcript}
                
                Quality assessment:
                {{
                    "content_quality": {{
                        "clarity": 0.0-1.0,
                        "coherence": 0.0-1.0,
                        "value_provided": 0.0-1.0,
                        "production_quality": 0.0-1.0,
                        "professional_level": 0.0-1.0
                    }},
                    "strengths": ["strength1", "strength2"],
                    "weaknesses": ["weakness1", "weakness2"],
                    "improvement_areas": [
                        {{
                            "area": "audio|visual|content|pacing|engagement",
                            "priority": "high|medium|low",
                            "suggestion": "specific improvement suggestion"
                        }}
                    ]
                }}
                """,
                
                'technical_analysis': """
                Analyze technical aspects for optimization:
                
                Transcript: {transcript}
                
                Technical analysis:
                {{
                    "speech_analysis": {{
                        "clarity": 0.0-1.0,
                        "pace": "too_slow|optimal|too_fast",
                        "filler_words": 0.0-1.0,
                        "articulation": 0.0-1.0
                    }},
                    "content_structure": {{
                        "introduction": 0.0-1.0,
                        "body": 0.0-1.0,
                        "conclusion": 0.0-1.0,
                        "flow": 0.0-1.0
                    }},
                    "optimization_suggestions": [
                        {{
                            "category": "audio|content|structure|pacing",
                            "suggestion": "specific technical improvement",
                            "impact": "high|medium|low"
                        }}
                    ]
                }}
                """
            }
        }
    
    def _load_custom_prompts(self, config_path: str):
        """Load custom prompts from configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                custom_prompts = json.load(f)
            
            # Merge custom prompts with default ones
            for category, prompts in custom_prompts.items():
                if category in self.prompts:
                    self.prompts[category].update(prompts)
                else:
                    self.prompts[category] = prompts
            
            self.logger.info(f"Loaded custom prompts from {config_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load custom prompts: {e}")
            raise
    
    def get_prompt(self, category: str, prompt_name: str, **kwargs) -> str:
        """
        Get a formatted prompt.
        
        Args:
            category: Prompt category (e.g., 'scene_detection', 'content_analysis')
            prompt_name: Name of the specific prompt
            **kwargs: Variables to format into the prompt
            
        Returns:
            Formatted prompt string
        """
        try:
            if category not in self.prompts:
                raise ValueError(f"Unknown prompt category: {category}")
            
            if prompt_name not in self.prompts[category]:
                raise ValueError(f"Unknown prompt name: {prompt_name} in category {category}")
            
            prompt_template = self.prompts[category][prompt_name]
            
            # Format the prompt with provided variables
            return prompt_template.format(**kwargs)
            
        except KeyError as e:
            self.logger.error(f"Missing variable for prompt formatting: {e}")
            raise ValueError(f"Missing required variable: {e}")
        except Exception as e:
            self.logger.error(f"Error getting prompt: {e}")
            raise
    
    def get_all_prompts(self, category: str = None) -> Dict:
        """
        Get all prompts, optionally filtered by category.
        
        Args:
            category: Optional category to filter by
            
        Returns:
            Dictionary of prompts
        """
        if category:
            return self.prompts.get(category, {})
        return self.prompts.copy()
    
    def add_custom_prompt(self, category: str, prompt_name: str, prompt_template: str):
        """
        Add a custom prompt template.
        
        Args:
            category: Prompt category
            prompt_name: Name for the prompt
            prompt_template: Template string with placeholders
        """
        if category not in self.prompts:
            self.prompts[category] = {}
        
        self.prompts[category][prompt_name] = prompt_template
        self.logger.info(f"Added custom prompt: {category}.{prompt_name}")
    
    def update_prompt(self, category: str, prompt_name: str, prompt_template: str):
        """
        Update an existing prompt template.
        
        Args:
            category: Prompt category
            prompt_name: Name of the prompt to update
            prompt_template: New template string
        """
        if category not in self.prompts or prompt_name not in self.prompts[category]:
            raise ValueError(f"Prompt {category}.{prompt_name} does not exist")
        
        self.prompts[category][prompt_name] = prompt_template
        self.logger.info(f"Updated prompt: {category}.{prompt_name}")
    
    def save_prompts(self, output_path: str):
        """
        Save current prompts to a file.
        
        Args:
            output_path: Path to save the prompts
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.prompts, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved prompts to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save prompts: {e}")
            raise
    
    def validate_prompt(self, category: str, prompt_name: str, required_variables: List[str]) -> bool:
        """
        Validate that a prompt contains all required variables.
        
        Args:
            category: Prompt category
            prompt_name: Name of the prompt
            required_variables: List of required variable names
            
        Returns:
            True if prompt is valid
        """
        try:
            prompt_template = self.prompts[category][prompt_name]
            
            # Check if all required variables are in the template
            for var in required_variables:
                if f"{{{var}}}" not in prompt_template:
                    self.logger.warning(f"Prompt {category}.{prompt_name} missing variable: {var}")
                    return False
            
            return True
            
        except KeyError:
            self.logger.error(f"Prompt {category}.{prompt_name} not found")
            return False
    
    def list_categories(self) -> List[str]:
        """Get list of available prompt categories."""
        return list(self.prompts.keys())
    
    def list_prompts(self, category: str) -> List[str]:
        """
        Get list of prompt names in a category.
        
        Args:
            category: Prompt category
            
        Returns:
            List of prompt names
        """
        return list(self.prompts.get(category, {}).keys())
    
    def get_prompt_info(self) -> Dict:
        """Get information about available prompts."""
        info = {}
        
        for category, prompts in self.prompts.items():
            info[category] = {
                'count': len(prompts),
                'prompts': list(prompts.keys())
            }
        
        return info
    
    def search_prompts(self, search_term: str) -> List[Dict]:
        """
        Search for prompts containing a specific term.
        
        Args:
            search_term: Term to search for
            
        Returns:
            List of matching prompts with metadata
        """
        matches = []
        search_term_lower = search_term.lower()
        
        for category, prompts in self.prompts.items():
            for prompt_name, prompt_template in prompts.items():
                if (search_term_lower in prompt_name.lower() or 
                    search_term_lower in prompt_template.lower()):
                    
                    matches.append({
                        'category': category,
                        'name': prompt_name,
                        'template': prompt_template[:200] + "..." if len(prompt_template) > 200 else prompt_template
                    })
        
        return matches
