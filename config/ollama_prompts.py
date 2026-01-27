# config/ollama_prompts.py
"""Ollama API prompts for content analysis and smart zoom decisions"""

ZOOM_ANALYSIS_PROMPTS = {
    "framing_strategy": """
    Based on this video transcript with timestamps: {transcription}
    For each segment, determine the optimal framing strategy:
    - "close_up": For personal moments, important points, single speaker
    - "medium_shot": For general dialogue, presentations
    - "wide_shot": For context, multiple people, demonstrations
    - "dynamic": For action sequences, movements
    
    Consider the emotional tone and content importance for each decision.
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    
    Return a JSON response with the following structure:
    {{
        "segments": [
            {{
                "start_time": float,
                "end_time": float,
                "framing_strategy": "close_up|medium_shot|wide_shot|dynamic",
                "confidence": float,
                "reasoning": "explanation for this framing choice"
            }}
        ]
    }}
    """,
    
    "subject_priority": """
    Analyze this video segment transcript: {transcription}
    Identify the primary subject(s) that should be the focus of framing:
    - Main speaker or presenter
    - Key visual elements mentioned in speech
    - Important actions or demonstrations
    - Secondary subjects to include when relevant
    
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    
    Return a JSON response with subject priorities:
    {{
        "primary_subjects": ["speaker", "demonstration", "visual_aid"],
        "secondary_subjects": ["audience", "background_elements"],
        "focus_keywords": ["keyword1", "keyword2"],
        "attention_moments": [
            {{
                "timestamp": float,
                "subject": "primary_subject",
                "action": "description of what to focus on"
            }}
        ]
    }}
    """,
    
    "zoom_transitions": """
    Based on this content flow: {transcription}
    Suggest smooth transition points for zoom changes:
    - Natural pause points for zoom adjustments
    - Topic changes requiring reframing
    - Emphasis moments needing close-ups
    - Context moments requiring wider shots
    
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    
    Return transition recommendations:
    {{
        "transitions": [
            {{
                "timestamp": float,
                "from_zoom": "current_zoom_level",
                "to_zoom": "target_zoom_level",
                "transition_type": "smooth|cut|fade",
                "duration": float,
                "trigger": "pause|topic_change|emphasis|context"
            }}
        ]
    }}
    """,
    
    "content_engagement": """
    Analyze this transcript for engagement potential: {transcription}
    Identify the most engaging segments that would work well as short videos:
    - High-energy moments
    - Key insights or revelations
    - Emotional peaks
    - Actionable content
    - Compelling narratives
    
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    Focus on segments that are 15-60 seconds long.
    
    Rate each segment's potential for viral content:
    {{
        "engagement_analysis": [
            {{
                "start_time": float,
                "end_time": float,
                "engagement_score": float,
                "content_type": "insight|story|action|emotion|humor",
                "viral_potential": "high|medium|low",
                "recommended_length": int,
                "key_moments": ["timestamp1", "timestamp2"]
            }}
        ]
    }}
    """,
    
    "scene_detection": """
    Based on this transcript: {transcription}
    Identify natural scene breaks and content segments:
    - Topic transitions
    - Speaker changes
    - Natural pauses
    - Content shifts
    - Story segments
    
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    
    Suggest optimal cutting points for short video creation:
    {{
        "scenes": [
            {{
                "start_time": float,
                "end_time": float,
                "scene_type": "introduction|main_content|conclusion|transition",
                "topic": "brief topic description",
                "completeness": "complete|partial|fragment",
                "suitable_for_short": boolean
            }}
        ]
    }}
    """
}

CONTENT_ANALYSIS_PROMPTS = {
    "topic_classification": """
    Classify the content type of this video transcript: {transcription}
    
    Determine:
    - Primary content category (education, entertainment, news, tutorial, etc.)
    - Target audience
    - Content style (formal, casual, technical, etc.)
    - Optimal framing approach for this content type
    
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    
    Return classification:
    {{
        "content_category": "category",
        "subcategory": "subcategory",
        "target_audience": "audience_description",
        "content_style": "style_description",
        "recommended_framing": "framing_approach",
        "ideal_short_length": int
    }}
    """,
    
    "emotional_analysis": """
    Analyze the emotional tone and intensity throughout this transcript: {transcription}
    
    Identify:
    - Emotional peaks and valleys
    - Tone changes
    - Intensity variations
    - Moments requiring close-up framing
    - Moments requiring wider context
    
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    
    Return emotional analysis:
    {{
        "emotional_timeline": [
            {{
                "timestamp": float,
                "emotion": "emotion_type",
                "intensity": float,
                "recommended_framing": "close_up|medium|wide",
                "duration": float
            }}
        ]
    }}
    """,
    
    "action_detection": """
    Identify action-oriented moments in this transcript: {transcription}
    
    Look for:
    - Demonstrations
    - Physical movements
    - Gesture-heavy segments
    - Visual references
    - Interactive moments
    
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    
    Return action analysis:
    {{
        "action_moments": [
            {{
                "timestamp": float,
                "action_type": "demonstration|gesture|movement|interaction",
                "description": "action_description",
                "framing_requirement": "dynamic|static|follow",
                "zoom_strategy": "zoom_in|zoom_out|track"
            }}
        ]
    }}
    """
}

QUALITY_PROMPTS = {
    "content_quality": """
    Assess the quality and suitability of this content for short-form video: {transcription}
    
    Evaluate:
    - Content completeness
    - Standalone value
    - Engagement potential
    - Clarity and coherence
    - Visual interest
    
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    
    Return quality assessment:
    {{
        "overall_quality": float,
        "completeness": float,
        "engagement": float,
        "clarity": float,
        "visual_interest": float,
        "recommendations": ["improvement1", "improvement2"],
        "suitable_for_shorts": boolean
    }}
    """
}

# Fast mode prompts - shorter and simpler for faster processing
FAST_ZOOM_ANALYSIS_PROMPTS = {
    "framing_strategy": """
    Video analysis task. Respond ONLY in valid JSON format:

    {{"segments": [{{"start_time": 0, "end_time": 10, "strategy": "close_up", "confidence": 0.8}}]}}

    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.

    Text: {transcription}
    """,
    
    "engagement_analysis": """
    Analyze this content to identify the most engaging 15-60 second segments for social media shorts:
    
    Content:
    {content}
    
    For each segment:
    1. Provide exact start and end timestamps in seconds
    2. Rate engagement value from 0-10
    3. Explain why this segment would perform well as a standalone short
    4. Indicate if this contains key information, emotional moments, or narrative payoff
    
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    Focus on segments that are 15-60 seconds long with clear narrative value.
    
    Format your response as JSON with an array of segments:
    {{
      "segments": [
        {{
          "start_time": start_time,
          "end_time": end_time,
          "engagement_score": score,
          "analysis": "explanation"
        }}
      ]
    }}
    """
}

FAST_CONTENT_ANALYSIS_PROMPTS = {
    "segment_quality": """
    Rate quality 0-1: {segment}
    
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    
    JSON: {{"quality": 0.8}}
    """,
    
    "viral_potential": """
    Viral score 0-1: {content}
    
    IMPORTANT: Ignore intro music, silence or generic greetings in the first 10% of the video.
    
    JSON: {{"viral": 0.7}}
    """
}

# Enhanced comprehensive analysis prompt for user prompt-based selection
ENHANCED_OLLAMA_PROMPTS = {
    "comprehensive_analysis": """
    You are a professional video editor specializing in creating viral short-form content.
    Analyze the video transcript to identify the most engaging 15-60 second segments for social media.
    Focus on segments with clear narrative structure, emotional moments, or key information.
    Explicitly ignore introductory music, silence, or generic greetings in the first 10% of the video.

    Transcript:
    {transcription}

    For each segment:
    1. Provide exact start and end timestamps in seconds
    2. Rate engagement value from 0-10
    3. Explain why this segment would perform well as a standalone short
    4. Indicate if this contains key information, emotional moments, or narrative payoff

    Format your response as JSON with an array of segments:
    {{
      "segments": [
        {{
          "start_time": start_time,
          "end_time": end_time,
          "engagement_score": score,
          "analysis": "explanation"
        }}
      ]
    }}

    IMPORTANT: Each segment should be 15-60 seconds long. Ignore intro music or generic greetings.
    """,
    
    "prompt_based_analysis": """
    You are an expert video content analyst. A user wants to create short videos with this specific request:
    
    USER REQUEST: "{user_prompt}"
    
    Analyze the video transcript to find segments that match the user's request.
    
    Transcript:
    {transcription}
    
    Consider:
    1. Does the content directly relate to what the user asked for?
    2. Would this segment work well as a standalone short video for social media?
    3. Does it capture the essence of the user's requested theme?
    4. Is the content engaging and complete enough for the user's intent?
    
    For each matching segment:
    - Provide exact start and end timestamps in seconds
    - Rate how well it matches the user's request (0.0 to 1.0)
    - Explain why it fits the user's request
    - Rate overall engagement potential (0-10)
    
    IMPORTANT: 
    - Only include segments that clearly match the user's request
    - Each segment should be 15-60 seconds long
    - Ignore intro music, silence, or generic greetings in the first 10% of the video
    - Focus on segments that would work as standalone short videos
    
    Format your response as JSON:
    {{
      "user_request_interpretation": "your understanding of what the user wants",
      "matching_segments": [
        {{
          "start_time": start_time,
          "end_time": end_time,
          "prompt_match_score": score_0_to_1,
          "engagement_score": score_0_to_10,
          "match_explanation": "why this segment fits the user's request",
          "standalone_viability": "why this works as a standalone short"
        }}
      ],
      "total_matches_found": number_of_segments
    }}
    """,
    
    "dual_modal_analysis": """
    You are a professional video editor analyzing both audio transcription and visual content for short-form video creation.
    
    Audio Transcription:
    {transcription}
    
    Visual Analysis:
    {vision_analysis}
    
    Combine both audio and visual insights to identify the most engaging 15-60 second segments for social media.
    Consider:
    - How well the visual content matches the audio content
    - Visual interest and engagement potential
    - Audio content quality and completeness
    - Combined storytelling value
    
    Prioritize segments where:
    - Visual content enhances the audio message
    - There are people visible during important speech
    - Action or demonstration aligns with spoken content
    - Emotional visual content matches emotional audio
    
    IMPORTANT: Ignore intro music, silence, or generic greetings in the first 10% of the video.
    
    Format your response as JSON:
    {{
      "segments": [
        {{
          "start_time": start_time,
          "end_time": end_time,
          "audio_score": score_0_to_1,
          "visual_score": score_0_to_1,
          "combined_score": score_0_to_1,
          "audiovisual_alignment": score_0_to_1,
          "analysis": "explanation combining both audio and visual insights",
          "recommendation": "why this works well for shorts"
        }}
      ]
    }}
    """,
    
    "prompt_based_dual_modal": """
    You are an expert video content analyst with access to both audio and visual information.
    
    USER REQUEST: "{user_prompt}"
    
    Audio Transcription:
    {transcription}
    
    Visual Analysis:
    {vision_analysis}
    
    Task: Find segments that match the user's specific request using both audio and visual information.
    
    Analysis approach:
    1. Identify segments where the spoken content relates to the user's request
    2. Verify that visual content supports or enhances the audio content
    3. Ensure segments work as standalone short videos
    4. Prioritize segments with strong audiovisual alignment
    
    Special considerations for the user's request:
    - If requesting "climax scenes": Look for intense audio + dramatic visuals
    - If requesting "comedy": Look for humorous speech + visual expressions/reactions
    - If requesting "emotional parts": Look for heartfelt audio + intimate visual moments
    - If requesting "educational": Look for explanatory speech + visual demonstrations
    - If requesting "action": Look for dynamic speech + movement in visuals
    
    IMPORTANT: 
    - Segments must be 15-60 seconds long
    - Ignore intro music, silence, or generic greetings in first 10% of video
    - Only include segments that clearly match the user's request
    - Ensure audiovisual content aligns well
    
    Format response as JSON:
    {{
      "user_request": "{user_prompt}",
      "analysis_approach": "how you interpreted and analyzed the request",
      "matched_segments": [
        {{
          "start_time": start_time,
          "end_time": end_time,
          "prompt_match_score": score_0_to_1,
          "audio_match_score": score_0_to_1,
          "visual_match_score": score_0_to_1,
          "audiovisual_alignment": score_0_to_1,
          "overall_score": combined_score_0_to_1,
          "match_explanation": "why this segment fits the user's request",
          "audiovisual_analysis": "how audio and visual content work together",
          "standalone_potential": "viability as standalone short video"
        }}
      ]
    }}
    """
}

# Enhanced comprehensive analysis prompt
ENHANCED_OLLAMA_PROMPTS = {
    "comprehensive_analysis": """
    You are a professional video editor specializing in creating viral short-form content.
    Analyze the video transcript to identify the most engaging 15-60 second segments for social media.
    Focus on segments with clear narrative structure, emotional moments, or key information.
    Explicitly ignore introductory music, silence, or generic greetings in the first 10% of the video.

    Transcript:
    {transcription}

    For each segment:
    1. Provide exact start and end timestamps in seconds
    2. Rate engagement value from 0-10
    3. Explain why this segment would perform well as a standalone short
    4. Indicate if this contains key information, emotional moments, or narrative payoff

    Format your response as JSON with an array of segments:
    {{
      "segments": [
        {{
          "start_time": start_time,
          "end_time": end_time,
          "engagement_score": score,
          "analysis": "explanation"
        }}
      ]
    }}

    IMPORTANT: Each segment should be 15-60 seconds long. Ignore intro music or generic greetings.
    """,
    
    "dual_modal_analysis": """
    You are a professional video editor analyzing both audio transcription and visual content for short-form video creation.
    
    Audio Transcription:
    {transcription}
    
    Visual Analysis:
    {vision_analysis}
    
    Combine both audio and visual insights to identify the most engaging 15-60 second segments for social media.
    Consider:
    - How well the visual content matches the audio content
    - Visual interest and engagement potential
    - Audio content quality and completeness
    - Combined storytelling value
    
    Prioritize segments where:
    - Visual content enhances the audio message
    - There are people visible during important speech
    - Action or demonstration aligns with spoken content
    - Emotional visual content matches emotional audio
    
    IMPORTANT: Ignore intro music, silence, or generic greetings in the first 10% of the video.
    
    Format your response as JSON:
    {{
      "segments": [
        {{
          "start_time": start_time,
          "end_time": end_time,
          "audio_score": score,
          "visual_score": score,
          "combined_score": score,
          "audiovisual_alignment": score,
          "analysis": "explanation combining both audio and visual insights",
          "recommendation": "why this works well for shorts"
        }}
      ]
    }}
    """
}

# Vision-specific prompts for dual-modal analysis
VISION_ENHANCED_PROMPTS = {
    "scene_analysis": """
    Analyze this video frame to understand the visual content:
    
    1. What type of scene is this? (person speaking, demonstration, slides, action, etc.)
    2. Are there people visible? How many? What are they doing?
    3. What is the emotional tone of the visual content?
    4. Rate the visual interest level from 1-10 for social media
    5. Would this work well in vertical (9:16) format?
    
    Keep response brief and focused. Respond in JSON format:
    {{
        "scene_type": "talking_head|demonstration|action|slides|other",
        "people_count": number,
        "emotional_tone": "neutral|positive|negative|energetic|calm",
        "visual_interest": number,
        "vertical_suitable": true/false,
        "key_elements": ["element1", "element2"],
        "description": "brief scene description"
    }}
    """,
    
    "engagement_assessment": """
    Assess this video frame for social media engagement potential:
    
    1. Is this visually compelling or attention-grabbing?
    2. Does it show clear emotion, action, or important visual information?
    3. Would this stop someone scrolling through social media?
    4. Rate engagement potential 1-10
    
    Brief JSON response:
    {{
        "engagement_score": number,
        "attention_grabbing": true/false,
        "has_clear_emotion": true/false,
        "has_action": true/false,
        "social_media_ready": true/false,
        "standout_elements": ["element1", "element2"]
    }}
    """,
    
    "content_context": """
    Analyze this frame in the context of the following spoken content:
    
    Audio content: {audio_context}
    
    Questions:
    1. How well does the visual content match the audio?
    2. Does the visual enhance or detract from the spoken message?
    3. Is this a good moment for a video cut or highlight?
    4. Would this frame represent the audio content well as a thumbnail?
    
    JSON response:
    {{
        "audio_visual_match": "excellent|good|fair|poor",
        "visual_enhances_audio": true/false,
        "good_cut_point": true/false,
        "thumbnail_quality": "excellent|good|fair|poor",
        "mismatch_reason": "explanation if match is poor",
        "enhancement_value": "how visuals add to the message"
    }}
    """,
    
    "efficient_segment_analysis": """
    Quickly analyze this video frame for short-form content creation:
    
    Rate 1-10:
    - Visual interest for social media
    - Likelihood to grab attention
    - Suitability for vertical format
    
    Identify:
    - Scene type (talking, demo, action, slides)
    - Number of people visible
    - Key visual elements
    
    Brief JSON only:
    {{
        "visual_interest": number,
        "attention_score": number,
        "vertical_score": number,
        "scene_type": "talking|demo|action|slides|other",
        "people_count": number,
        "engaging": true/false
    }}
    """,
    
    "batch_frame_analysis": """
    Analyze this frame for video segment selection. Focus on:
    1. Visual engagement (1-10)
    2. Scene type identification
    3. People presence and activity
    4. Social media suitability
    
    Concise JSON response:
    {{
        "engagement": number,
        "scene": "type",
        "people": number,
        "suitable": true/false
    }}
    """,
    
    # PHASE 1 ENHANCEMENT: New contextual analysis prompts
    "content_overview_analysis": """
    You are analyzing a video to understand its overall content and structure for intelligent short-form content creation.
    
    VIDEO METADATA:
    - Duration: {duration} seconds
    - Estimated speakers: {speaker_count}
    - Has visual analysis: {has_visual}
    
    TRANSCRIPTION SAMPLE:
    {transcription_sample}
    
    VISUAL CONTENT SUMMARY:
    {visual_summary}
    
    ANALYSIS TASK:
    Provide a comprehensive overview to help with intelligent scene selection:
    
    1. What type of content is this? (movie, tutorial, interview, presentation, sports, etc.)
    2. What is the overall narrative structure and flow?
    3. What are the main themes and topics discussed?
    4. Where are the most engaging/important moments likely to be?
    5. What would work best as short-form social media content?
    
    Consider the video's genre, pacing, and content characteristics.
    
    Respond in JSON format:
    {{
        "content_type": "movie|tutorial|interview|presentation|sports|documentary|other",
        "genre": "specific genre if applicable",
        "narrative_structure": {{
            "has_clear_beginning": boolean,
            "has_development": boolean,
            "has_climax_or_peak": boolean,
            "has_resolution": boolean,
            "content_flow": "linear|episodic|instructional|conversational"
        }},
        "main_themes": ["theme1", "theme2", "theme3"],
        "content_characteristics": {{
            "is_educational": boolean,
            "is_entertaining": boolean,
            "is_narrative_driven": boolean,
            "has_demonstrations": boolean,
            "interaction_level": "monologue|dialogue|interactive"
        }},
        "engagement_patterns": {{
            "peak_likely_at": "beginning|middle|end|distributed",
            "content_density": "high|medium|low",
            "emotional_variation": "high|medium|low"
        }},
        "short_form_potential": {{
            "best_segment_types": ["type1", "type2"],
            "ideal_segment_length": "15-30|30-60|60-90 seconds",
            "key_selection_criteria": ["criteria1", "criteria2"]
        }}
    }}
    """,
    
    "contextual_intent_analysis": """
    You are an expert video editor who understands user intent for creating short-form content.
    
    USER REQUEST: "{user_prompt}"
    
    VIDEO CONTENT OVERVIEW:
    {content_overview}
    
    VIDEO METADATA:
    - Duration: {duration} seconds
    - Type: {content_type}
    
    ANALYSIS TASK:
    Based on the user's request and the actual video content, determine:
    
    1. What the user's true intent is (beyond literal keywords)
    2. What type of content from this specific video would satisfy their request
    3. How to adapt their request to this video's content and structure
    4. What quality criteria should be used for segment selection
    
    Consider the video's actual content type and structure when interpreting the request.
    For example, "climax scenes" means different things for movies vs tutorials vs interviews.
    
    Provide detailed intent analysis:
    {{
        "intent_interpretation": {{
            "literal_request": "what they said",
            "contextual_intent": "what they actually want given this video type",
            "content_alignment": "how well this video can satisfy their request",
            "adaptation_strategy": "how to best fulfill their intent with available content"
        }},
        "selection_criteria": {{
            "primary_factors": ["most important selection criteria"],
            "secondary_factors": ["additional considerations"],
            "content_position_preference": "beginning|middle|end|peak|any",
            "quality_thresholds": {{
                "minimum_engagement": 0.4,
                "minimum_completeness": 0.3,
                "minimum_relevance": 0.5
            }}
        }},
        "content_requirements": {{
            "emotional_tone": "required emotional characteristics",
            "must_include": ["required elements"],
            "must_avoid": ["elements to avoid"],
            "duration_preference": "ideal length in seconds",
            "standalone_viability": "high|medium|low requirement"
        }},
        "confidence_assessment": {{
            "intent_clarity": 0.8,
            "content_availability": 0.7,
            "match_likelihood": 0.8,
            "overall_confidence": 0.8
        }}
    }}
    """
}
