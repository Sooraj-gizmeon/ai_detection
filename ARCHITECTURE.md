# Video-to-Shorts System Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [High-Level Architecture](#high-level-architecture)
3. [Core Processing Pipeline](#core-processing-pipeline)
4. [Component Details](#component-details)
5. [Data Flow](#data-flow)
6. [API & Queue System](#api--queue-system)
7. [AI/ML Integration](#aiml-integration)
8. [Storage & Caching](#storage--caching)
9. [Deployment Architecture](#deployment-architecture)

---

## System Overview

The Video-to-Shorts system is an AI-powered video processing pipeline that automatically converts long-form horizontal videos into engaging vertical short-form content. The system analyzes video content through multiple modalities (audio, visual, object detection) and intelligently selects or segments the most engaging portions for social media platforms.

### Key Capabilities

- **AI-Powered Content Analysis**: Multi-modal analysis (audio transcription + visual understanding + object detection)
- **Smart Segment Selection**: Identifies the most engaging portions based on user prompts or comprehensive analysis
- **Intelligent Framing**: AI-driven dynamic cropping and subject tracking for optimal composition
- **Flexible Output**: Supports both vertical (9:16) shorts and horizontal (16:9) clips
- **Template-Based Processing**: Specialized templates (e.g., podcast, interview)
- **Customizable Styling**: Kinetic captions, subtitles, overlays, intro/outro
- **Linear Cut Mode**: Fast segmentation without AI analysis for maximum speed
- **Scalable Architecture**: Distributed processing with API + worker pattern

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           CLIENT LAYER                              │
│  (REST API Clients, Web Frontend, Mobile Apps)                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         API SERVICE                                 │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  FastAPI Application (Port 8000/2000)                        │  │
│  │  - Video Processing Endpoints                                │  │
│  │  - Queue Management                                          │  │
│  │  - Status Tracking                                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      MESSAGE QUEUE LAYER                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Redis + Celery (Task Queue & Result Backend)               │  │
│  │  - Asynchronous Task Distribution                            │  │
│  │  - Task Status Tracking                                      │  │
│  │  - Result Caching                                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PROCESSING LAYER                              │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Celery Workers (GPU-Accelerated)                            │  │
│  │  - Video Processing Tasks                                    │  │
│  │  - AI Model Inference                                        │  │
│  │  - File Operations                                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      CORE PIPELINE LAYER                            │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  VideoToShortsProcessor (main.py)                            │  │
│  │  - Orchestrates all processing components                    │  │
│  │  - Manages processing workflow                               │  │
│  │  - Coordinates AI analysis                                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
└──┬─────────────┬──────────────┬──────────────┬──────────────┬───────┘
   │             │              │              │              │
   ▼             ▼              ▼              ▼              ▼
┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│ Audio   │ │ Vision  │ │ Content  │ │ Smart    │ │ Video        │
│ Analysis│ │ Analysis│ │ Analysis │ │ Zoom     │ │ Processing   │
└─────────┘ └─────────┘ └──────────┘ └──────────┘ └──────────────┘
     │           │              │            │              │
     └───────────┴──────────────┴────────────┴──────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STORAGE LAYER                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐                │
│  │ Wasabi S3   │  │ Local Cache  │  │ Model      │                │
│  │ (Input/     │  │ (Whisper,    │  │ Storage    │                │
│  │  Output)    │  │  Vision)     │  │ (YOLO, MP) │                │
│  └─────────────┘  └──────────────┘  └────────────┘                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Core Processing Pipeline

The system transforms long videos into shorts through a multi-stage pipeline:

### Pipeline Stages

```
┌──────────────────────────────────────────────────────────────────────┐
│                    1. VIDEO INGESTION                                │
│  Input: Long-form video file (MP4, AVI, MOV, etc.)                  │
│  Output: Validated video ready for processing                        │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    2. AUDIO ANALYSIS                                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  • Extract audio track from video                              │ │
│  │  • Transcribe speech using OpenAI Whisper                      │ │
│  │  • Detect speech boundaries & pauses                           │ │
│  │  • Identify pre-roll content                                   │ │
│  │  • Analyze content for engagement keywords                     │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  Output: Transcription + word timestamps + speech analysis          │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    3. SCENE DETECTION                                │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  • Detect visual scene changes                                 │ │
│  │  • Combine with audio speech boundaries                        │ │
│  │  • Filter out pre-roll content                                 │ │
│  │  • Generate natural breakpoints                                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  Output: Scene boundaries + combined breakpoints                     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    4. VISUAL ANALYSIS (Optional)                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  • Extract representative frames                               │ │
│  │  • Analyze visual content with vision models                   │ │
│  │  • Detect scene types (action, dialogue, landscape, etc.)      │ │
│  │  • Assess visual interest and engagement                       │ │
│  │  • Identify people and objects                                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  Output: Visual analysis + scene classifications                     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    5. CONTENT ANALYSIS & SEGMENTATION                │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Mode A: USER PROMPT-BASED (Theme-Specific)                   │ │
│  │  • Analyze prompt with LLM (Ollama/OpenAI)                     │ │
│  │  • Extract theme requirements                                  │ │
│  │  • Optional: Object detection for prompt-relevant objects      │ │
│  │  • Select segments matching user intent                        │ │
│  │  • Apply theme-specific quality criteria                       │ │
│  │                                                                 │ │
│  │  Mode B: COMPREHENSIVE (Automatic)                             │ │
│  │  • Generate ALL possible candidate segments (300+)             │ │
│  │  • Perform dual-modal analysis (audio + visual)                │ │
│  │  • Optional: General object detection                          │ │
│  │  • Score segments on multiple criteria:                        │ │
│  │    - Audio quality (speech clarity, keywords)                  │ │
│  │    - Visual quality (scene type, interest)                     │ │
│  │    - Engagement potential (timing, pacing)                     │ │
│  │    - Duration suitability                                      │ │
│  │  • Select top N segments above quality threshold               │ │
│  │                                                                 │ │
│  │  Mode C: LINEAR CUT (Fast)                                     │ │
│  │  • Skip ALL AI analysis                                        │ │
│  │  • Divide video into equal linear segments                     │ │
│  │  • Apply duration-based segmentation only                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  Output: Selected segments with quality scores + metadata            │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    6. SEGMENT EXTRACTION                             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  • Extract video segments from original file                   │ │
│  │  • Verify segment duration accuracy                            │ │
│  │  • Log extraction metrics                                      │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  Output: Individual segment video files                              │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    7. SMART ZOOM & FRAMING                           │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Framing Pipeline:                                             │ │
│  │                                                                 │ │
│  │  A. Object-Aware Zoom (if user prompt + object detection)      │ │
│  │     • Detect and track prompt-relevant objects                 │ │
│  │     • Apply AI reframing recommendations                       │ │
│  │     • Optimize crop for detected objects                       │ │
│  │                                                                 │ │
│  │  B. Standard Smart Zoom (fallback)                             │ │
│  │     • Detect faces, people, subjects                           │ │
│  │     • Track subject positions across frames                    │ │
│  │     • Calculate optimal crop positions                         │ │
│  │     • Apply smooth zoom transitions                            │ │
│  │                                                                 │ │
│  │  C. Basic Conversion (last resort)                             │ │
│  │     • Center crop to target aspect ratio                       │ │
│  │     • Resize to output dimensions                              │ │
│  │                                                                 │ │
│  │  D. Linear Cut Fast Path (if enabled)                          │ │
│  │     • Skip smart zoom entirely                                 │ │
│  │     • Fast center crop + resize (7-10x faster)                 │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  Output: Reframed videos in target aspect ratio                     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    8. TEMPLATE PROCESSING (Optional)                 │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Podcast Template:                                             │ │
│  │  • Detect 2 speakers in video                                  │ │
│  │  • Split detection: left/right speaker positioning             │ │
│  │  • Crop left speaker → top half of vertical video              │ │
│  │  • Crop right speaker → bottom half of vertical video          │ │
│  │  • Maintain speaker consistency across frames                  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  Output: Template-processed videos                                   │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    9. SUBTITLE & CAPTION OVERLAY                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  Standard Subtitles:                                           │ │
│  │  • Extract segment transcription from word timestamps          │ │
│  │  • Apply style template (cinematic, modern, minimal, etc.)     │ │
│  │  • Position captions at configured coordinates                 │ │
│  │  • Render using ASS subtitle format                            │ │
│  │                                                                 │ │
│  │  Kinetic Captions (Default):                                   │ │
│  │  • Word-by-word timing synchronization                         │ │
│  │  • Karaoke effect: highlight word being spoken                 │ │
│  │  • Typewriter effect: words appear as spoken                   │ │
│  │  • Fade effect: smooth transitions                             │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  Output: Videos with styled captions                                 │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    10. INTRO/OUTRO PROCESSING                        │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  • Download intro video from storage (if provided)             │ │
│  │  • Download outro video from storage (if provided)             │ │
│  │  • Concatenate: intro + clip + outro                           │ │
│  │  • Ensure consistent encoding                                  │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  Output: Final videos with intro/outro                               │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    11. AI TITLE & DESCRIPTION GENERATION             │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  • Extract segment transcription                               │ │
│  │  • Generate engaging title with LLM                            │ │
│  │  • Create compelling description                               │ │
│  │  • Generate relevant tags                                      │ │
│  │  • Classify content type                                       │ │
│  │  • Apply variation for diversity (if enabled)                  │ │
│  │  • Append part numbers (for linear cut)                        │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  Output: Metadata for each video                                     │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    12. FINALIZATION & UPLOAD                         │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │  • Verify output files exist and are valid                     │ │
│  │  • Generate thumbnails                                         │ │
│  │  • Upload to storage (Wasabi S3 / AWS S3)                      │ │
│  │  • Update database via API                                     │ │
│  │  • Log processing metrics                                      │ │
│  │  • Clean up temporary files                                    │ │
│  └────────────────────────────────────────────────────────────────┘ │
│  Output: Published shorts with metadata                              │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Audio Analysis Module (`src/audio_analysis/`)

**Purpose**: Extract and analyze audio content for transcription and speech understanding.

**Key Components**:

- **WhisperAnalyzer** (`whisper_analyzer.py`)
  - Uses OpenAI Whisper for speech-to-text
  - Supports multiple model sizes (tiny, base, small, medium, large)
  - Generates word-level timestamps
  - Caches transcriptions by video_id/uploaded_video_id
  - GPU-accelerated inference

- **AudioProcessor** (`audio_processor.py`)
  - Extracts audio from video files
  - Converts to Whisper-compatible format
  - Handles various audio codecs

- **SpeechAnalyzer** (`speech_analysis.py`)
  - Analyzes speech patterns
  - Detects natural speech boundaries
  - Identifies pauses and emphasis
  - Recognizes pre-roll content

**Technologies**: OpenAI Whisper, FFmpeg, PyTorch

---

### 2. Vision Analysis Module (`src/vision_analysis/`)

**Purpose**: Understand visual content through frame analysis and scene classification.

**Key Components**:

- **VisionProcessor** (`vision_processor.py`)
  - Extracts representative frames from video
  - Sends frames to vision models (LLaVA via Ollama)
  - Performs dual-modal analysis (audio + vision)
  - Configurable sampling rate

- **FrameAnalyzer** (`frame_analyzer.py`)
  - Analyzes individual frames
  - Detects scene types (action, dialogue, landscape)
  - Assesses visual interest
  - Identifies compositional elements

- **VisualContentAnalyzer** (`visual_content_analyzer.py`)
  - Aggregates frame analysis
  - Scores segment visual quality
  - Identifies engaging visual patterns

**Technologies**: LLaVA vision model, OpenCV, Ollama API

---

### 3. Content Analysis Module (`src/content_analysis/`)

**Purpose**: Intelligently select and segment video content based on quality criteria.

**Key Components**:

- **ContentAnalyzer** (`content_analyzer.py`)
  - Orchestrates segment generation and selection
  - Manages prompt-based vs. comprehensive analysis
  - Integrates object detection results
  - Applies quality thresholds

- **PromptBasedAnalyzer** (`prompt_based_analyzer.py`)
  - Parses user prompts
  - Extracts theme requirements
  - Matches segments to user intent
  - Applies theme-specific scoring

- **SegmentGenerator** (within `content_analyzer.py`)
  - Generates ALL possible candidate segments
  - Respects speech boundaries
  - Filters pre-roll content
  - Calculates preliminary quality scores

- **SegmentOptimizer** (`segment_optimizer.py`)
  - Removes overlapping segments
  - Optimizes for diversity
  - Balances quality vs. quantity
  - Handles edge cases

- **IntelligentClimaxDetector** (`intelligent_climax_detector.py`)
  - Specialized detector for dramatic moments
  - Analyzes emotional peaks
  - Identifies action sequences

**Technologies**: LLMs (Ollama/OpenAI), custom scoring algorithms

---

### 4. Smart Zoom Module (`src/smart_zoom/`)

**Purpose**: Apply intelligent dynamic framing and subject tracking.

**Key Components**:

- **SmartZoomProcessor** (`smart_zoom_processor.py`)
  - Coordinates framing pipeline
  - Manages subject detection
  - Calculates smooth zoom transitions
  - Handles aspect ratio conversion

- **ObjectAwareZoomProcessor** (`object_aware_zoom.py`)
  - Integrates object detection
  - Applies AI reframing recommendations
  - Optimizes crop for detected objects
  - Enhanced for prompt-based processing

- **ZoomCalculator** (`zoom_calculator.py`)
  - Computes optimal crop positions
  - Applies rule-of-thirds
  - Handles subject transitions
  - Generates smooth camera motion

- **FramingOptimizer** (`framing_optimizer.py`)
  - Optimizes composition
  - Balances multiple subjects
  - Maintains visual interest
  - Applies cinematic framing rules

- **EnhancedFrameProcessor** (`enhanced_frame_processor.py`)
  - Processes individual frames
  - Applies crops and transforms
  - Handles edge cases
  - Optimizes for performance

**Technologies**: OpenCV, MediaPipe, YOLO, custom algorithms

---

### 5. Subject Detection Module (`src/subject_detection/`)

**Purpose**: Detect and track faces, people, and objects in video frames.

**Key Components**:

- **SubjectDetector** (`subject_detector.py`)
  - Unified interface for all detection types
  - Coordinates multiple detectors
  - Tracks subjects across frames
  - Manages detection confidence

- **MediaPipeFaceDetector** (`mediapipe_face_detector.py`)
  - Fast face detection
  - Facial landmark estimation
  - Real-time performance
  - Handles multiple faces

- **PersonDetector** (`person_detector.py`)
  - Full-body person detection
  - Pose estimation
  - Movement tracking
  - Occlusion handling

**Technologies**: MediaPipe, YOLO, PyTorch

---

### 6. Object Detection Module (`src/object_detection/`)

**Purpose**: Detect and classify objects relevant to user prompts.

**Key Components**:

- **ObjectDetector**
  - Uses YOLO for object detection
  - Filters objects by relevance to prompt
  - Tracks object positions and confidence
  - Supports custom object classes

- **AIReframer**
  - Uses LLM to analyze prompt for framing needs
  - Generates reframing recommendations
  - Considers object positions and scene context

**Technologies**: YOLOv8, Ollama API

---

### 7. Scene Detection Module (`src/scene_detection/`)

**Purpose**: Identify natural breakpoints in video content.

**Key Components**:

- **SceneDetector** (`scene_detector.py`)
  - Detects visual scene changes
  - Combines with audio boundaries
  - Filters pre-roll content
  - Generates natural cut points

**Technologies**: PySceneDetect, custom algorithms

---

### 8. Video Processing Module (`src/video_processing/`)

**Purpose**: Core video manipulation operations.

**Key Components**:

- **VideoProcessor**
  - Video file I/O
  - Format conversion
  - Segment extraction
  - Encoding and compression
  - Metadata extraction

**Technologies**: FFmpeg, OpenCV

---

### 9. AI Integration Module (`src/ai_integration/`)

**Purpose**: Interface with LLM services for content analysis.

**Key Components**:

- **OllamaClient** (`ollama_client.py`)
  - Connects to Ollama API
  - Manages model selection
  - Handles prompt templates
  - Supports dual-modal analysis
  - Implements retry logic

- **ContentAnalyzer** (`content_analyzer.py`)
  - Generates content insights
  - Performs sentiment analysis
  - Identifies engagement triggers

**Technologies**: Ollama API (LLaMA, LLaVA), OpenAI API

---

### 10. Template System (`src/templates/`)

**Purpose**: Specialized video processing templates.

**Key Templates**:

- **PodcastTemplate** (`podcast_template.py`, `fixed_podcast_template.py`, `optimized_podcast_template.py`)
  - 2-speaker detection and splitting
  - Left/right speaker identification
  - Top/bottom half cropping for vertical format
  - Enhanced person splitting with face detection
  - Optimized processing pipeline

**Technologies**: MediaPipe, OpenCV, FFmpeg

---

### 11. Utility Modules (`src/utils/`)

**Purpose**: Supporting functionality and helpers.

**Key Components**:

- **SubtitleProcessor** (`subtitle_processor.py`)
  - Standard subtitle rendering
  - Style template management
  - ASS subtitle format

- **KineticSubtitleProcessor** (`kinetic_subtitle_processor.py`)
  - Word-by-word timing
  - Karaoke/typewriter/fade effects
  - Synchronized highlighting

- **TitleGenerator** (`title_generator.py`)
  - AI-powered title generation
  - Description creation
  - Tag generation
  - Content type classification

- **VariationManager** (`variation_manager.py`)
  - Output diversity management
  - Seed-based randomization
  - Prevents repetitive outputs

- **DatabaseAPIClient** (`database_api_client.py`)
  - API-based database operations
  - Clip metadata updates
  - Transcription storage

- **VideoAnalysisLogger** (`video_analysis_logger.py`)
  - Comprehensive analysis logging
  - Decision tracking
  - Performance metrics

- **IntroOutroHandler** (`intro_outro_handler.py`)
  - Download intro/outro videos
  - Concatenate with clips
  - Ensure consistent encoding

**Technologies**: Various Python libraries, custom code

---

## Data Flow

### Video Processing Data Flow

```
┌─────────────┐
│ Input Video │
│ (Long-form) │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. EXTRACTION PHASE                                         │
│                                                             │
│  Video ──→ Audio Track ──→ Whisper ──→ Transcription       │
│    │                                    + Word Timestamps   │
│    └──→ Frame Extraction ──→ Vision ──→ Visual Analysis    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. ANALYSIS PHASE                                           │
│                                                             │
│  Transcription ────┬──→ Scene Detection ──→ Breakpoints    │
│                    │                                        │
│  Visual Analysis ──┼──→ Object Detection ──→ Objects       │
│                    │                                        │
│  User Prompt ──────┴──→ LLM Analysis ─────→ Requirements   │
│                                                             │
│  All Inputs ───────────→ Segment Generation ──→ Candidates │
│                                             (300+ segments) │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. SELECTION PHASE                                          │
│                                                             │
│  Candidates ──→ Quality Scoring ──→ Top N Segments         │
│                 (multiple criteria)                         │
│                                                             │
│  Scoring Factors:                                           │
│  • Audio quality (speech clarity, keywords)                 │
│  • Visual quality (scene type, interest)                    │
│  • Prompt match (theme alignment)                           │
│  • Object relevance (detected objects)                      │
│  • Duration suitability                                     │
│  • Engagement potential                                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. EXTRACTION PHASE                                         │
│                                                             │
│  Selected Segments ──→ Extract from Video ──→ Segment Files│
│                                                             │
│  Segment 1: 00:15 - 01:20 (65s)                            │
│  Segment 2: 03:45 - 04:35 (50s)                            │
│  Segment 3: 07:22 - 08:15 (53s)                            │
│  ...                                                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. TRANSFORMATION PHASE                                     │
│                                                             │
│  Segment Files ──→ Smart Zoom ──→ Reframed Videos          │
│                    (Subject Detection + Tracking)           │
│                          │                                  │
│                          ▼                                  │
│                    Template Processing (Optional)           │
│                          │                                  │
│                          ▼                                  │
│                    Subtitle Overlay                         │
│                          │                                  │
│                          ▼                                  │
│                    Intro/Outro Addition (Optional)          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. FINALIZATION PHASE                                       │
│                                                             │
│  Videos ──→ Title Generation ──→ Metadata                   │
│        └──→ Thumbnail Generation                            │
│        └──→ Upload to Storage                               │
│        └──→ Database Update                                 │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Output Shorts    │
                │ (Social Media    │
                │  Ready)          │
                └──────────────────┘
```

---

## API & Queue System

### Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        API SERVICE                           │
│  (FastAPI - Port 8000/2000)                                  │
│                                                              │
│  Endpoints:                                                  │
│  • POST /api/queue/video      - Submit processing job       │
│  • GET  /api/queue/list       - List all jobs               │
│  • GET  /api/queue/status/:id - Check job status            │
│  • GET  /api/health           - Health check                │
│  • GET  /api/queue/micro-drama/:id - Query micro drama      │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ Redis Protocol
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                   REDIS MESSAGE QUEUE                        │
│  (Port 6379)                                                 │
│                                                              │
│  Queues:                                                     │
│  • video_processing - High-priority video tasks             │
│  • celery          - General background tasks               │
│                                                              │
│  Data:                                                       │
│  • Task metadata and parameters                             │
│  • Processing status                                         │
│  • Result storage                                            │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         │ Celery Protocol
                         │
┌────────────────────────▼─────────────────────────────────────┐
│                    CELERY WORKERS                            │
│  (GPU-Accelerated Processing)                                │
│                                                              │
│  Worker Tasks:                                               │
│  • process_video_task - Main processing pipeline            │
│  • cleanup_task       - Cleanup temporary files             │
│  • retry_task         - Retry failed jobs                   │
│                                                              │
│  Features:                                                   │
│  • GPU-accelerated AI inference                             │
│  • Automatic retry on failure                               │
│  • Progress tracking                                         │
│  • Resource management                                       │
└──────────────────────────────────────────────────────────────┘
```

### API Request Flow

```
Client Request
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ POST /api/queue/video                                       │
│                                                             │
│ {                                                           │
│   "video_id": "video_123",                                  │
│   "canvas_type": "shorts",                                  │
│   "no_of_videos": 5,                                        │
│   "final_video_ids": [1001, 1002, 1003, 1004, 1005],       │
│   "bucket_path": "videos/input.mp4",                        │
│   "storage_bucket": "my-bucket",                            │
│   "user_prompt": "climax scenes",                           │
│   "subtitle_overlay": true,                                 │
│   "kinetic_captions": true,                                 │
│   "min_duration": 15,                                       │
│   "max_duration": 60,                                       │
│   "intro_url": "intros/intro.mp4",                          │
│   "outro_url": "outros/outro.mp4"                           │
│ }                                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
                 Validate Request
                         │
                         ▼
                  Create Celery Task
                         │
                         ▼
                 Submit to Redis Queue
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Response:                                                    │
│ {                                                            │
│   "status": "success",                                       │
│   "task_id": "abc123...",                                    │
│   "video_id": "video_123",                                   │
│   "message": "Video processing started"                      │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
          Client polls status endpoint
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ GET /api/queue/status/video_123                             │
│                                                             │
│ Response:                                                    │
│ {                                                            │
│   "status": "processing",                                    │
│   "progress": "Audio analysis complete",                     │
│   "task_id": "abc123..."                                     │
│ }                                                            │
│                                                             │
│ Eventually:                                                  │
│ {                                                            │
│   "status": "completed",                                     │
│   "shorts_generated": 5,                                     │
│   "output_files": [                                          │
│     "video_123_1001_shorts.mp4",                            │
│     "video_123_1002_shorts.mp4",                            │
│     ...                                                      │
│   ],                                                         │
│   "metadata": { ... }                                        │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
```

### Micro Drama Mode (Auto ID Generation)

```
Client Request (No final_video_ids)
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│ POST /api/queue/video                                       │
│                                                             │
│ {                                                           │
│   "video_id": "drama_episode_1",                            │
│   "micro_drama_id": 123,      ← Triggers auto ID generation│
│   "linear_cut": true,                                       │
│   "linear_duration": 60,                                    │
│   "final_video_ids": [],      ← Empty - will be generated  │
│   "bucket_path": "videos/episode.mp4",                      │
│   "storage_bucket": "my-bucket"                             │
│ }                                                           │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
               Backend Processing:
               1. Get video duration (e.g., 600s)
               2. Calculate segments: 600s / 60s = 10
               3. Generate IDs: [123001, 123002, ..., 123010]
               4. Process with generated IDs
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│ Response includes generated IDs:                            │
│ {                                                            │
│   "status": "success",                                       │
│   "task_id": "xyz789...",                                    │
│   "video_id": "drama_episode_1",                             │
│   "micro_drama_id": 123,                                     │
│   "generated_final_video_ids": [                             │
│     123001, 123002, 123003, 123004, 123005,                 │
│     123006, 123007, 123008, 123009, 123010                  │
│   ],                                                         │
│   "segments_count": 10                                       │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
                         │
                         ▼
         Client uses generated IDs for updates
```

---

## AI/ML Integration

### LLM Services

**Ollama (Default)**:
- Local/self-hosted LLM inference
- Models: LLaMA 3, Mistral, etc.
- Vision models: LLaVA for image understanding
- Port: 11434
- Benefits: Privacy, no API costs, customizable

**OpenAI (Alternative)**:
- Cloud-based LLM service
- Models: GPT-4, GPT-4 Vision
- Benefits: Higher quality, no infrastructure needed
- Requires API key

### Model Usage

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Service Integration                   │
└─────────────────────────────────────────────────────────────┘

1. PROMPT ANALYSIS
   User Prompt ──→ LLM ──→ Theme Requirements
   "climax scenes" ──→ {
     theme: "dramatic",
     keywords: ["climax", "peak", "intense"],
     duration_preference: "shorter",
     visual_cues: ["action", "emotion"],
     audio_cues: ["music_build", "dialogue_peak"]
   }

2. CONTENT ANALYSIS
   Transcription + Vision ──→ LLM ──→ Content Understanding
   - Sentiment analysis
   - Topic extraction
   - Engagement prediction
   - Scene classification

3. DUAL-MODAL ANALYSIS
   Audio + Visual Context ──→ LLM ──→ Comprehensive Insights
   - Audio-visual alignment
   - Scene type classification
   - Engagement scoring
   - Quality assessment

4. TITLE GENERATION
   Segment Transcription ──→ LLM ──→ Titles + Descriptions
   - Engaging titles
   - SEO-optimized descriptions
   - Relevant tags
   - Content type classification

5. AI REFRAMING
   Prompt + Frame Analysis ──→ LLM ──→ Framing Recommendations
   - Object importance ranking
   - Crop position suggestions
   - Focus area identification
```

### Computer Vision Models

**OpenAI Whisper**:
- Purpose: Speech-to-text transcription
- Models: tiny, base, small, medium, large
- GPU-accelerated
- Word-level timestamps

**MediaPipe**:
- Purpose: Face detection and pose estimation
- Real-time performance
- Facial landmark detection
- Multi-face support

**YOLOv8**:
- Purpose: Object detection
- 80+ object classes
- Real-time inference
- GPU-optimized

**LLaVA (via Ollama)**:
- Purpose: Vision-language understanding
- Frame-level scene analysis
- Visual question answering
- Scene classification

---

## Storage & Caching

### Storage Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     STORAGE LAYERS                          │
└─────────────────────────────────────────────────────────────┘

1. REMOTE OBJECT STORAGE (Primary)
   
   Wasabi S3:
   • Input videos
   • Output shorts
   • Intro/outro videos
   • Thumbnails
   • Persistent storage
   
   AWS S3 (Alternative):
   • Same functionality as Wasabi
   • Configured via storage_type parameter

2. LOCAL CACHE (Performance)
   
   /cache/
   ├── whisper/        - Transcription cache
   │   └── {video_id}.json
   ├── vision/         - Vision analysis cache
   │   └── {video_id}_frames/
   └── ollama/         - LLM response cache
       └── {hash}.json

3. TEMPORARY STORAGE (Processing)
   
   /temp/
   ├── audio/          - Extracted audio tracks
   ├── segments/       - Extracted video segments
   ├── frames/         - Extracted frames for analysis
   └── processing/     - Intermediate files

4. MODEL STORAGE (AI Models)
   
   /models/
   ├── whisper/        - Whisper model weights
   ├── yolo/           - YOLO model weights
   └── mediapipe/      - MediaPipe models

5. OUTPUT STORAGE (Final Products)
   
   /output/
   └── {video_name}_{clip_id}_{canvas_type}.mp4
```

### Caching Strategy

**Transcription Cache**:
- Key: `video_id` or `uploaded_video_id`
- Avoids re-transcribing same video
- Significant time savings (30-60s per video)

**Vision Analysis Cache**:
- Key: `video_id` + segment timestamps
- Reuses frame analysis
- Reduces vision API calls

**LLM Response Cache**:
- Key: Hash of (prompt + context)
- Avoids redundant LLM calls
- Faster response times

---

## Deployment Architecture

### Docker Compose Setup

```yaml
services:
  # API Service
  api:
    - Exposes REST API
    - Port: 2000 (mapped to internal 8000)
    - Stateless, horizontally scalable
    
  # Celery Worker
  worker:
    - GPU-accelerated processing
    - CUDA support
    - Auto-restart on failure
    - Multiple workers can run in parallel
    
  # Redis
  redis:
    - Message queue broker
    - Result backend
    - Task status storage
    - Port: 6380 (mapped to internal 6379)
    
  # Ollama (Optional)
    - LLM inference service
    - Port: 11434
    - GPU-accelerated
    - Model auto-download on first use
```

### Infrastructure Requirements

**Minimum**:
- 16GB RAM
- 8GB GPU VRAM (NVIDIA)
- 100GB disk space
- CUDA 11.8+

**Recommended**:
- 32GB RAM
- 16GB+ GPU VRAM (RTX 3090/4090, A6000)
- 500GB SSD storage
- CUDA 12.0+
- Multi-GPU support

**Network**:
- High-bandwidth internet for S3 operations
- Low-latency connection to storage
- Optional: VPN for Ollama if remote

### Scaling Considerations

**Horizontal Scaling**:
- Add more Celery workers
- Use dedicated GPUs per worker
- Load balance API requests
- Distributed Redis cluster

**Vertical Scaling**:
- Larger GPU memory for bigger models
- More CPU cores for parallel processing
- Faster storage (NVMe SSDs)

**Performance Optimization**:
- Use smaller Whisper models for speed
- Enable linear cut mode for maximum throughput
- Batch process multiple videos
- Optimize FFmpeg settings
- Use hardware video encoding (NVENC)

---

## Processing Modes Comparison

| Feature | AI Mode (Prompt/Comprehensive) | Linear Cut Mode |
|---------|-------------------------------|-----------------|
| **Speed** | Slower (AI analysis overhead) | 7-10x faster |
| **Audio Analysis** | Full Whisper transcription | Optional (subtitles only) |
| **Scene Detection** | Full analysis | Skipped |
| **Vision Analysis** | Optional (comprehensive) | Skipped |
| **Object Detection** | Optional (prompt-based) | Skipped |
| **LLM Analysis** | Required | Skipped |
| **Segment Selection** | AI-driven quality-based | Time-based equal division |
| **Smart Zoom** | Full subject tracking | Optional fast crop |
| **Title Generation** | AI-powered, contextual | AI-powered with part numbers |
| **Best Use Case** | Quality content curation | High-volume batch processing |

---

## Key Design Principles

1. **Modularity**: Each component is independent and swappable
2. **Flexibility**: Multiple processing modes for different use cases
3. **Performance**: GPU acceleration and caching throughout
4. **Reliability**: Retry logic, error handling, fallback mechanisms
5. **Scalability**: Distributed processing with queue system
6. **Quality**: Multi-modal AI analysis for best segment selection
7. **Observability**: Comprehensive logging and analysis tracking

---

## Technology Stack Summary

**Core Technologies**:
- Python 3.8+
- FFmpeg (video processing)
- OpenCV (computer vision)
- PyTorch (deep learning)

**AI/ML**:
- OpenAI Whisper (speech-to-text)
- Ollama/OpenAI (LLM services)
- YOLOv8 (object detection)
- MediaPipe (face/pose detection)
- LLaVA (vision-language)

**Infrastructure**:
- FastAPI (REST API)
- Celery (distributed task queue)
- Redis (message broker)
- Docker (containerization)
- Wasabi S3 / AWS S3 (object storage)

**Supporting Libraries**:
- Pydantic (data validation)
- boto3 (S3 client)
- PySceneDetect (scene detection)
- Pillow (image processing)
- NumPy (numerical computing)

---

This architecture enables the system to transform long-form videos into engaging short-form content through intelligent AI-powered analysis and processing, while maintaining flexibility for different use cases and scale requirements.
