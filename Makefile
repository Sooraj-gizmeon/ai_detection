# Makefile for Video-to-Shorts Pipeline

.PHONY: help install setup run clean test lint format check-gpu api-run worker-run integrated-run integrated-single verify submit-job check-queue

# Default target
help:
	@echo "Video-to-Shorts Pipeline - Available Commands:"
	@echo ""
	@echo "  setup          - Set up virtual environment and install dependencies"
	@echo "  install        - Install/update dependencies"
	@echo "  run            - Run the pipeline on input directory"
	@echo "  run-single     - Run pipeline on single video file"
	@echo "  run-verbose    - Run with verbose logging"
	@echo "  run-no-zoom    - Run without smart zoom"
	@echo "  run-no-ai      - Run without Ollama AI analysis"
	@echo "  api-run        - Run the API service"
	@echo "  worker-run     - Run the Celery worker"
	@echo "  integrated-run - Run integrated processor (download + process)"
	@echo "  integrated-single - Run integrated processor once and exit"
	@echo "  download-run   - Run video downloader service"
	@echo "  verify         - Verify implementation"
	@echo "  submit-job     - Submit a test job to the API"
	@echo "  check-queue    - Check the current queue status"
	@echo "  test-thumbnail - Test thumbnail generation and upload"
	@echo "  update-db-schema - Update database schema for thumbnails"
	@echo "  check-gpu      - Check GPU compatibility"
	@echo "  test           - Run tests"
	@echo "  test-aspect-ratios - Test various canvas_type and content_aspect_ratio combinations"
	@echo "  lint           - Run code linting"
	@echo "  format         - Format code"
	@echo "  clean          - Clean temporary files and cache"
	@echo "  clean-all      - Clean everything including models"
	@echo "  requirements   - Generate requirements.txt"
	@echo ""
	@echo "Usage Examples:"
	@echo "  make setup                    # Initial setup"
	@echo "  make run                      # Process all videos in input/"
	@echo "  make run-single VIDEO=video.mp4  # Process single video"
	@echo "  make run-verbose              # Run with debug output"
	@echo "  make integrated-run           # Run continuous download and processing"
	@echo "  make test-aspect-ratios VIDEO=input/video.mp4  # Test aspect ratio combinations"
	@echo ""

# Virtual environment setup
VENV_DIR = video-shorts-pipeline
PYTHON = python3
PIP = $(VENV_DIR)/bin/pip
PYTHON_VENV = $(VENV_DIR)/bin/python

# Check if virtual environment exists
VENV_EXISTS := $(shell test -d $(VENV_DIR) && echo "exists")

setup:
	@echo "Setting up Video-to-Shorts Pipeline..."
	@if [ "$(VENV_EXISTS)" != "exists" ]; then \
		echo "Creating virtual environment..."; \
		$(PYTHON) -m venv $(VENV_DIR); \
	fi
	@echo "Activating virtual environment and installing dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
	@$(PIP) install -r requirements.txt
	@echo "Creating .env file from template..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "Creating directories..."
	@mkdir -p input output temp cache logs models
	@echo ""
	@echo "Setup complete! Please update .env file with your Ollama bearer token."
	@echo "Place your input videos in the 'input' directory."
	@echo "Run 'make run' to start processing."

install:
	@echo "Installing/updating dependencies..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "Dependencies updated."

# Main run commands
run:
	@echo "Running Video-to-Shorts Pipeline..."
	@$(PYTHON_VENV) main.py

run-single:
	@if [ -z "$(VIDEO)" ]; then \
		echo "Error: Please specify VIDEO parameter"; \
		echo "Usage: make run-single VIDEO=path/to/video.mp4"; \
		exit 1; \
	fi
	@echo "Processing single video: $(VIDEO)"
	@$(PYTHON_VENV) main.py --input $(VIDEO)

run-verbose:
	@echo "Running with verbose logging..."
	@$(PYTHON_VENV) main.py --verbose

run-no-zoom:
	@echo "Running without smart zoom..."
	@$(PYTHON_VENV) main.py --no-smart-zoom

run-no-ai:
	@echo "Running without Ollama AI analysis..."
	@$(PYTHON_VENV) main.py --no-ollama

# API and Worker commands
api-run:
	@echo "Starting API service..."
	@if command -v docker-compose >/dev/null 2>&1; then \
		echo "Using Docker Compose for API service"; \
		docker-compose up -d api; \
	else \
		echo "Installing FastAPI dependencies first..."; \
		$(PIP) install fastapi uvicorn; \
		$(PYTHON_VENV) -m src.api.app; \
	fi

worker-run:
	@echo "Starting Celery worker..."
	@if command -v docker-compose >/dev/null 2>&1; then \
		echo "Using Docker Compose for worker service"; \
		docker-compose up -d worker; \
	else \
		echo "Installing Celery dependencies first..."; \
		$(PIP) install celery redis; \
		$(PYTHON_VENV) -m celery -A src.queue_system.celery_app worker --loglevel=info --queue=video_processing; \
	fi

# Integrated processing commands
integrated-run:
	@echo "Starting integrated processor (download + process)..."
	@$(PYTHON_VENV) -m src.integrated_processor --verbose

integrated-single:
	@echo "Running integrated processor once..."
	@$(PYTHON_VENV) -m src.integrated_processor --single-run --verbose

download-run:
	@echo "Starting video downloader service..."
	@$(PYTHON_VENV) download_from_wasabi.py

run-custom:
	@echo "Running with custom parameters..."
	@$(PYTHON_VENV) main.py $(ARGS)

# Canvas and aspect ratio specific runs
run-shorts:
	@echo "Running with shorts canvas type (9:16)..."
	@$(PYTHON_VENV) main.py --aspect-ratio 9:16 --canvas-type shorts

run-clips:
	@echo "Running with clips canvas type (16:9)..."
	@$(PYTHON_VENV) main.py --aspect-ratio 16:9 --canvas-type clips

run-square:
	@echo "Running with square aspect ratio (1:1)..."
	@$(PYTHON_VENV) main.py --aspect-ratio 1:1 --canvas-type square

run-vertical:
	@echo "Running with vertical aspect ratio (4:5)..."
	@$(PYTHON_VENV) main.py --aspect-ratio 4:5 --canvas-type vertical

# Aspect ratio test commands
run-shorts-square:
	@echo "Running with shorts canvas type and 1:1 (square) content aspect ratio..."
	@$(PYTHON_VENV) main.py --canvas-type shorts --aspect-ratio 1:1

run-shorts-cinematic:
	@echo "Running with shorts canvas type and 2.35:1 (cinematic) content aspect ratio..."
	@$(PYTHON_VENV) main.py --canvas-type shorts --aspect-ratio 2.35:1

run-clips-vertical:
	@echo "Running with clips canvas type and 9:16 (vertical) content aspect ratio..."
	@$(PYTHON_VENV) main.py --canvas-type clips --aspect-ratio 9:16

# GPU and system checks
check-gpu:
	@echo "Checking GPU compatibility..."
	@$(PYTHON_VENV) -c "from src.utils.gpu_utils import get_system_info; import json; print(json.dumps(get_system_info(), indent=2))"

check-system:
	@echo "System Information:"
	@$(PYTHON_VENV) -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Development and testing
# Testing and example commands
test:
	@echo "Running tests..."
	@$(PYTHON_VENV) -m pytest tests/ -v

test-integrated:
	@echo "Testing integrated processor..."
	@$(PYTHON_VENV) test_integrated.py

submit-job:
	@echo "Submitting a test job to the API..."
	@$(PYTHON_VENV) submit_job.py --video-id test_video_001 --video-path input/video.mp4 --no-of-videos 3 --canvas-type shorts

check-queue:
	@echo "Checking queue status..."
	@$(PYTHON_VENV) submit_job.py --check-queue

lint:
	@echo "Running linting..."
	@$(PYTHON_VENV) -m flake8 src/ main.py --max-line-length=100

format:
	@echo "Formatting code..."
	@$(PYTHON_VENV) -m black src/ main.py --line-length=100

type-check:
	@echo "Running type checking..."
	@$(PYTHON_VENV) -m mypy src/ main.py --ignore-missing-imports

# Cleanup commands
clean:
	@echo "Cleaning temporary files..."
	@rm -rf temp/*
	@rm -rf cache/*
	@rm -rf logs/*
	@rm -rf __pycache__
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -type d -exec rm -rf {} +
	@echo "Cleanup complete."

clean-all: clean
	@echo "Cleaning all files including models..."
	@rm -rf models/*
	@rm -rf output/*
	@echo "Full cleanup complete."

clean-venv:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "Virtual environment removed."

# Requirements management
requirements:
	@echo "Generating requirements.txt..."
	@$(PIP) freeze > requirements.txt
	@echo "Requirements.txt updated."

# Docker commands
docker-build:
	@echo "Building Docker image..."
	@docker build -t video-to-shorts .

docker-run:
	@echo "Running in Docker container..."
	@docker run -v $(PWD)/input:/app/input -v $(PWD)/output:/app/output video-to-shorts

services-up:
	@echo "Starting all services with Docker..."
	@docker-compose up -d

services-down:
	@echo "Stopping all services..."
	@docker-compose down

services-restart:
	@echo "Restarting all services..."
	@docker-compose restart

services-logs:
	@echo "Showing services logs..."
	@docker-compose logs -f

# Development helpers
dev-setup: setup
	@echo "Setting up development environment..."
	@$(PIP) install pytest black flake8 mypy jupyter notebook
	@echo "Development environment ready."

notebook:
	@echo "Starting Jupyter notebook..."
	@$(PYTHON_VENV) -m jupyter notebook

# Monitoring and logs


monitor:
	@echo "Monitoring GPU usage..."
	@watch -n 1 nvidia-smi

# Verification
verify:
	@echo "Verifying implementation..."
	@$(PYTHON_VENV) verify_implementation.py

# Model management
download-models:
	@echo "Downloading required models..."
	@$(PYTHON_VENV) -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
	@$(PYTHON_VENV) -c "import whisper; whisper.load_model('base')"
	@echo "Models downloaded."

# Batch processing
batch-process:
	@echo "Processing all videos in input directory..."
	@for video in input/*.mp4 input/*.avi input/*.mov; do \
		if [ -f "$$video" ]; then \
			echo "Processing $$video..."; \
			$(PYTHON_VENV) main.py --input "$$video"; \
		fi; \
	done

# Quality checks
validate-output:
	@echo "Validating output videos..."
	@$(PYTHON_VENV) -c "from src.utils.video_utils import validate_video_files; validate_video_files('output')"

# Performance testing
benchmark:
	@echo "Running performance benchmark..."
	@$(PYTHON_VENV) -c "from src.utils.benchmark import run_benchmark; run_benchmark()"

# Configuration
config-check:
	@echo "Checking configuration..."
	@$(PYTHON_VENV) -c "from config.smart_zoom_settings import *; print('Configuration loaded successfully')"

# Help for specific commands
help-run:
	@echo "Run command options:"
	@echo "  make run                    - Process all videos in input/"
	@echo "  make run-single VIDEO=file  - Process single video"
	@echo "  make run-verbose            - Run with debug output"
	@echo "  make run-no-zoom            - Disable smart zoom"
	@echo "  make run-no-ai              - Disable AI analysis"
	@echo "  make run-custom ARGS='...'  - Run with custom arguments"

# Thumbnail functionality
test-thumbnail:
	@echo "Testing thumbnail generation and upload..."
	@if [ -z "$(VIDEO)" ]; then \
		echo "Using first video in output directory..."; \
		$(PYTHON_VENV) test_thumbnail.py; \
	else \
		echo "Testing with specified video: $(VIDEO)"; \
		$(PYTHON_VENV) test_thumbnail.py $(VIDEO); \
	fi

update-db-schema:
	@echo "Updating database schema to add thumbnail column..."
	@$(PYTHON_VENV) update_db_schema.py

help-setup:
	@echo "Setup commands:"
	@echo "  make setup          - Initial setup with virtual environment"
	@echo "  make install        - Install/update dependencies"
	@echo "  make download-models - Download required AI models"
	@echo "  make check-gpu      - Check GPU compatibility"

# Status check
status:
	@echo "Pipeline Status:"
	@echo "Virtual Environment: $(if $(VENV_EXISTS),✓ Exists,✗ Missing)"
	@echo "Configuration: $(if $(shell test -f .env && echo exists),✓ Exists,✗ Missing)"
	@echo "Input Directory: $(if $(shell test -d input && echo exists),✓ Exists,✗ Missing)"
	@echo "Output Directory: $(if $(shell test -d output && echo exists),✓ Exists,✗ Missing)"
	@echo "Models Directory: $(if $(shell test -d models && echo exists),✓ Exists,✗ Missing)"
	@echo ""
	@if [ "$(VENV_EXISTS)" = "exists" ]; then \
		echo "Python: $$($(PYTHON_VENV) --version)"; \
		echo "PyTorch: $$($(PYTHON_VENV) -c 'import torch; print(torch.__version__)')"; \
		echo "CUDA: $$($(PYTHON_VENV) -c 'import torch; print(torch.cuda.is_available())')"; \
	fi

# Quick start
quick-start: setup download-models
	@echo ""
	@echo "🚀 Quick Start Complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Update .env file with your Ollama bearer token"
	@echo "2. Place video files in the 'input' directory"
	@echo "3. Run 'make run' to start processing"
	@echo ""
	@echo "For help: make help"

download-mp4:
	source video-shorts-pipeline/bin/activate && python download_from_wasabi.py

# Test aspect ratios
test-aspect-ratios:
	@echo "Testing different canvas_type and content_aspect_ratio combinations..."
	@if [ -z "$(VIDEO)" ]; then \
		echo "Error: Please specify VIDEO parameter"; \
		echo "Usage: make test-aspect-ratios VIDEO=path/to/video.mp4"; \
		exit 1; \
	fi
	@$(PYTHON_VENV) test_aspect_ratios.py --input $(VIDEO) --output output/test_aspect_ratios

# Testing API with different aspect ratios
test-api-aspect-ratios:
	@echo "Testing API with different canvas_type and aspect_ratio combinations..."
	@$(PYTHON_VENV) test_aspect_ratio_api.py

test-canvas-type:
	@echo "Testing canvas_type and aspect_ratio handling..."
	@if [ -z "$(VIDEO)" ]; then \
		echo "Error: Please specify VIDEO parameter"; \
		echo "Usage: make test-canvas-type VIDEO=input/video.mp4"; \
		exit 1; \
	fi
	@$(PYTHON_VENV) test_canvas_type.py --input-video $(VIDEO)

up:
	@echo "Starting all services..."
	docker compose up -d

build:
	@echo "Building Docker images..."
	docker compose build --no-cache

bu:
	@echo "Building Docker images..."
	docker compose build
down:
	@echo "Stopping all services..."
	docker compose down

log:
	docker compose logs -f 

llm:
	docker compose exec ollama ollama run mistral-small3.2:latest --keepalive 999h

reset:
	make down && make bu && make up && make log
restart:
	make down && make bu && make up