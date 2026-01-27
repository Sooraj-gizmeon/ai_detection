#!/usr/bin/env python3
"""
Example script demonstrating prompt-based video processing for theme-specific short creation.

This script shows how to use the enhanced video processing system to create
shorts based on specific user prompts like "climax scenes", "comedy shorts", etc.
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import VideoToShortsProcessor


async def example_prompt_based_processing():
    """
    Demonstrate various prompt-based processing scenarios.
    """
    
    print("=== Prompt-Based Video Processing Examples ===\n")
    
    # Initialize processor
    processor = VideoToShortsProcessor()
    
    try:
        # Initialize components
        await processor.initialize_components()
        
        # Example video file (replace with your actual video path)
        video_path = "input/example_video.mp4"
        
        if not Path(video_path).exists():
            print(f"Example video not found at {video_path}")
            print("Please place a video file at the specified path or update the path in this script.")
            return
        
        # Example 1: Extract climax scenes
        print("Example 1: Extracting climax scenes...")
        print("Prompt: 'create shorts on the climax scenes'")
        
        result1 = await processor.process_video(
            video_path, 
            user_prompt="create shorts on the climax scenes"
        )
        
        print(f"✓ Generated {result1['shorts_generated']} climax scene shorts")
        print(f"  Processing time: {result1['processing_time']:.2f} seconds")
        print(f"  Analysis method: {result1.get('content_analysis_results', {}).get('analysis_method', 'unknown')}")
        print()
        
        # Example 2: Extract comedy content
        print("Example 2: Extracting comedy content...")
        print("Prompt: 'Create comedy shorts'")
        
        result2 = await processor.process_video(
            video_path, 
            user_prompt="Create comedy shorts"
        )
        
        print(f"✓ Generated {result2['shorts_generated']} comedy shorts")
        print(f"  Processing time: {result2['processing_time']:.2f} seconds")
        print()
        
        # Example 3: Extract emotional parts
        print("Example 3: Extracting emotional content...")
        print("Prompt: 'create a shorts so that it conveys the emotional parts of the movie'")
        
        result3 = await processor.process_video(
            video_path, 
            user_prompt="create a shorts so that it conveys the emotional parts of the movie"
        )
        
        print(f"✓ Generated {result3['shorts_generated']} emotional shorts")
        print(f"  Processing time: {result3['processing_time']:.2f} seconds")
        print()
        
        # Example 4: Extract educational content
        print("Example 4: Extracting educational content...")
        print("Prompt: 'extract the most educational segments'")
        
        result4 = await processor.process_video(
            video_path, 
            user_prompt="extract the most educational segments"
        )
        
        print(f"✓ Generated {result4['shorts_generated']} educational shorts")
        print(f"  Processing time: {result4['processing_time']:.2f} seconds")
        print()
        
        # Example 5: No prompt (comprehensive analysis)
        print("Example 5: Comprehensive analysis (no specific prompt)...")
        
        result5 = await processor.process_video(video_path)
        
        print(f"✓ Generated {result5['shorts_generated']} general shorts")
        print(f"  Processing time: {result5['processing_time']:.2f} seconds")
        print(f"  Analysis method: {result5.get('content_analysis_results', {}).get('analysis_method', 'unknown')}")
        print()
        
        # Summary
        total_shorts = sum([
            result1['shorts_generated'],
            result2['shorts_generated'], 
            result3['shorts_generated'],
            result4['shorts_generated'],
            result5['shorts_generated']
        ])
        
        print("=== Summary ===")
        print(f"Total shorts generated: {total_shorts}")
        print(f"Climax scenes: {result1['shorts_generated']}")
        print(f"Comedy content: {result2['shorts_generated']}")
        print(f"Emotional parts: {result3['shorts_generated']}")
        print(f"Educational content: {result4['shorts_generated']}")
        print(f"General analysis: {result5['shorts_generated']}")
        
        # Show supported themes
        print("\n=== Supported Themes ===")
        themes = processor.content_analyzer.get_supported_themes()
        for theme, description in themes.items():
            print(f"  {theme}: {description}")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        logging.error(f"Processing error: {e}", exc_info=True)
    
    finally:
        # Cleanup
        await processor.cleanup()


async def example_api_usage():
    """
    Show example of how to use the API endpoints for prompt-based processing.
    """
    
    print("\n=== API Usage Examples ===\n")
    
    # Example API request for climax scenes
    api_request_climax = {
        "video_id": "example_video_001",
        "user_prompt": "create shorts on the climax scenes",
        "canvas_type": "shorts",
        "no_of_videos": 5,
        "min_duration": 15,
        "max_duration": 60,
        "subtitle_overlay": True,
        "pubid": "example_publisher",
        "bucket_path": "videos/example_video.mp4",
        "final_video_ids": [1001, 1002, 1003, 1004, 1005]
    }
    
    print("Example API request for climax scenes:")
    print("POST /api/queue/video")
    print("Content-Type: application/json")
    print()
    import json
    print(json.dumps(api_request_climax, indent=2))
    print()
    
    # Example API request for comedy content
    api_request_comedy = {
        "video_id": "example_video_002", 
        "user_prompt": "Create comedy shorts",
        "canvas_type": "shorts",
        "no_of_videos": 3,
        "min_duration": 20,
        "max_duration": 45,
        "subtitle_overlay": False,
        "pubid": "example_publisher",
        "bucket_path": "videos/funny_video.mp4",
        "final_video_ids": [2001, 2002, 2003]
    }
    
    print("Example API request for comedy content:")
    print("POST /api/queue/video")
    print("Content-Type: application/json")
    print()
    print(json.dumps(api_request_comedy, indent=2))
    print()
    
    # Show supported themes endpoint
    print("Get supported themes:")
    print("GET /api/supported-themes")
    print()
    print("Response structure:")
    example_response = {
        "supported_themes": {
            "climax": "intense, high-energy, dramatic moments with emotional peaks",
            "comedy": "humorous, entertaining, light-hearted moments",
            "emotional": "emotionally engaging, touching, inspiring moments"
        },
        "examples": {
            "climax": "create shorts on the climax scenes",
            "comedy": "Create comedy shorts",
            "emotional": "create a shorts so that it conveys the emotional parts of the movie"
        },
        "usage_note": "Use these themes in the user_prompt field to get theme-specific content selection"
    }
    print(json.dumps(example_response, indent=2))


async def main():
    """Main function to run all examples."""
    
    print("Video-to-Shorts Enhanced Processing Examples")
    print("=" * 50)
    print()
    print("This script demonstrates the new prompt-based content selection features.")
    print("You can now create shorts based on specific themes like:")
    print("  • Climax scenes")
    print("  • Comedy content") 
    print("  • Emotional parts")
    print("  • Educational segments")
    print("  • Motivational moments")
    print("  • Action sequences")
    print("  • And more!")
    print()
    
    # Run the processing examples
    await example_prompt_based_processing()
    
    # Show API usage examples
    await example_api_usage()
    
    print("\n=== Complete! ===")
    print("Check the output directory for generated shorts.")
    print("Each prompt type will generate different content based on the theme.")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    asyncio.run(main())
