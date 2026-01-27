"""
Video Analysis Logger - Quality Analysis and Processing Details

This module provides detailed logging for video analysis pipeline,
capturing all inputs, outputs, decisions, and quality metrics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import os


class VideoAnalysisLogger:
    """
    Comprehensive logger for video analysis pipeline.
    
    Captures:
    - Source video metadata
    - All analysis inputs and outputs
    - LLM interactions and responses
    - Object detection results
    - Segment selection decisions
    - Quality metrics and scores
    """
    
    def __init__(self, video_name: str, logs_dir: str = "logs/video_analysis"):
        self.video_name = video_name
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_video_name = "".join(c for c in video_name if c.isalnum() or c in (' ', '-', '_')).rstrip()[:50]
        self.log_file = self.logs_dir / f"video_analysis_{safe_video_name}_{timestamp}.json"
        
        # Initialize analysis data structure
        self.analysis_data = {
            "video_info": {},
            "processing_metadata": {
                "timestamp": timestamp,
                "video_name": video_name,
                "processing_start": datetime.now().isoformat()
            },
            "user_request": {},
            "source_analysis": {},
            "llm_interactions": [],
            "object_detection": {},
            "vision_analysis": {},
            "segment_generation": {},
            "segment_selection": {},
            "final_outputs": {},
            "quality_metrics": {},
            "processing_summary": {}
        }
        
        # Setup logger
        self.logger = logging.getLogger(f"video_analysis_{safe_video_name}")
        
    def log_video_metadata(self, video_info: Dict, video_path: str):
        """Log source video metadata and properties."""
        self.analysis_data["video_info"] = {
            "source_path": video_path,
            "duration_seconds": video_info.get('duration', 0),
            "duration_formatted": self._format_duration(video_info.get('duration', 0)),
            "fps": video_info.get('fps', 0),
            "resolution": video_info.get('resolution', {}),
            "file_size_mb": self._get_file_size_mb(video_path),
            "video_metadata": video_info
        }
        
    def log_user_request(self, user_prompt: Optional[str], config: Dict, task_data: Dict = None):
        """Log user request details and configuration."""
        celebrity_path = None
        try:
            celebrity_path = config.get('celebrity_index_path') if isinstance(config, dict) else None
            if not celebrity_path and task_data:
                celebrity_path = task_data.get('celebrity_result_path')
        except Exception:
            celebrity_path = None

        self.analysis_data["user_request"] = {
            "user_prompt": user_prompt,
            "prompt_provided": user_prompt is not None,
            "config": config,
            "task_data": task_data or {},
            "celebrity_index_path": celebrity_path,
            "analysis_type": "prompt_based" if user_prompt else "generic_analysis"
        }
        
    def log_audio_analysis(self, audio_analysis: Dict):
        """Log audio transcription and analysis results."""
        transcription = audio_analysis.get('transcription', {})
        segments = transcription.get('segments', [])
        
        self.analysis_data["source_analysis"]["audio"] = {
            "transcription_available": bool(segments),
            "total_segments": len(segments),
            "total_words": sum(len(seg.get('text', '').split()) for seg in segments),
            "languages_detected": list(set(seg.get('language', 'unknown') for seg in segments)),
            "speakers_detected": len(set(seg.get('speaker', 'unknown') for seg in segments)),
            "transcription_sample": segments[:3] if segments else [],
            "silence_periods": transcription.get('silence_periods', []),
            "full_transcription_length": len(transcription.get('text', ''))
        }
        
    def log_llm_interaction(self, interaction_type: str, prompt: str, response: str, 
                           model: str, duration_seconds: float, metadata: Dict = None):
        """Log individual LLM interactions."""
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "type": interaction_type,
            "model": model,
            "duration_seconds": round(duration_seconds, 2),
            "prompt_length": len(prompt),
            "response_length": len(response),
            "prompt_preview": prompt[:200] + "..." if len(prompt) > 200 else prompt,
            "response_preview": response[:200] + "..." if len(response) > 200 else response,
            "full_prompt": prompt,
            "full_response": response,
            "metadata": metadata or {}
        }
        
        self.analysis_data["llm_interactions"].append(interaction)
        
    def log_object_detection(self, object_results: Dict, prompt_analysis: Dict = None):
        """Log object detection results and analysis."""
        detected_objects = object_results.get('detected_objects', [])
        object_tracks = object_results.get('object_tracks', {})
        
        # Summarize detected objects
        object_summary = {}
        for obj in detected_objects:
            class_name = obj.get('class_name', 'unknown')
            object_summary[class_name] = object_summary.get(class_name, 0) + 1
            
        # Summarize tracking results
        tracking_summary = {}
        for track_id, detections in object_tracks.items():
            if detections:
                class_name = detections[0].get('class_name', 'unknown')
                tracking_summary[class_name] = tracking_summary.get(class_name, 0) + 1
                
        self.analysis_data["object_detection"] = {
            "enabled": True,
            "total_detections": len(detected_objects),
            "unique_objects": len(object_summary),
            "object_counts": object_summary,
            "tracks_created": len(object_tracks),
            "tracking_summary": tracking_summary,
            "prompt_analysis": prompt_analysis or {},
            "detection_quality": object_results.get('detection_quality', 0),
            "processing_time": object_results.get('processing_time', 0),
            "frames_processed": object_results.get('frames_processed', 0),
            "sample_detections": detected_objects[:5]  # First 5 for review
        }
        
    def log_object_detection_disabled(self, reason: str = "not_enabled"):
        """Log that object detection was disabled or skipped."""
        self.analysis_data["object_detection"] = {
            "enabled": False,
            "total_detections": 0,
            "unique_objects": 0,
            "object_counts": {},
            "tracks_created": 0,
            "tracking_summary": {},
            "prompt_analysis": {},
            "detection_quality": 0,
            "processing_time": 0,
            "frames_processed": 0,
            "sample_detections": [],
            "disabled_reason": reason
        }
    
    def log_vision_analysis(self, vision_results: Dict):
        """Log vision analysis results."""
        segments = vision_results.get('segments', [])
        
        # Analyze vision metrics
        scene_types = {}
        visual_interest_scores = []
        
        for segment in segments:
            scene_type = segment.get('scene_type', 'unknown')
            scene_types[scene_type] = scene_types.get(scene_type, 0) + 1
            
            visual_interest = segment.get('visual_interest', 0)
            if visual_interest:
                visual_interest_scores.append(visual_interest)
                
        self.analysis_data["vision_analysis"] = {
            "total_segments_analyzed": len(segments),
            "scene_types_detected": scene_types,
            "average_visual_interest": sum(visual_interest_scores) / len(visual_interest_scores) if visual_interest_scores else 0,
            "visual_interest_range": {
                "min": min(visual_interest_scores) if visual_interest_scores else 0,
                "max": max(visual_interest_scores) if visual_interest_scores else 0
            },
            "analysis_method": vision_results.get('analysis_method', 'unknown'),
            "processing_status": vision_results.get('status', 'unknown')
        }
        
    def log_segment_generation(self, candidate_segments: List[Dict], generation_method: str):
        """Log candidate segment generation process."""
        if not candidate_segments:
            self.analysis_data["segment_generation"] = {
                "total_candidates": 0,
                "generation_method": generation_method,
                "error": "No candidates generated"
            }
            return
            
        # Analyze candidate segments
        durations = [seg.get('duration', 0) for seg in candidate_segments]
        quality_scores = [seg.get('quality_score', 0) for seg in candidate_segments if seg.get('quality_score')]
        engagement_scores = [seg.get('engagement_score', 0) for seg in candidate_segments if seg.get('engagement_score')]
        
        self.analysis_data["segment_generation"] = {
            "total_candidates": len(candidate_segments),
            "generation_method": generation_method,
            "duration_stats": {
                "min": min(durations) if durations else 0,
                "max": max(durations) if durations else 0,
                "average": sum(durations) / len(durations) if durations else 0
            },
            "quality_stats": {
                "count_with_scores": len(quality_scores),
                "average": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "min": min(quality_scores) if quality_scores else 0,
                "max": max(quality_scores) if quality_scores else 0
            },
            "engagement_stats": {
                "count_with_scores": len(engagement_scores),
                "average": sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0,
                "min": min(engagement_scores) if engagement_scores else 0,
                "max": max(engagement_scores) if engagement_scores else 0
            },
            "sample_candidates": candidate_segments[:3]  # First 3 for review
        }
        
    def log_segment_selection(self, selected_segments: List[Dict], selection_method: str, 
                            quality_threshold: float, original_count: int):
        """Log segment selection process and decisions."""
        if not selected_segments:
            self.analysis_data["segment_selection"] = {
                "selected_count": 0,
                "original_count": original_count,
                "selection_method": selection_method,
                "quality_threshold": quality_threshold,
                "selection_rate": 0,
                "error": "No segments selected"
            }
            return
            
        # Analyze selected segments
        selection_scores = []
        for seg in selected_segments:
            score_data = {
                "start_time": seg.get('start_time', 0),
                "end_time": seg.get('end_time', 0),
                "duration": seg.get('duration', 0),
                "quality_score": seg.get('quality_score', 0),
                "engagement_score": seg.get('engagement_score', 0),
                "prompt_match_score": seg.get('prompt_match_score', 0),
                "visual_interest": seg.get('visual_interest', 0),
                "object_relevance_score": seg.get('object_relevance_score', 0)
            }
            selection_scores.append(score_data)
            
        self.analysis_data["segment_selection"] = {
            "selected_count": len(selected_segments),
            "original_count": original_count,
            "selection_rate": len(selected_segments) / original_count if original_count > 0 else 0,
            "selection_method": selection_method,
            "quality_threshold": quality_threshold,
            "selected_segments": selection_scores,
            "total_duration": sum(seg.get('duration', 0) for seg in selected_segments),
            "average_quality": sum(seg.get('quality_score', 0) for seg in selected_segments) / len(selected_segments),
            "average_engagement": sum(seg.get('engagement_score', 0) for seg in selected_segments) / len(selected_segments)
        }
        
    def log_final_outputs(self, output_files: List[str], processing_time: float):
        """Log final output generation results."""
        self.analysis_data["final_outputs"] = {
            "output_count": len(output_files),
            "output_files": output_files,
            "processing_time_seconds": round(processing_time, 2),
            "processing_time_formatted": self._format_duration(processing_time),
            "success": len(output_files) > 0
        }
        
        # Update processing metadata
        self.analysis_data["processing_metadata"]["processing_end"] = datetime.now().isoformat()
        self.analysis_data["processing_metadata"]["total_processing_time"] = round(processing_time, 2)
        
    def log_quality_metrics(self, metrics: Dict):
        """Log comprehensive quality metrics."""
        self.analysis_data["quality_metrics"] = metrics
        
    def generate_summary(self):
        """Generate processing summary."""
        video_info = self.analysis_data.get("video_info", {})
        user_request = self.analysis_data.get("user_request", {})
        segment_generation = self.analysis_data.get("segment_generation", {})
        segment_selection = self.analysis_data.get("segment_selection", {})
        final_outputs = self.analysis_data.get("final_outputs", {})
        object_detection = self.analysis_data.get("object_detection", {})
        llm_interactions = self.analysis_data.get("llm_interactions", [])
        
        summary = {
            "source_video": {
                "duration": video_info.get("duration_formatted", "Unknown"),
                "size_mb": video_info.get("file_size_mb", 0)
            },
            "user_request": {
                "prompt_provided": user_request.get("prompt_provided", False),
                "prompt": user_request.get("user_prompt", "None"),
                "analysis_type": user_request.get("analysis_type", "unknown")
            },
            "llm_usage": {
                "total_interactions": len(llm_interactions),
                "total_llm_time": sum(interaction.get("duration_seconds", 0) for interaction in llm_interactions),
                "models_used": list(set(interaction.get("model", "unknown") for interaction in llm_interactions))
            },
            "object_detection": {
                "enabled": object_detection.get("enabled", False),
                "objects_detected": object_detection.get("total_detections", 0),
                "unique_objects": object_detection.get("unique_objects", 0),
                "top_objects": object_detection.get("object_counts", {})
            },
            "segment_analysis": {
                "candidates_generated": segment_generation.get("total_candidates", 0),
                "segments_selected": segment_selection.get("selected_count", 0),
                "selection_rate": f"{segment_selection.get('selection_rate', 0) * 100:.1f}%",
                "selection_method": segment_selection.get("selection_method", "unknown")
            },
            "final_results": {
                "output_videos": final_outputs.get("output_count", 0),
                "total_processing_time": final_outputs.get("processing_time_formatted", "Unknown"),
                "success": final_outputs.get("success", False)
            }
        }
        
        self.analysis_data["processing_summary"] = summary
        return summary
        
    def save_log(self):
        """Save the complete analysis log to file."""
        try:
            # Generate final summary
            self.generate_summary()
            
            # Write to JSON file
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.analysis_data, f, indent=2, ensure_ascii=False, default=str)
                
            # Also create a human-readable summary
            summary_file = self.log_file.with_suffix('.summary.txt')
            self._write_human_readable_summary(summary_file)
            
            self.logger.info(f"Video analysis log saved: {self.log_file}")
            self.logger.info(f"Human-readable summary saved: {summary_file}")
            
            return str(self.log_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save video analysis log: {e}")
            return None
            
    def _write_human_readable_summary(self, summary_file: Path):
        """Write a human-readable summary of the analysis."""
        summary = self.analysis_data.get("processing_summary", {})
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("VIDEO ANALYSIS QUALITY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Source video info
            f.write("SOURCE VIDEO:\n")
            f.write(f"  Name: {self.video_name}\n")
            f.write(f"  Duration: {summary.get('source_video', {}).get('duration', 'Unknown')}\n")
            f.write(f"  Size: {summary.get('source_video', {}).get('size_mb', 0):.1f} MB\n\n")
            
            # User request
            f.write("USER REQUEST:\n")
            user_req = summary.get('user_request', {})
            f.write(f"  Prompt Provided: {user_req.get('prompt_provided', False)}\n")
            f.write(f"  Prompt: {user_req.get('prompt', 'None')}\n")
            f.write(f"  Analysis Type: {user_req.get('analysis_type', 'unknown')}\n\n")
            
            # LLM interactions
            f.write("LLM ANALYSIS:\n")
            llm_usage = summary.get('llm_usage', {})
            f.write(f"  Total Interactions: {llm_usage.get('total_interactions', 0)}\n")
            f.write(f"  Total LLM Time: {llm_usage.get('total_llm_time', 0):.1f} seconds\n")
            f.write(f"  Models Used: {', '.join(llm_usage.get('models_used', []))}\n\n")
            
            # Object detection
            f.write("OBJECT DETECTION:\n")
            obj_det = summary.get('object_detection', {})
            f.write(f"  Enabled: {obj_det.get('enabled', False)}\n")
            f.write(f"  Objects Detected: {obj_det.get('objects_detected', 0)}\n")
            f.write(f"  Unique Object Types: {obj_det.get('unique_objects', 0)}\n")
            top_objects = obj_det.get('top_objects', {})
            if top_objects:
                f.write("  Top Objects:\n")
                for obj_type, count in sorted(top_objects.items(), key=lambda x: x[1], reverse=True)[:5]:
                    f.write(f"    {obj_type}: {count}\n")
            f.write("\n")
            
            # Segment analysis
            f.write("SEGMENT ANALYSIS:\n")
            seg_analysis = summary.get('segment_analysis', {})
            f.write(f"  Candidates Generated: {seg_analysis.get('candidates_generated', 0)}\n")
            f.write(f"  Segments Selected: {seg_analysis.get('segments_selected', 0)}\n")
            f.write(f"  Selection Rate: {seg_analysis.get('selection_rate', '0%')}\n")
            f.write(f"  Selection Method: {seg_analysis.get('selection_method', 'unknown')}\n\n")
            
            # Final results
            f.write("FINAL RESULTS:\n")
            final_results = summary.get('final_results', {})
            f.write(f"  Output Videos: {final_results.get('output_videos', 0)}\n")
            f.write(f"  Processing Time: {final_results.get('total_processing_time', 'Unknown')}\n")
            f.write(f"  Success: {final_results.get('success', False)}\n\n")
            
            # Selected segments details
            segment_selection = self.analysis_data.get("segment_selection", {})
            selected_segments = segment_selection.get("selected_segments", [])
            if selected_segments:
                f.write("SELECTED SEGMENTS:\n")
                for i, seg in enumerate(selected_segments, 1):
                    f.write(f"  Segment {i}:\n")
                    f.write(f"    Time: {self._format_time(seg.get('start_time', 0))} - {self._format_time(seg.get('end_time', 0))}\n")
                    f.write(f"    Duration: {seg.get('duration', 0):.1f}s\n")
                    f.write(f"    Quality Score: {seg.get('quality_score', 0):.2f}\n")
                    f.write(f"    Engagement Score: {seg.get('engagement_score', 0):.2f}\n")
                    f.write(f"    Prompt Match Score: {seg.get('prompt_match_score', 0):.2f}\n")
                    f.write(f"    Visual Interest: {seg.get('visual_interest', 0):.2f}\n")
                    if seg.get('object_relevance_score', 0) > 0:
                        f.write(f"    Object Relevance: {seg.get('object_relevance_score', 0):.2f}\n")
                    f.write("\n")
                    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{int(minutes)}m {secs:.1f}s"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            secs = seconds % 60
            return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"
            
    def _format_time(self, seconds: float) -> str:
        """Format time as MM:SS."""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"
        
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        try:
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                return size_bytes / (1024 * 1024)
        except Exception:
            pass
        return 0.0
