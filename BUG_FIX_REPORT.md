# Bug Fix Report: Missing load_object_index Function

## Issue Found
The production log analysis revealed a critical import error preventing the actor-only segment extraction from working:

```
2026-01-29 06:59:37 - src.content_analysis.prompt_based_analyzer - WARNING - 
Could not load celebrity/object index: cannot import name 'load_object_index' 
from 'src.face_insights.celebrity_index'
```

This error occurred on **line 281** in `prompt_based_analyzer.py` when trying to import `load_object_index`.

## Root Cause
The function `load_object_index()` was referenced in [prompt_based_analyzer.py](src/content_analysis/prompt_based_analyzer.py#L279) but was never implemented in [celebrity_index.py](src/face_insights/celebrity_index.py).

When the import failed with an exception, the try-except block at line 283 caught it silently, logging only a WARNING. This prevented:
1. Loading the celebrity index
2. Detecting actor-only requests
3. Using the ActorSegmentExtractor
4. Returning precomputed segments from Rekognition

Instead, the system fell back to generic LLM-based analysis, which then failed because no candidate segments were available (since actor-only mode was supposed to skip candidate generation).

## Solution Implemented
Added `load_object_index()` function to [src/face_insights/celebrity_index.py](src/face_insights/celebrity_index.py#L42) with the following features:

```python
def load_object_index(path: str) -> Tuple[Dict[str, List[Dict]], Dict[str, float], Dict[str, str]]:
    """
    Load result.json for object detection data.
    Returns:
      - appearances_per_object: object_id -> list of dicts with 'start_sec', 'end_sec', 'score'
      - object_conf: object_id -> confidence score
      - object_labels: object_id -> label/name
    """
```

### Implementation Details
- Parses the JSON result file for objects section
- Consolidates object appearances across multiple detections
- Returns structured data matching the format expected by `prompt_based_analyzer.py`
- Gracefully handles missing data (returns empty dictionaries if no objects exist)
- Uses same error-tolerant approach as `load_celebrity_index()`

## Impact
With this fix:
1. ✅ The import will succeed
2. ✅ Celebrity/object indices will be loaded correctly
3. ✅ Actor-only request detection will work
4. ✅ ActorSegmentExtractor will be properly invoked
5. ✅ Segments will be extracted from precomputed Rekognition results
6. ✅ No random candidate generation will occur
7. ✅ Precomputed confidence scores will be preserved

## File Modified
- [src/face_insights/celebrity_index.py](src/face_insights/celebrity_index.py)
  - Added `load_object_index()` function (lines 42-77)

## Verification
The fix can be verified by:
1. Running the video processing with a user prompt like "generate only clips with Rupert Grint"
2. Checking logs for successful import messages and actor detection
3. Verifying segments are generated from precomputed timestamps only
4. Confirming no fallback to candidate segment generation occurs

Expected log output after fix:
```
✅ Loaded celebrity index with X actors and 0 objects
🎯 STRICT ACTOR MODE: Extracting segments ONLY from precomputed 'Rupert Grint' timestamps
✅ Generated N segments from precomputed actor timestamps
```
