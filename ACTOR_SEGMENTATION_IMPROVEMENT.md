# Actor-Based Video Segmentation Accuracy Improvement

## Overview
This update implements **strict actor-based segmentation** to improve accuracy when users request "generate clips with [actor name]". The system now:

1. **Uses ONLY precomputed actor timestamps** from the celebrity detection results
2. **Never recomputes confidence scores** - uses precomputed values
3. **Skips random segment generation** - generates segments exclusively from actor appearance timestamps
4. **Handles multiple detections** - consolidates same actor detected multiple times with different face IDs

## Changes Made

### 1. New Module: `ActorSegmentExtractor` 
**File**: `src/content_analysis/actor_segment_extractor.py`

A dedicated module for extracting segments strictly from precomputed actor timestamps:

```python
extractor = ActorSegmentExtractor()
segments = extractor.extract_actor_only_segments(
    actor_name="Rupert Grint",
    appearances_per_actor=appearances,  # Dict of actor -> timestamps
    actor_conf=confidence,               # Dict of actor -> confidence scores
    min_duration=60,
    max_duration=120
)
```

**Key features:**
- Consolidates same actor across multiple face detections (e.g., Rupert Grint detected with face_0007.jpg and face_0027.jpg)
- Extracts segments centered on each appearance timestamp
- Marks all segments with `source: 'precomputed_detection'`
- Sets `prompt_match_score: 1.0` (maximum confidence from precomputed data)
- Never falls back to random segment generation

### 2. Enhanced Prompt Analyzer
**File**: `src/content_analysis/prompt_based_analyzer.py`

Modified to detect actor-only requests and use the `ActorSegmentExtractor`:

```python
if actor_matches:
    # STRICT IMPLEMENTATION: Use ONLY precomputed actor timestamps
    # Do NOT recompute confidence scores, do NOT use candidate segments
    
    extractor = ActorSegmentExtractor()
    selected = extractor.extract_multiple_actors_segments(...)
    
    return {
        'status': 'success',
        'analysis_method': 'actor_only_strict',
        'matched_actors': actor_matches,
        'segments': selected,
        'generation_note': 'Segments extracted exclusively from precomputed actor detection timestamps'
    }
```

### 3. Fixed Celebrity Index Loading
**File**: `src/face_insights/celebrity_index.py`

Updated `load_celebrity_index()` to properly consolidate appearances for actors detected multiple times:

```python
# Before: Same actor name would overwrite
appearances_per_actor[name] = times  # ❌ Overwrites previous entry

# After: Consolidates all appearances
if name not in appearances_per_actor:
    appearances_per_actor[name] = []
appearances_per_actor[name].extend(times)  # ✅ Consolidates

# Remove duplicates and sort
appearances_per_actor[actor] = sorted(set(appearances_per_actor[actor]))
```

### 4. Content Analyzer Optimization
**File**: `src/content_analysis/content_analyzer.py`

Added early detection and optimization for actor-only requests:

```python
# OPTIMIZATION: Detect actor-only requests early to skip expensive candidate generation
if actor_only:
    self.logger.info("⚡ ACTOR-ONLY MODE: Skipping comprehensive candidate generation")
    all_candidates = []  # Prompt analyzer will generate from precomputed timestamps
else:
    all_candidates = self.segment_generator.generate_all_possible_segments(...)
```

This saves significant computational cost by skipping the expensive comprehensive segment generation when we're doing actor-only extraction.

## Example Usage

### User Request: "Generate only clips with Rupert Grint"

**Processing Flow:**
1. ✅ System detects actor name "Rupert Grint" in user prompt
2. ✅ Loads precomputed celebrity results JSON
3. ✅ Finds all timestamps where Rupert Grint appears (8 appearances at: 190, 193, 295, 642, 691, 944, 1020, 1126 seconds)
4. ✅ Consolidates appearances (handles face_0007.jpg and face_0027.jpg as same person)
5. ✅ Generates exactly 16 segments (2 per appearance: primary + extended)
6. ✅ Returns segments with:
   - `source: 'precomputed_detection'`
   - `actor_confidence: 1.0` (from Rekognition result)
   - `prompt_match_score: 1.0` (highest - from precomputed)
   - `generation_method: 'actor_only_strict'`

**NO** candidate segment generation, **NO** confidence recomputation, **NO** random segments.

## Data Structure: Celebrity Results JSON

```json
{
  "celebrities": [
    {
      "name": "Rupert Grint",
      "face_id": "face_0007.jpg",
      "confidence": 1.0,
      "appearances": [
        {"timestamp_sec": 190},
        {"timestamp_sec": 193},
        {"timestamp_sec": 295},
        ...
      ]
    },
    {
      "name": "Rupert Grint",
      "face_id": "face_0027.jpg",
      "confidence": 1.0,
      "appearances": [
        {"timestamp_sec": 944}
      ]
    }
  ]
}
```

The system correctly merges both entries into a single actor entry with all 8 timestamps.

## Testing

Run the test script to verify actor extraction:

```bash
python test_actor_extraction.py
```

Expected output:
```
✅ Generated 16 segments for Rupert Grint
  Segment 1: 160.0s - 220.0s (appearance at 190s, method: actor_only_strict, score: 1.00)
  Segment 2: 130.0s - 250.0s (appearance at 190s, method: actor_only_strict_extended, score: 1.00)
  ...
```

## Accuracy Guarantees

When a user requests clips with a specific actor:
- ✅ **100% coverage** of precomputed actor appearances
- ✅ **No false negatives** - every detected appearance gets a segment
- ✅ **No confidence recomputation** - uses precomputed Rekognition scores
- ✅ **No random fallbacks** - segments generated only from actor timestamps
- ✅ **Consolidated detections** - same actor detected multiple times is properly merged

## Performance Impact

For actor-only requests:
- **Reduced computation**: Skips comprehensive segment generation (500+ candidates → 0 candidates)
- **Faster response**: Direct extraction from precomputed timestamps
- **Same quality output**: Uses high-confidence precomputed results

## Backward Compatibility

- Non-actor requests continue to use comprehensive segment generation
- Existing code paths unchanged for general/theme-based prompts
- All changes are additive - no breaking changes to existing APIs

## Migration Notes

If upgrading from previous version:
1. No database changes required
2. No configuration changes needed
3. Existing celebrity result JSON files work as-is
4. Same actor with different face_ids automatically consolidated

## Future Enhancements

1. **Multi-actor requests**: "Generate clips with both Rupert Grint AND Emma Watson"
2. **Actor interaction detection**: "Clips where these actors are together"
3. **Confidence filtering**: "Only clips where actor confidence > 0.95"
4. **Appearance filtering**: "Skip brief appearances < 2 seconds"
