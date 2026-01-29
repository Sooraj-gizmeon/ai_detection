# Non-Overlapping Actor Segment Generation - Implementation Guide

## Problem Statement

Previously, when generating video clips for a specific actor, the system could produce overlapping or duplicated segments. This occurred because:

1. Multiple timestamps within a 60-second window would each generate their own segment
2. Both "primary" (60s) and "extended" (120s) segments were created for every timestamp
3. There was no mechanism to prevent overlapping time ranges

**Example Issue:**
- Actor appears at: [18s, 21s, 25s, 31s, 749s, 754s, ...]
- With 60s min_duration, segments would be:
  - Timestamp 18s → segment 0-60s
  - Timestamp 21s → segment 0-60s *(DUPLICATE)*
  - Timestamp 25s → segment 0-60s *(DUPLICATE)*
  - Timestamp 31s → segment 1-61s *(OVERLAPS)*

This resulted in redundant clips with overlapping content.

## Solution Overview

The solution implements **non-overlapping segment generation** by:

1. **Tracking generated time ranges**: Maintain a list of all segment time ranges already generated
2. **Overlap detection**: Before creating a new segment, check if it overlaps with any existing segment
3. **Skipping overlapping candidates**: If a segment would overlap, skip it and try the next timestamp
4. **Per-actor tracking**: Reset tracking between different actors (independent deduplication)

## Implementation Details

### Key Changes to `actor_segment_extractor.py`

#### 1. Added Time Range Tracking

```python
def __init__(self):
    self.generated_time_ranges = []  # Track all generated segment time ranges
```

#### 2. Added Helper Methods

**`_segments_overlap(seg1_start, seg1_end, seg2_start, seg2_end)`**
- Checks if two time segments overlap
- Logic: `seg1_start < seg2_end AND seg2_start < seg1_end`

**`_has_overlap_with_existing(start_time, end_time)`**
- Checks if a new segment overlaps with any previously generated segment
- Iterates through `generated_time_ranges` and returns True if any overlap found

**`_record_segment(start_time, end_time)`**
- Records a newly created segment's time range
- Appends to `generated_time_ranges` to prevent future overlaps

**`reset_time_ranges()`**
- Clears tracking for new actor
- Called at the start of each actor's segment generation

#### 3. Updated Segment Generation Logic

In `extract_actor_only_segments()`:

```python
# Check for overlap: Skip this segment if it overlaps with any previously generated segment
if self._has_overlap_with_existing(segment_start, segment_end):
    self.logger.debug(f"⏭️  Skipping segment {segment_start:.2f}s-{segment_end:.2f}s (overlaps)")
    continue

# ... create segment ...

# Record this segment's time range to prevent future overlaps
self._record_segment(segment_start, segment_end)
```

#### 4. Updated Multi-Actor Extraction

In `extract_multiple_actors_segments()`:

```python
for actor_name in actor_names:
    # Reset time ranges for each actor to track overlaps independently
    self.reset_time_ranges()
    
    actor_segments = self.extract_actor_only_segments(...)
    all_segments.extend(actor_segments)
```

## Behavior Examples

### Example 1: Very Close Timestamps

**Input:**
- Actor appears at: [10s, 20s, 30s, 40s, 50s]
- min_duration = 60s
- max_duration = 120s

**Execution:**
1. Timestamp 10s → Creates segment 0-60s ✅
   - Also tries extended 0-120s, but skips (overlaps with 0-60s)
2. Timestamp 20s → Tries segment 0-60s, **OVERLAPS**, skips ❌
3. Timestamp 30s → Tries segment 0-60s, **OVERLAPS**, skips ❌
4. Timestamp 40s → Tries segment 10-70s, **OVERLAPS**, skips ❌
5. Timestamp 50s → Tries segment 20-80s, **OVERLAPS**, skips ❌

**Output:** 1 segment (0-60s covering timestamp 10s)

### Example 2: Well-Spaced Timestamps

**Input:**
- Actor appears at: [10s, 100s, 200s, 300s]
- min_duration = 60s

**Execution:**
1. Timestamp 10s → Creates segment 0-60s ✅
   - Extended 0-120s overlaps, skips
2. Timestamp 100s → Creates segment 70-130s ✅
   - No overlap with (0-60s)
   - Extended 40-160s overlaps with (70-130s), skips
3. Timestamp 200s → Creates segment 170-230s ✅
   - Extended skips (overlaps)
4. Timestamp 300s → Creates segment 270-330s ✅

**Output:** 4 segments (one per timestamp, all non-overlapping)

### Example 3: Production Data (Maggie Smith)

**Input:**
```
[18s, 21s, 25s, 31s, 749s, 754s, 755s, 756s, 761s, 762s, 765s, 766s, 767s]
```

**Behavior:**
- Timestamps 18-31s are all within a 60s window → only 1st creates segment
- Timestamps 749-767s are all within a 60s window → only 1st creates segment (extended skips)

**Output:** 2 segments
- [0-60s] (appearance at 18s)
- [719-779s] (appearance at 749s)

**Benefit:** Instead of potentially 13 segments (one per appearance) or 26 segments (13 × 2 with extended), we get only 2 non-overlapping clips.

## Test Results

All test cases pass:

```
✅ Test 1: Close Timestamps (5 appearances → 1 segment)
✅ Test 2: Spaced Timestamps (4 appearances → 4 segments)
✅ Test 3: Maggie Smith Production Data (13 appearances → 2 segments)
✅ Test 4: Dense Consecutive Timestamps (10 appearances → 1 segment)
```

**Key Finding:** The deduplication is aggressive - it prevents any overlapping segments, ensuring each generated clip is unique and distinct.

## Configuration Parameters

The segment generation is controlled by:

- **min_duration**: Minimum segment length (default: 60s)
- **max_duration**: Maximum segment length (default: 120s)

These determine:
- Primary segment: `min_duration` seconds
- Extended segment: min(max_duration, min_duration × 2) seconds

Smaller min_duration values will generate more segments from densely-spaced timestamps.

## Impact on User Experience

### Before (With Duplicates)
- Request: "Generate clips with Maggie Smith"
- Output: Potentially 26 clips (13 appearances × 2)
- Issue: Many clips have overlapping content, redundant

### After (Non-Overlapping)
- Request: "Generate clips with Maggie Smith"  
- Output: 2 clips (clusters of appearances)
- Benefit: Each clip is unique, no redundancy, optimal coverage

## Backward Compatibility

✅ **Fully backward compatible:**
- No changes to API or data structures
- Existing segment dictionaries remain unchanged
- Only prevents generation of overlapping segments
- Other features (LLM analysis, vision processing) unaffected

## Verification

To verify this is working in production:

1. Check logs for `⏭️  Skipping segment` messages
2. Count generated segments vs. input appearances
3. Verify all segment time ranges are non-overlapping:
   ```
   segment[i].end_time <= segment[i+1].start_time
   ```

## Testing

Run the included test suites:

```bash
# Standalone test (no dependencies)
python3 test_deduplication_standalone.py

# Full integration test (requires all dependencies)
python3 test_deduplication.py
```

## Future Enhancements

Possible optimizations:

1. **Configurable overlap tolerance**: Allow small overlaps (< 5s) for continuous action
2. **Time-gap awareness**: Increase segment size if gap is large between appearances
3. **Content-based deduplication**: Use visual similarity to detect actual redundancy
4. **User preference**: Let users choose between dense vs. sparse segment generation
