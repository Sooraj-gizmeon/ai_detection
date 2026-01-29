# Actor Timestamp Verification Logging

## Overview
Enhanced logging has been added to track and verify the exact timestamps where actors appear in the generated shorts. This helps you confirm that:
1. The system correctly identifies all actor appearances
2. The generated segments actually contain the actor
3. No appearances are missed or skipped

## Enhanced Logging Locations

### 1. Actor Segment Extractor (`src/content_analysis/actor_segment_extractor.py`)

#### All Appearance Timestamps
```
📍 Actor 'Matthew Lewis' appearance timestamps: [283s, 284s, 375s, 620s, 923s, 924s, 1207s, 1276s, 1302s]
```
**What this shows:** All timestamps from the precomputed detection where the actor appears.

#### Segment Generation with Coverage
```
  Segment 1: 253.00s-313.00s (covers actor appearance at 283s)
  Segment 2: 254.00s-314.00s (covers actor appearance at 284s)
  Segment 3: 315.00s-435.00s (covers actor appearance at 375s)
  ... and N more segments
```
**What this shows:** Each segment and which actor appearance timestamp it covers.

### 2. Prompt-Based Analyzer (`src/content_analysis/prompt_based_analyzer.py`)

#### Actor Appearance Summary
```
✓ Actor 'Matthew Lewis' appears at: [283s, 284s, 375s, 620s, 923s, 924s, 1207s, 1276s, 1302s]
✅ Generated 30 segments from precomputed actor timestamps (NO fallback to candidate segments, NO recomputed scores)
```
**What this shows:** Confirmation of all actor appearances and segment generation from those timestamps only.

### 3. Content Analyzer (`src/content_analysis/content_analyzer.py`)

#### Segment Coverage Details
```
📊 Prompt-matched segment coverage (Actor: Matthew Lewis):
  [1] 253.0s-313.0s (appearance at 283s, source: precomputed_detection)
  [2] 254.0s-314.0s (appearance at 284s, source: precomputed_detection)
  [3] 315.0s-435.0s (appearance at 375s, source: precomputed_detection)
  ... and 27 more segments
```
**What this shows:** Final segments that will be used, with their coverage of actor appearances.

## How to Verify Actor Coverage

### Example: Verifying "generate only clips with Matthew Lewis"

1. **Check the appearance timestamps log:**
   ```
   📍 Actor 'Matthew Lewis' appearance timestamps: [283s, 284s, 375s, 620s, 923s, 924s, 1207s, 1276s, 1302s]
   ```
   Count: 9 unique timestamps (note: some actors appear multiple times in consecutive frames)

2. **Verify segment generation:**
   ```
   Generated 30 segments exclusively from precomputed 'Matthew Lewis' timestamps (15 unique appearance points)
   ```
   If there are 15 unique appearance points but 9 appear in the log, it means 6 additional timestamps were consolidated from different face detections (same actor, different angles/frames)

3. **Check final segment coverage:**
   ```
   [1] 253.0s-313.0s (appearance at 283s, source: precomputed_detection)
   [2] 254.0s-314.0s (appearance at 284s, source: precomputed_detection)
   ...
   ```
   Each segment shows:
   - **Start-End time:** When the clip plays
   - **Appearance timestamp:** When the actor actually appears (should be within start-end range)
   - **Source:** Should always be `precomputed_detection` for actor-only requests

## Verification Checklist

### For Each Actor Request:
- [ ] All actor appearance timestamps are logged with 📍 emoji
- [ ] Segment count matches `unique appearance points × 2` (primary + extended segments)
- [ ] Each segment's appearance timestamp is within its start-end range
- [ ] All segments have `source: precomputed_detection`
- [ ] No segments have fallback sources (like `comprehensive_candidate`)
- [ ] Actor confidence is from Rekognition (0.0-1.0 range)

### Example Verification:

**Request:** "generate only clips with Matthew Lewis"

**Expected in logs:**
```
Loaded celebrity index with X actors and 0 objects
Detected actors from prompt: ['Matthew Lewis']
🎯 STRICT ACTOR MODE: Extracting segments ONLY from precomputed '['Matthew Lewis']' timestamps
📍 Actor 'Matthew Lewis' appearance timestamps: [283s, 284s, 375s, 620s, 923s, 924s, 1207s, 1276s, 1302s]
✓ Actor 'Matthew Lewis' appears at: [283s, 284s, 375s, 620s, 923s, 924s, 1207s, 1276s, 1302s]
Generated 30 segments exclusively from precomputed 'Matthew Lewis' timestamps (15 unique appearance points)
✅ Generated 30 segments from precomputed actor timestamps (NO fallback to candidate segments, NO recomputed scores)
📊 Prompt-matched segment coverage (Actor: Matthew Lewis):
  [1] 253.0s-313.0s (appearance at 283s, source: precomputed_detection)
  ...
```

**NOT Expected:**
- ❌ "Could not load celebrity/object index" (would indicate import error)
- ❌ "Using intelligent LLM-based analysis" (should use extractor instead)
- ❌ "LLM evaluated 0 segments" (should return precomputed)
- ❌ Segments with `source: comprehensive_candidate` (shouldn't use fallback)
- ❌ Missing appearance timestamps in the logs

## What the Timestamps Tell You

### Actor Appearance Time (283s = 4:43)
- The exact second in the video where the actor's face was detected
- This is from AWS Rekognition face detection

### Segment Time Range (253.0s-313.0s = 4:13 to 5:13)
- The 60-second clip that will be generated
- Centered around the actor appearance (283s is in the middle of 253s-313s)

### Confidence Score (0.83)
- From AWS Rekognition's celebrity recognition confidence
- Not recomputed; taken directly from precomputed results
- Higher = more confident the actor is correctly identified

## Troubleshooting with Timestamps

### Problem: "Actor appears but no segments generated"
Check logs for:
- Actor timestamps logged but no segments?
- Likely cause: Timestamps outside video bounds
- Solution: Increase video end time or adjust segment duration

### Problem: "Segment generated but doesn't contain actor"
Check:
- Does appearance timestamp fall within segment's start-end range?
- If not: Segment generation logic issue
- If yes: Check actual video to confirm actor is present

### Problem: "Some appearances missing from logs"
- Same actor detected multiple times (different face_ids)
- Timestamps consolidated from multiple detections
- Total appearances shown is correct; consolidated list may be subset

## Integration with Quality Assurance

These timestamps enable:
1. **Automated verification:** Check if generated segments contain appearances
2. **Manual QA:** Review specific timestamps in actual video
3. **Metrics:** Track coverage percentage (segments covering all appearances)
4. **Debugging:** Identify which appearances are missed

## Log Output Format Reference

```
Format: [timestamp] - logger_name - LEVEL - message

Timestamp Format: 2026-01-29 07:23:33
Appearance Timestamps: 283s (= 4 minutes 43 seconds)
Segment Times: 253.0s-313.0s (start-end in seconds with decimals)
```

Timestamps are always in **seconds from video start**, allowing easy conversion:
- Minutes = seconds ÷ 60
- Hours:Minutes:Seconds = seconds formatted as HH:MM:SS

## Future Enhancements

Potential additions:
- Timeline visualization of actor appearances
- Automated validation that each segment contains the appearance
- Comparison of appearance count vs. segment count
- Coverage percentage reporting
