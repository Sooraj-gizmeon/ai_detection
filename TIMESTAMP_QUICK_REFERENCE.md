# Quick Reference: Actor Timestamp Verification

## What You'll See in Logs

### 1️⃣ Actor Detection
```
Detected actors from prompt: ['Matthew Lewis']
```

### 2️⃣ Appearance Timestamps (MAIN REFERENCE)
```
📍 Actor 'Matthew Lewis' appearance timestamps: [283s, 284s, 375s, 620s, 923s, 924s, 1207s, 1276s, 1302s]
```
✅ **Use this to verify:** How many times the actor was detected in the video

### 3️⃣ Segment Generation
```
Extracting segments for actor 'Matthew Lewis' with 15 total precomputed appearances across 1 detections (max confidence: 0.83)
Generated 30 segments exclusively from precomputed 'Matthew Lewis' timestamps (15 unique appearance points)
```
✅ **Use this to verify:** Math should work out (appearances × 2 = segments)

### 4️⃣ Coverage Verification
```
Segment 1: 253.00s-313.00s (covers actor appearance at 283s)
Segment 2: 254.00s-314.00s (covers actor appearance at 284s)
```
✅ **Use this to verify:** Each appearance is covered by a segment

### 5️⃣ Final Summary
```
📊 Prompt-matched segment coverage (Actor: Matthew Lewis):
  [1] 253.0s-313.0s (appearance at 283s, source: precomputed_detection)
  [2] 254.0s-314.0s (appearance at 284s, source: precomputed_detection)
```
✅ **Use this to verify:** Final segments all use precomputed_detection source

## Timestamps Explained

| Format | Example | Meaning |
|--------|---------|---------|
| Seconds | `283s` | 4:43 in the video |
| As time | `283s` = `4m 43s` | 4 minutes, 43 seconds |
| Segment | `253.0s-313.0s` | From 4:13 to 5:13 |

## Quick Verification Checklist

For request: `"generate only clips with [ACTOR_NAME]"`

- [ ] Appearance timestamps logged (with 📍 emoji)
- [ ] Each segment shows appearance coverage
- [ ] Appearance timestamps fall within segment ranges
- [ ] All sources are `precomputed_detection`
- [ ] No errors about missing actor
- [ ] Generated count ≈ appearances × 2

## Red Flags to Watch For

❌ `Could not load celebrity/object index` → Import error  
❌ `Using intelligent LLM-based analysis` → Should use extractor  
❌ `LLM evaluated 0 segments` → LLM fallback instead of precomputed  
❌ `source: comprehensive_candidate` → Using fallback instead of precomputed  
❌ No appearance timestamps logged → Actor detection failed  

## How to Calculate Expected Results

```
Formula: Expected Segments = Unique Appearances × 2

Example:
- Matthew Lewis appears at: [283s, 284s, 375s, 620s, 923s, 924s, 1207s, 1276s, 1302s]
- But system says "15 unique appearance points"
- This means: 15 × 2 = 30 expected segments

Why the difference?
- Same actor detected with multiple face_ids
- Same frames detected multiple times
- Consolidated to 15 unique points
- Which generates 30 segments (15 primary + 15 extended)
```

## Segment Time Conversion

```
Convert seconds to MM:SS format:
- 283 seconds = 283 ÷ 60 = 4 minutes, 43 seconds = 04:43
- 620 seconds = 620 ÷ 60 = 10 minutes, 20 seconds = 10:20
- 1207 seconds = 1207 ÷ 60 = 20 minutes, 7 seconds = 20:07
```

## Example Log Analysis

**Request:** "generate only clips with Matthew Lewis"

**What to expect:**

```
✓ Appearance timestamps: [283s, 284s, 375s, 620s, 923s, 924s, 1207s, 1276s, 1302s]
  = 9 occurrences

✓ System consolidates to: 15 unique appearance points
  (some timestamps counted multiple times from different detections)

✓ Generates: 30 segments
  = 15 unique points × 2 (primary + extended)

✓ Each segment includes appearance timestamp verification:
  Segment 253.0s-313.0s (covers actor appearance at 283s) ✓
  Segment 254.0s-314.0s (covers actor appearance at 284s) ✓
```

## When Something's Wrong

**If you see:**
```
📍 Actor 'Matthew Lewis' appearance timestamps: [283s, 284s, ...]
But then:
Found 0 prompt-matched segments
```

**Likely causes:**
1. Segments generated but filtered out (too short/long)
2. Timestamps outside video bounds
3. Segment duration settings too restrictive

**Check:**
- `min_duration` and `max_duration` parameters
- Video total length (is it longer than appearance times?)
- Segment generation logic (any filters applied?)

## Contact Points for Timestamps

The following modules now log timestamps:

1. **actor_segment_extractor.py** → Generates segments, logs coverage
2. **prompt_based_analyzer.py** → Verifies appearances, returns list
3. **content_analyzer.py** → Logs final segment coverage

All use consistent timestamp format: **seconds from video start**

## Tools to Verify

Once you have timestamps from logs:

1. **Manual verification:** Open video, go to timestamp, watch
2. **Automated check:** Script to verify segment contains appearance time
3. **QA dashboard:** Could display appearance timeline and segment coverage
4. **Metrics:** Calculate coverage percentage (appearances covered / total)

---

**Key Takeaway:** The appearance timestamps (with 📍) are your ground truth. Every segment should be centered on one of these timestamps to guarantee the actor is in the clip.
