# Kinetic Captions Demo Files

This directory contains demo ASS files showcasing different kinetic caption effects.

## Files:
- `demo_karaoke.ass` - Karaoke effect (words change color as spoken)
- `demo_typewriter.ass` - Typewriter effect (words appear progressively)
- `demo_highlight.ass` - Highlight effect (current word is emphasized)

## Usage with FFmpeg:
```bash
# Apply kinetic captions to a video
ffmpeg -i input_video.mp4 -vf "ass=demo_karaoke.ass" -c:a copy output_with_captions.mp4
```

## Kinetic Caption Modes:

### Karaoke Mode
Words change color as they are spoken, creating a karaoke-style effect.

### Typewriter Mode  
Words appear progressively, creating a typewriter effect.

### Highlight Mode
The currently spoken word is highlighted in a different color.

## Technical Details:
- Uses ASS (Advanced SubStation Alpha) format
- Word-level timing from Whisper transcription
- FFmpeg compatible syntax
- Automatic fallback to standard captions if word timing unavailable
