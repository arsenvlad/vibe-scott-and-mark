#!/usr/bin/env python3
"""
Test the audio processor with the real downloaded audio file
"""

import os
import sys
from audio_processor import PodcastAudioProcessor

def test_real_audio_processing():
    print("Testing Real Audio Processing")
    print("=" * 50)
    
    # Path to the downloaded audio file
    audio_path = r"C:\Code\Python\vibe-coding\scott-and-mark\data\audio_cache\UmXW8nGG9ZE.webm"
    
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        return
    
    print(f"Processing audio file: {audio_path}")
    
    # Initialize the processor
    processor = PodcastAudioProcessor()
    
    try:
        # Process the audio - use correct method name
        print("\n🎙️ Starting audio analysis...")
        results = processor.process_episode("https://www.youtube.com/watch?v=UmXW8nGG9ZE", "UmXW8nGG9ZE")
        
        print(f"\n✅ Analysis complete!")
        print(f"Episode Results:")
        print(f"  Total Duration: {results.get('duration_seconds', 'Unknown')} seconds")
        print(f"  Total Words: {results.get('total_words', 'Unknown')}")
        print(f"  Scott's Words: {results.get('scott_words', 'Unknown')}")
        print(f"  Mark's Words: {results.get('mark_words', 'Unknown')}")
        
        # Show transcript sample
        transcript = results.get('transcript', '')
        if transcript:
            print(f"\n📝 Transcript Sample (first 500 chars):")
            print(f"{transcript[:500]}...")
            
        # Show speaker breakdown
        speaker_segments = results.get('speaker_segments', [])
        if speaker_segments:
            print(f"\n👥 Speaker Breakdown (first 5 segments):")
            for i, segment in enumerate(speaker_segments[:5]):
                print(f"  {i+1}. {segment}")
                
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_audio_processing()
