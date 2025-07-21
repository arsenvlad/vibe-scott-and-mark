#!/usr/bin/env python3
"""
Test the real YouTube downloader with the actual podcast episode
"""

import os
import sys
from youtube_downloader import YouTubeAudioDownloader

def test_real_download():
    print("Testing Real YouTube Audio Download")
    print("=" * 50)
    
    # Initialize the downloader
    downloader = YouTubeAudioDownloader()
    
    # Test URL - the actual podcast episode
    test_url = "https://www.youtube.com/watch?v=UmXW8nGG9ZE"
    video_id = "UmXW8nGG9ZE"  # Extract video ID from URL
    
    print(f"Downloading from: {test_url}")
    print(f"Video ID: {video_id}")
    print(f"Cache directory: {downloader.audio_cache_dir}")
    
    try:
        # Download the audio
        audio_path = downloader.download_audio(test_url, video_id)
        
        if audio_path:
            print(f"\n✅ SUCCESS!")
            print(f"Audio downloaded to: {audio_path}")
            
            # Check file size
            if os.path.exists(audio_path):
                file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MB
                print(f"File size: {file_size:.2f} MB")
            
            # Check cache
            print(f"\nCache metadata:")
            for video_id, data in downloader.cache_metadata.items():
                print(f"  {video_id}: {data}")
                
        else:
            print("❌ Download failed!")
            
    except Exception as e:
        print(f"❌ Error during download: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_download()
