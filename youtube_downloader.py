#!/usr/bin/env python3
"""
YouTube Audio Downloader - Standalone iteration for fixing download issues
Saves audio files to data folder to avoid re-downloading.
"""

import os
import sys
import logging
import json
from datetime import datetime
import yt_dlp
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YouTubeAudioDownloader:
    def __init__(self, data_dir=None):
        """Initialize the YouTube audio downloader."""
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), 'data')
        self.audio_cache_dir = os.path.join(self.data_dir, 'audio_cache')
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.audio_cache_dir, exist_ok=True)
        
        # Cache metadata file
        self.cache_metadata_file = os.path.join(self.audio_cache_dir, 'download_cache.json')
        self.cache_metadata = self.load_cache_metadata()
        
        logger.info(f"Audio cache directory: {self.audio_cache_dir}")
    
    def load_cache_metadata(self):
        """Load cache metadata to track what's been downloaded."""
        if os.path.exists(self.cache_metadata_file):
            try:
                with open(self.cache_metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache metadata: {e}")
                return {}
        return {}
    
    def save_cache_metadata(self):
        """Save cache metadata."""
        try:
            with open(self.cache_metadata_file, 'w') as f:
                json.dump(self.cache_metadata, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save cache metadata: {e}")
    
    def get_cached_audio_path(self, video_id):
        """Check if audio is already cached."""
        if video_id in self.cache_metadata:
            cached_path = self.cache_metadata[video_id].get('audio_path')
            if cached_path and os.path.exists(cached_path):
                logger.info(f"Found cached audio for {video_id}: {cached_path}")
                return cached_path
        return None
    
    def download_audio_v1(self, video_url, video_id):
        """Download method v1: Use specific mp4 audio format that works (233-x series)."""
        logger.info(f"Trying download method v1 for {video_id}")
        
        ydl_opts = {
            'format': '233-0/233-1/233-2/233-3/233-4/233-5/233-6/233-7/233-8/233-9/bestaudio/best',
            'outtmpl': os.path.join(self.audio_cache_dir, f"{video_id}.%(ext)s"),
            'writeinfojson': False,
            'writedescription': False,
            'writesubtitles': False,
            'quiet': False,  # Enable output for debugging
            'no_warnings': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            # Find the downloaded file
            for ext in ['mp4', 'webm', 'm4a', 'mp3', 'wav']:
                potential_file = os.path.join(self.audio_cache_dir, f"{video_id}.{ext}")
                if os.path.exists(potential_file):
                    logger.info(f"Downloaded {video_id} as {potential_file}")
                    return potential_file
            
            raise Exception("Audio file not created")
                
        except Exception as e:
            logger.warning(f"Download method v1 failed: {e}")
            return None
    
    def download_audio_v2(self, video_url, video_id):
        """Download method v2: With user agent and cookies."""
        logger.info(f"Trying download method v2 for {video_id}")
        
        output_path = os.path.join(self.audio_cache_dir, f"{video_id}.wav")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.audio_cache_dir, f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,  # Enable output for debugging
            'no_warnings': False,
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            },
            'cookiefile': None,
            'age_limit': None,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path):
                return output_path
            else:
                raise Exception("Audio file not created")
                
        except Exception as e:
            logger.warning(f"Download method v2 failed: {e}")
            return None
    
    def download_audio_v3(self, video_url, video_id):
        """Download method v3: With android client and no dash."""
        logger.info(f"Trying download method v3 for {video_id}")
        
        output_path = os.path.join(self.audio_cache_dir, f"{video_id}.wav")
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': os.path.join(self.audio_cache_dir, f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,
            'no_warnings': False,
            'extractor_args': {
                'youtube': {
                    'skip': ['dash', 'hls'],
                    'player_client': ['android']
                }
            },
            'http_headers': {
                'User-Agent': 'com.google.android.youtube/17.31.35 (Linux; U; Android 11) gzip'
            },
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path):
                return output_path
            else:
                raise Exception("Audio file not created")
                
        except Exception as e:
            logger.warning(f"Download method v3 failed: {e}")
            return None
    
    def download_audio_v4(self, video_url, video_id):
        """Download method v4: With ios client."""
        logger.info(f"Trying download method v4 for {video_id}")
        
        output_path = os.path.join(self.audio_cache_dir, f"{video_id}.wav")
        
        ydl_opts = {
            'format': 'bestaudio',
            'outtmpl': os.path.join(self.audio_cache_dir, f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,
            'extractor_args': {
                'youtube': {
                    'player_client': ['ios']
                }
            },
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path):
                return output_path
            else:
                raise Exception("Audio file not created")
                
        except Exception as e:
            logger.warning(f"Download method v4 failed: {e}")
            return None
    
    def download_audio_v6(self, video_url, video_id):
        """Download method v6: Use web client with cookies."""
        logger.info(f"Trying download method v6 for {video_id}")
        
        output_path = os.path.join(self.audio_cache_dir, f"{video_id}.wav")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.audio_cache_dir, f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,
            'extractor_args': {
                'youtube': {
                    'player_client': ['web'],
                    'skip': ['dash'],
                }
            },
            'http_headers': {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            },
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path):
                return output_path
            else:
                raise Exception("Audio file not created")
                
        except Exception as e:
            logger.warning(f"Download method v6 failed: {e}")
            return None
    
    def download_audio_v7(self, video_url, video_id):
        """Download method v7: Try with no extractor args."""
        logger.info(f"Trying download method v7 for {video_id}")
        
        output_path = os.path.join(self.audio_cache_dir, f"{video_id}.wav")
        
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio',
            'outtmpl': os.path.join(self.audio_cache_dir, f"{video_id}.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,
            'no_check_certificate': True,
            'ignoreerrors': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([video_url])
            
            if os.path.exists(output_path):
                return output_path
            else:
                raise Exception("Audio file not created")
                
        except Exception as e:
            logger.warning(f"Download method v7 failed: {e}")
            return None
    
    def download_audio_v8(self, video_url, video_id):
        """Download method v8: Try different video altogether to test if it's video-specific."""
        logger.info(f"Trying download method v8 for {video_id} - testing with different video")
        
        # Try a test video that might not be restricted
        test_video_id = "dQw4w9WgXcQ"  # Rick Roll - often used for testing
        test_url = f"https://www.youtube.com/watch?v={test_video_id}"
        
        output_path = os.path.join(self.audio_cache_dir, f"{test_video_id}_test.wav")
        
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.audio_cache_dir, f"{test_video_id}_test.%(ext)s"),
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'quiet': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([test_url])
            
            if os.path.exists(output_path):
                logger.info(f"Test download successful - YouTube access works!")
                # Clean up test file
                os.remove(output_path)
                return "TEST_SUCCESS"
            else:
                raise Exception("Test audio file not created")
                
        except Exception as e:
            logger.warning(f"Download method v8 (test) failed: {e}")
            return None
    
    def download_audio(self, video_url, video_id):
        """
        Download audio using multiple fallback methods.
        Returns path to downloaded audio file or None if all methods fail.
        """
        # Check cache first
        cached_path = self.get_cached_audio_path(video_id)
        if cached_path:
            return cached_path
        
        logger.info(f"Downloading audio for video {video_id}")
        logger.info(f"URL: {video_url}")
        
        # Try different download methods
        download_methods = [
            self.download_audio_v1,
            self.download_audio_v2,
            self.download_audio_v3,
            self.download_audio_v4,
            self.download_audio_v6,
            self.download_audio_v7,
            self.download_audio_v8,
        ]
        
        for i, method in enumerate(download_methods, 1):
            try:
                logger.info(f"Attempting download method {i}/{len(download_methods)}")
                audio_path = method(video_url, video_id)
                
                if audio_path and os.path.exists(audio_path):
                    # Cache successful download
                    self.cache_metadata[video_id] = {
                        'audio_path': audio_path,
                        'downloaded_at': datetime.now().isoformat(),
                        'method_used': f"v{i}",
                        'file_size': os.path.getsize(audio_path)
                    }
                    self.save_cache_metadata()
                    
                    logger.info(f"✅ Successfully downloaded audio using method v{i}: {audio_path}")
                    return audio_path
                    
            except Exception as e:
                logger.warning(f"Method v{i} failed: {e}")
                continue
        
        logger.error(f"❌ All download methods failed for video {video_id}")
        return None
    
    def get_video_info(self, video_url):
        """Get video information without downloading."""
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'id': info.get('id', ''),
                }
        except Exception as e:
            logger.error(f"Could not get video info: {e}")
            return None
    
    def list_cached_files(self):
        """List all cached audio files."""
        logger.info("Cached audio files:")
        for video_id, metadata in self.cache_metadata.items():
            file_path = metadata['audio_path']
            if os.path.exists(file_path):
                size_mb = metadata['file_size'] / (1024 * 1024)
                logger.info(f"  {video_id}: {file_path} ({size_mb:.1f} MB)")
            else:
                logger.warning(f"  {video_id}: File missing - {file_path}")
        
        return self.cache_metadata

def test_download():
    """Test the YouTube downloader with Episode 20."""
    downloader = YouTubeAudioDownloader()
    
    # Test video - Episode 20
    video_id = "UmXW8nGG9ZE"
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    
    print("="*60)
    print("TESTING YOUTUBE AUDIO DOWNLOADER")
    print("="*60)
    
    # Test getting video info first
    print("\n1. Getting video information...")
    info = downloader.get_video_info(video_url)
    if info:
        print(f"Title: {info['title']}")
        print(f"Duration: {info['duration']} seconds")
        print(f"Uploader: {info['uploader']}")
    else:
        print("❌ Could not get video info")
        return False
    
    # Test downloading
    print(f"\n2. Downloading audio for {video_id}...")
    audio_path = downloader.download_audio(video_url, video_id)
    
    if audio_path:
        print(f"✅ Download successful!")
        print(f"Audio file: {audio_path}")
        print(f"File size: {os.path.getsize(audio_path) / (1024*1024):.1f} MB")
        
        # Test cache
        print(f"\n3. Testing cache...")
        cached_path = downloader.get_cached_audio_path(video_id)
        if cached_path == audio_path:
            print(f"✅ Cache working correctly!")
        else:
            print(f"⚠️ Cache issue: {cached_path} != {audio_path}")
        
        return True
    else:
        print(f"❌ Download failed!")
        return False

if __name__ == "__main__":
    success = test_download()
    
    print("\n" + "="*60)
    if success:
        print("🎉 YouTube downloader is working!")
    else:
        print("💥 YouTube downloader failed!")
    print("="*60)
    
    sys.exit(0 if success else 1)
