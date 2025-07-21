#!/usr/bin/env python3
"""
Focused download test with latest yt-dlp
"""

import yt_dlp
import os
import tempfile

def test_download_with_format_selection():
    """Test download with specific format selection."""
    
    # Test with Scott and Mark episode
    video_url = "https://www.youtube.com/watch?v=UmXW8nGG9ZE"
    video_id = "UmXW8nGG9ZE"
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, f"{video_id}.%(ext)s")
    
    print(f"Testing download to: {temp_dir}")
    
    # First, let's see what formats are available
    print("\n1. Checking available formats...")
    try:
        ydl_opts = {'quiet': True, 'listformats': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            
            formats = info.get('formats', [])
            audio_formats = [f for f in formats if f.get('acodec') != 'none' and f.get('vcodec') == 'none']
            
            print(f"Found {len(audio_formats)} audio-only formats:")
            for i, fmt in enumerate(audio_formats[:5]):  # Show first 5
                size_info = f"~{fmt.get('filesize', 'unknown')} bytes" if fmt.get('filesize') else "unknown size"
                print(f"  {i+1}. {fmt.get('format_id')}: {fmt.get('ext')} {fmt.get('abr', 'unknown')}kbps ({size_info})")
            
            if audio_formats:
                # Try downloading the first available audio format
                best_audio_format = audio_formats[0]['format_id']
                print(f"\n2. Attempting download with format {best_audio_format}...")
                
                download_opts = {
                    'format': best_audio_format,
                    'outtmpl': output_path,
                    'quiet': False,
                }
                
                with yt_dlp.YoutubeDL(download_opts) as ydl:
                    ydl.download([video_url])
                
                # Check if file was created
                downloaded_files = os.listdir(temp_dir)
                if downloaded_files:
                    downloaded_file = os.path.join(temp_dir, downloaded_files[0])
                    file_size = os.path.getsize(downloaded_file)
                    print(f"✅ Download successful!")
                    print(f"File: {downloaded_file}")
                    print(f"Size: {file_size / (1024*1024):.1f} MB")
                    
                    # Clean up
                    os.remove(downloaded_file)
                    os.rmdir(temp_dir)
                    return True
                else:
                    print("❌ No file was downloaded")
                    return False
            else:
                print("❌ No audio formats found")
                return False
                
    except Exception as e:
        print(f"❌ Download test failed: {e}")
        # Clean up temp dir
        try:
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        except:
            pass
        return False

def test_simple_download():
    """Test with simpler approach - no post-processing."""
    print("\n3. Testing simple download (no conversion)...")
    
    video_url = "https://www.youtube.com/watch?v=UmXW8nGG9ZE"
    video_id = "UmXW8nGG9ZE"
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        ydl_opts = {
            'format': 'bestaudio',
            'outtmpl': os.path.join(temp_dir, f"{video_id}.%(ext)s"),
            'quiet': False,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])
        
        # Check if file was created
        downloaded_files = os.listdir(temp_dir)
        if downloaded_files:
            downloaded_file = os.path.join(temp_dir, downloaded_files[0])
            file_size = os.path.getsize(downloaded_file)
            print(f"✅ Simple download successful!")
            print(f"File: {downloaded_file}")
            print(f"Size: {file_size / (1024*1024):.1f} MB")
            
            # Clean up
            os.remove(downloaded_file)
            os.rmdir(temp_dir)
            return True
        else:
            print("❌ No file was downloaded")
            return False
            
    except Exception as e:
        print(f"❌ Simple download failed: {e}")
        # Clean up
        try:
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
        except:
            pass
        return False

if __name__ == "__main__":
    print("Testing YouTube Download Functionality")
    print("=" * 50)
    
    test1 = test_download_with_format_selection()
    
    if not test1:
        test2 = test_simple_download()
    else:
        test2 = True
    
    print("\n" + "=" * 50)
    if test1 or test2:
        print("🎉 YouTube download is working!")
    else:
        print("💥 YouTube download is blocked")
