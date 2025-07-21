#!/usr/bin/env python3
"""
Simplified YouTube Scraper Module
Handles fetching playlist data from YouTube videos without complex dependencies.
"""

import yt_dlp
import os
import json
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class YouTubeScraperSimple:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_playlist_episodes(self, playlist_url):
        """
        Fetch all episodes from a YouTube playlist
        Returns a list of episode dictionaries with metadata
        """
        episodes = []
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'ignoreerrors': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Fetching playlist info from: {playlist_url}")
                playlist_info = ydl.extract_info(playlist_url, download=False)
                
                if 'entries' not in playlist_info:
                    logger.error("No entries found in playlist")
                    return episodes
                
                for entry in playlist_info['entries']:
                    if entry is None:
                        continue
                    
                    try:
                        episode = self._extract_episode_data(entry)
                        if episode:
                            episodes.append(episode)
                            logger.info(f"Extracted episode: {episode['title']}")
                    except Exception as e:
                        logger.warning(f"Failed to extract episode data: {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"Error fetching playlist: {str(e)}")
            raise
        
        # Sort episodes by upload date (newest first)
        episodes.sort(key=lambda x: x.get('upload_date', ''), reverse=True)
        
        logger.info(f"Successfully fetched {len(episodes)} episodes")
        return episodes
    
    def _extract_episode_data(self, entry):
        """Extract relevant data from a YouTube video entry"""
        try:
            # Basic video information
            video_id = entry.get('id', '')
            title = entry.get('title', 'Unknown Title')
            description = entry.get('description', '')
            duration = entry.get('duration', 0)
            view_count = entry.get('view_count', 0)
            upload_date = entry.get('upload_date', '')
            uploader = entry.get('uploader', '')
            thumbnail = entry.get('thumbnail', '')
            
            # Format upload date
            formatted_date = self._format_upload_date(upload_date)
            
            # Extract episode number from title if possible
            episode_number = self._extract_episode_number(title)
            
            # Build YouTube URL
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            episode = {
                'video_id': video_id,
                'title': title,
                'description': description,
                'duration_seconds': duration,
                'duration_formatted': self._format_duration(duration),
                'view_count': view_count,
                'upload_date': upload_date,
                'upload_date_formatted': formatted_date,
                'uploader': uploader,
                'thumbnail': thumbnail,
                'url': url,
                'episode_number': episode_number,
                'fetched_at': datetime.now().isoformat()
            }
            
            return episode
        
        except Exception as e:
            logger.error(f"Error extracting episode data: {str(e)}")
            return None
    
    def _format_upload_date(self, upload_date):
        """Format upload date string to readable format"""
        if not upload_date:
            return "Unknown"
        
        try:
            # upload_date is typically in YYYYMMDD format
            date_obj = datetime.strptime(upload_date, '%Y%m%d')
            return date_obj.strftime('%B %d, %Y')
        except ValueError:
            return upload_date
    
    def _format_duration(self, duration_seconds):
        """Format duration from seconds to readable format"""
        if not duration_seconds:
            return "Unknown"
        
        hours = duration_seconds // 3600
        minutes = (duration_seconds % 3600) // 60
        seconds = duration_seconds % 60
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes}:{seconds:02d}"
    
    def _extract_episode_number(self, title):
        """Try to extract episode number from title"""
        # Look for patterns like "Episode 1", "Ep 1", "#1", etc.
        patterns = [
            r'episode\s*(\d+)',
            r'ep\s*(\d+)',
            r'#(\d+)',
            r'part\s*(\d+)',
            r'(\d+)\s*-',
            r'(\d+)\s*:',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title, re.IGNORECASE)
            if match:
                return int(match.group(1))
        
        return None
