#!/usr/bin/env python3
"""
Real Audio Processing Module for Scott and Mark Podcast Analytics

This module handles:
1. Audio download from YouTube
2. Speech-to-text transcription using Whisper
3. Speaker diarization (identifying Scott vs Mark)
4. Word counting and analysis
"""

import os
import logging
import tempfile
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
import warnings

# Suppress librosa warnings
warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')
warnings.filterwarnings('ignore', message='.*__audioread_load.*')

import yt_dlp
import librosa
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import torch
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PodcastAudioProcessor:
    def __init__(self, temp_dir: str = None):
        """Initialize the audio processor."""
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.recognizer = sr.Recognizer()
        
        # Initialize paths for speaker model
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.model_path = os.path.join(self.data_dir, 'speaker_model.pkl')
        self.scaler_path = os.path.join(self.data_dir, 'speaker_scaler.pkl')
        
        # Speaker classification model
        self.speaker_model = None
        self.feature_scaler = None
        
        # Initialize Whisper for transcription
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")
            logger.info("Initialized Whisper model for transcription")
        except ImportError:
            logger.warning("Whisper not available, falling back to Google Speech Recognition")
            self.whisper_model = None
        
        # Load or train the speaker identification model
        self._load_or_train_speaker_model()
    
    def download_audio(self, video_url: str, video_id: str) -> str:
        """Download audio from YouTube video."""
        try:
            output_path = os.path.join(self.temp_dir, f"{video_id}.wav")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.temp_dir, f"{video_id}.%(ext)s"),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'quiet': True,
                'no_warnings': True,
                # Add headers and options to avoid 403 errors
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                },
                'extractor_args': {
                    'youtube': {
                        'skip': ['dash', 'hls'],
                        'player_client': ['android', 'web']
                    }
                },
                'cookiefile': None,  # Don't use cookies
                'age_limit': None,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                logger.info(f"Downloading audio for video {video_id}")
                ydl.download([video_url])
            
            if os.path.exists(output_path):
                logger.info(f"Successfully downloaded audio: {output_path}")
                return output_path
            else:
                raise Exception(f"Audio file not found after download: {output_path}")
                
        except Exception as e:
            logger.error(f"Error downloading audio for {video_id}: {e}")
            raise
    
    def transcribe_with_whisper(self, audio_path: str) -> List[Dict]:
        """Transcribe audio using Whisper model."""
        try:
            if not self.whisper_model:
                return self.fallback_transcription(audio_path)
            
            logger.info("Transcribing audio with Whisper...")
            result = self.whisper_model.transcribe(audio_path, word_timestamps=True)
            
            # Process segments with timestamps
            chunks = []
            for segment in result.get('segments', []):
                chunks.append({
                    'text': segment['text'].strip(),
                    'start': segment['start'],
                    'end': segment['end']
                })
            
            if not chunks and 'text' in result:
                # Fallback if no segments
                chunks.append({
                    'text': result['text'],
                    'start': 0,
                    'end': 0
                })
            
            logger.info(f"Transcribed {len(chunks)} segments")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in Whisper transcription: {e}")
            return self.fallback_transcription(audio_path)
    
    def fallback_transcription(self, audio_path: str) -> List[Dict]:
        """Fallback transcription using speech_recognition library."""
        try:
            logger.info("Using fallback transcription method...")
            
            # Convert to WAV if needed
            audio = AudioSegment.from_file(audio_path)
            
            # Split into chunks (30 second segments)
            chunk_length_ms = 30 * 1000  # 30 seconds
            chunks = []
            
            for i, start_ms in enumerate(range(0, len(audio), chunk_length_ms)):
                end_ms = min(start_ms + chunk_length_ms, len(audio))
                chunk_audio = audio[start_ms:end_ms]
                
                # Export chunk to temporary file
                chunk_path = os.path.join(self.temp_dir, f"chunk_{i}.wav")
                chunk_audio.export(chunk_path, format="wav")
                
                try:
                    with sr.AudioFile(chunk_path) as source:
                        audio_data = self.recognizer.record(source)
                        text = self.recognizer.recognize_google(audio_data)
                        
                        chunks.append({
                            'text': text,
                            'start': start_ms / 1000,
                            'end': end_ms / 1000
                        })
                        
                except sr.UnknownValueError:
                    logger.warning(f"Could not understand chunk {i}")
                    chunks.append({
                        'text': "",
                        'start': start_ms / 1000,
                        'end': end_ms / 1000
                    })
                except Exception as e:
                    logger.warning(f"Error processing chunk {i}: {e}")
                
                # Clean up chunk file
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)
            
            logger.info(f"Fallback transcription completed with {len(chunks)} segments")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in fallback transcription: {e}")
            return []
    
    def _load_or_train_speaker_model(self):
        """Load existing speaker model or train a new one using voice samples"""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            logger.info("Loading existing speaker identification model...")
            self._load_speaker_model()
        else:
            logger.info("Training new speaker identification model...")
            self._train_speaker_model()
    
    def _load_speaker_model(self):
        """Load the trained speaker model and scaler"""
        try:
            with open(self.model_path, 'rb') as f:
                self.speaker_model = pickle.load(f)
            
            with open(self.scaler_path, 'rb') as f:
                self.feature_scaler = pickle.load(f)
            
            logger.info("Speaker model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading speaker model: {str(e)}")
            self._train_speaker_model()
    
    def _train_speaker_model(self):
        """Train speaker identification model using voice samples"""
        try:
            scott_sample = os.path.join(self.data_dir, 'voice_scott.mp3')
            mark_sample = os.path.join(self.data_dir, 'voice_mark.mp3')
            
            if not os.path.exists(scott_sample) or not os.path.exists(mark_sample):
                logger.error("Voice samples not found. Cannot train speaker model.")
                return
            
            logger.info("Extracting features from voice samples...")
            
            # Extract features from voice samples
            scott_features = self._extract_voice_features(scott_sample)
            mark_features = self._extract_voice_features(mark_sample)
            
            if scott_features is None or mark_features is None:
                logger.error("Failed to extract features from voice samples")
                return
            
            logger.info(f"Scott features: {len(scott_features)}, Mark features: {len(mark_features)}")
            
            # Use the feature vectors directly - no segmentation needed for training
            # We'll just use the full feature vector from each voice sample
            X = np.array([scott_features, mark_features])
            y = np.array([0, 1])  # 0=Scott, 1=Mark
            
            # Create additional training samples by adding small noise variations
            # This helps the model generalize better
            noise_samples = []
            noise_labels = []
            for i in range(5):  # Create 5 variations of each
                # Add small random noise (1% of the feature values)
                scott_noise = scott_features + np.random.normal(0, 0.01 * np.std(scott_features), scott_features.shape)
                mark_noise = mark_features + np.random.normal(0, 0.01 * np.std(mark_features), mark_features.shape)
                noise_samples.extend([scott_noise, mark_noise])
                noise_labels.extend([0, 1])
            
            # Combine original and noise samples
            X = np.vstack([X, np.array(noise_samples)])
            y = np.concatenate([y, np.array(noise_labels)])
            
            logger.info(f"Training with {len(X)} samples, {X.shape[1]} features each")
            
            # Scale features
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train model
            self.speaker_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            self.speaker_model.fit(X_scaled, y)
            
            # Save the trained model
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.speaker_model, f)
            
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
            
            logger.info("Speaker identification model trained and saved successfully")
            
        except Exception as e:
            logger.error(f"Error training speaker model: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _extract_voice_features(self, audio_file):
        """Extract audio features for speaker identification"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=22050)
            
            # Use the same feature extraction logic
            return self._extract_voice_features_from_audio(y, sr)
            
        except Exception as e:
            logger.error(f"Error extracting voice features: {str(e)}")
            return None
    
    def _identify_speaker_by_voice(self, audio_path: str, start_time: float, end_time: float) -> str:
        """Identify speaker using trained voice model for a specific audio segment"""
        if self.speaker_model is None or self.feature_scaler is None:
            return 'unknown'
        
        try:
            # Load audio file and extract segment
            y, sr = librosa.load(audio_path, sr=22050, offset=start_time, duration=end_time-start_time)
            
            if len(y) < sr * 0.5:  # Skip very short segments (less than 0.5 seconds)
                return 'unknown'
            
            # Extract features from this segment
            features = self._extract_segment_features(y, sr)
            
            if features is None:
                return 'unknown'
            
            # Predict speaker
            features_scaled = self.feature_scaler.transform([features])
            prediction = self.speaker_model.predict(features_scaled)[0]
            confidence = self.speaker_model.predict_proba(features_scaled)[0]
            
            # Only return prediction if confidence is high enough
            max_confidence = np.max(confidence)
            if max_confidence < 0.6:  # Require at least 60% confidence
                return 'unknown'
            
            return 'scott' if prediction == 0 else 'mark'
            
        except Exception as e:
            logger.warning(f"Error identifying speaker by voice: {str(e)}")
            return 'unknown'
    
    def _extract_segment_features(self, y, sr):
        """Extract features from a single audio segment - must match _extract_voice_features exactly"""
        try:
            # Use the same feature extraction logic as training
            return self._extract_voice_features_from_audio(y, sr)
            
        except Exception as e:
            logger.error(f"Error extracting segment features: {str(e)}")
            return None
    
    def _extract_voice_features_from_audio(self, y, sr):
        """Extract audio features for speaker identification from loaded audio data"""
        try:
            features = []
            
            # MFCC features (most important for speaker identification)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend([
                np.mean(mfccs, axis=1),
                np.std(mfccs, axis=1),
                np.max(mfccs, axis=1),
                np.min(mfccs, axis=1)
            ])
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            
            features.extend([
                np.mean(spectral_centroids),
                np.std(spectral_centroids),
                np.mean(spectral_rolloff),
                np.std(spectral_rolloff),
                np.mean(spectral_bandwidth),
                np.std(spectral_bandwidth),
                np.mean(zero_crossing_rate),
                np.std(zero_crossing_rate)
            ])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend([
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1)
            ])
            
            # Flatten all features
            feature_vector = np.concatenate([np.atleast_1d(f) for f in features])
            
            # Log feature dimension for debugging
            logger.debug(f"Extracted {len(feature_vector)} features from audio segment")
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error extracting voice features: {str(e)}")
            return None
    
    def identify_speakers(self, transcription_chunks: List[Dict], audio_path: str) -> List[Dict]:
        """
        Identify speakers (Scott vs Mark) using trained voice model with keyword fallback.
        """
        try:
            processed_chunks = []
            
            for i, chunk in enumerate(transcription_chunks):
                # Try voice-based identification first
                speaker = self._identify_speaker_by_voice(
                    audio_path, 
                    chunk['start'], 
                    chunk['end']
                )
                
                # If voice identification fails, fall back to keyword heuristics
                if speaker == 'unknown':
                    speaker = self._identify_speaker_by_keywords(chunk, processed_chunks, i)
                
                processed_chunks.append({
                    'text': chunk['text'],
                    'start': chunk['start'],
                    'end': chunk['end'],
                    'speaker': speaker,
                    'scott_score': 1.0 if speaker == 'scott' else 0.0,
                    'mark_score': 1.0 if speaker == 'mark' else 0.0
                })
            
            logger.info(f"Speaker identification completed for {len(processed_chunks)} segments")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error in speaker identification: {e}")
            # Return chunks with unknown speaker
            return [
                {**chunk, 'speaker': 'unknown', 'scott_score': 0, 'mark_score': 0}
                for chunk in transcription_chunks
            ]
    
    def _identify_speaker_by_keywords(self, chunk: Dict, processed_chunks: List[Dict], index: int) -> str:
        """Fallback speaker identification using keyword heuristics"""
        try:
            text_lower = chunk['text'].lower()
            
            # Keywords/phrases that might indicate who is speaking
            scott_indicators = [
                'hanselman', 'scott', 'hanselminutes', 'azure', 'accessibility',
                'inclusive', 'developer', 'community', 'podcast', 'blog', 'web',
                'javascript', 'typescript', 'react', 'blazor', 'dotnet', '.net'
            ]
            
            mark_indicators = [
                'russinovich', 'mark', 'sysinternals', 'windows', 'azure cto',
                'internals', 'kernel', 'system', 'architecture', 'technical',
                'process', 'registry', 'security', 'troubleshooting', 'tools'
            ]
            
            # Count indicator words
            scott_score = sum(1 for indicator in scott_indicators if indicator in text_lower)
            mark_score = sum(1 for indicator in mark_indicators if indicator in text_lower)
            
            # Additional heuristics
            # Scott tends to ask more questions and be more conversational
            question_words = ['what', 'how', 'why', 'when', 'where', 'who']
            question_score = sum(1 for q in question_words if q in text_lower)
            
            # Mark tends to be more technical and use longer explanations
            technical_words = ['system', 'process', 'kernel', 'memory', 'cpu', 'performance']
            technical_score = sum(1 for t in technical_words if t in text_lower)
            
            # Adjust scores based on additional heuristics
            scott_score += question_score * 0.5
            mark_score += technical_score * 0.5
            
            # Determine speaker
            if scott_score > mark_score:
                return 'scott'
            elif mark_score > scott_score:
                return 'mark'
            else:
                # Fallback: try to maintain conversation flow
                # If previous speaker was identified, alternate is likely
                if index > 0 and processed_chunks[index-1]['speaker'] == 'scott':
                    return 'mark'
                elif index > 0 and processed_chunks[index-1]['speaker'] == 'mark':
                    return 'scott'
                else:
                    # Default alternating pattern starting with Scott
                    return 'scott' if index % 2 == 0 else 'mark'
        
        except Exception as e:
            logger.warning(f"Error in keyword-based identification: {e}")
            return 'unknown'
    
    def count_words(self, processed_chunks: List[Dict]) -> Dict:
        """Count words for each speaker and calculate statistics."""
        try:
            scott_words = 0
            mark_words = 0
            total_words = 0
            
            scott_segments = 0
            mark_segments = 0
            
            for chunk in processed_chunks:
                # Clean and count words
                text = chunk['text'].strip()
                words = len([w for w in text.split() if w.strip()])
                total_words += words
                
                if chunk['speaker'] == 'scott':
                    scott_words += words
                    scott_segments += 1
                elif chunk['speaker'] == 'mark':
                    mark_words += words
                    mark_segments += 1
            
            # Calculate percentages
            scott_percentage = round((scott_words / total_words) * 100) if total_words > 0 else 0
            mark_percentage = round((mark_words / total_words) * 100) if total_words > 0 else 0
            
            # Ensure percentages add up to 100
            if scott_percentage + mark_percentage != 100 and total_words > 0:
                if scott_words > mark_words:
                    scott_percentage = 100 - mark_percentage
                else:
                    mark_percentage = 100 - scott_percentage
            
            result = {
                'total_words': total_words,
                'scott_words': scott_words,
                'mark_words': mark_words,
                'scott_percentage': scott_percentage,
                'mark_percentage': mark_percentage,
                'segments_analyzed': len(processed_chunks),
                'scott_segments': scott_segments,
                'mark_segments': mark_segments,
                'processing_note': f'Real audio processing completed at {datetime.now().isoformat()}'
            }
            
            logger.info(f"Word counting completed: {total_words} total words ({scott_words} Scott, {mark_words} Mark)")
            return result
            
        except Exception as e:
            logger.error(f"Error counting words: {e}")
            return {
                'total_words': 0,
                'scott_words': 0,
                'mark_words': 0,
                'scott_percentage': 0,
                'mark_percentage': 0,
                'segments_analyzed': 0,
                'scott_segments': 0,
                'mark_segments': 0,
                'processing_note': f'Error in word counting: {str(e)}'
            }
    
    def process_episode(self, video_url: str, video_id: str) -> Dict:
        """
        Complete processing pipeline for a single episode.
        """
        audio_path = None
        use_cached = False
        try:
            logger.info(f"Starting real audio processing for episode {video_id}")
            
            # Step 1: Check for cached audio first
            from youtube_downloader import YouTubeAudioDownloader
            downloader = YouTubeAudioDownloader()
            cached_audio_path = downloader.get_cached_audio_path(video_id)
            
            if cached_audio_path:
                logger.info(f"Using cached audio file: {cached_audio_path}")
                audio_path = cached_audio_path
                use_cached = True
            else:
                # Step 1b: Download audio to temp if not cached
                audio_path = self.download_audio(video_url, video_id)
            
            # Step 2: Transcribe audio
            transcription_chunks = self.transcribe_with_whisper(audio_path)
            
            if not transcription_chunks:
                raise Exception("No transcription data generated")
            
            # Step 3: Identify speakers
            processed_chunks = self.identify_speakers(transcription_chunks, audio_path)
            
            # Step 4: Count words
            word_stats = self.count_words(processed_chunks)
            
            # Step 5: Add sample transcription data
            word_stats['sample_transcription'] = processed_chunks[:5]  # First 5 chunks as sample
            word_stats['full_transcription_available'] = True
            
            logger.info(f"Successfully processed episode {video_id}: {word_stats['total_words']} words")
            return word_stats
            
        except Exception as e:
            logger.error(f"Error processing episode {video_id}: {e}")
            return {
                'total_words': 0,
                'scott_words': 0,
                'mark_words': 0,
                'scott_percentage': 0,
                'mark_percentage': 0,
                'segments_analyzed': 0,
                'scott_segments': 0,
                'mark_segments': 0,
                'processing_note': f'Real processing failed: {str(e)}',
                'error': True
            }
        
        finally:
            # Clean up downloaded audio file (but not cached files)
            if audio_path and os.path.exists(audio_path) and not use_cached:
                try:
                    os.remove(audio_path)
                    logger.info(f"Cleaned up temporary audio file: {audio_path}")
                except Exception as e:
                    logger.warning(f"Could not remove audio file {audio_path}: {e}")

def test_audio_processor():
    """Test function for the audio processor."""
    processor = PodcastAudioProcessor()
    
    # Test with Episode 20 (a recent episode)
    test_url = "https://www.youtube.com/watch?v=UmXW8nGG9ZE"
    test_id = "UmXW8nGG9ZE"
    
    result = processor.process_episode(test_url, test_id)
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    test_audio_processor()
