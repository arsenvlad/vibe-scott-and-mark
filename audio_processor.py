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

import yt_dlp
import librosa
import speech_recognition as sr
from pydub import AudioSegment
import torch
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PodcastAudioProcessor:
    def __init__(self, temp_dir: str = None):
        """Initialize the audio processor."""
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.recognizer = sr.Recognizer()
        
        # Initialize Whisper for transcription
        try:
            import whisper
            self.whisper_model = whisper.load_model("base")
            logger.info("Initialized Whisper model for transcription")
        except ImportError:
            logger.warning("Whisper not available, falling back to Google Speech Recognition")
            self.whisper_model = None
    
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
    
    def identify_speakers(self, transcription_chunks: List[Dict]) -> List[Dict]:
        """
        Identify speakers (Scott vs Mark) in transcription chunks.
        Uses keyword-based heuristics and speaking patterns.
        """
        try:
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
            
            processed_chunks = []
            
            for i, chunk in enumerate(transcription_chunks):
                text_lower = chunk['text'].lower()
                
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
                    speaker = 'scott'
                elif mark_score > scott_score:
                    speaker = 'mark'
                else:
                    # Fallback: try to maintain conversation flow
                    # If previous speaker was identified, alternate is likely
                    if i > 0 and processed_chunks[i-1]['speaker'] == 'scott':
                        speaker = 'mark'
                    elif i > 0 and processed_chunks[i-1]['speaker'] == 'mark':
                        speaker = 'scott'
                    else:
                        # Default alternating pattern starting with Scott
                        speaker = 'scott' if i % 2 == 0 else 'mark'
                
                processed_chunks.append({
                    'text': chunk['text'],
                    'start': chunk['start'],
                    'end': chunk['end'],
                    'speaker': speaker,
                    'scott_score': scott_score,
                    'mark_score': mark_score
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
        try:
            logger.info(f"Starting real audio processing for episode {video_id}")
            
            # Step 1: Download audio
            audio_path = self.download_audio(video_url, video_id)
            
            # Step 2: Transcribe audio
            transcription_chunks = self.transcribe_with_whisper(audio_path)
            
            if not transcription_chunks:
                raise Exception("No transcription data generated")
            
            # Step 3: Identify speakers
            processed_chunks = self.identify_speakers(transcription_chunks)
            
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
            # Clean up downloaded audio file
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                    logger.info(f"Cleaned up audio file: {audio_path}")
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
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.model_path = os.path.join(self.data_dir, 'speaker_model.pkl')
        self.scaler_path = os.path.join(self.data_dir, 'speaker_scaler.pkl')
        
        # Voice recognition setup
        self.recognizer = sr.Recognizer()
        
        # Speaker classification model
        self.speaker_model = None
        self.feature_scaler = None
        
        # Load or train the speaker identification model
        self._load_or_train_speaker_model()
    
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
            
            # Create training data
            # We'll create multiple segments from each sample for better training
            scott_segments = self._segment_audio_features(scott_features)
            mark_segments = self._segment_audio_features(mark_features)
            
            # Prepare training data
            X = np.vstack([scott_segments, mark_segments])
            y = np.array([0] * len(scott_segments) + [1] * len(mark_segments))  # 0=Scott, 1=Mark
            
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
    
    def _extract_voice_features(self, audio_file):
        """Extract audio features for speaker identification"""
        try:
            # Load audio file
            y, sr = librosa.load(audio_file, sr=22050)
            
            # Extract various audio features
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
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error extracting voice features: {str(e)}")
            return None
    
    def _segment_audio_features(self, features, segment_size=50):
        """Create multiple feature segments for training"""
        segments = []
        feature_length = len(features)
        
        for i in range(0, feature_length - segment_size + 1, segment_size // 2):
            segment = features[i:i + segment_size]
            if len(segment) == segment_size:
                segments.append(segment)
        
        if not segments:
            segments = [features[:segment_size] if len(features) >= segment_size else features]
        
        return np.array(segments)
    
    def analyze_episode(self, audio_file, video_id):
        """
        Analyze an episode audio file for speaker identification and word counting
        """
        try:
            logger.info(f"Starting audio analysis for episode {video_id}")
            
            # Convert to WAV if needed and load audio
            audio_data = self._prepare_audio(audio_file)
            if audio_data is None:
                return {'error': 'Failed to prepare audio'}
            
            # Split audio into segments based on silence
            segments = self._split_audio_segments(audio_data)
            logger.info(f"Split audio into {len(segments)} segments")
            
            # Analyze each segment
            scott_words = 0
            mark_words = 0
            total_words = 0
            segment_analysis = []
            
            for i, segment in enumerate(segments):
                if len(segment) < 1000:  # Skip very short segments
                    continue
                
                # Transcribe segment
                transcript = self._transcribe_segment(segment)
                if not transcript:
                    continue
                
                # Count words in transcript
                words_in_segment = len(transcript.split())
                total_words += words_in_segment
                
                # Identify speaker for this segment
                speaker = self._identify_speaker(segment)
                
                if speaker == 'scott':
                    scott_words += words_in_segment
                elif speaker == 'mark':
                    mark_words += words_in_segment
                
                segment_analysis.append({
                    'segment': i,
                    'speaker': speaker,
                    'words': words_in_segment,
                    'transcript': transcript[:100] + '...' if len(transcript) > 100 else transcript
                })
                
                # Log progress every 10 segments
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(segments)} segments")
            
            # Calculate speaking time percentages
            scott_percentage = (scott_words / max(total_words, 1)) * 100
            mark_percentage = (mark_words / max(total_words, 1)) * 100
            
            analysis_result = {
                'total_words': total_words,
                'scott_words': scott_words,
                'mark_words': mark_words,
                'scott_percentage': round(scott_percentage, 1),
                'mark_percentage': round(mark_percentage, 1),
                'segments_analyzed': len(segment_analysis),
                'processing_date': os.path.getctime(audio_file) if os.path.exists(audio_file) else None
            }
            
            logger.info(f"Analysis complete for {video_id}: {total_words} total words, {scott_words} Scott, {mark_words} Mark")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error analyzing episode audio: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_audio(self, audio_file):
        """Prepare audio file for processing"""
        try:
            # Load audio file with pydub for better format support
            audio = AudioSegment.from_file(audio_file)
            
            # Convert to mono and normalize sample rate
            audio = audio.set_channels(1).set_frame_rate(22050)
            
            # Normalize volume
            audio = audio.normalize()
            
            return audio
            
        except Exception as e:
            logger.error(f"Error preparing audio: {str(e)}")
            return None
    
    def _split_audio_segments(self, audio_data, silence_thresh=-40, min_silence_len=500):
        """Split audio into segments based on silence"""
        try:
            segments = split_on_silence(
                audio_data,
                min_silence_len=min_silence_len,
                silence_thresh=silence_thresh,
                keep_silence=100
            )
            
            # Filter out very short segments (less than 1 second)
            segments = [seg for seg in segments if len(seg) > 1000]
            
            return segments
            
        except Exception as e:
            logger.error(f"Error splitting audio segments: {str(e)}")
            return [audio_data]  # Return original audio as single segment
    
    def _transcribe_segment(self, audio_segment):
        """Transcribe an audio segment to text"""
        try:
            # Export segment to temporary file for speech recognition
            temp_file = os.path.join(self.data_dir, 'temp_segment.wav')
            audio_segment.export(temp_file, format='wav')
            
            # Use speech recognition
            with sr.AudioFile(temp_file) as source:
                audio = self.recognizer.record(source)
                transcript = self.recognizer.recognize_google(audio)
                
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            return transcript
            
        except sr.UnknownValueError:
            # Speech recognition could not understand audio
            return ""
        except sr.RequestError as e:
            logger.warning(f"Speech recognition error: {str(e)}")
            return ""
        except Exception as e:
            logger.warning(f"Error transcribing segment: {str(e)}")
            return ""
    
    def _identify_speaker(self, audio_segment):
        """Identify whether the speaker is Scott or Mark"""
        if self.speaker_model is None or self.feature_scaler is None:
            # If no model available, return unknown
            return 'unknown'
        
        try:
            # Export segment to temporary file for feature extraction
            temp_file = os.path.join(self.data_dir, 'temp_speaker.wav')
            audio_segment.export(temp_file, format='wav')
            
            # Extract features
            features = self._extract_voice_features(temp_file)
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            if features is None:
                return 'unknown'
            
            # Predict speaker
            features_scaled = self.feature_scaler.transform([features])
            prediction = self.speaker_model.predict(features_scaled)[0]
            
            return 'scott' if prediction == 0 else 'mark'
            
        except Exception as e:
            logger.warning(f"Error identifying speaker: {str(e)}")
            return 'unknown'
