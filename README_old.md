# Scott and Mark Podcast Analytics

A modern web application for analyzing the "Scott and Mark Learn To..." podcast episodes with comprehensive statistics, speaker identification, and beautiful visualizations.

## Features

- 🎧 **YouTube Integration**: Automatically fetch episodes from the YouTube playlist
- 📊 **Rich Analytics**: View counts, duration trends, and episode statistics
- 🎤 **Speaker Identification**: AI-powered analysis to distinguish Scott vs Mark's speaking time
- 🗣️ **Word Counting**: Count total words spoken and individual contributions
- 📈 **Interactive Charts**: Beautiful visualizations powered by Chart.js
- 💾 **Local Storage**: Download and cache audio files to avoid reprocessing
- 🎨 **Modern UI**: Clean, responsive design with smooth animations
- ⚡ **Fast Processing**: Optimized for CPU-only processing (no GPU required)

## Technology Stack

### Backend
- **Flask**: Python web framework
- **yt-dlp**: YouTube video/audio downloading
- **librosa**: Audio processing and feature extraction
- **scikit-learn**: Machine learning for speaker identification
- **SpeechRecognition**: Audio-to-text transcription
- **pydub**: Audio manipulation and processing

### Frontend
- **HTML5/CSS3**: Modern web standards
- **JavaScript (ES6+)**: Interactive functionality
- **Chart.js**: Data visualization
- **Font Awesome**: Icons
- **Google Fonts**: Typography

## Setup Instructions

### Prerequisites

1. **Python 3.8+** installed on your system
2. **FFmpeg** for audio processing:
   - Windows: Download from https://ffmpeg.org/download.html
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

### Installation

1. **Clone or download the project files**

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare voice samples** (already included):
   - `data/voice_scott.mp3` - Sample of Scott's voice
   - `data/voice_mark.mp3` - Sample of Mark's voice
   
   These samples are used to train the speaker identification model.

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Usage

### Getting Started

1. **Fetch Episodes**: Click "Fetch New Episodes" to download the latest episode data from YouTube
2. **View Statistics**: Browse the overview statistics and charts
3. **Analyze Audio**: 
   - Click "Analyze" on individual episodes, or
   - Click "Process All Audio" to analyze all episodes at once

### Features Explained

#### Episode Analytics
- **View Counts**: Track popularity across episodes
- **Duration Trends**: See how episode lengths have evolved
- **Upload Timeline**: Visualize release schedule consistency

#### Speaker Analysis
- **Word Counting**: Total words spoken in each episode
- **Speaker Identification**: Distinguish between Scott and Mark using voice samples
- **Speaking Balance**: See who talks more in each episode
- **Aggregate Statistics**: Overall speaking patterns across all episodes

#### Data Management
- **Local Caching**: Audio files are downloaded once and cached in the `data/` folder
- **Incremental Processing**: Only new episodes are processed when you click "Fetch New Episodes"
- **Processing Status**: Visual indicators show which episodes have been analyzed

## Project Structure

```
scott-and-mark/
├── app.py                 # Main Flask application
├── youtube_scraper.py     # YouTube playlist and video handling
├── audio_processor.py     # Audio analysis and speaker identification
├── podcast_analyzer.py    # Analytics and insights generation
├── requirements.txt       # Python dependencies
├── data/                  # Data storage directory
│   ├── voice_scott.mp3   # Scott's voice sample
│   ├── voice_mark.mp3    # Mark's voice sample
│   ├── youtube_playlist_url.txt
│   ├── episodes.json     # Episode metadata cache
│   ├── processed_episodes.json # Processing status
│   └── [audio_files].wav # Downloaded episode audio
└── static/               # Frontend files
    ├── index.html        # Main web page
    ├── style.css         # Styling and layout
    └── script.js         # JavaScript functionality
```

## API Endpoints

- `GET /api/episodes` - Get all episodes data
- `POST /api/fetch-new-episodes` - Fetch new episodes from YouTube
- `POST /api/process-audio/<video_id>` - Process audio for specific episode
- `GET /api/statistics` - Get aggregate statistics

## Configuration

### Voice Samples
The speaker identification system uses the provided voice samples in the `data/` folder. For best results:
- Samples should be clear speech without background music
- 10-30 seconds of audio is sufficient
- Higher quality audio improves identification accuracy

### Processing Settings
Audio processing settings can be adjusted in `audio_processor.py`:
- `silence_thresh`: Threshold for splitting audio segments
- `min_silence_len`: Minimum silence duration for splitting
- Speaker model parameters can be tuned for better accuracy

## Performance Notes

- **CPU Optimization**: Designed to work efficiently without GPU acceleration
- **Memory Usage**: Audio files are processed in segments to manage memory
- **Processing Time**: Expect 2-5 minutes per episode depending on length and hardware
- **Storage**: Each episode audio file is ~10-50MB depending on duration

## Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   - Ensure FFmpeg is installed and in your system PATH
   - Restart your terminal/command prompt after installation

2. **YouTube download errors**:
   - Check internet connection
   - YouTube may occasionally block requests; wait and retry
   - Update yt-dlp: `pip install --upgrade yt-dlp`

3. **Speech recognition failures**:
   - Requires internet connection for Google Speech Recognition API
   - Some audio segments may be unclear and fail to transcribe

4. **Speaker identification issues**:
   - Ensure voice samples are clear and representative
   - Model accuracy improves with more training data
   - Consider replacing voice samples with better quality ones

### Debug Mode
Run with debug mode for detailed logging:
```bash
python app.py
```
Check console output for detailed error messages and processing status.

## Contributing

Feel free to contribute improvements:
1. Better speaker identification algorithms
2. Additional chart types and visualizations
3. Export functionality (CSV, PDF reports)
4. Improved audio processing performance
5. Additional podcast metadata extraction

## License

This project is for educational and personal use. Respect YouTube's terms of service when downloading content.

---

Built with ❤️ for podcast analytics enthusiasts!
