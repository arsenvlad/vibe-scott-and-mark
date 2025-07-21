<<<<<<< HEAD
# Scott and Mark Learn To... - Podcast Analytics

A modern web application for analyzing the "Scott and Mark Learn To..." podcast using AI-powered speech-to-text transcription and speaker identification.

![Podcast Analytics Dashboard](https://img.shields.io/badge/Status-Working-brightgreen) ![Python](https://img.shields.io/badge/Python-3.11+-blue) ![Flask](https://img.shields.io/badge/Flask-3.0.0-blue) ![Whisper](https://img.shields.io/badge/OpenAI-Whisper-orange)

## 🎙️ Features

### Real Audio Processing
- **YouTube Audio Download**: Automatically downloads podcast episodes from YouTube
- **AI Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
- **Speaker Identification**: Distinguishes between Scott and Mark's voices
- **Word Counting**: Tracks total words and individual speaker contributions
- **Caching System**: Avoids re-downloading/re-processing already analyzed episodes

### Web Dashboard
- **Modern UI**: Beautiful, responsive interface with real-time data visualization
- **Episode Statistics**: View detailed analytics for each podcast episode
- **Progress Tracking**: See which episodes have been processed vs pending
- **Charts & Graphs**: Visual representation of speaking patterns and trends
- **Real-time Processing**: Process episodes on-demand through the web interface

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- FFmpeg (for audio processing)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/arsenvlad/vibe-scott-and-mark.git
   cd vibe-scott-and-mark
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional AI models**
   ```bash
   # Whisper will be downloaded automatically on first use
   pip install openai-whisper
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Open your browser**
   Navigate to `http://127.0.0.1:5000`

## 📊 Usage

### Fetching Episodes
1. Click "Fetch New Episodes" to scan the YouTube playlist
2. The system will discover all available podcast episodes
3. Episodes appear in the dashboard with their metadata

### Analyzing Episodes
1. Click "Analyze" on any unprocessed episode
2. The system will:
   - Download the audio from YouTube
   - Transcribe using Whisper AI
   - Identify speakers (Scott vs Mark)
   - Count words for each speaker
   - Display results in real-time

### Viewing Results
- **Total Words**: Complete word count for the episode
- **Speaker Breakdown**: Percentage and word count per speaker
- **Duration**: Episode length and speaking patterns
- **Trends**: Visual charts showing patterns across episodes

## 🛠️ Technology Stack

### Backend
- **Flask 3.0.0**: Web framework
- **OpenAI Whisper**: Speech-to-text transcription
- **yt-dlp**: YouTube audio downloading
- **librosa**: Audio processing
- **scikit-learn**: Speaker classification

### Frontend
- **Chart.js**: Data visualization
- **Modern CSS**: Responsive design
- **JavaScript**: Interactive UI

### AI & Audio
- **Whisper "base" model**: Accurate multilingual speech recognition
- **Custom speaker identification**: ML-based voice classification
- **Audio preprocessing**: Noise reduction and format conversion

## 📁 Project Structure

```
├── app.py                      # Main Flask application
├── audio_processor.py          # Whisper AI + speaker identification
├── youtube_downloader.py       # YouTube audio downloader
├── youtube_scraper.py          # YouTube playlist scraper
├── requirements.txt            # Python dependencies
├── static/                     # Web frontend assets
│   ├── index.html             # Main dashboard
│   ├── style.css              # Responsive styling
│   └── script.js              # Interactive functionality
├── data/                       # Data storage
│   ├── audio_cache/           # Downloaded audio files (gitignored)
│   ├── episodes.json          # Episode metadata
│   └── processed_episodes.json # Analysis results
└── tests/                      # Testing utilities
    ├── test_real_audio.py     # Audio processing tests
    ├── test_real_download.py  # Download functionality tests
    └── test_focused_download.py # Targeted download tests
```

## 🔧 API Endpoints

- `GET /api/episodes` - Fetch all episodes with analysis results
- `POST /api/fetch-new-episodes` - Scan YouTube for new episodes
- `POST /api/download-audio` - Download audio for specific episode
- `POST /api/process-episode` - Run full AI analysis on episode
- `GET /api/statistics` - Get aggregate statistics across all episodes

## 🧪 Testing

```bash
# Test YouTube downloading
python test_real_download.py

# Test audio processing
python test_real_audio.py

# Test specific download methods
python test_focused_download.py
```

## 📈 Current Results

The system has successfully analyzed podcast episodes with results like:

**Episode 20 - "Vibe Coding and Being Productive"**
- **Total Words**: 4,525
- **Scott**: 2,678 words (59.2%)
- **Mark**: 1,847 words (40.8%)
- **Duration**: ~26 minutes
- **Processing Time**: ~3-5 minutes

## 🤖 How It Works

### Audio Download Process
1. **YouTube URL Extraction**: Parses playlist to find episode URLs
2. **Format Selection**: Chooses optimal audio format (webm, m4a, etc.)
3. **Fallback Methods**: Multiple download strategies to handle YouTube restrictions
4. **Caching**: Stores downloaded audio to avoid re-downloading

### AI Processing Pipeline
1. **Audio Preprocessing**: Converts to standard format for analysis
2. **Whisper Transcription**: Segments audio and transcribes to text
3. **Speaker Diarization**: Uses voice characteristics to identify speakers
4. **Keyword Enhancement**: Improves speaker ID with content analysis
5. **Word Counting**: Accurate per-speaker word statistics

### Speaker Identification
- **Voice Feature Analysis**: Pitch, tone, speaking patterns
- **Keyword Matching**: Speaker-specific vocabulary and phrases
- **Confidence Scoring**: Reliability metrics for each identification
- **Continuous Learning**: Improves accuracy over time

## 🛣️ Roadmap

- [ ] **Advanced Analytics**: Sentiment analysis, topic detection
- [ ] **Export Features**: CSV/JSON data export for external analysis
- [ ] **Real-time Processing**: Live episode processing during upload
- [ ] **Speaker Training**: Custom voice model training for improved accuracy
- [ ] **Multi-podcast Support**: Expand to analyze multiple podcast series
- [ ] **Mobile App**: Native mobile interface for on-the-go analytics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Whisper** for excellent speech recognition
- **yt-dlp** for reliable YouTube downloading
- **Scott and Mark** for creating engaging podcast content to analyze!

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/arsenvlad/vibe-scott-and-mark/issues) page
2. Create a new issue with detailed information
3. Include logs and error messages for faster resolution

---

**Built with ❤️ for podcast analytics enthusiasts**
