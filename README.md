Captrix – AI Powered Video Summarization System

Captrix is an AI-powered web application that automatically generates
concise natural-language summaries for videos. The system analyzes video
frames, selects important keyframes, generates captions, and produces a
coherent summary with optional audio narration.

The project integrates modern vision-language models, deep learning
techniques, and a Django-based web framework to create a complete
end-to-end video summarization platform.

  --------------------------------------------------
  FEATURES
  --------------------------------------------------
  - Upload videos directly from the browser 
  - Drag-and-drop video upload with progress tracking 
  - Automatic AI-based video summarization 
  - Keyframe extraction and caption generation 
  - Human-readable paragraph summary generation 
  - Text-to-speech narration of generated summary 
  - User authentication (signup, login, logout) 
  - Personalized summary history for each user 
  - Modern responsive UI with animated result display 
  - Persistent database storage for
    summaries

  --------------------------------------------------

SYSTEM ARCHITECTURE

Video Input
  | 
Frame Sampling 
  |
CLIP Feature Extraction
  | 
Keyframe Selection (MLP Selector)
  | 
BLIP Caption Generation
  | 
Caption Aggregation
  | 
LLM Summary Generation
  |
Text-to-Speech Conversion
  | 
Result Display (Web Interface)

  -------------------
  TECHNOLOGIES USED
  -------------------

Backend - Python 3.12.4 - Django - PyTorch - Transformers (HuggingFace)

AI Models - CLIP (Vision-Language Embeddings) - BLIP (Image
Captioning) - LLAMA / Local LLM (Summary Generation)

Frontend - HTML - CSS - JavaScript

Other Tools - OpenCV (Video Processing) - gTTS (Text-to-Speech) -
NumPy - Scikit-learn

  -------------------
  PROJECT STRUCTURE
  -------------------

video_summarization_project
|--captrix
   |--backend
   |--captrix
   |--frontend
   |--outputs
   |--static
   |--uploads
   |--db.sqlite3
   |--manage.py
|--data
|--dataset
   |--video
   |--msrvtt_test_1k.json
   |--msrvtt_train_7k.json
   |--msrvtt_train_9k.json
|--models
|--outputs
|--scripts
   |--__pycache__
   |--brightness.py
   |--compute_pseudo_labels.py
   |--convert_msrvtt_to_meta.py
   |--evaluate.py
   |--extract_features.py
   |--infer_summary.py
   |--text_to_speech.py
   |--train_selector.py
   |--venv


NOTE : The dataset used here is MSRVTT. After downloading the dataset, it must be saved in the dataset folder. The video folder should contain 10000 videos
       and the .json files should contain the captions for each frame of all the videos.
  
  --------------
  INSTALLATION
  --------------

1.  Clone the repository

git clone
https://github.com/yourusername/captrix-video-summarization.git cd
captrix-video-summarization

2.  Create virtual environment

python -m venv venv

Activate environment

Windows: venv

Linux / Mac: source venv/bin/activate

3.  Install dependencies

pip install -r requirements.txt

Main dependencies include:

torch transformers opencv-python numpy scikit-learn django gtts

  ---------------------
  RUNNING THE PROJECT
  ---------------------

Navigate to the Django directory

cd captrix

Run the server

python manage.py runserver

Open the application

http://127.0.0.1:8000

  -------
  USAGE
  -------

1.  Open the web application
2.  Create an account or log in
3.  Upload a video using the upload interface
4.  Wait for the upload to complete
5.  Click “Summarize Video”
6.  The AI system analyzes the video
7.  View the generated summary, key events, and audio narration
8.  Access previous summaries through the history page

  ----------------
  DATABASE MODEL
  ----------------

The system stores user summaries in the ‘summary_details’ table.

Attributes include: 
- user_name 
- file_name 
- duration 
- size 
- summary 
- audio_url 
- image_urls (list of keyframe images) 
- image_captions (list of captions for keyframes)

  ----------------
  KEY HIGHLIGHTS
  ----------------

-   Combines visual understanding and language generation
-   Fully integrated AI pipeline with web interface
-   Supports multi-user personalized history
-   Designed for scalability and modular extension

  ---------------------
  FUTURE IMPROVEMENTS
  ---------------------

-   WebSocket based real-time progress updates
-   Video preview and keyframe gallery
-   Downloadable summary reports
-   Improved keyframe visualization
-   Mobile-responsive UI optimization
-   Deployment using Docker and cloud services

  --------------
  APPLICATIONS
  --------------

-   Educational video summarization
-   Media content analysis
-   Surveillance footage summarization
-   Content recommendation systems
-   Large video archive management
