# ---------- PYTHON PATH FIX FOR AI SCRIPTS ----------
import sys
from pathlib import Path

# Project root: dual_summarization_project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Scripts directory: dual_summarization_project/scripts
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(SCRIPTS_DIR))
from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import os
import cv2
import uuid
import threading
from scripts.infer_summary import run_inference
from django.contrib.auth.models import User
from django.contrib import messages
from django.contrib.auth import authenticate, login as auth_login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache
from .models import VideoSummary

PROCESS_STATUS = {}
PROCESS_RESULTS = {}

def welcome(request):
    return render(request, 'welcome.html')

@never_cache
@login_required(login_url='login')
def home(request):
    return render(request, 'home.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def w_about(request):
    return render(request, 'w_about.html')

def features(request):
    return render(request, 'features.html')    

def login(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        # Authenticate using email as username
        user = authenticate(request, username=email, password=password)

        if user is not None:
            auth_login(request, user)
            return redirect('home')
        else:
            messages.error(request, "Invalid email or password.")
            return redirect('login')

    return render(request, 'login.html')

def signup(request):
    if request.method == 'POST':
        fullname = request.POST.get('fullname')
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')

        # Password match check
        if password != confirm_password:
            messages.error(request, "Passwords do not match.")
            return redirect('signup')

        # Check if email already exists
        if User.objects.filter(username=email).exists():
            messages.error(request, "Email already registered.")
            return redirect('signup')

        # Create user
        user = User.objects.create_user(
            username=email,   # using email as username
            email=email,
            password=password,
            first_name=fullname
        )

        user.save()

        # Auto login after signup
        auth_login(request, user)

        return redirect('home')

    return render(request, 'signup.html')

def pr(request):
    return render(request, 'pr.html')  

def logout_view(request):
    print("Logout Initiated")

    logout(request)

    # Remove the user's session data by setting all keys to None
    request.session.flush()

    if request.user.is_authenticated:
        # User is logged in
        print("Authenticated")
    else:
        # User is logged out
        print("Not Authenticated")

    return redirect('welcome')
  
def tnc(request):
    return render(request, 'tnc.html')   

def extract_video_metadata(video_path):
    # Title (file name only)
    filename = os.path.basename(video_path)
    title = filename.split("_", 1)[1] 

    # File size (MB)
    size_bytes = os.path.getsize(video_path)
    size_mb = round(size_bytes / (1024 * 1024), 2)

    # Duration
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = frame_count / fps if fps else 0
    cap.release()

    minutes = int(duration // 60)
    seconds = int(duration % 60)

    formatted_duration = f"{minutes}:{str(seconds).zfill(2)}"

    return {
        "title": title,
        "duration": formatted_duration,
        "size": f"{size_mb} MB"
    } 

@csrf_exempt
def upload_video(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request method'}, status=400)

    if 'video' not in request.FILES:
        return JsonResponse({'error': 'No video file uploaded'}, status=400)

    video = request.FILES['video']

    upload_dir = os.path.join(settings.BASE_DIR, 'uploads')
    os.makedirs(upload_dir, exist_ok=True)

    video_id = uuid.uuid4().hex
    video_path = os.path.join(upload_dir, f"{video_id}_{video.name}")

    with open(video_path, 'wb+') as destination:
        for chunk in video.chunks():
            destination.write(chunk)

    #  NO AI HERE
    return JsonResponse({
        'message': 'Upload successful',
        'video_id': video_id,
        'video_path': video_path
    })

@csrf_exempt
def process_video(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Invalid request'}, status=400)

    video_id = request.POST.get('video_id')
    video_path = request.POST.get('video_path')

    if not video_id or not video_path:
        return JsonResponse({'error': 'Missing parameters'}, status=400)

    user_id = request.user.id

    output_dir = os.path.join(settings.BASE_DIR, 'outputs', video_id)
    os.makedirs(output_dir, exist_ok=True)

    PROCESS_STATUS[video_id] = 'processing'

    def run(user_id):
        try:
            original_cwd = os.getcwd()
            os.chdir(PROJECT_ROOT)

            metadata = extract_video_metadata(video_path)

            summary_text, keyframe_captions = run_inference(
                video_path=video_path,
                out_dir=output_dir,
                samples=96,
                keyframes=32
            )

            title = metadata.get("title", "")
            audio_filename = video_id + "_" + os.path.splitext(title)[0] + ".mp3"

            audio_url = f"/media/{video_id}/{audio_filename}"
            image_urls = []

            for item in keyframe_captions:
                frame_index = item["frame"] if isinstance(item, dict) else None

                if frame_index is not None:
                    image_filename = f"key_{frame_index}.jpg"
                    image_url = f"/media/{video_id}/{image_filename}"
                    image_urls.append(image_url)
                else:
                    image_urls.append(None)

            PROCESS_STATUS[video_id] = 'completed'

            PROCESS_RESULTS[video_id] = {
                "summary": summary_text,
                "captions": keyframe_captions,
                "metadata": metadata,
                "audio_url":audio_url,
                "image_urls": image_urls
            }

            user = User.objects.get(id=user_id)

            VideoSummary.objects.create(
                user=user,
                video_file_name=metadata.get("title", ""),
                duration=metadata.get("duration", ""),
                file_size=metadata.get("size", ""),
                summary=summary_text,
                audio_file_url=audio_url,
                image_urls=image_urls,
                captions=keyframe_captions
            )

        except Exception as e:
            PROCESS_STATUS[video_id] = 'failed'
            print(e)

        finally:
            os.chdir(original_cwd)

    threading.Thread(target=run, args=(user_id,), daemon=True).start()

    return JsonResponse({'status': 'started'})

def check_status(request, video_id):
    status = PROCESS_STATUS.get(video_id, 'unknown')

    if status == 'completed':
        result = PROCESS_RESULTS.get(video_id, {})
        return JsonResponse({
            "status": status,
            "summary": result.get("summary", ""),
            "captions": result.get("captions", []),
            "metadata": result.get("metadata", {}),
            "audio_url": result.get("audio_url", ""),
            "image_urls": result.get("image_urls", []),
        })

    return JsonResponse({"status": status})

@login_required
def your_summaries(request):
    summaries = VideoSummary.objects.filter(user=request.user).order_by('-created_at')

    for summary in summaries:
        if summary.image_urls and summary.captions:
            summary.events = zip(summary.image_urls, summary.captions)
        else:
            summary.events = []

    return render(request, "your_summaries.html", {
        "summaries": summaries,
        "total_count": summaries.count()
    })
