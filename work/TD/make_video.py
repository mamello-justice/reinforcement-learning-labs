import os, subprocess

os.chdir("./images")
subprocess.call([
    'ffmpeg', '-framerate', '8', '-i', '*.png', '-r', '30', '-pix_fmt', 'yuv420p',
    'video_name.mp4'
])