from moviepy.editor import *
from moviepy.audio.AudioClip import AudioArrayClip
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import io
import random
import string
import soundfile as sf
from MasteringRPC import *

def _temp_file_name(ext):
    return ("./tmp-video-"
            + "".join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
            + ext)
def getAudioFromVideo(byteData):
    temp_file = _temp_file_name(ext=".mp4")
    with open(temp_file, "wb") as file:
        file.write(data)
    videoclip = VideoFileClip(temp_file)
    # os.remove(temp_file)
    audioclip = videoclip.audio
    audioclip.write_audiofile("./test_origin.wav")
    print(audioclip)
    print(audioclip.fps) #sample rate
    print(audioclip.nchannels)

    return audioclip.to_soundarray(), audioclip.fps, videoclip, temp_file


if (__name__ == '__main__'):
    path = "./Test/videoplayback.mp4"
    f = open(path, 'rb')
    data = f.read()
    f.close()

    y, sr, originVideoClip, tmpFileName = getAudioFromVideo(data)
    
    masteringRPC = MasteringRPC()
    masteredAudio = masteringRPC.getMasteredAudioFromY(y,sr)
    mastered_y, samplerate = sf.read(file=io.BytesIO(masteredAudio), dtype='float32')
    
    
    mastered_audioclip = AudioArrayClip(mastered_y, fps=sr)
    mastered_audioclip.write_audiofile("./test.wav")

    masteredVideoClip = originVideoClip.set_audio(mastered_audioclip)
    masteredVideoClip.write_videofile("./tmpvideo.mp4")

    os.remove(tmpFileName)
