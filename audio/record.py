import sys
import sounddevice
import numpy
from scipy.io.wavfile import write as scipy_wavfile_write
from pydub import AudioSegment
from io import BytesIO
import queue

# Settings
RATE = int(sounddevice.query_devices(kind='input')['default_samplerate']) # Choosing rate(Hz) by device support
DURATION = 2  # Duration of recording or audio segments to process(seconds)
RECORD_DURATION = int(RATE * DURATION) #  Recording duration(seconds)
CHANNELS = 1  # Number of channels (1 = mono, 2 = stereo)
CHUNK_SIZE = 1024  # Frames per buffer
segment_counter = 0
start_time = None
OUTPUT_FILENAME = "output.wav"
OUTPUT_MP3 = "output.mp3"
audio_queue = queue.Queue() # Queue to hold audio segments for processing


def to_file(file_format:str='wav') -> str:
    if file_format not in ['mp3', 'wav']:
        raise ValueError("format must be wav or mp3")

    print("Recording...")
    # Record audio
    audio_data = sounddevice.rec(RECORD_DURATION, samplerate=RATE, channels=CHANNELS, dtype='int16')
    sounddevice.wait()  # Wait until recording is finished
    print("Finished recording")

    if file_format == 'wav': # Save as .wav file
        scipy_wavfile_write(OUTPUT_FILENAME, RATE, audio_data)
        print(f"Audio saved as {OUTPUT_FILENAME}")

    if file_format == 'mp3': # Save as .mp3 file
        # Convert WAV to MP3
        audio_bytes = audio_data.tobytes()
        audio_segment = AudioSegment(
            data=audio_bytes,
            sample_width=audio_data.dtype.itemsize,  # 2 bytes for 'int16'
            frame_rate=RATE,
            channels=CHANNELS
        )
        # Export the audio directly to an MP3 file in memory
        mp3_buffer = BytesIO()
        audio_segment.export(mp3_buffer, format="mp3")
        mp3_data = mp3_buffer.getvalue()

        # Save the MP3 data to a file
        with open(OUTPUT_MP3, "wb") as f:
            f.write(mp3_data)

        print("MP3 file saved as output.mp3")

    return OUTPUT_FILENAME


# Save stream segments to wav file
def save_stream_audio(time):
    """
    Function to process the audio data.
    Modify this to apply any transformations or effects.
    """
    global segment_counter
    chunk = []

    while len(chunk) < RECORD_DURATION:
        chunk.extend(audio_queue.get_nowait())

    audio_data = (numpy.array(chunk) * 32767).astype(numpy.int16)  # Convert to 16-bit
    timestamp_seconds = int(time)  # Seconds since stream start

    # Save the segment to a file
    scipy_wavfile_write(filename=f"seg/segment_{segment_counter}_{timestamp_seconds}.wav", rate=RATE, data=audio_data)

    # Increment the segment counter
    segment_counter += 1


# Callback function for recording and processing
def audio_callback(indata, frames, time, status):
    global start_time

    if start_time is None:
        start_time = time.inputBufferAdcTime

    if status:
        print(f"Error: {status}", file=sys.stderr)
    try:
        # Add input audio to the queue
        audio_queue.put(indata.copy())

        # Check if queue has something
        if audio_queue.empty():
            raise RuntimeError('Not recording')

        # Check if recording duration is reached
        if audio_queue.qsize() * frames >= RECORD_DURATION:
            elapsed_time = time.inputBufferAdcTime - start_time
            save_stream_audio(time=elapsed_time)
        else:
            print('Recording duration too short')
    except Exception as e:
        print(f"Callback error: {e}")


def to_stream():  # TODO: loopback stream from desktop
    stream = sounddevice.InputStream(samplerate=RATE, channels=CHANNELS, dtype='float32', blocksize=CHUNK_SIZE, callback=audio_callback)
    with stream:
        print("Recording and processing audio in real-time...")
        try:
            while True:
                sounddevice.sleep(100)  # Keep the program running
        except KeyboardInterrupt:
            print("\nStopped.")


to_stream()
# to_file(file_format='wav')
