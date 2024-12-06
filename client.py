import pyaudio
import numpy as np
import socket
import time
import struct
from scipy.signal import butter, lfilter

# Configuration
SAMPLE_RATE = 8000  # 8 kHz
FRAME_SIZE = 1024  # Audio frame size
PEER_IP = "192.168.1.139"  # IP address of the other client (change as needed)
PEER_PORT = 5006  # Port to communicate with the other client
BUFFER_SIZE = 2048  # Buffer size for receiving RTP packets
LOCAL_PORT = 5005  # Local port to receive audio packets
BITRATE = 64000  # Placeholder, bitrate depends on codec and compression


# Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=400, highcut=3999, fs=8000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def quantize(data, levels=256):
    data = data / np.max(np.abs(data))
    quantized = np.round(data * (levels / 2))
    return quantized.astype(np.int16)

def compress(data):
    compressed = np.sign(data) * (np.log1p(np.abs(data)) / np.log(256))
    return (compressed * 127).astype(np.int8)

def silence_suppression(data, threshold=0.01):
    return data if np.abs(data).mean() > threshold else np.zeros_like(data)

# RTP packetization
def rtp_packetize(data, seq_num=0, timestamp=0):
    version = 2
    payload_type = 0  # PCM
    ssrc = 12345
    header = struct.pack('!BBHII', (version << 6) | payload_type, seq_num, len(data), timestamp, ssrc)
    return header + data.tobytes()

def rtp_depacketize(packet):
    payload = packet[12:]  # Strip off the first 12 bytes of RTP header
    return np.frombuffer(payload, dtype=np.int8)

# Send RTP audio packet
def send_audio_packet(data, seq_num, timestamp):
    rtp_packet = rtp_packetize(data, seq_num, timestamp)
    sock.sendto(rtp_packet, (PEER_IP, PEER_PORT))

# PyAudio setup for capturing audio
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=FRAME_SIZE)

# Socket for sending/receiving RTP packets
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('', LOCAL_PORT))  # Bind to local port to receive data

# Handshake function to ensure readiness
def handshake(sock, peer_ip, peer_port):
    print("Starting handshake process...")
    sock.settimeout(5)
    
    while True:
        try:
            # Send 'ready' to the peer
            sock.sendto(b'ready', (peer_ip, peer_port))
            print("Sent 'ready' to peer, waiting for peer to be ready...")
            
            # Try to receive 'ready' from the peer
            data, _ = sock.recvfrom(BUFFER_SIZE)
            if data == b'ready':
                print("Received 'ready' from peer. Connection is established.")
                break
        except socket.timeout:
            print("No response yet, continuing to wait...")
        except ConnectionResetError:
            print("Connection reset by peer, retrying...")
        time.sleep(1)

    sock.settimeout(None)  # Remove timeout after handshake

# Perform the handshake before starting audio communication
handshake(sock, PEER_IP, PEER_PORT)

# PyAudio setup for playing received audio
play_stream = audio.open(format=pyaudio.paInt16,
                         channels=1,
                         rate=SAMPLE_RATE,
                         output=True)

seq_num = 0
timestamp = 0

try:
    while True:
        # Capture audio
        audio_frame = np.frombuffer(stream.read(FRAME_SIZE), dtype=np.int16)

        # Process the audio: filter, quantize, compress, etc.
        filtered_audio = bandpass_filter(audio_frame)
        quantized_audio = quantize(filtered_audio)
        compressed_audio = compress(quantized_audio)
        processed_audio = silence_suppression(compressed_audio)

        # Send the processed audio directly to the peer client
        send_audio_packet(processed_audio, seq_num, timestamp)
        
        # Increment sequence number and timestamp
        seq_num += 1
        timestamp += FRAME_SIZE

        # Receive RTP packet from the peer client
        data, _ = sock.recvfrom(BUFFER_SIZE)
        received_audio = rtp_depacketize(data)

        # Play the received audio
        play_stream.write(received_audio.tobytes())

except KeyboardInterrupt:
    print("Client shutting down...")

finally:
    # Cleanup
    stream.stop_stream()
    stream.close()
    play_stream.stop_stream()
    play_stream.close()
    audio.terminate()
    sock.close()