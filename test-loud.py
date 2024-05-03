import soundfile as sf
import pyloudnorm as pyln

# 加载音频文件
data, rate = sf.read(r"D:\Downloads\output_v2_zh.wav")

# 峰值归一化至 -1 dB
peak_normalized_audio = pyln.normalize.peak(data, -1.0)

# 测量响度
meter = pyln.Meter(rate)
loudness = meter.integrated_loudness(data)

# 响度归一化至 -12 dB LUFS
loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -12.0)

sf.write("./normalized_audio.wav", loudness_normalized_audio, rate)
