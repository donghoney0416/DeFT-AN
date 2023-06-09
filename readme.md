# DeFT-AN: Dense Frequency-Time Attentive Network for multichannel speech enhancement
[D. Lee and J-W. Choi, "DeFT-AN: Dense Frequency-Time Attentive Network for Multichannel Speech Enhancement," IEEE Signal Processing Letters vol.30, pp.155-159, 2023](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10042963)

![Speech enhancement diagram](speech_enhancement.png)


In this study, we propose a dense frequencytime attentive network (DeFT-AN) for multichannel speech enhancement. DeFT-AN is a mask estimation network that
predicts a complex spectral masking pattern for suppressing the noise and reverberation embedded in the short-time Fourier transform (STFT) of an input signal. The proposed
mask estimation network incorporates three different types of blocks for aggregating information in the spatial, spectral, and temporal dimensions. It utilizes a spectral transformer
with a modified feed-forward network and a temporal conformer with sequential dilated convolutions. The use of dense blocks and transformers dedicated to the three different characteristics of audio signals enables more comprehensive enhancement in noisy and reverberant environments. The remarkable performance of DeFT-AN over
state-of-the-art multichannel models is demonstrated based on two popular noisy and reverberant datasets in terms of various metrics for speech quality and intelligibility

http://www.sound.kaist.ac.kr
