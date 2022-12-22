from midi2audio import FluidSynth

audios = []

for audio in audios:
	fs = FluidSynth()
	fs.midi_to_audio(audio + '.midi', audio + '.wav')
