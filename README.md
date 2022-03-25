# ling120-final-project
My final project for linguistics 120

For this project I used [Mozilla's Common Voice Project][1]
to see if I could create an AI to generate speech better than the current state of the art.

## Background
There are two common technologies for synthesizing speech, concatenation synthesis, and format synthesis.
For this project format synthesis was used because it does not rely on prerecorded samples ([wikipedia][4]).
Format synthesis relies on spectrograms and autoencoders to generate speech. 
A form of this method of synethsis is known as deep learning synthesis. Which is what will be focused on, 
this tries to model the human voice by training deep learning models ([wikipedia][4]).

### Diffculties

One of the most common difficulties for text to speech is text normalization, wikipedia specifies that the majority of systems do the following,

*"Most text-to-speech (TTS) systems do not generate semantic representations of their input texts, 
as processes for doing so are unreliable, poorly understood, and computationally ineffective. As a result, 
various heuristic techniques are used to guess the proper way to disambiguate homographs, 
like examining neighboring words and using statistics about frequency of occurrence."* ([wikipedia][4]) 

In contrast to my methodology, I do plan to create a semantic representation of the input texts, 
this will be done using a combination of both automated and human methods.

Valardo also metions that long distance relationships are often difficult to simulate using raw text data, 
these include things like pitch, prosody, rhythm, etc ([Valardo][3]).

### Spectrograms

Raw sound data is very dense, thus spectrograms are used. 
A spectrogram is a lossy compression method that squeezes a short
sequence of sound data into a Fast Fourier Transform (FFT) which results in an intensity graph displaying 
frequency components of the sound sample given. 
([Valardo][3])

### Autoencoders

An autoencoder is a method of machine learning that uses spectrograms to generate speech. 
The basic concept is that you can train a model to compress data by 
creating an architecture that consists of an input layer, 
an output layer that is the same size as the input layer, 
and a hidden layer with less nodes than the input layer, 
Valardo refers to this hidden layer as a bottleneck. 
A helpful way to think about it is to consider two different combinations of these layers, the encoder, 
representing the input and the hidden layer, and the decoder, representing the hidden layer and the output layer.
The encoder is responsible for compression of the sound data and the decoder is responsible for the decompression of 
the sound data. For training an autoencoder, it is actually pretty ingenious, audio data is fed into the encoder, 
and the output is expected to be the same, this forces the model to figure out how to compress data losslessly.
([Valardo][2])

## Proposed Methodology

My novel methodology in this experiment will try to use raw speech data instead of spectrograms on the output side of 
an autoencoder to see if I can make it sound more human than the current state of the art.

## Usage
### [ipa.py](./ipa.py)
This file is used to append Mozilla's TSV files with the International Phonetic Alphabet (IPA) of the sentences.
It makes use of multiprocessing to speed it up, but it still takes around 3 hours for conversion.
```
python ipa.py "path to tsv file" [--batch_size=256]
```
In this example, `batch_size` is the number of entries that are queued up at each process once the task is split.

## Works Cited

“Speech Synthesis.” *Wikipedia*, Wikimedia Foundation, 12 Mar. 2022, [https://en.wikipedia.org/wiki/Speech_synthesis][5]. 

Valardo, Valerio, "Autoencoders Explained Easily", *YouTube*, commentary by Valerio Valardo, 3, December, 2020, [xwrzh4e8DLs][2]

Valardo, Valerio, "Sound Generation with Deep Learning || Approaches and Challenges", *YouTube*, commentary by Valerio Valardo, 20, November, 2020, [pwV8K9wXY2E][3]

[1]: https://commonvoice.mozilla.org/en
[2]: https://youtu.be/xwrzh4e8DLs
[3]: https://youtu.be/pwV8K9wXY2E
[4]: https://en.wikipedia.org/wiki/Speech_synthesis#Synthesizer_technologies
[5]: https://en.wikipedia.org/wiki/Speech_synthesis
[6]: https://librosa.org/doc/latest/index.html