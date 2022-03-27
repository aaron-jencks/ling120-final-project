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

## Hidden Markov Models

One common way that people find to process speech is with the use of Hidden Markov Models (HMM).
Specifically in the interest of this project, I'm looking at them for forced phoneme alignment. ([Jiahong][7])

> Hidden Markov models (HMMs), named after the Russian mathematician Andrey Andreyevich Markov, 
> who developed much of relevant statistical theory, are introduced and studied in the early 1970s. 
> They were first used in speech recognition and have been successfully applied to the 
> analysis of biological sequences since late 1980s. Nowadays, they are considered as a specific form of dynamic 
> Bayesian networks, which are based on the theory of Bayes.
> ([Franzese][8])

## Proposed Methodology

I would like to propose a novel solution for speech synthesis that makes use of raw audio instead of spectrograms.
This will allow the models to learn nuances of speech that have been previously unavailable. To train the model,
[Mozilla's Common Voice][1] speech dataset was used.

### Preprocessing

In order to make the dataset easier to train with, phonemic information was applied to the dataset. This consisted of
two things:

1. The python package `eng-to-ipa` was used to add a column to the common voice dataset containing the IPA of each sentence.
2. A combination of [`librosa`][6] and `soundfile` were then used to trim empty audio data off and convert to WAV files.
3. [Penn's p2FA][7] was used to forcibly align the audio with phoneme markup.
4. These forced alignments were broken into their phonemes, and then each phoneme was stored in a directory of the same phonemes for training.

### Training Phoneme Generation

## Usage
### [ipa.py](./ipa.py)
This file is used to append Mozilla's TSV files with the International Phonetic Alphabet (IPA) of the sentences.
It makes use of multiprocessing to speed it up, but it still takes around 3 hours for conversion.
```
python ipa.py "path to tsv file" [--batch_size=256]
```
In this example, `batch_size` is the number of entries that are queued up at each process once the task is split.

## Works Cited

Franzese, Monica. “Hidden Markov Models.” Encyclopedia of Bioinformatics and Computational Biology, edited by Antonalla Luliano, Elsevier Inc., 2019. 

Jiahong Yuan and Mark Liberman. 2008. Speaker identification on the SCOTUS corpus. Proceedings of Acoustics '08.

“Speech Synthesis.” *Wikipedia*, Wikimedia Foundation, 12 Mar. 2022, [https://en.wikipedia.org/wiki/Speech_synthesis][5]. 

Valardo, Valerio, "Autoencoders Explained Easily", *YouTube*, commentary by Valerio Valardo, 3, December, 2020, [xwrzh4e8DLs][2]

Valardo, Valerio, "Sound Generation with Deep Learning || Approaches and Challenges", *YouTube*, commentary by Valerio Valardo, 20, November, 2020, [pwV8K9wXY2E][3]

[1]: https://commonvoice.mozilla.org/en
[2]: https://youtu.be/xwrzh4e8DLs
[3]: https://youtu.be/pwV8K9wXY2E
[4]: https://en.wikipedia.org/wiki/Speech_synthesis#Synthesizer_technologies
[5]: https://en.wikipedia.org/wiki/Speech_synthesis
[6]: https://librosa.org/doc/latest/index.html
[7]: https://babel.ling.upenn.edu/phonetics/old_website_2015/p2fa/index.html
[8]: https://www.sciencedirect.com/topics/medicine-and-dentistry/hidden-markov-model