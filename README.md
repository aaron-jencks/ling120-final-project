# ling120-final-project
My final project for linguistics 120

For this project I used [Mozilla's Common Voice Project][1]
to see if I could create an AI to generate speech better than the current state of the art.

## Background
According to sources I was able to find (see below), the current state of the art in NLG uses a combination of
spectrograms and autoencoders to generate speech. 

### Spectrograms

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

Valardo, Valerio, "Autoencoders Explained Easily", *YouTube*, commentary by Valerio Valardo, 3, December, 2020, [xwrzh4e8DLs][2]

Valardo, Valerio, "Sound Generation with Deep Learning || Approaches and Challenges", *YouTube*, commentary by Valerio Valardo, 20, November, 2020, [pwV8K9wXY2E][3]

[1]: https://commonvoice.mozilla.org/en
[2]: https://youtu.be/xwrzh4e8DLs
[3]: https://youtu.be/pwV8K9wXY2E