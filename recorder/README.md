# Phoneme Dataset Generator
This program can assist you in generating a phoneme dataset from the p2fa dictionary

## Installation
You can install the dependencies by using `pip install -r requirements.txt`. Please not that in addition to this,
you'll need to have the [HTK](https://htk.eng.cam.ac.uk/), regardless of your OS you'll want to download the HTK 3.4 version,
not 3.4.1. At the bottom of the download screen there's a link to the archive. Then you need to download the .tar.gz
version, the .zip version had issues in the makefiles for me.

### Windows
I found a stack overflow answer that was particularly helpful. You can find it 
[here](https://stackoverflow.com/questions/61765832/htk-installing-in-windows10-not-able-to-find-vc98), as long as the
compilation is successful you can skip the demo portion, make sure to add the `bin.win32` directory to your `PATH`

### Linux

I found the installation instructions on HTK's site to work for me.

## Running
You can run the program by using `python display.py`. Note for this to work you'll need a correctly formatted `settings.json`
file in the `recorder` directory. This json file should look something like this:

```json
{
  "paths": [  // Tells the config method to convert these into pathlib.Path's
    "word_dict",
    "recording_dir"
  ],
  "word_dict": "...\\p2fa_py3\\p2fa\\model\\dict", // This is the path to the p2fa dictionary
  "start_index": 0,                                // Holds the index of the first phrase you haven't done yet
  "recording_seconds": 3,                          // Number of seconds to record for
  "recording_rate": 48000,                         // The sample rate in Hz
  "recording_dir": "path\\to\\phoneme\\folders"    // The directory where you want to store the phonemes
}
```

Once you get it running, you'll be prompted with a word, it's pronunciation (according to the dict),
a graph representation of the sample, along with `pyfoal`'s alignment, and two bars for trimming the sample down.

Once you're done you can either skip or save the sample's phonemes. In addition to this there are also buttons for
clearing/recording your sample and for playing your sample.

## Playback
If you want to test the playback of your phonemes you can, though, I found that a default installation of windows 10
does not include the necessary encoders for playback, the VLC media player worked just fine though.
