# phonapps
Interactive phonetic applications that run in a Jupyter notebook locally or
on a JupyterHub instance, e.g. [Binder](https://mybinder.org).

These apps use Bokeh for interactive plotting.

Try launching the current master branch with Binder and exploring notebooks:

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rsprouse/phonapps/master)

## Try a live notebook in appmode:

A notebook launched in appmode executes all cells and displays the results. Markdown cells are rendered normally, and the code is hidden.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rsprouse/phonapps/master?urlpath=%2Fapps%2Fwaveform_and_spectrogram/waveform_and_spectrogram.ipynb) An app demonstrating client-side (javascript) visualization and playback of an audio waveform and spectrogram

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rsprouse/phonapps/master?urlpath=%2Fapps%2Fspectrum_slice/spectrum_slice.ipynb) Demonstration of the relationship between a spectrogram and a spectrum slice.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rsprouse/phonapps/master?urlpath=%2Fapps%2Fwavelets/explore_wavelets.ipynb) Explore wavelets

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rsprouse/phonapps/master?urlpath=%2Fapps%2Fsimple_wavapp/simple_wavapp.ipynb) Simple app demonstrating client-side (javascript) visualization and playback of an audio waveform (deprecated for using AudioPlot)

## Notes

* Speech analysis is provided by [Praat](https://github.com/praat/praat).

  > Boersma, P., & Weenink, D. (2018). Praat: doing phonetics by computer [Computer program]. Version 6.0.43, retrieved 8 September 2018 from [http://www.praat.org/](http://www.praat.org/)

* Access to Praat's speech analysis routines in Python is provided by [Parselmouth](https://github.com/YannickJadoul/Parselmouth).

  > Jadoul, Y., Thompson, B., & de Boer, B. (2018). Introducing Parselmouth: A Python interface to Praat. *Journal of Phonetics*, *71*, 1-15. https://doi.org/10.1016/j.wocn.2018.07.001

* The audio file [*The North Wind and the Sun*](resource/the_north_wind_and_the_sun.wav) is a [Wikimedia Commons audio file](https://en.wikipedia.org/wiki/File:Recording_of_speaker_of_British_English_(Received_Pronunciation).ogg) licensed by the International Phonetic Association under the [Creative Commons Attribution-Share Alike 3.0 Unported license](https://creativecommons.org/licenses/by-sa/3.0/deed.en).
