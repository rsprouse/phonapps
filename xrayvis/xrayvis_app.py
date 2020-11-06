# The xrayvis bokeh app

import os
import numpy as np
import pandas as pd
import requests
import yaml
from tempfile import TemporaryDirectory, NamedTemporaryFile
from base64 import b64decode
import parselmouth

from bokeh_phon.utils import remote_jupyter_proxy_url_callback, set_default_jupyter_url
from bokeh_phon.models.audio_button import AudioButton
from phonlab.array import nonzero_groups

from bokeh.core.query import find
from bokeh.plotting import figure
from bokeh.colors import RGB
from bokeh.models import BoxAnnotation, BoxSelectTool, BoxZoomTool, Button, Circle, \
    ColumnDataSource, CrosshairTool, Div, FileInput, HoverTool, LinearColorMapper, \
    LogColorMapper, MultiLine, MultiSelect, RadioButtonGroup, Range1d, RangeSlider, \
    Select, Slider, Span, Spinner, PanTool, ResetTool, TapTool, \
    WheelZoomTool, ZoomInTool, ZoomOutTool
from bokeh.models.widgets import DataTable, NumberFormatter, Panel, Tabs, TableColumn
from bokeh.io import show, output_notebook, push_notebook
from bokeh.layouts import column, gridplot, row
from bokeh.events import MouseMove, SelectionGeometry, Tap
from bokeh.transform import linear_cmap
from bokeh.palettes import Greens, Greys, Greys256, Reds
r_Greens9 = list(reversed(Greens[9]))
r_Greys9 = list(reversed(Greys[9]))
r_Greys256 = list(reversed(Greys256))
r_Reds9 = list(reversed(Reds[9]))

# The remote_jupyter_proxy_url function is required when running on a BinderHub instance.
# Use the set_default_jupyter_url function to set the hostname of your instance after it has
# started. The current value is the most frequent result when launching from mybinder.org.
# Change to another value if running Binder host is not this most common one:
# set_default_jupyter_url('https://datahub.berkeley.edu/')

# Or if you are running locally (on localhost), ignore the previous line and set
# `local_notebook` to True:
# local_notebook = False

output_notebook()

# bad values in .txy files are 1000000 (scaled to 1000)
# TODO:
# when bokeh can handle plots with NaN, use that to filter instead of badval
badval = 1000

params = {
    'low_thresh_color': 'white',
    'low_thresh_power': 3.5,
    'window_size': 5.0,
    'spslice_lastx': 0.0,
    'downsample_rate': 20000
}
snd = None
wavname = None
dfs = {}

# Info for loading/caching audio files
tempdirobj = TemporaryDirectory()
tempdir = tempdirobj.name
resource_url = 'https://linguistics.berkeley.edu/phonapps/resource/'
manifest_name = 'manifest.yaml'
manifest_key = 'resources'

timesource = ColumnDataSource(
        {
            'T1x': [], 'T1y': [], 'T2x': [], 'T2y': [],
            'T3x': [], 'T3y': [], 'T4x': [], 'T4y': [],
            'ULx': [], 'ULy': [], 'LLx': [], 'LLy': [],
            'MIx': [], 'MIy': [], 'MMx': [], 'MMy': [],
            'color': [], 'sec': []
        }
    )

allwddf = pd.read_feather('all_words.feather')
allphdf = pd.read_feather('all_phones.feather')
fileopt_demo_dtypes = {
    'speaker': 'category', 'wavname': 'category', 'uttid': 'category', 'rep': 'category',
    'bytes': np.int16,
    'subject': 'category', 'sex': 'category', 'dialect_base_state': 'category', 'dialect_base_city': 'category'
}
fileoptsdf = pd.read_csv('file_opts.csv', dtype=fileopt_demo_dtypes)
demo = pd.merge(
    pd.read_csv('speaker_demographics1.csv', dtype=fileopt_demo_dtypes),
    pd.read_csv('speaker_demographics2.csv', dtype=fileopt_demo_dtypes),
    on='subject'
)
# Remove subjects that are not in the transcribed speaker set.
demo = demo[demo.subject.isin(allphdf.speaker.cat.categories)]

def snd2specgram(snd, winsize, pre_emphasize=True):
    '''Return a spectrogram created from snd.'''
    if pre_emphasize is True:
        specsnd = snd.copy()
        specsnd.pre_emphasize()
    else:
        specsnd = snd
    return specsnd.to_spectrogram(window_length=winsize, maximum_frequency=params['downsample_rate']/2)

def get_cached_fname(fname, base_url, speaker):
    '''Get filename from tempdir, or base_url and fname if not already in cache.'''
# Load from e.g. https://linguistics.berkeley.edu/phonapps/xray_microbeam_database/JW11/tp001.txy
    tempspkrdir = os.path.join(tempdir, speaker)
    os.makedirs(tempspkrdir, exist_ok=True)
    cachefile = os.path.join(tempspkrdir, fname)
    if not os.path.isfile(cachefile):
        if not base_url.endswith('/'):
            base_url += '/'
        # Get .wav
        r = requests.get(base_url + fname)
        tempwav = os.path.join(tempspkrdir, 'temp.wav')
        with open(tempwav, 'wb') as twav:
            twav.write(r.content)
        tempsnd = parselmouth.Sound(tempwav)
        ds_snd = tempsnd.resample(params['downsample_rate'], 50)
        ds_snd.save(cachefile, parselmouth.SoundFileFormat.WAV)
        # Get .txy
        txyfile = cachefile.replace('.wav', '.txy')
        r = requests.get(base_url + fname.replace('.wav', '.txy'))
        with open(txyfile, 'wb') as tfile:
            tfile.write(r.content)
    for f in ('PAL.DAT', 'PHA.DAT'):
        landmarkfile = os.path.join(tempspkrdir, f)
        if not os.path.isfile(landmarkfile):
            r = requests.get(base_url + f)
            with open(landmarkfile, 'wb') as tfile:
               tfile.write(r.content)
    return cachefile

def xrayvis_app(doc):
    def load_wav_cb(attr, old, new):
        '''Handle selection of audio file to be loaded.'''
        if new == '':
            return
        global wavname
        global snd
        spkr, fname = os.path.split(new)
        wavname = get_cached_fname(
            fname, 
            f'https://linguistics.berkeley.edu/phonapps/xray_microbeam_database/{spkr}',
            spkr
        )
#        wavname = new
        if not wavname.endswith('.wav'):
            return
        snd = parselmouth.Sound(wavname)
        srcdf = pd.DataFrame(dict(
            seconds=snd.ts().astype(np.float32),
            ch0=snd.values[0,:].astype(np.float32),
        ))
#! TODO: do file caching
        phdf = allphdf.loc[allphdf.wavpath == new, :].copy()
        phdf['t1'] = phdf['t1'].astype(np.float32)
        wddf = allwddf.loc[allwddf.wavpath == new, :].copy()
        wddf['t1'] = wddf['t1'].astype(np.float32)
        uttdiv.text = '<b>Utterance:</b> ' + ' '.join(wddf.word.str.replace('sp', '')).strip()
        phwddf = pd.merge_asof(
            phdf[['t1', 'phone']],
            wddf[['t1', 'word']],
            on='t1',
            suffixes=['_ph', '_wd']
        )
# TODO: separate t1_ph and t1_wd columns
        srcdf = pd.merge_asof(srcdf, phwddf, left_on='seconds', right_on='t1')
        srcdf[['phone', 'word']] = srcdf[['phone', 'word']].fillna('')
        srcdf = srcdf.drop('t1', axis='columns')
        dfs['srcdf'] = srcdf
        source.data = srcdf
        tngsource.data = {'x': [], 'y': []}
        othsource.data = {'x': [], 'y': []}
        timesource.data = {k: [] for k in timesource.data.keys()}
        lasttngtimesource.data = {'x': [], 'y': []}
        lastothtimesource.data = {'x': [], 'y': []}
        playvisbtn.channels = channels
        playvisbtn.disabled = False
        playselbtn.channels = channels
        playselbtn.disabled = False
        playvisbtn.fs = snd.sampling_frequency
        playvisbtn.start = snd.start_time
        playvisbtn.end = snd.end_time
        playselbtn.fs = snd.sampling_frequency
        playselbtn.start = 0.0
        playselbtn.end = 0.0
        selbox.left = 0.0
        selbox.right = 0.0
        selbox.visible = False
        cursor.location = 0.0
        cursor.visible = False
        ch0.visible = True
        update_sgram()
        load_artic()
        set_limits(0.0, srcdf['seconds'].max())

    def load_artic():
        '''Load articulation data.'''
        trace.title.text = 'Static trace'
        traj.title.text = 'Trajectories'
        tngfile = os.path.splitext(wavname)[0] + '.txy'
        palfile = os.path.join(os.path.dirname(wavname), 'PAL.DAT')
        phafile = os.path.join(os.path.dirname(wavname), 'PHA.DAT')
        tngdf = pd.read_csv(
                tngfile,
                sep='\t',
                names=[
                    'sec', 'ULx', 'ULy', 'LLx', 'LLy', 'T1x', 'T1y', 'T2x', 'T2y',
                    'T3x', 'T3y', 'T4x', 'T4y', 'MIx', 'MIy', 'MMx', 'MMy'
                ]
            )
        # Convert to seconds
        tngdf['sec'] = tngdf['sec'] / 1e6
        tngdf = tngdf.set_index(['sec'])
        # Convert to mm
        tngdf[[
            'ULx', 'ULy', 'LLx', 'LLy', 'T1x', 'T1y', 'T2x', 'T2y',
            'T3x', 'T3y', 'T4x', 'T4y', 'MIx', 'MIy', 'MMx', 'MMy'
        ]] = tngdf[[
            'ULx', 'ULy', 'LLx', 'LLy', 'T1x', 'T1y', 'T2x', 'T2y',
            'T3x', 'T3y', 'T4x', 'T4y', 'MIx', 'MIy', 'MMx', 'MMy'
        ]] * 1e-3
        # Find global x/y max/min in this recording to set axis limits.
        # Exclude bad values (1000000 in data file; 1000 mm in scaled dataframe).
        cmpdf = tngdf[tngdf < badval]
        xmax = np.max(
            np.max(
                cmpdf[['ULx','LLx','T1x', 'T2x', 'T3x', 'T4x', 'MIx', 'MMx']]
            )
        )
        xmin = np.min(
            np.min(
                cmpdf[['ULx','LLx','T1x', 'T2x', 'T3x', 'T4x', 'MIx', 'MMx']]
            )
        )
        ymax = np.max(
            np.max(
                cmpdf[['ULy','LLy','T1y', 'T2y', 'T3y', 'T4y', 'MIy', 'MMy']]
            )
        )
        ymin = np.min(
            np.min(
                cmpdf[['ULy','LLy','T1y', 'T2y', 'T3y', 'T4y', 'MIy', 'MMy']]
            )
        )
    
        paldf = pd.read_csv(palfile, sep='\s+', header=None, names=['x', 'y'])
        paldf = paldf * 1e-3
        palsource.data = {'x': paldf['x'], 'y': paldf['y']}
        phadf = pd.read_csv(phafile, sep='\s+', header=None, names=['x', 'y'])
        phadf = phadf * 1e-3
        phasource.data = {'x': phadf['x'], 'y': phadf['y']}
    
        xmin = np.min([xmin, np.min(paldf['x']), np.min(phadf['x'])])
        xmax = np.max([xmax, np.max(paldf['x']), np.max(phadf['x'])])
        ymin = np.min([ymin, np.min(paldf['y']), np.min(phadf['y'])])
        ymax = np.max([ymax, np.max(paldf['y']), np.max(phadf['y'])])
        xsz = xmax - xmin
        ysz = ymax - ymin
        xrng = [xmin - (xsz * 0.05), xmax + (xsz * 0.05)]
        yrng = [ymin - (ysz * 0.05), ymax + (ysz * 0.05)]
        dfs['tngdf'] = tngdf
        dfs['paldf'] = paldf
        dfs['phadf'] = phadf

    def update_sgram():
        '''Update spectrogram based on current values.'''
        if snd.end_time < 15:
            sgrams[0] = snd2specgram(snd, 0.005)
            specsource.data = dict(
                sgram0=[sgrams[0].values.astype(np.float32)]
            )
            spec0img.glyph.dw = sgrams[0].x_grid().max()
            spec0img.glyph.dh = sgrams[0].y_grid().max()
            spec0cmap.low = _low_thresh()
            spec0.visible = True
        else:
            specsource.data = dict(
                sgram0=[]
            )
            spec0.visible = False

    def update_trace():
        '''Update the static trace at the cursor time.'''
        trace.title.text = f'Static trace ({cursor.location:0.4f})'
        tidx = dfs['tngdf'].index.get_loc(cursor.location, method='nearest')
        row = dfs['tngdf'].iloc[tidx]
        tngsource.data = {
            'x': [row.T1x, row.T2x, row.T3x, row.T4x],
            'y': [row.T1y, row.T2y, row.T3y, row.T4y]
        }
        othsource.data = {
            'x': [row.ULx, row.LLx, row.MIx, row.MMx],
            'y': [row.ULy, row.LLy, row.MIy, row.MMy]
        }

    def update_traj():
        '''Update the trajectories during the selected time range.'''
        traj.title.text = f'Trajectories ({selbox.left:0.4f} - {selbox.right:0.4f})'
        seldf = dfs['tngdf'].loc[
            (dfs['tngdf'].index >= selbox.left) & (dfs['tngdf'].index <= selbox.right)
        ]
        dfs['seldf'] = seldf
        pts = (
            'T1x', 'T1y', 'T2x', 'T2y', 'T3x', 'T3y', 'T4x', 'T4y',
            'ULx', 'ULy', 'LLx', 'LLy', 'MIx', 'MIy', 'MMx', 'MMy'
        )
        # Create a list of line segments for each tracked element.
        newdata = {
            pt: list(np.squeeze(np.dstack((seldf[pt].iloc[:-1], seldf[pt].iloc[1:])))) \
                for pt in pts
        }
        newdata['color'] = np.arange(1, len(seldf))
        newdata['sec'] = seldf.index[1:]
        timesource.data = newdata
        anim_slider.start = seldf.index[0]
        anim_slider.end = seldf.index[-1]
        anim_slider.step = np.diff(newdata['sec']).mean()
        anim_slider.value = anim_slider.end
        anim_slider.disabled = False
        anim_btn.disabled = False
        lastrow = seldf.iloc[-1]
        lasttngtimesource.data = {
            'x': [lastrow.T1x, lastrow.T2x, lastrow.T3x, lastrow.T4x],
            'y': [lastrow.T1y, lastrow.T2y, lastrow.T3y, lastrow.T4y]
        }
        lastothtimesource.data = {
            'x': [lastrow.ULx, lastrow.LLx, lastrow.MIx, lastrow.MMx],
            'y': [lastrow.ULy, lastrow.LLy, lastrow.MIy, lastrow.MMy]
        }

    # TODO: this is a workaround until we can set x_range, y_range directly
    # See https://github.com/bokeh/bokeh/issues/4014
    def set_limits(xstart, xend):
        '''Set axis limits.'''
        ch0.x_range.start = xstart
        ch0.x_range.end = xend
        ch0.axis[0].bounds = (xstart, xend)

    def update_select_widgets(clicked_x=None):
        '''Update widgets based on current selection. Use the clicked_x param to
        designate the cursor location if this function is called as the result of
        a Tap event. If clicked_x is None, then use the existing cursor location
        to set the center of the selection.'''
        mode = selmodebtn.labels[selmodebtn.active]
        if clicked_x is None and cursor.visible:
            x_loc = cursor.location
        elif clicked_x is not None:
            x_loc = clicked_x
        else:
            return
        if mode == '200ms':
            start = x_loc - 0.100
            end = x_loc + 0.100
            cursor.location = x_loc
        else:  # 'word' or 'phone'
            idx = np.abs(source.data['seconds'] - x_loc).argmin()
# TODO: clean up the redundancy
            fld = {'word': 'word', 'phone': 'phone'}[mode]
            label = source.data[fld][idx]
            indexes = nonzero_groups(source.data[fld]==label, include_any=idx)
            secs = source.data['seconds'][indexes]
            start = secs.min()
            end = secs.max()
            cursor.location = secs.mean()
        playselbtn.start = start
        playselbtn.end = end
        selbox.left = start
        selbox.right = end
        selbox.visible = True
        cursor.visible = True

    def spkr_select_cb(attr, old, new):
        '''Respond to changes in speaker multiselect.'''
        try:
            spkrs = demo[
                (demo.sex.isin(sex_select.value) \
                & demo.dialect_base_state.isin(state_select.value) \
                & (demo.dialect_base_city.isin(city_select.value)))
            ].subject.unique()
            new_opts = [''] + [
                (f.value, f.label) for f in fileoptsdf[fileoptsdf.speaker.isin(spkrs)].itertuples()
            ]
            fselect.options = new_opts
            fselect.value = ''
        except NameError as e:
            pass   # Values not set yet, so ignore

    def cursor_cb(e):
        '''Handle cursor mouse click in the waveform.'''
        update_select_widgets(clicked_x=e.x)
        update_trace()
        update_traj()

    def x_range_cb(attr, old, new):
        '''Handle change of x range in waveform/spectrogram.'''
        if attr == 'start':
            playvisbtn.start = new
        elif attr == 'end':
            playvisbtn.end = new

    def selection_cb(e):
        '''Handle data range selection event.'''
#! TODO: handle situation in which selection is too short, i.e. len(seldf) <= 1
        cursor.location = (e.geometry['x0'] + e.geometry['x1']) / 2
        cursor.visible = True
        playselbtn.start = e.geometry['x0']
        playselbtn.end = e.geometry['x1']
        selbox.left = e.geometry['x0']
        selbox.right = e.geometry['x1']
        selbox.visible = True
        update_trace()
        update_traj()

    def selmode_cb(attr, old, new):
        '''Handle change in click selection value.'''
        update_select_widgets(clicked_x=None)

    def anim_cb(attr, old, new):
        '''Handle change in the animation slider.'''
        idx = np.argmin(np.abs(timesource.data['sec'] - new))
        n = len(timesource.data['color'])
        active = np.arange(n - idx, n + 1)
        timesource.data['color'] = np.pad(
            active, (0, n - len(active)), constant_values=0
        )
        anim_cmap = LinearColorMapper(palette=r_Greys256, low=1, high=n+1, low_color='white')
        for tag, palette in (('anim_tng', r_Reds9), ('anim_oth', r_Greens9)):
            for a in find(traj.references(), {'tags': tag}):
                a.line_color = linear_cmap('color', palette, low=1, high=n+1, low_color='white')
        lasttngtimesource.data = {
            'x': [timesource.data[pt][idx][1] for pt in ('T1x', 'T2x', 'T3x', 'T4x')],
            'y': [timesource.data[pt][idx][1] for pt in ('T1y', 'T2y', 'T3y', 'T4y')]
        }
        lastothtimesource.data = {
            'x': [timesource.data[pt][idx][1] for pt in ('ULx', 'LLx', 'MIx', 'MMx')],
            'y': [timesource.data[pt][idx][1] for pt in ('ULy', 'LLy', 'MIy', 'MMy')]
        }

    def anim_btn_cb():
        '''Handle click of anim_btn animate trajectories of selected audio.'''
        values = np.linspace(anim_slider.start, anim_slider.end, len(timesource.data['T1x']))
        for v in values:
            anim_slider.value = v

    def low_thresh_cb(attr, old, new):
        '''Handle change in threshold slider to fade out low spectrogram values.'''
        params['low_thresh_power'] = new
        lt = _low_thresh()
        spec0cmap.low = lt

    def _low_thresh():
        return sgrams[0].values.min() \
               + sgrams[0].values.std()**params['low_thresh_power']

    step = None
    rate = orig_rate = None
#    dfs = {}
    xrng = []
    yrng = []
    width = 1000
    height = 200
    cutoff = 50
    order = 3
    tngcolor = 'DarkRed'
    othcolor = 'Indigo'
    fselect = Select(options=[], value='')
    fselect.on_change('value', load_wav_cb)
    sex_select = MultiSelect(
        options=[('F', 'female'), ('M', 'male')],
        value=['F', 'M']
    )
    state_select = MultiSelect(
        options=list(demo.dialect_base_state.cat.categories),
        value=list(demo.dialect_base_state.cat.categories)
    )
    city_select = MultiSelect(
        options=list(demo.dialect_base_city.cat.categories),
        value=list(demo.dialect_base_city.cat.categories)
    )
    sex_select.on_change('value', spkr_select_cb)
    state_select.on_change('value', spkr_select_cb)
    city_select.on_change('value', spkr_select_cb)
    spkr_select_cb('', '', '')

    source = ColumnDataSource(data=dict(seconds=[], ch0=[]))
    channels = ['ch0']

    playvisbtn = AudioButton(
        label='Play visible signal', source=source, channels=channels,
        width=120, disabled=True
    )
    playselbtn = AudioButton(
        label='Play selected signal', source=source, channels=channels,
        width=120, disabled=True
    )
    selmodebtn = RadioButtonGroup(labels=['200ms', 'word', 'phone'], active=1)
    selmodebtn.on_change('active', selmode_cb)
    # Instantiate and share specific select/zoom tools so that
    # highlighting is synchronized on all plots.
    boxsel = BoxSelectTool(dimensions='width')
    spboxsel = BoxSelectTool(dimensions='width')
    boxzoom = BoxZoomTool(dimensions='width')
    zoomin = ZoomInTool(dimensions='width')
    zoomout = ZoomOutTool(dimensions='width')
    crosshair = CrosshairTool(dimensions='height')
    shared_tools = [
        'xpan', boxzoom, boxsel, crosshair, zoomin, zoomout, 'reset'
    ]

    uttdiv = Div(text='')

    figargs = dict(
        tools=shared_tools,
    )
    cursor = Span(dimension='height', line_color='red', line_dash='dashed', line_width=1)
    wavspec_height = 280
    ch0 = figure(
        name='ch0',
        tooltips=[('time', '$x{0.0000}'), ('word', '@word'), ('phone', '@phone')],
        height=wavspec_height,
        **figargs
    )
    ch0.toolbar.logo = None
    ch0.line(x='seconds', y='ch0', source=source, nonselection_line_alpha=0.6)
    # Link pan, zoom events for plots with x_range.
    ch0.x_range.on_change('start', x_range_cb)
    ch0.x_range.on_change('end', x_range_cb)
    ch0.on_event(SelectionGeometry, selection_cb)
    ch0.on_event(Tap, cursor_cb)
    ch0.add_layout(cursor)
    wavtab = Panel(child=ch0, title='Waveform')
    selbox = BoxAnnotation(
        name='selbox',
        left=None, right=None,
        fill_color='green', fill_alpha=0.1,
        line_color='green', line_width=1.5, line_dash='dashed',
        visible=False
    )
    ch0.add_layout(selbox)
    sgrams = [np.ones((1, 1))]
    specsource = ColumnDataSource(data=dict(sgram0=[sgrams[0]]))
    spec0 = figure(
        name='spec0',
        x_range=ch0.x_range, # Keep times synchronized
        tooltips=[("time", "$x{0.0000}"), ("freq", "$y{0.0000}"), ("value", "@sgram0{0.000000}")],
        height=wavspec_height,
        **figargs
    )
    spec0.toolbar.logo = None
    spec0.x_range.on_change('start', x_range_cb)
    spec0.x_range.on_change('end', x_range_cb)
    spec0.on_event(SelectionGeometry, selection_cb)
    spec0.on_event(Tap, cursor_cb)
    spec0.add_layout(cursor)
    spec0.x_range.range_padding = spec0.y_range.range_padding = 0
    spec0cmap = LogColorMapper(palette=r_Greys256, low_color=params['low_thresh_color'])
    low_thresh_slider = Slider(
        start=1.0, end=12.0, step=0.03125, value=params['low_thresh_power'],
        title='Spectrogram threshold'
    )
    low_thresh_slider.on_change('value', low_thresh_cb)
    spec0img = spec0.image(
        image='sgram0',
        x=0, y=0,
        color_mapper=spec0cmap,
        level='image',
        source=specsource
    )
    spec0.grid.grid_line_width = 0.0
    spec0.add_layout(selbox)
    sgramtab = Panel(child=spec0, title='Spectrogram')

    tngsource = ColumnDataSource(data={'x': [], 'y': []})
    othsource = ColumnDataSource(data={'x': [], 'y': []})
    timesource = ColumnDataSource({
        'T1x': [], 'T1y': [], 'T2x': [], 'T2y': [],
        'T3x': [], 'T3y': [], 'T4x': [], 'T4y': [],
        'ULx': [], 'ULy': [], 'LLx': [], 'LLy': [],
        'MIx': [], 'MIy': [], 'MMx': [], 'MMy': [],
        'color': [], 'sec': []
    })
    lasttngtimesource = ColumnDataSource(data={'x': [], 'y': []})
    lastothtimesource = ColumnDataSource(data={'x': [], 'y': []})
    palsource = ColumnDataSource(pd.DataFrame({'x': [], 'y': []}))
    phasource = ColumnDataSource(pd.DataFrame({'x': [], 'y': []}))

    trace = figure(
        width=500, height=300,
        title='Static trace',
        x_range=(-100.0,25.0), y_range=(-37.650,37.650),
        tools=[],
        tags=['xray', 'static_fig']
    )
    trace.toolbar.logo = None
    trace.circle('x', 'y', source=tngsource, size=3, color=tngcolor, tags=['update_xray'])
    trace.circle('x', 'y', source=othsource, size=3, color=othcolor, tags=['update_xray'])
    trace.line('x', 'y', source=tngsource, color=tngcolor, tags=['update_xray'])
    trace.line('x', 'y', source=palsource, color='black')
    trace.line('x', 'y', source=phasource, color='black')

    traj = figure(
        width=500, height=300,
        title='Trajectories',
        x_range=(-100.0,25.0), y_range=(-37.650,37.650),
        tools=[],
        tags=['xray', 'trajectory_fig']
    )
    traj.toolbar.logo = None
    traj.multi_line('T1x', 'T1y', source=timesource, tags=['anim_tng'])
    traj.multi_line('T2x', 'T2y', source=timesource, tags=['anim_tng'])
    traj.multi_line('T3x', 'T3y', source=timesource, tags=['anim_tng'])
    traj.multi_line('T4x', 'T4y', source=timesource, tags=['anim_tng'])
    traj.multi_line('ULx', 'ULy', source=timesource, tags=['anim_oth'])
    traj.multi_line('LLx', 'LLy', source=timesource, tags=['anim_oth'])
    traj.multi_line('MIx', 'MIy', source=timesource, tags=['anim_oth'])
    traj.multi_line('MMx', 'MMy', source=timesource, tags=['anim_oth'])
    traj.circle('x', 'y', source=lasttngtimesource, color=tngcolor)
    traj.circle('x', 'y', source=lastothtimesource, color=othcolor)
    traj.line('x', 'y', source=lasttngtimesource, color='lightgray')
    traj.line('x', 'y', source=palsource, color='black')
    traj.line('x', 'y', source=phasource, color='black')

    anim_slider = Slider(
        start=0, end=100, step=1, value=0, 
        width=240, format='0.000f',
        title='Selected trajectory', orientation='horizontal', disabled=True
    )
    anim_slider.on_change('value', anim_cb)
    anim_btn = Button(label='Animate', width=120, disabled=True)
    anim_btn.on_click(anim_btn_cb)

    audtabs = Tabs(tabs=[wavtab, sgramtab])
    mainLayout = column(
        row(sex_select, state_select, city_select),
        row(fselect),
        row(
            column(uttdiv, audtabs),
            column(
#!                row(anim_slider, anim_btn),
                column(Div(text='Click selection mode:'), selmodebtn, low_thresh_slider),
                row(playvisbtn, playselbtn))
        ),
        row(trace, traj),
        name='mainLayout'
    )
    doc.add_root(mainLayout)
    return doc
