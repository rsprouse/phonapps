import os
import numpy as np
import requests
import yaml
from tempfile import TemporaryDirectory, NamedTemporaryFile
from base64 import b64decode
import parselmouth

from bokeh_phon.utils import remote_jupyter_proxy_url_callback, set_default_jupyter_url
from bokeh_phon.models.audio_button import AudioButton

from bokeh.plotting import figure
from bokeh.colors import RGB
from bokeh.models import BoxAnnotation, BoxSelectTool, BoxZoomTool, Circle, \
    ColumnDataSource, CrosshairTool, FileInput, HoverTool, LogColorMapper, Range1d, \
    RangeSlider, Select, Slider, Span, Spinner, ZoomInTool, ZoomOutTool
from bokeh.models.widgets import DataTable, NumberFormatter, TableColumn
from bokeh.io import show, output_notebook, push_notebook
from bokeh.layouts import column, gridplot, row
from bokeh.events import MouseMove, SelectionGeometry, Tap
from bokeh.palettes import Greys256
r_Greys256 = list(reversed(Greys256))

# The remote_jupyter_proxy_url function is required when running on a BinderHub instance.
# Use the set_default_jupyter_url function to set the hostname of your instance after it has
# started. The current value is the most frequent result when launching from mybinder.org.
# Change to another value if running Binder host is not this most common one:
# set_default_jupyter_url('https://datahub.berkeley.edu/')

# Or if you are running locally (on localhost), ignore the previous line and set
# `local_notebook` to True:
# local_notebook = False

output_notebook()


params = {
    'low_thresh_color': RGB(0, 0, 255, 0.25),
    'low_thresh_power': 6.0,
    'window_size': 5.0,
    'spslice_lastx': 0.0,
    'downsample_rate': 20000
}
snd = None
wavname = None
sgrams = []
spslice = None

# Info for loading/caching audio files
tempdirobj = TemporaryDirectory()
tempdir = tempdirobj.name
resource_url = 'https://linguistics.berkeley.edu/phonapps/resource/'
manifest_name = 'manifest.yaml'
manifest_key = 'resources'

def snd2specgram(snd, winsize, pre_emphasize=True):
    '''Return a spectrogram created from snd.'''
    if pre_emphasize is True:
        specsnd = snd.copy()
        specsnd.pre_emphasize()
    else:
        specsnd = snd
    return specsnd.to_spectrogram(window_length=winsize, maximum_frequency=params['downsample_rate']/2)

def get_cached_fname(fname, base_url, tempdir):
    '''Get filename from tempdir, or base_url and fname if not already in cache.'''
    cachefile = os.path.join(tempdir, fname)
    if not os.path.isfile(cachefile):
        if not base_url.endswith('/'):
            base_url += '/'
        r = requests.get(base_url + fname)
        with NamedTemporaryFile() as tempfile:
            tempfile.write(r.content)
            tempsnd = parselmouth.Sound(tempfile.name)
        ds_snd = tempsnd.resample(params['downsample_rate'], 50)
        ds_snd.save(cachefile, parselmouth.SoundFileFormat.WAV)
    return cachefile

def spectrum_slice_app(doc):
    def load_wav_cb(attr, old, new):
        '''Handle selection of audio file to be loaded.'''
        global wavname
        base_url, fname = os.path.split(new)
        wavname = get_cached_fname(fname, base_url, tempdir)        
        if not wavname.endswith('.wav'):
            return
        update_snd()
        playvisbtn.channels = channels
        playvisbtn.visible = True
        playselbtn.channels = channels
        playselbtn.visible = True
        playvisbtn.fs = snd.sampling_frequency
        playvisbtn.start = snd.start_time
        playvisbtn.end = snd.end_time
        playselbtn.fs = snd.sampling_frequency
        playselbtn.start = 0.0
        playselbtn.end = 0.0
        ch0.visible = True
        update_sgram()
        update_spslice(t=None)

    def file_input_cb(attr, old, new):
        '''Handle audio file upload.'''
        with NamedTemporaryFile() as tempfile:
            tempfile.write(b64decode(new))
            tempsnd = parselmouth.Sound(tempfile.name)
        ds_snd = tempsnd.resample(params['downsample_rate'], 50)
        cachefile = os.path.join(tempdir, file_input.filename)
        ds_snd.save(cachefile, parselmouth.SoundFileFormat.WAV)    
        options = fselect.options.copy()
        options += [(cachefile, file_input.filename)]
        fselect.options = options
        fselect.value = fselect.options[-1][0]

    def update_snd():
        '''Update the sound (waveform and audio button).'''
        global snd
        snd = parselmouth.Sound(wavname)
        if snd.n_channels > 1:
            snd = snd.convert_to_mono()
        if filter_sel.value not in ('no filter', 'no filter (clear)') and spselbox.right is not None:
            if filter_sel.value.startswith('stopband'):
                func = 'Filter (stop Hann band)...'
            if filter_sel.value.startswith('passband'):
                func = 'Filter (pass Hann band)...'
            snd = parselmouth.praat.call(
                snd,
                func,
                spselbox.left,
                spselbox.right,
                100.0
            )
        source.data = dict(
            seconds=snd.ts().astype(np.float32),
            ch0=snd.values[0,:].astype(np.float32),
        )

    def update_sgram():
        '''Update spectrogram based on current values.'''
        if filter_sel.value == 'no filter (clear)':
            sgselbox.bottom = None
            sgselbox.top = None
            sgselbox.visible = False
        else:
            sgselbox.visible = True
        sgrams[0] = snd2specgram(
            snd,
            winsize=winsize_slider.value * 10**-3
        )
        specsource.data = dict(
            sgram0=[sgrams[0].values.astype(np.float32)]
        )
        spec0img.glyph.dw = sgrams[0].x_grid().max()
        spec0img.glyph.dh = sgrams[0].y_grid().max()
        spec0cmap.low = _low_thresh()
        spec0.visible = True

    def update_spslice(t=None):
        '''Update spslice plot with spectrum slice at time t.'''
        if t is not None:
            slidx = np.round(
                parselmouth.praat.call(sgrams[0], 'Get frame number from time...', t)
            ).astype(int)
            spslice = sgrams[0].values[:, slidx]
            spdata = dict(
                freq=np.arange(sgrams[0].values.shape[0]) * sgrams[0].dy,
                power=spslice
            )
            spselbox.visible = True
        else:
            spdata = dict(freq=np.array([]), power=np.array([]))
            spec0_fq_marker.visible = False
            spslice0_fq_marker.visible = False
            spselbox.visible = False
        if filter_sel.value == 'no filter (clear)':
            spselbox.left = None
            spselbox.right = None
            spselbox.visible = False
        spslice_source.data = spdata
        spslice0.x_range = Range1d(0.0, sgrams[0].get_highest_y())
        spslice0.y_range = Range1d(0.0, sgrams[0].get_maximum())
        thresh_box.top = _low_thresh()
        try:
            fqidx = np.abs(
                spslice_source.data['freq'] - fq_marker_source.data['freq'][0]
            ).argmin()
            fq_marker_source.data['power'] = [spslice_source.data['power'][fqidx]]
        except ValueError:
            pass # Not set yet

    def cursor_cb(e):
        '''Handle cursor mouse click that creates the spectrum slice.'''
        cursor.location = e.x
        update_spslice(t=e.x)
        idx = np.abs(spslice_source.data['freq'] - e.y).argmin()
        fq_marker_source.data = dict(
            freq=[e.y],
            power=[spslice_source.data['power'][idx]],
            time=[e.x]
        )
        params['spslice_lastx'] = e.y
        spec0_fq_marker.visible = True
        spslice0_fq_marker.visible = True

    def spslice_move_cb(e):
        '''Handle a MouseMove event on spectrum slice crosshair tool.'''
        try:
            if params['spslice_lastx'] != e.x and e.x >= 0 and e.x <= spslice_source.data['freq'][-1]:
                params['spslice_lastx'] = e.x
                idx = np.abs(spslice_source.data['freq'] - e.x).argmin()
                fq_marker_source.data['freq'] = [spslice_source.data['freq'][idx]]
                fq_marker_source.data['power'] = [spslice_source.data['power'][idx]]
        except IndexError:  # data not loaded yet
            pass

    def x_range_cb(attr, old, new):
        '''Handle change of x range in waveform/spectrogram.'''
        if attr == 'start':
            playvisbtn.start = new
        elif attr == 'end':
            playvisbtn.end = new

    def selection_cb(e):
        '''Handle data range selection event.'''
        playselbtn.start = e.geometry['x0']
        playselbtn.end = e.geometry['x1']
        selbox.left = e.geometry['x0']
        selbox.right = e.geometry['x1']
        selbox.visible = True

    def low_thresh_cb(attr, old, new):
        '''Handle change in threshold slider to fade out low spectrogram values.'''
        params['low_thresh_power'] = new
        lt = _low_thresh()
        spec0cmap.low = lt
        thresh_box.top = lt

    def _low_thresh():
        return sgrams[0].values.min() \
               + sgrams[0].values.std()**params['low_thresh_power']

    def winsize_cb(attr, old, new):
        '''Handle change in winsize slider to change spectrogram analysis window.'''
        params['window_size'] = new
        update_sgram()
        if cursor.location is not None:
            update_spslice(t=cursor.location)
            idx = np.abs(spslice_source.data['freq'] - params['spslice_lastx']).argmin()
            fq_marker_source.data = dict(
                freq=[spslice_source.data['freq'][idx]],
                power=[spslice_source.data['power'][idx]],
                time=[cursor.location]
            )

    def filter_sel_cb(e):
        '''Handle change of filter range.'''
        lowfq = e.geometry['x0']
        highfq = e.geometry['x1']
        sgselbox.bottom = lowfq
        sgselbox.top = highfq
        spselbox.left = lowfq
        spselbox.right = highfq
        range_text = f' ({lowfq:.0f}-{highfq:.0f} Hz)'
        # Force assignment of new options so that Bokeh detects the values have changed
        # and synchronizes the JS.
        options = filter_sel.options.copy()
        for idx, opt in enumerate(options):
            if 'stopband' in opt:
                options[idx] = f'stopband {range_text}'
                if 'stopband' in filter_sel.value:
                    filter_sel.value = options[idx]
            if 'passband' in opt:
                options[idx] = f'passband {range_text}'
                if 'passband' in filter_sel.value:
                    filter_sel.value = options[idx]
        filter_sel.options = options
        update_snd()
        update_sgram()
        update_spslice(t=cursor.location)

    def filter_type_cb(attr, old, new):
        '''Handle change in filter type.'''
        if 'clear' in new:
            # Force assignment of new options so that Bokeh detects the values have changed
            # and synchronizes the JS.
            options = filter_sel.options.copy()
            for idx, opt in enumerate(options):
                if 'passband' in opt:
                    options[idx] = 'passband'
                    if 'passband' in filter_sel.value:
                        filter_sel.value = 'passband'
                if 'stopband' in opt:
                    options[idx] = 'stopband'
                    if 'stopband' in filter_sel.value:
                        filter_sel.value = 'stopband'
            filter_sel.options = options
        update_snd()
        update_sgram()
        update_spslice(t=cursor.location)

    manifest_text = requests.get(resource_url + manifest_name).text
    manifest = yaml.safe_load(manifest_text)[manifest_key]
    options = [('', 'Choose an audio file to display')] + [
        (resource_url + opt['fname'], opt['label']) for opt in manifest
    ]
    fselect = Select(options=options, value='')
    fselect.on_change('value', load_wav_cb)
    file_input = FileInput(accept=".wav")
    fselect_row = row(fselect, file_input)
    file_input.on_change('value', file_input_cb)
    source = ColumnDataSource(data=dict(seconds=[], ch0=[]))
    channels = ['ch0']

    playvisbtn = AudioButton(
        label='Play visible signal', source=source, channels=channels,
        width=120, visible=False
    )
    playselbtn = AudioButton(
        label='Play selected signal', source=source, channels=channels,
        width=120, visible=False
    )
    
    # Instantiate and share specific select/zoom tools so that
    # highlighting is synchronized on all plots.
    boxsel = BoxSelectTool(dimensions='width')
    spboxsel = BoxSelectTool(dimensions='width')
    boxzoom = BoxZoomTool(dimensions='width')
    zoomin = ZoomInTool(dimensions='width')
    zoomout = ZoomOutTool(dimensions='width')
    crosshair = CrosshairTool(dimensions='height')
    shared_tools = [
        'xpan', boxzoom, boxsel, crosshair, 'undo', 'redo',
        zoomin, zoomout, 'reset'
    ]

    figargs = dict(
        tools=shared_tools,
    )
    cursor = Span(dimension='height', line_color='red',
        line_dash='dashed', line_width=1)
    ch0 = figure(name='ch0', tooltips=[("time", "$x{0.0000}")], **figargs)
    ch0.line(x='seconds', y='ch0', source=source, nonselection_line_alpha=0.6)
    # Link pan, zoom events for plots with x_range.
    ch0.x_range.on_change('start', x_range_cb)
    ch0.x_range.on_change('end', x_range_cb)
    ch0.on_event(SelectionGeometry, selection_cb)
    ch0.on_event(Tap, cursor_cb)
    ch0.add_layout(cursor)
    low_thresh = 0.0
    sgrams = [np.ones((1, 1))]
    specsource = ColumnDataSource(data=dict(sgram0=[sgrams[0]]))
    fq_marker_source = ColumnDataSource(data=dict(freq=[0.0], power=[0.0], time=[0.0]))
    spec0 = figure(
        name='spec0',
        x_range=ch0.x_range, # Keep times synchronized
        tooltips=[("time", "$x{0.0000}"), ("freq", "$y{0.0000}"), ("value", "@sgram0{0.000000}")],
        **figargs
    )
    spec0.add_layout(cursor)
    spec0_fq_marker = spec0.circle(x='time', y='freq', source=fq_marker_source, 
        size=6, line_color='red', fill_color='red', visible=False)
    spec0.x_range.range_padding = spec0.y_range.range_padding = 0
    spec0cmap = LogColorMapper(palette=r_Greys256, low_color=params['low_thresh_color'])
    low_thresh_slider = Slider(
        start=1.0, end=12.0, step=0.125, value=params['low_thresh_power'],
        title='Low threshold'
    )
    winsize_slider = Slider(
        start=5.0, end=40.0, step=5.0, value=params['window_size'],
        title='Analysis window (ms)'
    )
    filter_sel = Select(
        options=['no filter (clear)', 'no filter', 'passband', 'stopband'],
        value='no filter (clear)'
    )
    spec0img = spec0.image(
        image='sgram0',
        x=0, y=0,
        color_mapper=spec0cmap,
        level='image',
        source=specsource
    )
    spec0.grid.grid_line_width = 0.0
    low_thresh_slider.on_change('value', low_thresh_cb)
    winsize_slider.on_change('value', winsize_cb)
    filter_sel.on_change('value', filter_type_cb)
    selbox = BoxAnnotation(
        name='selbox',
        left=None, right=None,
        fill_color='green', fill_alpha=0.1,
        line_color='green', line_width=1.5, line_dash='dashed',
        visible=False
    )
    sgselbox = BoxAnnotation(
        name='sgselbox',
        top=None, bottom=None,
        fill_color='red', fill_alpha=0.1,
        line_color='red', line_width=1.5, line_dash='dashed',
        visible=False
    )
    ch0.add_layout(selbox)
    spec0.add_layout(selbox)
    spec0.add_layout(sgselbox)
    spec0.on_event(SelectionGeometry, selection_cb)
    spec0.on_event(Tap, cursor_cb)
    grid = gridplot(
        [ch0, spec0],
        ncols=1,
        plot_height=200,
        toolbar_location='left',
        toolbar_options={'logo': None},
        merge_tools=True
    )
    spslice_chtool = CrosshairTool(dimensions='height')
    spslice0 = figure(
        name='spslice0',
        plot_width=400, plot_height=250,
        y_axis_type='log',
        y_range=(10**-9, 1),
        tools=[spboxsel, spslice_chtool],
        toolbar_location='left'
    )
    spslice0.toolbar.logo = None
    spslice_source = ColumnDataSource(data=dict(freq=np.array([]), power=np.array([])))
    spslice0.line(x='freq', y='power', source=spslice_source)
    spselbox = BoxAnnotation(
        name='spselbox',
        left=None, right=None,
        fill_color='red', fill_alpha=0.1,
        line_color='red', line_width=1.5, line_dash='dashed',
        visible=False
    )
    spslice0.add_layout(spselbox)
    spslice0.on_event(SelectionGeometry, filter_sel_cb)
    thresh_box = BoxAnnotation(fill_color=params['low_thresh_color'])
    spslice0.add_layout(thresh_box)
    spslice0.on_event(MouseMove, spslice_move_cb)
    spslice0_fq_marker = spslice0.circle(x='freq', y='power', source=fq_marker_source, 
        size=6, line_color='red', fill_color='red', visible=False)
    num_fmtr = NumberFormatter(format='0.0000')
    det_num_fmtr = NumberFormatter(format='0.000000000')
    fq_marker_table = DataTable(
        source=fq_marker_source,
        columns=[
            TableColumn(field="freq", title="Frequency", formatter=num_fmtr),
            TableColumn(field="power", title="Power", formatter=det_num_fmtr),
            TableColumn(field="time", title="Time", formatter=num_fmtr),
        ],
        width=300
    )
    control_col = column(
        row(playvisbtn, playselbtn), low_thresh_slider, winsize_slider,
        filter_sel, fq_marker_table
    )
    grid2 = gridplot(
        [spslice0, control_col],
        ncols=2
    )
    
    mainLayout = column(
        fselect_row, grid, #low_thresh_slider, winsize_slider,
        grid2, name='mainLayout'
    )
    doc.add_root(mainLayout)
    return doc
