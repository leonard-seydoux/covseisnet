"""Read and pre-process seismic data."""

import obspy


class ArrayStream(obspy.Stream):
    """Custom Stream class for seismic arrays."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._array = None
