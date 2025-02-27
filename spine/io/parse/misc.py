"""Module that contains parsers that do not fit in other categories.

Contains the following parsers:
- :class:`Meta2DParser`
- :class:`Meta3DParser`
- :class:`RunInfoParser`
- :class:`OpFlashParser`
- :class:`CRTHitParser`
- :class:`TriggerParser`
"""


from spine import Meta, RunInfo, Flash, CRTHit, Trigger, ObjectList

from spine.utils.conditional import larcv

from .base import ParserBase

__all__ = ['MetaParser', 'RunInfoParser', 'FlashParser',
           'CRTHitParser', 'TriggerParser']


class MetaParser(ParserBase):
    """Get the metadata information to translate into real world coordinates.

    Each entry in a dataset is a cube, where pixel/voxel coordinates typically
    go from 0 to some integer N in each dimension. If you wish to translate
    these pixel/voxel coordinates back into real world coordinates, you can use
    the output of this parser to compute it.

    .. code-block. yaml

        schema:
          meta:
            parser: meta
            sparse_event: sparse3d_pcluster
    """
    name = 'meta'
    aliases = ['meta2d', 'meta3d']

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

    def __init__(self, projection_id=None, **kwargs):
        """Initialize the parser.

        Parameters
        ----------
        projection_id : int, optional
            Projection ID to get the 2D image from (if fetching from 2D)
        **kwargs : dict, optional
            Data product arguments to be passed to the `process` function
        """
        # Initialize the parent class
        super().__init__(**kwargs)

        # Store the revelant attributes
        self.projection_id = projection_id

    def process(self, sparse_event=None, cluster_event=None):
        """Fetches the metadata from one object that has it.

        Parameters
        ----------
        sparse_event : Union[larcv.EventSparseTensor2D
                             larcv.EventSparseTensor3D], optional
            Tensor which contains the metadata information as an attribute
        cluster_event : Union[larcv.EventClusterPixel2D,
                              larcv.EventClusterVoxel3D], optional
            Cluster which contains the metadata information as an attribute

        Returns
        -------
        Meta
            Metadata information for one image
        """
        # Check on the input, pick a source for the metadata
        assert (sparse_event is not None) ^ (cluster_event is not None), (
                "Must specify either `sparse_event` or `cluster_event`")
        ref_event = sparse_event if sparse_event is not None else cluster_event

        # Fetch a specific projection, if needed
        if isinstance(ref_event,
                      (larcv.EventSparseTensor2D, larcv.EventClusterPixel2D)):
            ref_event = ref_event.sparse_tensor_2d(self.projection_id)

        return Meta.from_larcv(ref_event.meta())


class RunInfoParser(ParserBase):
    """Parse run information (run, subrun, event number).

    .. code-block. yaml

        schema:
          run_info:
            parser: run_info
            sparse_event: sparse3d_pcluster
    """
    name = 'run_info'

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

    def process(self, sparse_event=None, cluster_event=None):
        """Fetches the run information from one object that has it.

        Parameters
        ----------
        sparse_event : Union[larcv.EventSparseTensor2D
                             larcv.EventSparseTensor3D], optional
            Tensor which contains the run information as an attribute
        cluster_event : Union[larcv.EventClusterPixel2D,
                              larcv.EventClusterVoxel3D], optional
            Cluster which contains the run information as an attribute

        Returns
        -------
        RunInfo
            Run information object
        """
        # Check on the input, pick a source for the run information
        assert (sparse_event is not None) ^ (cluster_event is not None), (
                "Must specify either `sparse_event` or `cluster_event`")
        ref_event = sparse_event if sparse_event is not None else cluster_event

        return RunInfo.from_larcv(ref_event)


class FlashParser(ParserBase):
    """Copy construct Flash and return an array of `Flash`.

    .. code-block. yaml
        schema:
          flashes_cryoE:
            parser: flash
            flash_event: flash_cryoE

    """
    name = 'flash'
    aliases = ['opflash']

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

    def process(self, flash_event=None, flash_event_list=None):
        """Fetches the list of optical flashes.

        Parameters
        -------------
        flash_event : larcv.EventFlash, optional
            Optical flash event which contains a list of flash objects
        flash_event_list : List[larcv.EventFlash], optional
            List of optical flash events, each a list of flash objects

        Returns
        -------
        List[Flash]
            List of optical flash objects
        """
        # Check on the input, aggregate the sources for the optical flashes
        assert ((flash_event is not None) ^
                (flash_event_list is not None)), (
                "Must specify either `flash_event` or `flash_event_list`")
        if flash_event is not None:
            flash_list = flash_event.as_vector()
        else:
            flash_list = []
            for flash_event in flash_event_list:
                flash_list.extend(flash_event.as_vector())

        # Output as a list of LArCV optical flash objects
        flashes = [Flash.from_larcv(larcv.Flash(f)) for f in flash_list]

        return ObjectList(flashes, Flash())


class CRTHitParser(ParserBase):
    """Copy construct CRTHit and return an array of `CRTHit`.

    .. code-block. yaml
        schema:
          crthits:
            parser: crthit
            crthit_event: crthit_crthit
    """
    name = 'crthit'

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

    def process(self, crthit_event):
        """Fetches the list of CRT hits.

        Parameters
        ----------
        crthit_event : larcv.CRTHitEvent

        Returns
        -------
        List[CRTHit]
            List of CRT hit objects
        """
        # Output as a list of LArCV CRT hit objects
        crthit_list = crthit_event.as_vector()
        crthits = [CRTHit.from_larcv(larcv.CRTHit(c)) for c in crthit_list]

        return ObjectList(crthits, CRTHit())


class TriggerParser(ParserBase):
    """Copy construct Trigger and return a `Trigger`.

    .. code-block. yaml
        schema:
          trigger:
            parser: trigger
            trigger_event: trigger_base
    """
    name = 'trigger'

    def __call__(self, trees):
        """Parse one entry.

        Parameters
        ----------
        trees : dict
            Dictionary which maps each data product name to a LArCV object
        """
        return self.process(**self.get_input_data(trees))

    def process(self, trigger_event):
        """Fetches the trigger information.

        Parameters
        ----------
        trigger_event : larcv.TriggerEvent

        Returns
        -------
        Trigger
            Trigger object
        """
        # Output as a trigger objects
        trigger = Trigger.from_larcv(larcv.Trigger(trigger_event))

        return trigger
