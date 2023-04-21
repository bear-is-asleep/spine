import numpy as np
import pandas as pd

from typing import Counter, List, Union
from mlreco.utils.globals import SHAPE_LABELS, PID_LABELS


class Particle:
    '''
    Data structure for managing particle-level
    full chain output information.

    Attributes
    ----------
    id : int, default -1
        Unique ID of the particle
    fragment_ids : np.ndarray, default np.array([])
        List of ParticleFragment IDs that make up this particle
    num_fragments: int
        Total number of fragments in this particle
    interaction_id : int, default -1
        ID of the particle's parent interaction
    nu_id : int, default -1
        ID of the particle's parent neutrino
    volume_id : int, default -1
        ID of the detector volume the particle lives in
    image_id : int, default -1
        ID of the image the particle lives in
    size : int
        Total number of voxels that belong to this particle
    index : np.ndarray, default np.array([])
        (N) IDs of voxels that correspondn to the particle within the image coordinate tensor that
    depositions : np.ndarray, defaul np.array([])
        (N) Array of energy deposition values for each voxel (rescaled, ADC units)
    depositions_sum : float
        Sum of energy depositions
    semantic_type : int, default -1
        Semantic type (shower (0), track (1),
        michel (2), delta (3), low energy (4)) of this particle.
    pid : int
        PDG Type (Photon (0), Electron (1), Muon (2),
        Charged Pion (3), Proton (4)) of this particle
    pid_scores : np.ndarray
        (P) Array of softmax scores associated with each of particle class
    is_primary : bool
        Indicator whether this particle is a primary from an interaction
    primary_scores : np.ndarray
        (2) Array of softmax scores associated with secondary and primary
    start_point : np.ndarray, default np.array([-1, -1, -1])
        (3) Particle start point
    end_point : np.ndarray, default np.array([-1, -1, -1])
        (3) Particle end point
    start_dir : np.ndarray, default np.array([-1, -1, -1])
        (3) Particle direction estimate w.r.t. the start point
    end_dir : np.ndarray, default np.array([-1, -1, -1])
        (3) Particle direction estimate w.r.t. the end point
    energy_sum : float, default -1
        Energy reconstructed from the particle deposition sum
    momentum_range : float, default -1
        Momentum reconstructed from the particle range
    momentum_mcs : float, default -1
        Momentum reconstructed using the MCS method
    match : List[int]
        List of TruthParticle IDs for which this particle is matched to
    '''
    def __init__(self, 
                 group_id: int = -1, 
                 fragment_ids: np.ndarray = np.empty(0, dtype=np.int64),
                 interaction_id: int = -1, 
                 nu_id: int = -1,
                 volume_id: int = -1,
                 image_id: int = -1, 
                 semantic_type: int = -1, 
                 pid: int = -1,
                 is_primary: int = -1,
                 index: np.ndarray = np.empty(0, dtype=np.int64), 
                 depositions: np.ndarray = np.empty(0, dtype=np.float32), 
                 pid_scores: np.ndarray = -np.ones(np.max(list((PID_LABELS.keys())))+1, dtype=np.float32),
                 primary_scores: np.ndarray = -np.ones(2, dtype=np.float32),
                 start_point: np.ndarray = -np.ones(3, dtype=np.float32),
                 end_point: np.ndarray = -np.ones(3, dtype=np.float32),
                 start_dir: np.ndarray = -np.ones(3, dtype=np.float32),
                 end_dir: np.ndarray = -np.ones(3, dtype=np.float32),
                 momentum_range: float = -1.,
                 momentum_mcs: float = -1.):

        # Initialize private attributes to be assigned through setters only
        self._num_fragments = None
        self._size = None
        self._index = None
        self._depositions = None
        self._pid_scores = None
        self._primary_scores = None

        # Initialize attributes
        self.id             = group_id
        self.fragment_ids   = fragment_ids
        self.interaction_id = interaction_id
        self.nu_id          = nu_id
        self.image_id       = image_id
        self.volume_id      = volume_id
        self.semantic_type  = semantic_type

        self.index          = index
        self.depositions    = depositions

        self.pid_scores     = pid_scores
        if np.all(pid_scores < 0):
            self.pid = pid
        self.primary_scores = primary_scores
        if np.all(primary_scores < 0):
            self.is_primary = is_primary

        self.start_point    = start_point
        self.end_point      = end_point
        self.start_dir      = start_dir
        self.end_dir        = end_dir
        self.momentum_range = momentum_range
        self.momentum_mcs   = momentum_mcs

        # Quantities to be set by the particle matcher
        self.match = np.empty(0, np.int64)
        self._match_counts = np.empty(0, np.float32)

    def __repr__(self):
        msg = "Particle(image_id={}, id={}, pid={}, size={})".format(self.image_id, self.id, self.pid, self.size)
        return msg

    def __str__(self):
        fmt = "Particle( Image ID={:<3} | Particle ID={:<3} | Semantic_type: {:<15}"\
                " | PID: {:<8} | Primary: {:<2} | Interaction ID: {:<2} | Size: {:<5} | Volume: {:<2} )"
        msg = fmt.format(self.image_id, self.id,
                         SHAPE_LABELS[self.semantic_type] if self.semantic_type in list(range(len(SHAPE_LABELS))) else "None",
                         PID_LABELS[self.pid] if self.pid in PID_LABELS else "None",
                         self.is_primary,
                         self.interaction_id,
                         self.size,
                         self.volume_id)
        return msg

    @property
    def num_fragments(self):
        '''
        Number of particle fragments getter. This attribute has no setter,
        as it can only be set by providing a list of fragment ids.
        '''
        return self._num_fragments

    @property
    def fragment_ids(self):
        '''
        ParticleFragment indices getter/setter. The setter also sets
        the number of fragments.
        '''
        return self._index

    @fragment_ids.setter
    def fragment_ids(self, fragment_ids):
        # Count the number of fragments
        self._fragment_ids = fragment_ids
        self._num_fragments = len(fragment_ids)

    @property
    def size(self):
        '''
        Particle size (i.e. voxel count) getter. This attribute has no setter,
        as it can only be set by providing a set of voxel indices.
        '''
        return self._size

    @property
    def index(self):
        '''
        Particle voxel indices getter/setter. The setter also sets
        the particle size, i.e. the voxel count.
        '''
        return self._index

    @index.setter
    def index(self, index):
        # Count the number of voxels
        self._index = index
        self._size = len(index)

    @property
    def depositions_sum(self):
        '''
        Total amount of charge/energy deposited. This attribute has no setter,
        as it can only be set by providing a set of depositions.
        '''
        return self._size

    @property
    def depositions(self):
        '''
        Particle depositions getter/setter. The setter also sets
        the particle depositions sum.
        '''
        return self._depositions

    @depositions.setter
    def depositions(self, depositions):
        # Sum all the depositions
        self._depositions = depositions
        self._depositions_sum = np.sum(depositions)

    @property
    def pid_scores(self):
        '''
        Particle ID scores getter/setter. The setter converts the
        scores to an particle ID prediction through argmax.
        '''
        return self._pid_scores

    @pid_scores.setter
    def pid_scores(self, pid_scores):
        # If no PID scores are providen, the PID is unknown
        if pid_scores[0] < 0.:
            self._pid_scores = pid_scores
            self.pid = -1
        
        # Store the PID scores and give a best guess
        self._pid_scores = pid_scores
        self.pid = np.argmax(pid_scores)

    @property
    def primary_scores(self):
        '''
        Primary ID scores getter/setter. The setter converts the
        scores to a primary prediction through argmax.
        '''
        return self._primary_scores

    @primary_scores.setter
    def primary_scores(self, primary_scores):
        # If no primary scores are given, the primary status is unknown
        if primary_scores[0] < 0.:
            self._primary_scores = primary_scores
            self.is_primary = -1
        
        # Store the PID scores and give a best guess
        self._primary_scores = primary_scores
        self.is_primary = np.argmax(primary_scores)
