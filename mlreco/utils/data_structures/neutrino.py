"""Module with a data class object which represents true neutrino information.

This copies the internal structure of :class:`larcv.Neutrino`.
"""

import numpy as np
from dataclasses import dataclass
from larcv import larcv
from warnings import warn

from mlreco.utils.globals import NU_CURR_TYPE, NU_INT_TYPE

from .meta import Meta


@dataclass
class Neutrino:
    """Neutrino truth information.

    Attributes
    ----------
    id : int
        Index of the neutrino in the list
    event_id : int
        Index of the event in which this neutrino was generate
    vertex_id : int
        Index of the
    mct_index : int
        Index in the original MCTruth array from whence it came
    track_id : int
        Geant4 track ID of the neutrino
    lepton_track_id : int
        Geant4 track ID of the lepton (if CC)
    pdg_code : int
        PDG code of the neutrino
    current_type : int
        Enumerated current type of the neutrino interaction
    interaction_mode : int
        Enumerated neutrino interaction mode
    interaction_type : int
        Enumerated neutrino interaction type
    target : int
        PDG code of the target object
    nucleon : int
        PDG code of the target nucleon (if QE)
    quark : int
        PDG code of the target quark (if DIS)
    energy_init : float
        Energy of the neutrino at its interaction point in GeV
    hadronic_invariant_mass : float
        Hadronic invariant mass (W) in GeV/c^2
    bjorken_x : float
        Bjorken scaling factor (x)
    inelasticity : float
        Inelasticity (y)
    momentum_transfer : float
        Squared momentum transfer (Q^2) in (GeV/c)^2
    momentum_transfer_mag : float
        Magnitude of the momentum transfer (Q3) in GeV/c
    energy_transfer : float
        Energy transfer (Q0) in GeV
    theta : float
        Angle between incoming and outgoing leptons in radians
    creation_process : str
        Creation process of the neutrino
    position : np.ndarray
        Location of the neutrino interaction
    momentum : np.ndarray
        3-momentum of the neutrino at its interaction point
    units : str
        Units in which the position coordinates are expressed
    units: str = 'cm'
    """
    # Attributes
    id: int = -1
    event_id: int = -1
    vertex_id: int = -1
    mct_index: int = -1
    track_id: int = -1
    lepton_track_id: int = -1
    pdg_code: int = -1
    lepton_pdg_code: int = -1
    current_type: int = -1
    interaction_mode: int = -1
    interaction_type: int = -1
    target: int = -1
    nucleon: int = -1
    quark: int = -1
    energy_init: float = -1.
    hadronic_invariant_mass: float = -1.
    bjorken_x: float = -1.
    inelasticity: float = -1.
    momentum_transfer: float = -1.
    momentum_transfer_mag: float = -1.
    energy_transfer: float = -1.
    theta: float = -1.
    creation_process: str = ''
    position: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    momentum: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    lepton_momentum: np.ndarray = np.full(3, -np.inf, dtype=np.float32)
    units: str = 'cm'

    # Fixed-length attributes
    _fixed_length_attrs = ['position', 'momentum', 'lepton_momentum']

    # Attributes specifying coordinates
    _pos_attrs = ['position']

    # Enumerated attributes
    _enum_attrs = {
            'current_type': {v : k for k, v in NU_CURR_TYPE.items()},
            'interaction_mode': {v : k for k, v in NU_INT_TYPE.items()},
            'interaction_type': {v : k for k, v in NU_INT_TYPE.items()}
    }

    def to_cm(self, meta):
        """Converts the coordinates of the positional attributes to cm.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self.units != 'cm', "Units already expressed in cm"
        self.units = 'cm'
        for attr in self._pos_attrs:
            setattr(self, attr, meta.to_cm(getattr(self, attr)))

    def to_pixel(self, meta):
        """Converts the coordinates of the positional attributes to pixel.

        Parameters
        ----------
        meta : Meta
            Metadata information about the rasterized image
        """
        assert self.units != 'pixel', "Units already expressed in pixels"
        self.units = 'pixel'
        for attr in self._pos_attrs:
            setattr(self, attr, meta.to_pixel(getattr(self, attr)))

    @classmethod
    def from_larcv(cls, neutrino):
        """Builds and returns a Neutrino object from a LArCV Neutrino object.

        Parameters
        ----------
        neutrino : larcv.Neutrino
            LArCV-format neutrino object

        Returns
        -------
        Neutrino
            Neutrino object
        """
        # Initialize the dictionary to initialize the object with
        obj_dict = {}

        # Load the scalar attributes
        for key in ['id', 'event_id', 'vertex_id', 'mct_index', 'track_id',
                    'lepton_track_id', 'pdg_code', 'lepton_pdg_code',
                    'current_type', 'interaction_mode', 'interaction_type',
                    'target', 'nucleon', 'quark', 'energy_init',
                    'hadronic_invariant_mass', 'bjorken_x', 'inelasticity',
                    'momentum_transfer', 'momentum_transfer_mag',
                    'energy_transfer', 'theta', 'creation_process']:
            if not hasattr(neutrino, key):
                warn(f"The LArCV Neutrino object is missing the {key} "
                      "attribute. It will miss from the Neutrino object.")
                continue
            obj_dict[key] = getattr(neutrino, key)()

        # Load the positional attribute
        pos_attrs = ['x', 'y', 'z']
        for key in cls._pos_attrs:
            vector = getattr(neutrino, key)()
            obj_dict[key] = np.asarray(
                    [getattr(vector, a)() for a in pos_attrs], dtype=np.float32)
            
        # Load the other array attributes (special care needed)
        mom_attrs = ['px', 'py', 'pz']
        for prefix in ['', 'lepton_']:
            key = prefix + 'momentum'
            if not hasattr(neutrino, key):
                warn(f"The LArCV Neutrino object is missing the {key} "
                      "attribute. It will miss from the Neutrino object.")
                continue
            obj_dict[key] = np.asarray(
                    [getattr(neutrino, prefix + a)() for a in mom_attrs],
                    dtype=np.float32)

        return cls(**obj_dict)