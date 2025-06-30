"""Analysis module template.

Use this template as a basis to build your own analysis script. An analysis
script takes the output of the reconstruction and the post-processors and
performs basic selection cuts and store the output to a CSV file.
"""

# Add the imports specific to this module here
import numpy as np

# Must import the analysis script base class
from spine.ana.base import AnaBase

# Must list the post-processor(s) here to be found by the factory.
# You must also add it to the list of imported modules in the
# `spine.ana.factories`!
__all__ = ['numuccr2tAna']


class numuccr2tAna(AnaBase):
    """Script to dump generalized selection variables"""
    name = 'numuccr2t'

    def __init__(self, dummyarg, **kwargs):
        """Initialize the analysis script.
        """

        dummyarg : int
        
        # Initialize the parent class
        super().__init__('interaction', 'both',**kwargs)

        self.dummyarg = dummyarg
        
        # Initialize the CSV writer(s) you want
        self.initialize_writer('numucc')
        self.update_keys({'interaction_matches_r2t': True}) 

    def process(self, data):
        """Pass data products corresponding to one entry through the analysis.

        Parameters
        ----------
        data : dict
            Dictionary of data products
        """
        # Loop over matched interactions (r2t)      
        interaction_matches_r2t = data['interaction_matches_r2t']
        for match in interaction_matches_r2t:
            
            # Get match components
            reco_inter = match[0]
            true_inter = match[1]
            if true_inter == None:
                continue

            # Storage
            reco_dict = {}
            reco_dict['reco_interaction_id'] = reco_inter.id
            reco_dict['is_contained'] = reco_inter.is_contained
            #reco_dict['is_fiducial'] = reco_inter.is_fiducial
            is_reco_fiducial = True
            if not reco_inter.is_fiducial:
                is_reco_fiducial = False
            reco_dict['is_fiducial'] = is_reco_fiducial
            
            #Flash info
            #Initialize flash info
            reco_dict['reco_flash_score'] = -9999
            reco_dict['reco_flash_total_pe'] = -1
            reco_dict['reco_flash_hypo_pe'] = -1
            for i in range(2): #2 volumes
                reco_dict[f'flash_volume_id{i}'] = -1
                reco_dict[f'flash_time{i}'] = -9999
                #More ...
            
            for i, fid in enumerate(reco_inter.flash_volume_ids):
                reco_dict[f'flash_volume_id{fid}'] = fid
                reco_dict[f'flash_time{fid}'] = reco_inter.flash_times[i]
            is_flash_matched = False
            is_quality_flash = False
            if reco_inter.is_flash_matched:
                reco_dict[f'reco_flash_total_pe'] = reco_inter.flash_total_pe
                reco_dict[f'reco_flash_hypo_pe'] = reco_inter.flash_hypo_pe
                reco_dict[f'reco_flash_score'] = (reco_inter.flash_total_pe - reco_inter.flash_hypo_pe) / reco_inter.flash_total_pe
                fts = np.array([ft for ft in reco_inter.flash_times])
                _fms = (fts < 1.6) & (fts > 0.0) #BNB window us
                if np.sum(_fms) > 0: #At least one flash in the BNB window
                    is_flash_matched = True
                if (reco_dict[f'reco_flash_score'] > -0.1):
                    is_quality_flash = True
            reco_dict['reco_fmatched'] = is_flash_matched
            reco_dict['reco_fquality'] = is_quality_flash
                

            reco_muons = [p for p in reco_inter.particles if (p.pid == 2) and (p.is_primary)]# and (p.csda_ke > 143.425)]
            #Find best muon by it's length
            best_muon = None
            longest_muon_length = -np.inf
            for i,m in enumerate(reco_muons):
                if (m.length > longest_muon_length):
                    best_muon = m
                    longest_muon_length = m.length

            reco_dict['num_reco_muons'] = len(reco_muons)
            
            if (len(reco_muons) >= 1) and is_flash_matched and (reco_inter.is_fiducial):     
                reco_dict['topology'] = True
                reco_dict['reco_muon_id'] = best_muon.id
                reco_dict['reco_muon_is_contained'] = best_muon.is_contained

                reco_dict['reco_muon_momentum'] = best_muon.p
                reco_dict['reco_muon_costheta'] = best_muon.start_dir[2] #cos(theta) = pz/p
                
            else:
                reco_dict['topology'] = False
                reco_dict['reco_muon_id'] = -1
                reco_dict['reco_muon_is_contained'] = False
                
                reco_dict['reco_muon_momentum'] = -9999
                reco_dict['reco_muon_costheta'] = -9999

            ### Append matched truth information to our reco dictionary

            # neutrino event
            is_true_neutrino = False
            if true_inter.nu_id > -1:
                is_true_neutrino = True

            # CC
            is_true_cc = False
            if true_inter.current_type == 0:
                is_true_cc = True

            # fiducial
            is_true_fiducial = True
            if not true_inter.is_fiducial:
                is_true_fiducial = False
                
            # active volume
            is_true_active = (abs(true_inter.vertex[0]) <= 200 and abs(true_inter.vertex[1]) <= 200 and true_inter.vertex[2] <= 500 and true_inter.vertex[2] >= 0)
            
            # true muon count
            true_muons = [tp for tp in true_inter.particles if (tp.pid == 2) and (tp.is_primary)]# and (tp.ke > 143.425)]
            num_true_muons = len(true_muons)

            # classification
            # TODO: Update to use FV
            cat = 4 # cosmic
            if is_true_neutrino:
                #Signal - numucc
                if (num_true_muons == 1) and (is_true_cc) and (is_true_active): #(true_muons[0].is_contained)
                    # numucc
                    cat = 0                     
                elif (not is_true_active):
                    # Dirt
                    cat = 1
                elif (num_true_muons == 0) and (is_true_cc) and (is_true_active):
                    # nuecc
                    cat = 2
                elif (not is_true_cc) and (is_true_active):
                    # NC
                    cat = 3
                else:
                    # Other nu interactions
                    cat = -1
            
            # Output
            reco_dict['true_category'] = cat
            
            # Append row to CSV
            self.append(f'numucc', **reco_dict)
            