"""Analysis module template.

Use this template as a basis to build your own analysis script. An analysis
script takes the output of the reconstruction and the post-processors and
performs basic selection cuts and store the output to a CSV file.
"""

# Add the imports specific to this module here
import numpy as np

# Must import the analysis script base class
from spine.ana.base import AnaBase

from spine.utils.globals import MUON_PID
from spine.utils.gnn.cluster import cluster_dedx

# Must list the post-processor(s) here to be found by the factory.
# You must also add it to the list of imported modules in the
# `spine.ana.factories`!
__all__ = ['numuccr2tAna']


class numuccr2tAna(AnaBase):
    """Script to dump generalized selection variables"""
    name = 'numuccr2t'

    def __init__(self, ke_threshold=25, length_threshold=0., use_muon_start_point=True,fv=None,include_crthit=False, max_dist_dedx=4., **kwargs):
        """Initialize the analysis script.

        Parameters
        ----------
        ke_threshold : int
            Minimum kinetic energy for a muon to be considered
        length_threshold : float
            Minimum length for a muon to be considered (applied to reco only)
        use_muon_start_point : bool
            If True, use the start point of the muon to determine if the interaction is in the fiducial volume
        fv : list[float]
            Fiducial volume margin in cm. Format: [[x_min,x_max],[y_min,y_max],[z_min,z_max]]
        include_crthit : bool
            If True, include CRT hit information
        max_dist_dedx : float
            Maximum distance to use for dedx calculation
        """
        self.ke_threshold = ke_threshold
        self.length_threshold = length_threshold
        self.use_muon_start_point = use_muon_start_point
        if fv is None:
            self.fv = [[-180,180],[-180,180],[30,450]]
        else:
            self.fv = fv
        self.include_crthit = include_crthit
        self.max_dist_dedx = max_dist_dedx
        # Initialize the parent class
        super().__init__('interaction', 'both',**kwargs)
        
        # Initialize the CSV writer(s) you want
        self.initialize_writer('numucc')
        self.update_keys({'interaction_matches_r2t': True
                          }) 
        if self.include_crthit:
            self.update_keys({'crthits': True})

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
            
            #Flash info
            #Initialize flash info
            reco_dict['reco_flash_score'] = -9999
            reco_dict['reco_flash_total_pe'] = -1
            reco_dict['reco_flash_hypo_pe'] = -1
            for i in range(2): #2 volumes
                reco_dict[f'flash_volume_id{i}'] = -1
                reco_dict[f'flash_time{i}'] = -9999
            reco_dict['reco_flash_reduced_score'] = -9999
            
            for i, fid in enumerate(reco_inter.flash_volume_ids):
                reco_dict[f'flash_volume_id{fid}'] = fid
                reco_dict[f'flash_time{fid}'] = reco_inter.flash_times[i]
            is_flash_matched = False
            if reco_inter.is_flash_matched:
                reco_dict[f'reco_flash_total_pe'] = reco_inter.flash_total_pe
                reco_dict[f'reco_flash_hypo_pe'] = reco_inter.flash_hypo_pe
                reco_dict[f'reco_flash_score'] = reco_inter.flash_scores[0]
                reco_dict['reco_flash_reduced_score'] = (reco_inter.flash_total_pe - reco_inter.flash_hypo_pe) / reco_inter.flash_total_pe
                fts = np.array([ft for ft in reco_inter.flash_times])
                _fms = (fts < 1.6) & (fts > 0.0) #BNB window us
                if np.sum(_fms) > 0: #At least one flash in the BNB window
                    is_flash_matched = True
            reco_dict['reco_fmatched'] = is_flash_matched

            #CRT info
            if self.include_crthit:
                crthits = data['crthits']
                min_time = np.inf
                for i,crthit in enumerate(crthits):
                    if abs(crthit.ts0_ns) < abs(min_time):
                        min_time = crthit.ts0_ns
                        min_time_ind = i
                        total_pe = crthit.total_pe
                        center = crthit.center
                        width = crthit.width

                reco_dict['reco_crthit_ind'] = min_time_ind
                reco_dict['reco_crthit_total_pe'] = total_pe
                reco_dict['reco_crthit_min_time_ns'] = min_time
                reco_dict['reco_crthit_center_x'] = center[0]
                reco_dict['reco_crthit_center_y'] = center[1]
                reco_dict['reco_crthit_center_z'] = center[2]
                reco_dict['reco_crthit_width_x'] = width[0]
                reco_dict['reco_crthit_width_y'] = width[1]
                reco_dict['reco_crthit_width_z'] = width[2]
            
            reco_muons = [p for p in reco_inter.particles if (p.pid == MUON_PID) and (p.is_primary) and (p.ke > self.ke_threshold) and (p.length > self.length_threshold)]# and (p.csda_ke > 143.425)]
            #Find best muon by it's length
            best_muon = None
            longest_muon_length = -np.inf
            for i,m in enumerate(reco_muons):
                if (m.length > longest_muon_length):
                    best_muon = m
                    longest_muon_length = m.length

            reco_dict['num_reco_muons'] = len(reco_muons)

            #Determine if the reco interaction is in the fiducial volume
            if self.use_muon_start_point and (len(reco_muons) >= 1):
                is_reco_fv = reco_inter.is_fiducial & (abs(best_muon.start_point[0]) < self.fv[0][1]) & (abs(best_muon.start_point[1]) < self.fv[1][1]) & (best_muon.start_point[2] > self.fv[2][0]) & (best_muon.start_point[2] < self.fv[2][1])
            else:
                is_reco_fv = reco_inter.is_fiducial
            reco_dict['reco_interaction_is_fiducial'] = is_reco_fv
            reco_dict['is_fiducial'] = reco_inter.is_fiducial

            
            reco_dict['reco_interaction_vertex_x'] = reco_inter.vertex[0]
            reco_dict['reco_interaction_vertex_y'] = reco_inter.vertex[1]
            reco_dict['reco_interaction_vertex_z'] = reco_inter.vertex[2]

            reco_dict['true_interaction_vertex_x'] = true_inter.vertex[0]
            reco_dict['true_interaction_vertex_y'] = true_inter.vertex[1]
            reco_dict['true_interaction_vertex_z'] = true_inter.vertex[2]

            if (len(reco_muons) >= 1):     
                reco_dict['reco_muon_id'] = best_muon.id
                reco_dict['reco_muon_is_contained'] = best_muon.is_contained

                reco_dict['reco_muon_momentum'] = best_muon.p
                reco_dict['reco_muon_costheta'] = best_muon.start_dir[2]
                reco_dict['reco_muon_ke'] = best_muon.ke
                reco_dict['reco_muon_length'] = best_muon.length

                reco_dict['reco_muon_start_x'] = best_muon.start_point[0]
                reco_dict['reco_muon_start_y'] = best_muon.start_point[1]
                reco_dict['reco_muon_start_z'] = best_muon.start_point[2]

                reco_dict['reco_muon_end_x'] = best_muon.end_point[0]
                reco_dict['reco_muon_end_y'] = best_muon.end_point[1]
                reco_dict['reco_muon_end_z'] = best_muon.end_point[2]

                reco_dict['reco_muon_start_dedx'] = cluster_dedx(best_muon.points,best_muon.depositions,best_muon.start_point,max_dist=self.max_dist_dedx)
                reco_dict['reco_muon_end_dedx'] = cluster_dedx(best_muon.points,best_muon.depositions,best_muon.end_point,max_dist=self.max_dist_dedx)
                
            else:
                reco_dict['reco_muon_id'] = -1
                reco_dict['reco_muon_is_contained'] = False
                
                reco_dict['reco_muon_momentum'] = -9999
                reco_dict['reco_muon_costheta'] = -9999
                reco_dict['reco_muon_ke'] = -9999
                reco_dict['reco_muon_length'] = -9999

                reco_dict['reco_muon_start_x'] = -9999
                reco_dict['reco_muon_start_y'] = -9999
                reco_dict['reco_muon_start_z'] = -9999

                reco_dict['reco_muon_end_x'] = -9999
                reco_dict['reco_muon_end_y'] = -9999
                reco_dict['reco_muon_end_z'] = -9999

                reco_dict['reco_muon_start_dedx'] = -9999
                reco_dict['reco_muon_end_dedx'] = -9999

            ### Append matched truth information to our reco dictionary

            # CC
            is_true_cc = False
            if true_inter.current_type == 0:
                is_true_cc = True
                
            # active volume
            is_true_active = (abs(true_inter.vertex[0]) <= 200 and abs(true_inter.vertex[1]) <= 200 and true_inter.vertex[2] <= 500 and true_inter.vertex[2] >= 0)
            is_true_fv = true_inter.is_fiducial

            # true muon count
            true_muons = [tp for tp in true_inter.particles if (tp.pid == MUON_PID) and (tp.is_primary) and (tp.ke > self.ke_threshold)]

            num_true_muons = len(true_muons)

            # classification
            cat = 6 # cosmic
            if true_inter.nu_id > -1:
                #Signal - numucc
                if (num_true_muons == 1) and (is_true_cc) and (is_true_active): #(true_muons[0].is_contained)
                    # numucc in phase space
                    if (is_true_fv & (len(true_muons) == 1)):
                        if (true_muons[0].is_contained):
                            cat = 0
                        else:
                            cat = 1
                    # numucc out of phase space
                    else:
                        cat = 2
                elif (not is_true_active):
                    # Dirt
                    cat = 3
                elif (num_true_muons == 0) and (is_true_cc) and (is_true_active):
                    # nuecc
                    cat = 4
                elif (not is_true_cc) and (is_true_active):
                    # NC
                    cat = 5
                else:
                    # Other nu interactions
                    cat = -1
            
            # Output
            reco_dict['true_category'] = cat
            
            # Append row to CSV
            self.append(f'numucc', **reco_dict)
            