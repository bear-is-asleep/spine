"""Module to convert a point cloud into an convex hull envelope."""

import numpy as np
import plotly.graph_objs as go


def hull_trace(points, color=None, intensity=None, showscale=False,
               alphahull=0, **kwargs):
    """Converts a cloud of points into a 3D convex hull.

    This function represents a point cloud by forming a mesh with the points
    that belong to the convex hull of the point cloud.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) Array of point coordinates
    color : Union[str, float, np.ndarray], optional
        Color of hull
    intensity : Union[int, float], optional
        Color intensity of the box along the colorscale axis
    showscale : bool, default False
        If True, show the colorscale of the :class:`plotly.graph_objs.Mesh3d`
    alphahull : float, default 0
        Parameter that sets how to define the hull. 0 is the convex hull,
        larger numbers correspond to alpha-shapes.
    **kwargs : dict, optional
        Additional parameters to pass to the underlying
        :class:`plotly.graph_objs.Mesh3d` object
    """
    # Convert the color provided to a set of intensities, if needed
    if color is not None and not isinstance(color, str):
        assert intensity is None, (
                "Must not provide both `color` and `intensity`.")
        intensity = np.full(len(ell_points), color)
        color = None

    # Append Mesh3d object
    return go.Mesh3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2], intensity=intensity,
        alphahull=alphahull, showscale=showscale, **kwargs)
