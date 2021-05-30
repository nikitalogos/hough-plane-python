import numpy as np
import math
from plotly.offline import iplot
from plotly import graph_objs as go

def _show_data(data, show_zero=True, is_hough_space=False):
    if show_zero:
        data.append(
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode='markers',
                marker=dict(
                    size=10,
                    color=(255, 180, 60)
                )
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        height=400,
        paper_bgcolor="LightSteelBlue",
    )
    if is_hough_space:
        fig.update_layout(
            scene=go.Scene(
                xaxis=go.XAxis(title='φ'),
                yaxis=go.YAxis(title='θ'),
                zaxis=go.ZAxis(title='d')
            )
        )

    iplot(fig)

def show_points(points, is_hough_space=False):
    '''Shows points with or without color'''

    l,d = points.shape
    marker = {
        'size':1
    }
    if d == 4:
        marker['color'] = points[:, 3]
        marker['colorbar'] = dict(thickness=50)

    data = [
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=marker,
        )
    ]
    _show_data(data, is_hough_space=is_hough_space)

def _make_points2(points1, points2):
    '''Make data for two types of points'''

    data = [
        go.Scatter3d(
            x=points1[:, 0],
            y=points1[:, 1],
            z=points1[:, 2],
            mode='markers',
            marker=dict(
                size=1,
            )
        ),
        go.Scatter3d(
            x=points2[:, 0],
            y=points2[:, 1],
            z=points2[:, 2],
            mode='markers',
            marker=dict(
                size=5,
            )
        ),
    ]
    return data

def show_points_and_plane_vectors(points, plane_vectors):
    data = _make_points2(points, plane_vectors)
    _show_data(data)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def vec_norm(vec):
    x, y, z = vec
    vec_len = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    return vec / vec_len

def vec_len(vec):
    x,y,z = vec
    len_ = (x**2 + y**2 + z**2)**0.5
    return len_

def get_cos(vec1, vec2):
    x, y, z = vec1
    xx, yy, zz = vec2
    dot = x * xx + y * yy + z * zz

    len1 = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    len2 = (xx ** 2 + yy ** 2 + zz ** 2) ** 0.5

    cos = dot / (len1 * len2 + 1e-6)

    return cos

def get_plane_polygon_from_vector(vector, radius, steps=20):
    orts = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], dtype=np.float)

    for ort in orts:
        cos = abs(get_cos(ort, vector))
        if cos < 0.1:
            continue

        vector1 = np.cross(ort, vector)
        vector2 = np.cross(vector, vector1)
        break

    vector1_norm = vec_norm(vector1)
    vector2_norm = vec_norm(vector2)

    vectors_out = []
    for i in range(steps):
        direction = np.pi * 2 / steps * i
        vector_out = vector + \
                     math.cos(direction) * vector1_norm * radius + \
                     math.sin(direction) * vector2_norm * radius
        vectors_out.append(vector_out)
    vectors_out.append(vectors_out[0])
    vectors_out = np.array(vectors_out)

    return vectors_out

def visualize_plane(points, vectors):
    data = _make_points2(points, vectors)

    for vec in vectors:
        vec_len_ = vec_len(vec)
        for radius in [vec_len_ / 2, vec_len_, vec_len_ * 1.5]:
            poly = get_plane_polygon_from_vector(vec[:3], radius)
            data.append(
                go.Scatter3d(
                    x=poly[:,0],
                    y=poly[:,1],
                    z=poly[:,2],
                    mode='lines',
                ),
            )

    _show_data(data)