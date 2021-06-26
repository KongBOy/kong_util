import numpy as np

def draw_3D_xy_plane_by_mesh_f(ax, row, col, mesh_x_f, mesh_y_f, z=0, alpha=0.5):
    '''
    ax： 要是 projection=3d 的那種 ax
    row/col： mesh 的 row/col
    mesh_f ： mesh 拉平的樣子， shape 為 (row * col, 2)
    z      ： xy平面 z 的高度
    alpha  ： 透明度
    '''
    ax.plot_surface(mesh_x_f.reshape(row, col),
                    mesh_y_f.reshape(row, col),
                    z * np.ones(shape=(row, col)),
                    alpha=0.5)
