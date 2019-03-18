import matplotlib.pyplot as pplot

# A util function:
reshape_to_grid = lambda ngrid , grid_xy , z:  [grid_xy[:,0].reshape([ngrid,ngrid]),grid_xy[:,1].reshape([ngrid,ngrid]),z.reshape([ngrid,ngrid])]


def plotgrid(grid_xy,z = None, ngrid = 50):
    grid_data = reshape_to_grid(ngrid, grid_xy, z);
    pplot.pcolor(grid_data[0],grid_data[1],grid_data[2]);
