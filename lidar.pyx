import numpy as np

#cython syntax used to assign types to variables


#basic function for lidar. given a point (x0,y0) on the track and a direction dirang computes the
#distance of the border along the given direction
cpdef dist_grid(double x0,double y0,double dirang, map,float step=1./250,verbose=False):
    cdef double stepx = step*np.cos(dirang)
    cdef double stepy = step*np.sin(dirang)
    cdef double x = x0
    cdef double y = y0
    cdef int xg = int(x * 500) + 650
    cdef int yg = int(y * 500) + 650
    # get map bounds (assume numpy array)
    cdef int nx = map.shape[0]
    cdef int ny = map.shape[1]
    cdef bint check
    cdef bint up
    cdef bint down
    cdef bint left
    cdef bint right
    cdef bint center
    cdef int i = 0
    # guard initial access
    if xg < 0 or xg >= nx or yg < 0 or yg >= ny:
        # off-map, return current point
        return x, y
    if not map[xg, yg]:
        # if starting point is off the route, still proceed but guard
        check = False
    else:
        check = True
    while (check):
        if i == 10:
            step = 1./100
            stepx = step*np.cos(dirang)
            stepy = step*np.sin(dirang)
        x += stepx
        y += stepy
        xg = int(x * 500) + 650
        yg = int(y * 500) + 650
        # if we've moved off the map, stop tracing
        if xg < 0 or xg >= nx or yg < 0 or yg >= ny:
            check = False
            break
        # safe neighbor checks
        up = False
        down = False
        left = False
        right = False
        center = False
        if yg + 1 < ny:
            up = map[xg, yg+1]
        if yg - 1 >= 0:
            down = map[xg, yg-1]
        if xg - 1 >= 0:
            left = map[xg-1, yg]
        if xg + 1 < nx:
            right = map[xg+1, yg]
        center = map[xg, yg]
        # require center and all four neighbors to continue
        check = center and up and down and left and right
        i += 1
    x -= stepx
    y -= stepy
    if step == 1./100:
        #print("reducing step")
        x, y = dist_grid(x,y, dirang,map,step=1./500,verbose=False)
    if verbose:
        print("start at = {}, cross border at {}".format((x0,y0),(x,y)))
    return x,y


cpdef lidar_grid(double x,double y,double vx,double vy,map,float angle=np.pi/3, int pins=19):
    cdef double dirang = np.arctan2(vy, vx) #car direction
    obs = np.zeros(pins)
    cdef int i = 0
    cdef double a = dirang - angle/2
    cdef float astep = angle/(pins-1)
    cdef double cx
    cdef double cy
    for i in range(pins):
        cx,cy = dist_grid(x,y,a,map,verbose=False)
        obs[i] = ((cx-x)**2+(cy-y)**2)**.5
        a += astep
    return obs