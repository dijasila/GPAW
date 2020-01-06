
    


def Dget_alphas(Z_ig):
    # Assume we always cross?
    # No
    # Return alpha_ig, did_cross?, crossing_index
    if not cross_all(Z_ig):
        return None, False, None

    nlesser = (Z_ig < -1).sum(axis=0)
    ngreater = (Z_ig >= -1).sum(axis=0)

    ipart_ig = np.argpartition(np.abs(Z_ig - (-1)).reshape(len(Z_ig), -1), kth=2, axis=0)
    part_ig = np.take_along_axis(Z_ig, ipart_ig, axis=Z_ig)[:2, ...]
    iclosest_g = ipart_ir[0, ...]
    inclosest_g = ipart_ir[1, ...]
 
    alpha_g = np.zeros((2,) + Z_ig.shape[1:])
    alpha_g[0, ...] = (part_ig[1, ...] - (-1)) / (part_ig[1, ...] - part_ig[0, ...])
    alpha_g[1, ...] = ((-1) - part_ig[0, ...]) / (part_ig[1, ...] - part_ig[0, ...])
    
    return alpha_g, True, (iclosest_g, inclosest_g)


def get(Z_ig):

    alpha_ir = np.zeros_like(Z_ig).reshape(len(Z_ig), -1)
    crossed_r = np.zeros(Z_ig.shape[1:]).reshape(-1)
    for ir, Z_i in enumerate(Z_ig.reshape(len(Z_ig), -1).T):
        if not (Z_i >= -1).any() and (Z_i < -1).any():
            alpha_ig[:, ir] = 0
            crossed_r[ir] = 0
        else:
            crossed_r[ir] = 1
            # DO INTERPOLATION
            ic, inc = np.argpartition(np.abs(Z_i - (-1)), kth=2)[:2]
            
            
            
            

def cross_all(Z_ig):
    Z_ir = Z_ig.reshape(len(Z_ig), -1)
    AND = np.logical_and
    return AND((Z_ir >= -1).sum(axis=0), (Z_ir <= -1).sum(axis=0)).all()
