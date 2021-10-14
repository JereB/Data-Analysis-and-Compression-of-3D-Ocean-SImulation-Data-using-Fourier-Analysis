import numpy as np

def create_list_array(shape):
    fv = np.frompyfunc(lambda x: None, 1, 1)
    return fv(np.zeros(shape))


def convert_2d_to_4d(data: np.array, land_sea_mask: np.array) -> np.array:
    land_sea_mask = np.transpose(land_sea_mask)
    n1, n2 = np.shape(land_sea_mask)
    n3 = int(np.amax(land_sea_mask))

    v3d = create_list_array((n1, n2, n3))

    offset = 0

    for j in range(n2):
        for i in range(n1):
            depth = int(land_sea_mask[i, j])

            for d in range(depth):
                v3d[i, j, d] = data[offset]
                offset += 1

    return v3d

