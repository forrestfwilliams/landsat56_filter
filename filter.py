import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.transform import rotate, AffineTransform, warp
from scipy.fft import fft2, fftshift, ifft2, ifftshift
import scipy.ndimage as nd
from scipy.spatial.distance import pdist


def find_max(corner):
    indices = np.where(np.max(corner) == corner)
    return indices[0][0], indices[1][0]


def locate_max_in_subset(arr, bounds):
    m_min, m_max, n_min, n_max = bounds
    subset = arr[m_min:m_max, n_min:n_max].copy()
    indices = np.where(np.max(subset) == subset)
    m, n = (indices[0][0], indices[1][0])
    m += m_min
    n += n_min
    return m, n


def calculate_slope(point1, point2):
    slope = np.arctan((point1[0] - point2[0]) / (point1[1] - point2[1]))
    return slope


def fft_filter(valid_domain):
    m, n = valid_domain.shape

    labeled_img, num_labels = nd.label(valid_domain)
    props = regionprops_table(labeled_img, properties=('centroid', 'area'))
    props = pd.DataFrame(props)
    s = props.loc[props['area'] == props['area'].max(), ['centroid-0', 'centroid-1']]

    foo = np.full((m, n), True)
    foo[int(round(s['centroid-0'])), int(round(s['centroid-1']))] = False
    foo = nd.distance_transform_edt(foo)
    foo[~(valid_domain > 0)] = 0

    center_m = int(round(m / 2))
    center_n = int(round(n / 2))

    regions = {'top_left': (0, center_m, 0, center_n),
               'top_right': (0, center_m, center_n + 1, n),
               'bottom_left': (center_m + 1, m, 0, center_n),
               'bottom_right': (center_m + 1, m, center_n + 1, n)}

    max_location = {}
    for name, bounds in regions.items():
        max_location[name] = locate_max_in_subset(foo, bounds)

    corners = [max_location['bottom_left'], max_location['bottom_right'], max_location['top_left'],
               max_location['top_right']]
    sep = pdist(corners, 'euclidean')

    if any(sep < center_m):
        regions = {'top_left': (0, round(center_m / 2), 0, center_n),
                   'top_right': (0, m, 0, round(center_m / 2)),
                   'bottom_left': (round(center_m / 2 * 3), m, 0, n),
                   'bottom_right': (0, m, round(center_n / 2 * 3), n)}

        for name, bounds in regions.items():
            max_location[name] = locate_max_in_subset(foo, bounds)

        corners = [max_location['bottom_left'], max_location['bottom_right'], max_location['top_left'],
                   max_location['top_right']]

        sep = pdist(corners, 'euclidean')
        if any(sep < center_m):
            # print('error encountered but disabled')
            raise ValueError('two or more recovered image corner locations are too close to eachother'
                             '\n add to cleanLandsatDataDir corruptImages list then run cleanLandsatDataDir')

    slope1 = calculate_slope(max_location['bottom_right'], max_location['bottom_left'])
    slope2 = calculate_slope(max_location['top_right'], max_location['top_left'])
    slope3 = calculate_slope(max_location['top_left'], max_location['bottom_left'])
    slope4 = calculate_slope(max_location['bottom_right'], max_location['top_right'])

    fooA = np.full((m, n), False)
    fooA[center_m - 70:center_m + 70, :] = 1
    fooA[:, center_n - 100:center_n + 100] = 0

    A = rotate(fooA, np.rad2deg(np.nanmax([slope1, slope2])), resize=False)
    B = rotate(fooA, np.rad2deg(np.nanmax([slope3, slope4])), resize=False)

    shiftCtr = (round(s['centroid-0'] - center_m), round(s['centroid-1'] - center_n))

    matrix = np.array([(1, 0, 0), (0, 1, 0), (shiftCtr[0], shiftCtr[1], 1)])
    tform = AffineTransform(matrix=matrix)
    A = warp(A, tform)
    B = warp(B, tform)

    Ix = valid_domain.copy()  # unsure how this works
    for k in range(Ix.shape[2]):
        fftIm = Ix[:, :, k]
        fftIm[fftIm > 3] = 3
        fftIm[fftIm < -3] = -3
        fftIm[np.isnull(fftIm)] = 0
        fftIm = fftshift(fft2(np.float(fftIm)))
        P = np.abs(fftIm)
        mP = np.nanmean(P)
        stdP = np.nanstd(P)
        P = (P - mP) > 10 * stdP
        sA = np.nansum(P[A])
        sB = np.nansum(P[B])
        if (sA / sB >= 2 | sB / sA >= 2) & (sA > 500 | sB > 500):
            if sA > sB:
                mask = A.copy()
            elif sB > sA:
                mask = B.copy()

            foo1 = np.isnan(valid_domain[:, :, k])
            foo2 = np.real(ifft2(ifftshift(fftIm * (1 - (mask)))))
            foo2[foo1] = np.nan

            Ix[:, :, k] = foo2

    return Ix


if __name__ == '__main__':
    valid_domain = np.zeros((1000, 1000))
    valid_domain[300:700, 300:700] = 1
    valid_domain[900:950, 900:950] = 1

    out = fft_filter(valid_domain)
    print(out)
