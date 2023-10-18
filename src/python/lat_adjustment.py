from functools import reduce
import astropy.units as u
# from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.coordinates import Helioprojective
import time
# from tqdm.auto import tqdm
# import sunpy
import numpy as np


def trans(ts, xs, ys):
    c = Helioprojective(xs, ys, obstime=ts, observer='earth')
    # if the above version get a warning, use the bellow version
    # and import SkyCoord
    # the above version may not be allowed in a newer Sunpy or Astropy version
    # c = SkyCoord(xs, ys, frame='helioprojective', obstime=ts,
    #              observer='earth')
    with Helioprojective.assume_spherical_screen(c.observer,
                                                 only_off_disk=True):
        c_hgs = c.transform_to(frames.HeliographicStonyhurst)
    return c_hgs


def main() -> None:
    # e.g. the latitude of prominences are stored in `../../data/P_full.txt`
    srcs = ['../../data/P_full', '../../data/AR_full']
    for (i, src) in enumerate(srcs):
        print('#', i + 1, src)
        with open(src + '.txt', 'r') as f, open(src + '_lat_trans.txt',
                                                'w') as g:
            __st = time.time()
            lines = f.readlines()
            days, xs, ys = list(zip(*[t.split() for t in lines]))
            days = np.array(days)
            xs = np.array(list(map(float, xs))) * u.arcsec
            ys = np.array(list(map(float, ys))) * u.arcsec
            c = trans(days, xs, ys)
            lats = c.lat.degree
            lats = reduce(lambda x, y: str(x) + '\n' + str(y), lats)
            g.write(lats)
            __et = time.time()
            print(__et - __st)
    return None


if __name__ == '__main__':
    main()
