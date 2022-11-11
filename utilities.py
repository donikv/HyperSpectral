import cv2
import numpy as np
import colour

def load_png(file):
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 2**14  # type: ignore
    return img

def load_spec(file):
    s = np.load(file, allow_pickle=True).item()
    spd = colour.SpectralDistribution(s)
    return spd

def load_camera_spec(file):
    mspds = np.loadtxt(file)
    mspds = {x[0]: (x[1], x[2], x[3]) for x in mspds}
    mspd = colour.MultiSpectralDistributions(mspds)  # type: ignore
    return mspd

def create_spectral_dict(file):
    graph = cv2.imread(file)
    graph_mask = graph[:,:,0] == graph[:,:,1]
    graph_mask = np.invert(graph_mask)
    graph = graph * graph_mask[:,:,np.newaxis]

    gray = cv2.cvtColor(graph, cv2.COLOR_BGR2GRAY)[:1190,170:1880]
    gray = gray[::-1,:]

    adapt_thresh = cv2.adaptiveThreshold(gray, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY_INV, 7, -2)
    adapt_thresh = np.invert(adapt_thresh)
    
    lines = np.linspace(0, adapt_thresh.shape[1], num=400, endpoint=False).astype(np.int32)
    dots = [np.max(np.append(np.flatnonzero(adapt_thresh[:,i]), [0])) for i in lines]
    spd_dict = dict(zip(range(380, 780), dots / np.max(dots)))
    return spd_dict

def parse_camspec(lines):
    out = {}
    while len(lines) > 0:
        try:
            line = lines.pop(0)
            msds = None
            if line[0].isalpha():
                x = np.array(list(range(400,730,10)))
                r = np.array([float(x) for x in lines.pop(0).strip().split("\t")])
                g = np.array([float(x) for x in lines.pop(0).strip().split("\t")])
                b = np.array([float(x) for x in lines.pop(0).strip().split("\t")])
                d = {x[i]:(r[i], g[i], b[i]) for i in range(len(x))}
                msds = colour.MultiSpectralDistributions(d)
            out[line.strip()] = msds
        except IndexError as e: 
            return out
    return out

class FilteredImageCollection():
    def __init__(self, images, filter_specs, camera_sensitivity, illumination) -> None:
        self.images = images
        self.filter_specs = filter_specs
        self.camera_sensitivity = camera_sensitivity
        self.illumination = illumination