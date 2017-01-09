import ffmpeg
import nmslib_vector
from PIL import Image
import json
import random
import numpy as np
from concurrent import futures
from skimage.feature import hessian_matrix
from skimage import img_as_float

def frame_patches(frame, patch_size):
    for x in xrange(0, frame.shape[0]-patch_size[0]):
        for y in xrange(0, frame.shape[1]-patch_size[1]):
            yield frame[x:(x+patch_size[0]), y:(y+patch_size[1]), :]

def video_patches(video_file, n_patches=100000, patch_size=(32, 32), frame_size=(800, 600)):
    reader = ffmpeg.VideoReader(video_file, frame_size[0], frame_size[1], offset=1000)
    prev_frame = None
    patch_count = 0
    while patch_count < n_patches:
        for patch in frame_patches(reader.get_frame(), patch_size):
            yield patch
            patch_count += 1
            if patch_count >= n_patches:
                break
    reader.close()

def new_index():
    return  [nmslib_vector.init(
            'l1', [], 'small_world_rand',
            nmslib_vector.DataType.VECTOR,
            nmslib_vector.DistType.FLOAT), None, None]

def init_index(idx, k=10):
    index_param = ['NN=%d' % k, 'initIndexAttempts=3', 'indexThreadQty=8']
    query_time_param = ['initSearchAttempts=3']
    nmslib_vector.createIndex(idx[0], index_param)
    nmslib_vector.setQueryTimeParams(idx[0], query_time_param)
    idx[2] = k

def mean_color(patch):
    size = patch.shape[0] * patch.shape[1]
    return (
            np.sum(patch[:, :, 0]) / size,
            np.sum(patch[:, :, 1]) / size,
            np.sum(patch[:, :, 2]) / size)

def color_distance(patch, color):
    r = (patch[:, :, 0] - color[0]) ** 2
    g = (patch[:, :, 1] - color[1]) ** 2
    b = (patch[:, :, 2] - color[2]) ** 2
    return np.sqrt(r + g + b)

def rgb_stack(mat2d):
    return mat2d.reshape((mat2d.shape[0], mat2d.shape[1], 1)).repeat(3, 2)


def hdr_blend(patch_a, patch_b):
    assert patch_a.shape == patch_b.shape, repr((patch_a.shape, patch_b.shape))

    patch_a = patch_a.astype(np.float32)
    patch_b = patch_b.astype(np.float32)

    #return (patch_a + patch_b) / 2

    mean_b = mean_color(patch_b)
    mean_a = mean_color(patch_a)
    mean = ((mean_a[0] + mean_b[0])/2,
            (mean_a[1] + mean_b[1])/2,
            (mean_a[2] + mean_b[2])/2)

    dist_a = color_distance(patch_a, mean)
    dist_b = color_distance(patch_b, mean)
    
    #res = np.zeros(patch_a.shape, dtype=patch_a.dtype)
    #mask_a = dist_a >= dist_b
    #mask_b = np.logical_not(mask_a)
    #res[mask_a] = patch_a[mask_a]
    #res[mask_b] = patch_b[mask_b]
    #return res

    dist_total = dist_a + dist_b

    a_frac = dist_a / dist_total
    b_frac = dist_b / dist_total

    mat_a = rgb_stack(a_frac)
    mat_b = rgb_stack(b_frac)

    term_a = patch_a * mat_a
    term_b = patch_b * mat_b

    return term_a + term_b

def single_hessian(patch):
    mats = []
    patch = patch.astype(np.float64)
    gray = (patch[:, :, 0]  + patch[:, :, 1] + patch[:, :, 2]) / 3
    if patch.shape[0] < 2 or patch.shape[1] < 2:
        return gray
    gray = gray / 128. - 1
    for el in np.gradient(gray):
        mats.append(el)
    return reduce(lambda a, b: a + b, mats) / len(mats)

def hessian_blend(patch_a, patch_b):
    assert patch_a.shape == patch_b.shape, repr((patch_a.shape, patch_b.shape))
    #hessian_a = 1. / single_hessian(patch_a)
    #hessian_b = 1. / single_hessian(patch_b)
    #total_hessian = (hessian_a + hessian_b)
    #weights_a = hessian_a / total_hessian
    #weights_b = hessian_b / total_hessian

    hessian_a = single_hessian(patch_a)
    hessian_b = single_hessian(patch_b)
    hessian_diff = (hessian_a - hessian_b)**2
    hessian_diff_norm = hessian_diff / np.sum(hessian_diff)

    #res = np.zeros(patch_a.shape, dtype=patch_a.dtype)
    #mask_a = hessian_diff_norm <= 0.5
    #mask_b = np.logical_not(mask_a)
    #res[mask_a] = patch_a[mask_a]
    #res[mask_b] = patch_b[mask_b]
    #return res

    weights_a = 1 / hessian_diff
    weights_b = hessian_diff

    r = patch_a[:, :, 0] * weights_a + patch_b[:, :, 0] * weights_b
    g = patch_a[:, :, 1] * weights_a + patch_b[:, :, 1] * weights_b
    b = patch_a[:, :, 2] * weights_a + patch_b[:, :, 2] * weights_b
    return np.transpose((r, g, b))

def patch_offset(patch, target):
    patch_target_row = patch[0, :, :]
    target_rep = np.tile(patch_target_row, (patch.shape[1], 1, 1))

    try:
        deltas = (target - target_rep) ** 2
        deltas = deltas.reshape((deltas.shape[0], deltas.shape[1] * deltas.shape[2]))
        dists = np.sqrt(np.sum(deltas, axis=0))
        x_idx = np.argmin(dists)
        x_delta = (patch.shape[0] - x_idx)
        if x_delta < 1:
            x_delta = 1
        return x_delta
    except:
        return 0

def apply_mask(mask, image):
    nz = np.nonzero(mask)
    max_x = np.amax(nz[0])
    min_x = np.amin(nz[0])
    max_y = np.amax(nz[1])
    min_y = np.amin(nz[1])
    return image[min_x:max_x, min_y:max_y, :]

work_pool = futures.ThreadPoolExecutor(max_workers=3)

class ThreadWorker(object):

    def __init__(self, worker_fn):
        self.worker_fn = worker_fn

    def call(self, *args, **kwargs):
        return work_pool.submit(self.worker_fn, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs).result()

class PatchList(object):

    def __init__(self):
        self.frozen = False
        self.patch_list = []
        self.patch_array = None
        self.height = None
        self.width = None

    def add(self, patch):
        assert not self.frozen
        height = patch.shape[1]
        width = patch.shape[0]
        if self.height is None:
            self.height = height
        assert self.height == height
        if self.width is None:
            self.width = width
        assert self.width == width
        res = len(self.patch_list)
        self.patch_list.append(patch)
        return res

    def freeze(self):
        self.patch_array = np.empty((self.width * len(self.patch_list), self.height, 3), dtype=np.uint8)
        offset = 0
        idx = 0
        while offset < self.patch_array.shape[0]:
            patch = self.patch_list[idx]
            self.patch_array[offset:(offset + patch.shape[0]), :, :] = patch
            offset += patch.shape[0]
            self.patch_list[idx] = None
            idx += 1
        self.patch_list = None
        self.frozen = True

    def __getitem__(self, idx):
        if self.frozen:
            offset = self.width * idx
            return self.patch_array[offset:(offset + self.width), :, :]
        else:
            return self.patch_list[idx]

    def __len__(self):
        if self.frozen:
            return self.patch_array.shape[0] / self.width
        else:
            return len(self.patch_list)

class PatchIndex(object):

    def __init__(self, side_margin=16, patch_size=(32, 32)):
        self.side_margin = side_margin
        self.patch_size = patch_size
        self.patches = PatchList()
        self.index_x = new_index()
        self.index_y = new_index()
        self.index_prev = new_index()
        self.x_worker = ThreadWorker(self.match_x)
        self.y_worker = ThreadWorker(self.match_y)
        self.prev_worker = ThreadWorker(self.match_prev)

    @classmethod
    def copy_from(cls, other):
        res = cls()
        res.side_margin = other.side_margin
        res.patch_size = other.patch_size
        res.patches = other.patches
        res.index_x = other.index_x
        res.index_y = other.index_y
        return res

    def add_vector(self, index, _id, vector):
        if index[1] is None:
            index[1] = vector.shape
        assert index[1] == vector.shape
        nmslib_vector.addDataPoint(index[0], _id, list(vector))

    def knn_query(self, index, vector):
        assert index[1] == vector.shape, repr((index[1], vector.shape))
        return nmslib_vector.knnQuery(index[0], index[2], list(vector))

    def init_indexes(self):
        print 'init x'
        init_index(self.index_x)
        print 'init y'
        init_index(self.index_y)
        print 'init prev'
        init_index(self.index_prev)

    def vectorize_x(self, patch):
        margin = patch[0:self.side_margin, :, :]
        return margin.astype(np.float32).flatten()

    def vectorize_y(self, patch):
        margin = patch[:, 0:self.side_margin, :]
        return margin.astype(np.float32).flatten()

    def vectorize(self, patch):
        return patch.astype(np.float32).flatten()

    def match_x(self, patch):
        max_x = patch.shape[0]
        margin = patch[(max_x - self.side_margin):max_x, :, :]
        vector = margin.astype(np.float32).flatten()
        ids = self.knn_query(self.index_x, vector)
        return self.patches[random.choice(ids)]

    def match_y(self, patch):
        max_y = patch.shape[1]
        margin = patch[:, (max_y - self.side_margin):max_y, :]
        vector = margin.astype(np.float32).flatten()
        ids = self.knn_query(self.index_y, vector)
        patches = [self.patches[v] for v in ids]
        return random.choice(patches)

    def match_prev(self, patch):
        vector = self.vectorize(patch)
        ids = self.knn_query(self.index_prev, vector)
        return self.patches[random.choice(ids)]

    def generate_image(self, width, height, prev=None, stepsize=16):
        res = np.empty((width, height, 3), dtype=np.float32)
        mask = np.zeros((width, height), dtype=np.bool_)
        if prev is None:
            patch = random.choice(self.patches)
        else:
            patch = prev[0:self.patch_size[0], 0:self.patch_size[1], :]
        res[0:patch.shape[0], 0:patch.shape[1], :] = patch
        x_offset = stepsize
        y_offset = 0

        while 1:
            try:
                if x_offset > width - self.patch_size[0]:
                    x_offset = 0
                    y_offset += stepsize

                if y_offset >= height - self.patch_size[1]:
                    break
                patch_x = self.x_worker.call(patch)

                if y_offset > patch.shape[0]:
                    prev_y = res[
                            x_offset:(x_offset+patch.shape[0]),
                            (y_offset-stepsize):(y_offset + patch.shape[1] - stepsize), :]
                    try:
                        patch_y = self.y_worker.call(prev_y)
                    except AssertionError:
                        print x_offset, y_offset

                else:
                    patch_y = None

                if prev is not None:
                    prev_inp = prev[
                            x_offset:(x_offset + patch.shape[0]), y_offset:(y_offset + patch.shape[1]), :]
                    prev_patch = self.prev_worker(prev_inp).astype(np.float32)
                else:
                    prev_patch = None

                patch_x = patch_x.result().astype(np.float32)
                if patch_y is not None:
                    patch_y = patch_y.result().astype(np.float32)
                    patch = hdr_blend(patch_x, patch_y)
                else:
                    patch = patch_x

                if prev_patch is not None:
                    patch = hdr_blend(patch, prev_patch)

                max_x = x_offset + self.patch_size[0]
                max_y = y_offset + self.patch_size[1]

                #patch_clip = patch[:(max_x - x_offset), :(max_y - y_offset), :]
                patch_clip = patch

                patch_mask = mask[x_offset:max_x, x_offset:max_y]

                if patch_mask.shape[0] > 0 and patch_mask.shape[1] > 0 and np.count_nonzero(patch_mask) > 0:
                    #print mask_t
                    bg_overlap = apply_mask(patch_mask, res[x_offset:max_x, y_offset:max_y, :])
                    patch_overlap = apply_mask(patch_mask, patch_clip)
                    #print patch_overlap.shape, bg_overlap.shape, patch_clip.shape, patch_mask.shape, np.count_nonzero(patch_mask)
                    patch_clip = hessian_blend(patch_overlap, bg_overlap)

                if patch_clip.shape != patch.shape:
                    patch_clip = patch

                res[x_offset:max_x, y_offset:max_y, :] = patch_clip
                mask[x_offset:max_x, y_offset:max_y] = 1
                x_offset += stepsize
            except KeyboardInterrupt:
                break
        return res.astype(np.uint8)

    def generate_video(self, width, height, frame_count):
        prev = None
        for i in xrange(frame_count):
            res = self.generate_image(width, height, prev)
            prev = res
            yield res

    def load_video(self, fname, frame_size=(640, 480), offset=0, frame_count=30 * 5):
        reader = ffmpeg.VideoReader(fname, frame_size[0], frame_size[1], offset=offset)
        prev_frame = None
        prev_shape = None
        for frame_idx in xrange(frame_count):
            frame = reader.get_frame()
            for x in xrange(0, frame.shape[0]-1, self.patch_size[0]):
                for y in xrange(0, frame.shape[1]-1, self.patch_size[1]):
                    patch = frame[x:(x + self.patch_size[0]), y:(y + self.patch_size[1]), :]
                    if prev_shape is not None and patch.shape != prev_shape:
                        continue
                    if prev_shape is None:
                        prev_shape = patch.shape
                    idx = self.patches.add(patch)
                    self.add_vector(self.index_x, idx, self.vectorize_x(patch))
                    self.add_vector(self.index_y, idx, self.vectorize_y(patch))
                    if prev_frame is not None:
                        prev_patch = prev_frame[x:(x + self.patch_size[0]), y:(y + self.patch_size[1]), :]
                        self.add_vector(self.index_prev, idx, self.vectorize(prev_patch))
            prev_frame = frame
        reader.close()
        self.patches.freeze()

