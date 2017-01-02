import ffmpeg
import nmslib_vector
from PIL import Image
import json
import random
import numpy as np

def frame_patches(frame, patch_size):
    for x in xrange(0, frame.shape[0]-patch_size[0]):
        for y in xrange(0, frame.shape[1]-patch_size[1]):
            yield frame[x:(x+patch_size[0]), y:(y+patch_size[1]), :]

def video_patches(video_file, n_patches=100000, patch_size=(32, 32), frame_size=(800, 600)):
    reader = ffmpeg.VideoReader(video_file, frame_size[0], frame_size[1], offset=1000)
    patch_count = 0
    while patch_count < n_patches:
        for patch in frame_patches(reader.get_frame(), patch_size):
            yield patch
            patch_count += 1
            if patch_count >= n_patches:
                break
    reader.close()

def new_index():
    return  nmslib_vector.init(
            'l1', [], 'small_world_rand',
            nmslib_vector.DataType.VECTOR,
            nmslib_vector.DistType.FLOAT)

def init_index(idx):
    index_param = ['NN=10', 'initIndexAttempts=3', 'indexThreadQty=8']
    query_time_param = ['initSearchAttempts=3']
    nmslib_vector.createIndex(idx, index_param)
    nmslib_vector.setQueryTimeParams(idx, query_time_param)

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

    dist_total = dist_a + dist_b

    a_frac = dist_a / dist_total
    b_frac = dist_b / dist_total

    mat_a = rgb_stack(a_frac)
    mat_b = rgb_stack(b_frac)

    term_a = patch_a * mat_a
    term_b = patch_b * mat_b

    return term_a + term_b

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

class PatchIndex(object):

    def __init__(self, side_margin=16, patch_size=(32, 32)):
        self.side_margin = side_margin
        self.patch_size = patch_size
        self.patches = []
        self.index_x = new_index()
        self.index_y = new_index()

    @classmethod
    def copy_from(cls, other):
        res = cls()
        res.side_margin = other.side_margin
        res.patch_size = other.patch_size
        res.patches = other.patches
        res.index_x = other.index_x
        res.index_y = other.index_y
        return res

    def init_indexes(self):
        init_index(self.index_x)
        init_index(self.index_y)

    def vectorize_x(self, patch):
        margin = patch[0:self.side_margin, :, :]
        return margin.astype(np.float32).flatten()

    def vectorize_y(self, patch):
        margin = patch[:, 0:self.side_margin, :]
        return margin.astype(np.float32).flatten()

    def match_x(self, patch):
        max_x = patch.shape[0]
        margin = patch[(max_x - self.side_margin):max_x, :, :]
        vector = list(margin.astype(np.float32).flatten())
        ids = nmslib_vector.knnQuery(self.index_x, 10, vector)
        return self.patches[random.choice(ids)]

    def match_y(self, patch):
        max_y = patch.shape[1]
        margin = patch[:, (max_y - self.side_margin):max_y, :]
        vector = list(margin.astype(np.float32).flatten())
        ids = nmslib_vector.knnQuery(self.index_y, 10, vector)
        return self.patches[random.choice(ids)]

    def generate_image(self, width, height):
        res = np.empty((width, height, 3), dtype=np.float32)
        patch = random.choice(self.patches)
        res[0:patch.shape[0], 0:patch.shape[1], :] = patch
        x_offset = patch.shape[0]
        y_offset = 0

        while 1:
            try:
                if x_offset > width - patch.shape[0]:
                    x_offset = 0
                    y_offset += patch.shape[1]

                if y_offset >= height:
                    break
                patch_x = self.match_x(patch).astype(np.float32)

                if y_offset > 0:
                    prev_y = res[x_offset:(x_offset+patch.shape[0]), (y_offset-patch.shape[1]):y_offset, :]
                    patch_y = self.match_y(prev_y).astype(np.float32)
                    patch = hdr_blend(patch_x, patch_y)

                else:
                    patch = patch_x

                patch = patch.astype(np.float32)

                #if x_offset > 0:
                #    target = res[x_offset:(x_offset + patch.shape[0]), y_offset:(y_offset + patch.shape[1]), :]
                #    x_delta = min(16, patch_offset(patch, target))

                #     x_start = x_offset - x_delta

                #    patch_overlap = patch[:x_delta, :, :]
                #    target_overlap = res[x_start:(x_start + x_delta), y_offset:(y_offset + patch.shape[1]), :]
                #    if target_overlap.shape == patch_overlap.shape:
                #        overlap_pixels = hdr_blend(patch_overlap, target_overlap)
                #        patch[:x_delta, :, :] = overlap_pixels
                #else:
                x_start = x_offset

                max_x = min(x_start + patch.shape[0], width)
                max_y = min(y_offset + patch.shape[1], height)

                res[x_start:max_x, y_offset:max_y, :] = patch[:(max_x - x_start), :(max_y - y_offset), :]
                x_offset = x_start + patch.shape[0]
            except KeyboardInterrupt:
                break
        return res.astype(np.uint8)

    def load_video(self, fname, **kwargs):
        for idx, patch in enumerate(video_patches(fname, **kwargs)):
            self.patches.append(patch)
            nmslib_vector.addDataPoint(self.index_x, idx, list(self.vectorize_x(patch)))
            nmslib_vector.addDataPoint(self.index_y, idx, list(self.vectorize_y(patch)))

    def save(self, fname):
        nms_x_fname = '%s_x.nms' % fname
        nms_p_fname = '%s_x.nms' % fname
        image_fname = '%s.png' % fname

        nmslib_vector.saveIndex(self.index_x, nms_x_fname)
        nmslib_vector.saveIndex(self.index_y, nms_y_fname)
        patch_image = np.empty((patch_size[0] * len(self.patches), patch_size[1], 3), dtype=np.uint8)
        offset = 0
        for patch in self.patches:
            patch_image[offset:(offset + patch.shape[0]), :, :] = patch
            offset += patch.shape[0]
        outp_image = Image.fromarray(patch_image)
        outp_image.save(image_fname)
        config = dict(
                nms_x_fname=nms_x_fname,
                nms_y_fname=nms_y_fname,
                image_fname=image_fname,
                patch_size=self.patch_size,
                side_margin=self.side_margin)
        with open(fname, 'w') as outf:
            json.dump(config, outf)

    @classmethod
    def load(cls, fname):
        with open(fname, 'r') as inf:
            config = json.load(inf)
        res = cls(side_margin=config['side_margin'], patch_size=config['patch_size'])
        nmslib_vector.loadIndex(res.index, config['nms_fname'])
        image_data = np.asarray(Image.open(config['image_fname']).convert('RGB'))
        patch_size = config['patch_size']
        res.patches = [image_data[x:(x+patch_size[0]), :, :] for x in \
                xrange(0, image_data.shape[0]-patch_size[0], patch_size[0])]
        return res

