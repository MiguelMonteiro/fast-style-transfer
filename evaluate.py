from __future__ import print_function
import numpy as np, os
from src import transform
import tensorflow as tf
from src.utils import save_img, get_img, exists, list_files
from argparse import ArgumentParser
from collections import defaultdict
import json
import subprocess
import numpy
from PIL import Image

BATCH_SIZE = 4
DEVICE = '/gpu:0'


def read_frames(pipe_in, batch_size, height, width):
    nbytes = height * width * 3
    count = 0
    frames = list()
    while count < batch_size:
        raw_image = pipe_in.stdout.read(nbytes)

        if len(raw_image) != nbytes:
            return frames
        else:
            image = numpy.fromstring(raw_image, dtype='uint8')
            image = image.reshape((height, width, 3))
            frames.append(image)
            count += 1
    return frames


def from_pipe(opts):
    command = ["ffprobe",
               '-v', "quiet",
               '-print_format', 'json',
               '-show_streams', opts.in_path]
    print(opts.in_path)
    # info = json.loads(str(subprocess.check_output(command), encoding="utf8"))
    info = json.loads(str(subprocess.check_output(command)))
    width = int(info["streams"][0]["width"])
    height = int(info["streams"][0]["height"])
    fps = round(eval(info["streams"][0]["r_frame_rate"]))

    command = ["ffmpeg",
               '-loglevel', "quiet",
               '-i', opts.in_path,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vcodec', 'rawvideo', '-']

    pipe_in = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10 ** 9, stdin=None, stderr=None)

    command = ["ffmpeg",
               '-loglevel', "info",
               '-y',  # (optional) overwrite output file if it exists
               '-f', 'rawvideo',
               '-vcodec', 'rawvideo',
               '-s', str(width) + 'x' + str(height),  # size of one frame
               '-pix_fmt', 'rgb24',
               '-r', str(fps),  # frames per second
               '-i', '-',  # The imput comes from a pipe
               '-an',  # Tells FFMPEG not to expect any audio
               '-c:v', 'libx264',
               '-preset', 'slow',
               '-crf', '18',
               opts.out]

    pipe_out = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=None, stderr=None)
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with g.as_default(), g.device(opts.device), tf.Session(config=soft_config) as sess:

        batch_shape = (None, height, width, 3)
        img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
        tf_preds = transform.net(img_placeholder)

        saver = tf.train.Saver()

        if os.path.isdir(opts.checkpoint):
            ckpt = tf.train.get_checkpoint_state(opts.checkpoint)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise Exception("No checkpoint found...")
        else:
            saver.restore(sess, opts.checkpoint)
        f = 0
        while True:
            frames = read_frames(pipe_in, opts.batch_size, height, width)
            if not frames:
                break
            f += len(frames)
            preds = sess.run(tf_preds, feed_dict={img_placeholder: np.stack(frames)})
            print("Number of frames processed %s" % f)
            for i in range(len(preds)):
                img = np.clip(preds[i], 0, 255).astype(np.uint8)
                try:
                    pipe_out.stdin.write(img)
                except IOError as err:
                    ffmpeg_error = pipe_out.stderr.read()
                    error = (str(err) + ("\n\nFFMPEG encountered"
                                         "the following error while writing file:"
                                         "\n\n %s" % ffmpeg_error))
                    print(error)

        pipe_out.terminate()
        pipe_in.terminate()
        pipe_out.stdin.close()
        pipe_in.stdout.close()
        del pipe_in
        del pipe_out


def input_fn(input_filenames, out_filenames, batch_shape):
    print(input_filenames)
    input_filename_queue = tf.train.string_input_producer(input_filenames, num_epochs=1)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(input_filename_queue)

    image = tf.cast(tf.image.decode_jpeg(image_file, channels=3), tf.float32)
    image = tf.reshape(image, batch_shape[1:])
    output_filename_queue = tf.train.string_input_producer(out_filenames, num_epochs=1)
    output_file_name = output_filename_queue.dequeue()
    batch_size = batch_shape[0]

    capacity = 10 * batch_size
    image, output_file_name = tf.train.batch([image, output_file_name], batch_size=batch_size, capacity=capacity,
                                             allow_smaller_final_batch=True)
    return image, output_file_name


def ffwd(data_in, paths_out, checkpoint_dir, device_t='/gpu:0', batch_size=4):
    assert len(paths_out) > 0
    is_paths = type(data_in[0]) == str
    if is_paths:
        assert len(data_in) == len(paths_out)
        img_shape = get_img(data_in[0]).shape
    else:
        assert data_in.size[0] == len(paths_out)
        img_shape = X[0].shape

    g = tf.Graph()

    batch_size = min(len(paths_out), batch_size)
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with g.as_default(), g.device(device_t):
        batch_shape = (batch_size,) + img_shape

        tf_images, tf_output_paths = input_fn(data_in, paths_out, batch_shape)
        # img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')

        tf_preds = transform.net(tf_images / 255.0)
        tf_preds = tf.cast(tf.clip_by_value(tf_preds, 0, 255), dtype=tf.uint8)
        tf_preds = tf.map_fn(tf.image.encode_jpeg, tf_preds, dtype=tf.string)
        saver = tf.train.Saver()

        with tf.train.MonitoredSession(session_creator=tf.train.ChiefSessionCreator(config=soft_config)) as sess:

            if os.path.isdir(checkpoint_dir):
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    raise Exception("No checkpoint found...")
            else:
                saver.restore(sess, checkpoint_dir)

            while not sess.should_stop():
                preds, output_paths = sess.run([tf_preds, tf_output_paths])
                print(output_paths)
                for pred, output_path in zip(preds, output_paths):
                    save_img(output_path, pred)


def ffwd_to_img(in_path, out_path, checkpoint_dir, device='/cpu:0'):
    paths_in, paths_out = [in_path], [out_path]
    ffwd(paths_in, paths_out, checkpoint_dir, batch_size=1, device_t=device)


def ffwd_different_dimensions(in_path, out_path, checkpoint_dir, device_t=DEVICE, batch_size=4):
    in_path_of_shape = defaultdict(list)
    out_path_of_shape = defaultdict(list)
    for i in range(len(in_path)):
        in_image = in_path[i]
        out_image = out_path[i]
        shape = "%dx%d" % Image.open(in_image).size
        # shape = "%dx%dx%d" % get_img(in_image).shape
        in_path_of_shape[shape].append(in_image)
        out_path_of_shape[shape].append(out_image)
    for shape in in_path_of_shape:
        print('Processing images of shape %s' % shape)
        ffwd(in_path_of_shape[shape], out_path_of_shape[shape], checkpoint_dir, device_t, batch_size)


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        dest='checkpoint_dir',
                        help='dir or .ckpt file to load checkpoint from',
                        metavar='CHECKPOINT', required=True)

    parser.add_argument('--in-path', type=str,
                        dest='in_path', help='dir or file to transform',
                        metavar='IN_PATH', required=True)

    help_out = 'destination (dir or file) of transformed file or files'
    parser.add_argument('--out-path', type=str,
                        dest='out_path', help=help_out, metavar='OUT_PATH',
                        required=True)

    parser.add_argument('--device', type=str,
                        dest='device', help='device to perform compute on',
                        metavar='DEVICE', default=DEVICE)

    parser.add_argument('--batch-size', type=int,
                        dest='batch_size', help='batch size for feedforwarding',
                        metavar='BATCH_SIZE', default=BATCH_SIZE)

    parser.add_argument('--allow-different-dimensions', action='store_true',
                        dest='allow_different_dimensions',
                        help='allow different image dimensions')

    return parser


def check_opts(opts):
    exists(opts.checkpoint_dir, 'Checkpoint not found!')
    exists(opts.in_path, 'In path not found!')
    if os.path.isdir(opts.out_path):
        exists(opts.out_path, 'out dir not found!')
        assert opts.batch_size > 0


def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts(opts)

    if not os.path.isdir(opts.in_path):
        if os.path.exists(opts.out_path) and os.path.isdir(opts.out_path):
            out_path = os.path.join(opts.out_path, os.path.basename(opts.in_path))
        else:
            out_path = opts.out_path
        ffwd_to_img(opts.in_path, out_path, opts.checkpoint_dir, device=opts.device)
    else:
        files = list_files(opts.in_path)
        full_in = [os.path.join(opts.in_path, x) for x in files]
        full_out = [os.path.join(opts.out_path, x) for x in files]
        if opts.allow_different_dimensions:
            ffwd_different_dimensions(full_in, full_out, opts.checkpoint_dir,
                                      device_t=opts.device, batch_size=opts.batch_size)
        else:
            ffwd(full_in, full_out, opts.checkpoint_dir, device_t=opts.device,
                 batch_size=opts.batch_size)


if __name__ == '__main__':
    main()
