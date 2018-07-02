//
// Created by ay27 on 17/4/9.
//

#include <stdio.h>
#include <jpeglib.h>
#include <setjmp.h>
#include <fstream>
#include <util/image_reader.h>
#include <util/binary_dumper.h>
#include <signal.h>
#include <random>
#include <chrono>
#include <algorithm>
#include <pthread.h>
#include <thread>

using namespace std;
using namespace SuperNeurons;

/* follow the instruction from <https://github.com/ellzey/libjpeg/blob/master/example.c> */

struct my_error_mgr {
    struct jpeg_error_mgr pub;    /* "public" fields */

    jmp_buf setjmp_buffer;    /* for return to caller */
};

typedef struct my_error_mgr *my_error_ptr;

METHODDEF(void)
my_error_exit(j_common_ptr cinfo) {
    /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
    my_error_ptr myerr = (my_error_ptr) cinfo->err;

    /* Always display the message. */
    /* We could postpone this until after returning, if we chose. */
    (*cinfo->err->output_message)(cinfo);

    /* Return control to the setjmp point */
    longjmp(myerr->setjmp_buffer, 1);
}


GLOBAL(int)
read_JPEG_file(const char *filename, uint8_t *dst, volatile size_t height = 0, volatile size_t width = 0, const size_t CC = 3) {
    struct jpeg_decompress_struct cinfo;
    struct my_error_mgr jerr;
    /* More stuff */
    FILE *infile;        /* source file */
    JSAMPARRAY buffer;        /* Output row buffer */
    size_t row_stride;        /* physical row width in output buffer */

    if ((infile = fopen(filename, "rb")) == NULL) {
//        fprintf(stderr, "can't open %s\n", filename);
        return 0;
    }

    /* Step 1: allocate and initialize JPEG decompression object */
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = my_error_exit;
    if (setjmp(jerr.setjmp_buffer)) {
        jpeg_destroy_decompress(&cinfo);
        fclose(infile);
        return 0;
    }

    jpeg_create_decompress(&cinfo);

    /* Step 2: specify data source (eg, a file) */
    jpeg_stdio_src(&cinfo, infile);

    /* Step 3: read file parameters with jpeg_read_header() */
    (void) jpeg_read_header(&cinfo, TRUE);

    bool CMYK_to_RGB = false;
    bool Gray_to_RGB = false;
    /* Step 4: set parameters for decompression */
    if (cinfo.jpeg_color_space == JCS_CMYK || cinfo.jpeg_color_space == JCS_YCCK) {
        // CMYK
        CMYK_to_RGB = true;
    } else if (cinfo.jpeg_color_space == JCS_RGB || cinfo.jpeg_color_space == JCS_YCbCr) {
        // RGB
    } else if (cinfo.jpeg_color_space == JCS_GRAYSCALE) {
        // Grayscale
        Gray_to_RGB = true;
    } else {
        // error condition here...
    }

    /* Step 5: Start decompressor */
    // Attention: the numrows argument must be small, it's best to be 1
    (void) jpeg_start_decompress(&cinfo);

    size_t C = (size_t) cinfo.output_components, H = (size_t )cinfo.output_height, W = (size_t )cinfo.output_width;
//    if (C != CC) {
//        printf("channel mismatch, C=%d, CC=%d\n", C, CC);
//    }

    row_stride = C * W;
    buffer = (*cinfo.mem->alloc_sarray)
            ((j_common_ptr) &cinfo, JPOOL_IMAGE, row_stride, H);

//    printf("height=%d, width=%d, components=%d\n", H, W, C);

    height = (height == 0) ? H : height;
    width = (width == 0) ? W : width;

    /* Now, we set the scale value of height and width */
    float scale_h = (float) H / (float) height;
    float scale_w = (float) W / (float) width;

    /* Step 6: while (scan lines remain to be read) */
    /* Every scan line stores the pixel in RGB continuously, every channel is 8 bit.
     * For example:
     * RGB RGB RGB ...
     * */

    /* Should be known that it's best practice to read one line every time,
     * since it's reported bugs in this function.
     */
    for (size_t h = 0; h < H; ++h) {
        (void) jpeg_read_scanlines(&cinfo, buffer + h, 1);
    }

    uint8_t *bilinear = new uint8_t[C * height * width];

    for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
            for (size_t c = 0; c < C; ++c) {
                int tmp = 0;
                int cnt = 1;
                tmp += buffer[(int) (h * scale_h)][(int) (w * scale_w) * C + c];
                if ((h * scale_h + 1) < H) {
                    tmp += buffer[(int) (h * scale_h + 1)][(int) (w * scale_w) * C + c];
                    cnt++;
                }
                if ((w * scale_w + 1) < W) {
                    tmp += buffer[(int) (h * scale_h)][(int) (w * scale_w + 1) * C + c];
                    cnt++;
                }
                if ((h * scale_h + 1) < H && (w * scale_w + 1) < W) {
                    tmp += buffer[(int) (h * scale_h + 1)][(int) (w * scale_w + 1) * C + c];
                    cnt++;
                }
                bilinear[(h * width + w) * C + c] = (uint8_t) (tmp / cnt);
            }
        }
    }

    /* Change the arrangement of rgb value, and scale down.
     * Now, the arrangement is:
     * rrr...ggg...bbb...
     */
    size_t index = 0;
    int k, tmp;
    for (size_t c = 0; c < min(C, CC); ++c) {
        for (size_t i = c; i < height * width * C; i += C) {
            if (CMYK_to_RGB) {
                k = bilinear[i + CC - c], tmp = bilinear[i];
                tmp = k - ((255 - tmp) * k >> 8);
                dst[index++] = (uint8_t) tmp;
            } else {
                dst[index++] = (uint8_t) bilinear[i];
            }
        }
    }
    if (Gray_to_RGB) {
//        printf("gray to rgb\n");
        for (size_t i = C; i < CC; ++i) {
            memcpy(dst + height * width * i, dst, height * width);
        }
    }

    /* Step 7: Finish decompression */
    (void) jpeg_finish_decompress(&cinfo);

    /* Step 8: Release JPEG decompression object */
    /* This is an important step since it will release a good deal of memory. */
    jpeg_destroy_decompress(&cinfo);
    delete[] bilinear;

    fclose(infile);
    return 1;
}

volatile bool keep_running = true;
string src_folder, dst_folder;
vector<size_t> processed_pos;
vector<size_t> end_pos;
pthread_mutex_t lock_for_processed_pos;
// the label type must be label_t, or else it may be truncated
vector<pair<string, label_t >> file_label_dict;

const string CHECKPOINT_FILENAME = "checkpoint";
const string SHUFFLED_FILENAME = "shuffled_list";

inline string gen_data_filepath(string file_prefix, size_t block_index) {
    return dst_folder + "/" + file_prefix + "_data_" + to_string(block_index) + ".bin";
}

inline string gen_label_filepath(string file_prefix, size_t block_index) {
    return dst_folder + "/" + file_prefix + "_label_" + to_string(block_index) + ".bin";
}

inline string gen_checkpoint_filepath() {
    return src_folder + "/" + CHECKPOINT_FILENAME;
}

inline string gen_shuffled_filepath() {
    return src_folder + "/" + SHUFFLED_FILENAME;
}

bool read_checkpoint_file(string checkpoint_path) {
    ifstream checkpoint_file(checkpoint_path, ios::in | ios::binary);
    if (checkpoint_file.fail()) {
        return false;
    }
    size_t cnt, tmp1, tmp2;
    checkpoint_file >> cnt;
    for (size_t i = 0; i < cnt; ++i) {
        checkpoint_file >> tmp1 >> tmp2;
        processed_pos.push_back(tmp1);
        end_pos.push_back(tmp2);
    }
    printf("------read checkpoint--------\n");
    for (size_t i = 0; i < cnt; ++i) {
        printf("%zu ", processed_pos[i]);
    }
    printf("\n");

    checkpoint_file.close();

    printf("read previous checkpoint success\n");

    return true;
}

bool read_list_file(string shuffled_path) {
    ifstream shuffled_file(shuffled_path, ios::in | ios::binary);
    if (shuffled_file.fail()) {
        return false;
    }

    string line;
    size_t space_loc;
    label_t label;
    while (getline(shuffled_file, line)) {
        space_loc = line.find_last_of(' ');
        label = atoi(line.substr(space_loc + 1).c_str());
        file_label_dict.push_back(make_pair(line.substr(0, space_loc), label));
    }
    shuffled_file.close();

    return true;
}

void intHandler(int sig) {
    printf("write checkpoint file and exit.\n");
    ofstream checkpoint_file(gen_checkpoint_filepath(), ios::out);

    pthread_mutex_lock(&lock_for_processed_pos);
    {
        checkpoint_file << processed_pos.size() << endl;
        for (size_t i = 0; i < processed_pos.size(); ++i) {
            checkpoint_file << processed_pos[i] << " " << end_pos[i] << endl;
        }
        checkpoint_file.close();
        keep_running = false;
    }
    pthread_mutex_unlock(&lock_for_processed_pos);

    ofstream list_file(gen_shuffled_filepath(), ios::out);
    char tmp[200];
    for (size_t i = 0; i < file_label_dict.size(); ++i) {
        sprintf(tmp, "%s %u\n", file_label_dict[i].first.c_str(), file_label_dict[i].second);
        list_file.write(tmp, strlen(tmp));
    }
    list_file.close();

}

void convert_one_block(const char *dst_data_path, const char *dst_label_path,
                       const size_t block_index,
                       size_t N, const size_t C, const size_t height, const size_t width,
                       const bool append, const bool should_delete) {
    printf("converter %zu, will process %zu files\n", block_index, N);

    size_t old_N;
    if (append) {
        ifstream read_meta(dst_data_path, ios::in | ios::binary);
        read_meta.read((char *) (&old_N), 8);
        printf("previous N: %zu\n", old_N);
        read_meta.close();

        N = old_N;
    }

    Dumper data_dumper(N, C, height, width, dst_data_path, append);
    // the label C should be 1
    Dumper label_dumper(N, 1, 1, 1, dst_label_path, append);

    uint8_t *dst = new uint8_t[height * width * C];
    int res;
    int failed_cnt = 0, success_cnt = 0;
    size_t start = processed_pos[block_index];
    size_t end = end_pos[block_index];
    for (size_t index = start; (index < end) && keep_running; ++index) {
        res = read_JPEG_file((src_folder + "/" + file_label_dict[index].first).c_str(), dst, height, width, C);
        if (res != 1) {
            printf("thread %lu : process file %s failed\n", block_index,
                   (src_folder + "/" + file_label_dict[index].first).c_str());
            failed_cnt += 1;
            continue;
        }

        // to avoid danger condition
        bool flag = false;
        pthread_mutex_lock(&lock_for_processed_pos);
        if (keep_running) {
            processed_pos[block_index] = index + 1;
            flag = true;
        }
        pthread_mutex_unlock(&lock_for_processed_pos);

        if (!flag) {
            break;
        }

        data_dumper.dump_image((const char *) dst, C * height * width);
        label_dumper.dump_label((&file_label_dict[index].second), 1);

        success_cnt += 1;
        if (success_cnt % 1000 == 0) {
            printf("thread %lu: success %d, failed %d\n", block_index, success_cnt, failed_cnt);
        }
        if (should_delete) {
            if (remove((src_folder + "/" + file_label_dict[index].first).c_str())) {
                printf("can not delete file %s\n", (src_folder + "/" + file_label_dict[index].first).c_str());
            } else {
//            printf("delete file %s\n", (src_folder + "/" + file_label_dict[index].first).c_str());
            }
        }

    }

    if (!append) {
        data_dumper.fix_N(success_cnt);
        label_dumper.fix_N(success_cnt);
    } else {
        data_dumper.fix_N(old_N + success_cnt);
        label_dumper.fix_N(old_N + success_cnt);
    }

    printf("convert %zu, failed cnt: %d, success cnt: %d, total: %d\n", block_index, failed_cnt, success_cnt,
           failed_cnt + success_cnt);

}

int main(int argc, char **argv) {
    // install the interrupt signal handler
    signal(SIGINT, intHandler);

    if (argc < 8) {
        fprintf(stderr, "Usage:\n"
                "convert_jpeg [src_folder] [list_file] [dst_folder] [out_file_prefix] [resize_height] [resize_width] [number_blocks_to_split] [-d]\n"
                "-d: if set it, delete files\n"
        );
        exit(-1);
    }
    src_folder = argv[1];
    string list_file_path = argv[2];
    dst_folder = argv[3];
    string file_prefix = argv[4];
    const size_t height = (const size_t) atoi(argv[5]),
            width = (const size_t) atoi(argv[6]),
            total_block = (const size_t) atoi(argv[7]),
            C = 3;
    bool should_delete = false;
    if (argc == 9) {
        if (strcmp(argv[8], "-d") == 0) {
            should_delete = true;
        }
    }
    printf("should delete ? %d\n", should_delete);

    bool exist = read_checkpoint_file(gen_checkpoint_filepath());
    exist = exist && read_list_file(gen_shuffled_filepath());

    bool should_append = exist;

    // check binary file exist
    if (exist && (processed_pos.size() != 0)) {
        for (size_t i = 0; i < processed_pos.size(); ++i) {
            ifstream check_data_file_exist(gen_data_filepath(file_prefix, (size_t) i));
            ifstream check_label_file_exist(gen_label_filepath(file_prefix, (size_t) i));
            if ((check_data_file_exist.fail()) || (check_label_file_exist.fail())) {
                should_append = false;
                printf("previous works loss, the binary files %s or %s does not exist.\n",
                       gen_data_filepath(file_prefix, (size_t) i).c_str(),
                       gen_label_filepath(file_prefix, (size_t) i).c_str());
                break;
            }
        }
    }

    // first time to convert, or previous work loss
    if (!should_append) {
        printf("start a new convert process\n");
        exist = read_list_file(list_file_path);
        if (!exist) {
            fprintf(stderr, "failed to open list file %s\n", list_file_path.c_str());
            exit(-1);
        }
        unsigned int seed;
        int shuffle_time = 100;
        for (int i = 0; i < shuffle_time; ++i) {
            seed = (unsigned int) std::chrono::system_clock::now().time_since_epoch().count();
            std::shuffle(file_label_dict.begin(), file_label_dict.end(), std::default_random_engine(seed));
        }

        printf("shuffle list %d time finish\n", shuffle_time);

        processed_pos.clear();
        end_pos.clear();

        size_t block_size = (size_t) floor(file_label_dict.size() / total_block);

        for (size_t i = 0; i < total_block; ++i) {
            size_t start = block_size * i, end = block_size * (i + 1);
            processed_pos.push_back(start);
            end_pos.push_back(end);
        }
        end_pos.pop_back();
        end_pos.push_back(file_label_dict.size());
    } else {
        printf("restart from previous work\n");
    }

    vector<thread> threads;
    for (size_t i = 0; i < total_block; ++i) {
        threads.push_back(thread([&](size_t block_id) {
            string dst_file_name = gen_data_filepath(file_prefix, block_id);
            string dst_label_name = gen_label_filepath(file_prefix, block_id);

            convert_one_block(dst_file_name.c_str(), dst_label_name.c_str(), block_id,
                              end_pos[block_id] - processed_pos[block_id], C, height, width,
                              should_append, should_delete);
        }, i));
    }

    for (size_t i = 0; i < total_block; ++i) {
        threads[i].join();
        printf("thread %zu finish\n", i);
    }

}
