//
// Created by ay27 on 7/24/17.
//

#include <util/binary_dumper.h>
#include <cinttypes>
#include <string.h>

namespace SuperNeurons {


Dumper::Dumper(size_t n, size_t c, size_t h, size_t w, const char *dst_path, bool append) :
        n(n), c(c), h(h), w(w), dst_path(dst_path) {
    // test if file exist
    std::ifstream file_exist(dst_path, std::ios::in);
    if (file_exist.is_open() && !append) {
        fprintf(stderr, "file \"%s\" exists, exit and do nothing\n", dst_path);
        exit(-1);
    }

    file.open(dst_path, std::ios::out | std::ios::binary | std::ios::app);
    if (file.fail()) {
        fprintf(stderr, "failed to open file %s\n", dst_path);
        exit(-1);
    }

    if (!append) {
        write_meta();
    }

    if (append && file_exist.fail()) {
        write_meta();
    }

    file_exist.close();
}

Dumper::~Dumper() {
    printf("free dumper\n");
    if (file.is_open()) {
        file.close();
    }
}

/**
 * write meta data in meta_t type. this function **must** keep consistency with base_reader::read_meta !!!
 */
void Dumper::write_meta() {
    printf("in dumper, n=%" PRIu64 ", c=%" PRIu64 ", h=%" PRIu64 ", w=%" PRIu64 "\n", n, c, h, w);
    char meta[MAGIC_LEN];
    memset(meta, 0, MAGIC_LEN);
    memcpy(meta, &n, META_ATOM);
    memcpy(meta + META_ATOM, &c, META_ATOM);
    memcpy(meta + META_ATOM * 2, &h, META_ATOM);
    memcpy(meta + META_ATOM * 3, &w, META_ATOM);

    file.write(meta, MAGIC_LEN);
}


void Dumper::dump_image(const char *src_data, size_t data_cnt) {
    file.write(src_data, data_cnt);
    file.flush();
}

void Dumper::dump_label(const char *src_data, size_t data_cnt) {
    label_t *tmp = new label_t[data_cnt];
    expand_byte(src_data, tmp, data_cnt);
    file.write((const char *) tmp, data_cnt * sizeof(label_t));
    file.flush();
}

void Dumper::dump_label(const label_t *src_data, size_t data_cnt) {
    file.write((const char *) src_data, data_cnt * sizeof(label_t));
    file.flush();
}

/**
 * Re-fix the meta data N with given value.
 * Sometimes when dump some images, we can not give the correct N before all images are dumped finish,
 * so we should re-fix the correct N when it's finished.
 */
void Dumper::fix_N(size_t n) {
    std::ofstream tmp_file(dst_path, std::ios::in | std::ios::out);
    tmp_file.seekp(0, std::ios::beg);
    tmp_file.write((char *) (&n), META_ATOM);
    tmp_file.close();
}

} // namespace SuperNeurons