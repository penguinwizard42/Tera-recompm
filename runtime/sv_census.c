// Scarlet and Violet's Oodle decoder through the host: the shader census, the wild Pokemon population, the version
// exclusives and the trade evolutions.
//
// The shader census: every shader program of the player's own data.trpfs, extracted once at install time so that
// totk.exe --prepare-shaders can translate and build all of them before play (no stutter and no invisible effects the
// first time a scene, a move or a menu draws a shader).
//
// The population (setting population): how many wild Pokemon exist around the player at once. The game keeps it in
// world/data/system/object_generation_range, a small Oodle-packed FlatBuffers file (157 bytes packed, 312 unpacked,
// told apart by an FNV-1a 64 of its packed bytes); right after the game unpacks it, the count (15 in 4.0.0) and the
// distances at which wild Pokemon appear and go are written into the unpacked bytes. switch is the game's own; switch2
// (40) has the values of the fan mod Pokemon Compass for those fields, enhanced (80) and extreme (120) raise only the
// count. Offsets and values from KarPur's runtime/data_patch.c, which reports 120 together with 4096 shadows playing
// fine on its maintainer's PC; nothing is reserved for them, the game's own memory budget holds them.
//
// No version exclusives (setting no_exclusives): the wild Pokemon of both editions spawn. The three species tables of
// world/data/encount (pokedata*_array.bin, told apart by unpacked size and an FNV-1a 64 of their first packed bytes)
// give each entry versiontable { A: bool, B: bool }; an exclusive stores its own edition's flag and leaves the
// other's field absent, which FlatBuffers reads as false. Both editions ship the same tables. excl_fix makes both
// flags read the one stored byte without rewriting the file: the vtable of the exclusives that has room for both slots
// gets its absent slot pointed at the stored flag's offset, and the entries whose vtable is too short for the second
// slot are pointed at that vtable. It walks the table itself and leaves it alone unless it looks exactly like that.
//
// No trade evolutions (setting no_trade_evo): with no trading in the port, the Pokemon that evolve only by trading
// evolve on a level or while holding an item instead. See evo_fix below.
//
// Adapted from KarPur's runtime/shader_census.c and runtime/data_watch.c (https://github.com/Grunttanamo/KarPur @
// c2b2c0d738f2c769c24e2ceddb60592aba6e7778, MIT, Copyright (c) 2026 KarPur contributors): the Oodle hook, the
// archive walk, the BNSH extraction, the seed records, the population fields, excl_fix and evo_fix with its tables
// (the levels and substitute items are KarPur's maintainer's choices). Left out: KarPur's search for programs recorded in play, its
// container report, the team icons and the growth curves.
//
// How it works. Scarlet and Violet keep their shaders in BNSH containers inside Oodle-packed files of arc/data.trpfs
// (standalone .bnsh/.trsha files, SARC UI archives and VFXB effect files), and only the game's own decoder unpacks
// Oodle. The patch set sv_oodle (recomp/patches-v400.txt) sends the game's two calls of OodleLZ_Decompress
// (main+0xcc8750) to host_oodle below; with TOTK_SHADER_CENSUS=1 (sv_oodle_init turns that set on) the first call runs
// the census on that thread: it walks every pack of data.trpfs, unpacks each file with the game's decoder (the call's own
// extra arguments, the stack ones stay where they are: rt_call leaves sp alone) into a buffer below RT_ADDRESS_LIMIT,
// where guest addresses are host addresses, and finds every BNSH in the result. A program's shader info holds u64
// control offset, u64 code offset and u32 code length from the BNSH's start, and the control section's length field
// (+0x704) repeats that length. The stage comes from the Maxwell shader program header after the code's first 0x30
// bytes (type 1 with stage 1..4, type 2 with stage 5), else compute with a sane local size.
//
// Output, next to totk.exe (the player's own data, never shared): seed-programs.bin and seed-answers.bin in the record
// layouts of runtime/gfx_shader.c, one default answers record per program (no Scarlet/Violet shader asks for texture or
// image types, so float vertex attributes, float colour outputs and the control section's compute local size at
// +0x748). The log gets counts only. The process ends when the files are written.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "fs.h"
#include "gfx.h"
#include "host.h"

#if __has_include("patches.h")
#include "patches.h"
#include "platform/platform.h"
#endif

#ifdef RT_PATCH_PAGE

#define GUEST_OODLE 0x7000cc8750ull       // Scarlet/Violet 4.0.0 main: OodleLZ_Decompress
#define HOST_OODLE (RT_PATCH_PAGE + 0xf60)
#define CONTROL 0x880
#define PRG_HEAD (12 + 32 + CONTROL)       // gfx_shader.c PRG_HEAD
#define ANS_SIZE (4 + 32 + 32 + 147 + 1)   // gfx_shader.c ANS_REC
#define BUF_SIZE (1ull << 30)              // one packed + unpacked file, guest-visible
#define OUT_CHUNK (64u << 20)

static int census_wanted(void) {
    static int on = -1;
    if (on < 0) on = RT_SV && rt_knob("SHADER_CENSUS", "0");
    return on;
}

static uint32_t u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t u64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }

// ---- FlatBuffers, read-only and bounds-checked (data.trpfd, the data.trpfs root and each pack are FlatBuffers)
typedef struct { const uint8_t *b; uint64_t n, pos; uint16_t nslots; const uint8_t *vt; } Fb;

static int fb_at(Fb *t, const uint8_t *b, uint64_t n, uint64_t pos) {
    if (pos + 4 > n) return 0;
    int32_t back;
    memcpy(&back, b + pos, 4);
    uint64_t vt = pos - (int64_t)back;
    if (vt + 4 > n) return 0;
    uint16_t vsize;
    memcpy(&vsize, b + vt, 2);
    t->b = b, t->n = n, t->pos = pos, t->vt = b + vt, t->nslots = vsize >= 4 ? (uint16_t)((vsize - 4) / 2) : 0;
    return 1;
}
static int fb_root(Fb *t, const uint8_t *b, uint64_t n) { return n >= 4 && fb_at(t, b, n, u32(b)); }
static uint16_t fb_slot(const Fb *t, int i) {
    uint16_t s = 0;
    if (i >= 0 && i < t->nslots) memcpy(&s, t->vt + 4 + 2 * i, 2);
    return s;
}
static int fb_vec(const Fb *t, int i, uint64_t *start, uint64_t *count) {   // vector field i: its elements, its count
    uint16_t s = fb_slot(t, i);
    if (!s) return 0;
    uint64_t at = t->pos + s;
    if (at + 4 > t->n) return 0;
    uint64_t v = at + u32(t->b + at);
    if (v + 4 > t->n) return 0;
    *start = v + 4, *count = u32(t->b + v);
    return 1;
}
static int64_t fb_i8(const Fb *t, int i, int64_t def) {
    uint16_t s = fb_slot(t, i);
    return s && t->pos + s < t->n ? (int8_t)t->b[t->pos + s] : def;
}
static uint64_t fb_u64(const Fb *t, int i) {
    uint16_t s = fb_slot(t, i);
    return s && t->pos + s + 8 <= t->n ? u64(t->b + t->pos + s) : 0;
}
static int fb_table(const Fb *t, uint64_t vstart, uint64_t k, Fb *out) {
    uint64_t e = vstart + 4 * k;
    return e + 4 <= t->n && fb_at(out, t->b, t->n, e + u32(t->b + e));
}
static const char *fb_string(const Fb *t, uint64_t vstart, uint64_t k, uint32_t *len) {
    uint64_t e = vstart + 4 * k;
    if (e + 4 > t->n) return NULL;
    uint64_t s = e + u32(t->b + e);
    if (s + 4 > t->n) return NULL;
    *len = u32(t->b + s);
    return s + 4 + *len <= t->n ? (const char *)t->b + s + 4 : NULL;
}

static uint64_t fnv1a64(const char *s, uint32_t n) {   // a pack's name hash in data.trpfs
    uint64_t h = 0xCBF29CE484222645ull;
    for (uint32_t i = 0; i < n; i++) h = (h ^ (uint8_t)s[i]) * 0x100000001B3ull;
    return h;
}

// ---- extraction
#define HSET_BITS 20
typedef struct { uint8_t h[32]; uint8_t used; } HSlot;
static HSlot *seen;   // code hashes already written
static uint64_t n_bnsh, n_infos, n_out, n_stage[6], n_not_shader, n_fail, out_bytes;
static FILE *prg_f, *ans_f;
static uint8_t *out_buf;
static size_t out_n;

static int seen_put(const uint8_t h[32]) {   // 1 when new
    uint64_t i;
    memcpy(&i, h, 8);
    for (i &= (1u << HSET_BITS) - 1;; i = (i + 1) & ((1u << HSET_BITS) - 1)) {
        if (!seen[i].used) { memcpy(seen[i].h, h, 32), seen[i].used = 1; return 1; }
        if (!memcmp(seen[i].h, h, 32)) return 0;
    }
}

static int guess_stage(const uint8_t *code, uint32_t len) {
    if (len < 0x30 + 0x50) return GFX_STAGE_CS;
    uint32_t w = u32(code + 0x30), type = w & 0x1f, ver = (w >> 5) & 0x1f, st = (w >> 10) & 0xf;
    if (ver == 0) return GFX_STAGE_CS;
    if (type == 1 && st >= 1 && st <= 4) return (int)st - 1;   // vertex, tessellation control / evaluation, geometry
    if (type == 2 && st == 5) return GFX_STAGE_FS;
    return GFX_STAGE_CS;
}

static void out_flush(void) {
    if (prg_f && out_n) fwrite(out_buf, 1, out_n, prg_f);
    out_n = 0;
}

static void take_program(const uint8_t *ctrl, const uint8_t *code, uint32_t len) {
    int g = guess_stage(code, len);
    if (g == GFX_STAGE_CS) {   // no shader program header: compute only with a sane local size (KarPur found the others,
        uint64_t total = 1;    // with sizes like 37054 x 0 x 36, fail to translate and are no shaders)
        for (int k = 0; k < 3; k++) {
            uint32_t l = u32(ctrl + 0x748 + 4 * k);
            total *= l > 1024 ? 1u << 20 : l ? l : 1;
        }
        if (total > 1024) { n_not_shader++; return; }
    }
    uint8_t h[32];
    if (!gfx_shader_code_hash(g, ctrl, code, h) || n_out >= (7u << HSET_BITS) / 8) { n_fail++; return; }
    if (!seen_put(h)) return;
    uint32_t c1_off = u32(ctrl + 0x700), c1 = len - c1_off;
    size_t rec = PRG_HEAD + (size_t)len + c1;
    if (rec > OUT_CHUNK) { n_fail++; return; }
    if (out_n + rec > OUT_CHUNK) out_flush();
    n_out++, n_stage[g]++;
    uint8_t *o = out_buf + out_n;
    uint32_t head[3] = {(uint32_t)g, len, c1};
    memcpy(o, head, 12), memcpy(o + 12, h, 32), memcpy(o + 44, ctrl, CONTROL), memcpy(o + PRG_HEAD, code, len);
    memcpy(o + PRG_HEAD + len, code + c1_off, c1);
    out_n += rec, out_bytes += rec;
    // its answers record: stage, code hash, input id 0 (the runtime finds it), packed default answers, priority 2
    uint8_t a[ANS_SIZE] = {0};
    a[0] = (uint8_t)g;
    memcpy(a + 4, h, 32);
    if (g == GFX_STAGE_CS)   // pack_answers: the local size's three u16 after the 136 bytes before it
        for (int k = 0; k < 3; k++) {
            uint32_t l = u32(ctrl + 0x748 + 4 * k);
            l = l ? l : 1;
            a[68 + 136 + 2 * k] = (uint8_t)l, a[68 + 137 + 2 * k] = (uint8_t)(l >> 8);
        }
    a[ANS_SIZE - 1] = 2;
    if (ans_f) fwrite(a, sizeof a, 1, ans_f);
}

static void extract(const uint8_t *b, uint64_t n) {
    for (uint64_t base = 0; base + 0x60 <= n; base += 4) {
        // "BNSH", 0, byte-order mark 0xFEFF at +0xC, file size at +0x1C
        if (u32(b + base) != 0x48534E42u || u32(b + base + 4) || (u32(b + base + 0xC) & 0xFFFF) != 0xFEFF) continue;
        uint64_t size = u32(b + base + 0x1C);
        if (size < 0x60 || size > n - base) continue;
        const uint8_t *bn = b + base;
        n_bnsh++;
        for (uint64_t j = 0x20; j + 24 <= size; j += 8) {
            uint64_t co = u64(bn + j), xo = u64(bn + j + 8);
            uint32_t len = u32(bn + j + 16);
            if (!len || (xo & 0xFF) || (co & 7) || co > size || size - co < CONTROL || xo >= size || len > size - xo)
                continue;
            if (u32(bn + co + 0x704) != len || u32(bn + co + 0x700) > len) continue;
            n_infos++;
            take_program(bn + co, bn + xo, len);
        }
        base += size - 4;   // on to the next BNSH
    }
}

static uint8_t *read_file(uint64_t h, uint64_t off, uint64_t n) {
    uint8_t *b = malloc(n ? n : 1);
    uint64_t got = 0;
    if (b && (romfs_read(&got, h, (int64_t)off, b, n) != 0 || got != n)) { free(b); b = NULL; }
    return b;
}

__attribute__((noreturn)) static void census_fail(const char *why) {
    rt_log("shader census: %s; no seed written", why);
    fflush(NULL);
    exit(2);
}

static void census_run(Cpu *c, const uint64_t a[8]) {
    uint64_t t0 = plat_now_ns();
    rt_log("shader census: start (the process ends when it is done)");
    char path[PLAT_PATH_MAX * 2];
    int pn = plat_exe_path(path, sizeof path - 32);
    char *cut = pn ? plat_last_sep(path) : NULL;
    if (!cut) census_fail("no path of totk.exe");
    strcpy(cut + 1, "seed-programs.bin");
    prg_f = fopen(path, "wb");
    strcpy(cut + 1, "seed-answers.bin");
    ans_f = fopen(path, "wb");
    seen = calloc(1u << HSET_BITS, sizeof *seen), out_buf = malloc(OUT_CHUNK);
    if (!prg_f || !ans_f || !seen || !out_buf) census_fail("cannot write the seed files next to totk.exe");
    struct { char magic[8]; uint32_t version, rec_size; } hp = {"TOTKSPRG", 1, 0}, ha = {"TOTKSANS", 1, ANS_SIZE};
    fwrite(&hp, sizeof hp, 1, prg_f);
    fwrite(&ha, sizeof ha, 1, ans_f);

    uint64_t gbuf = 0;   // a guest-visible buffer at a free address below the guest address limit
    for (uint64_t at = 0x7E00000000ull; at >= 0x4000000000ull && !gbuf; at -= 0x200000000ull)
        if (plat_mem_reserve(at, BUF_SIZE, 0) && plat_mem_commit(at, BUF_SIZE)) gbuf = at;
    if (!gbuf) census_fail("no buffer below the guest address limit");

    char fd_path[64], fs_path[64];
    snprintf(fd_path, sizeof fd_path, "%s:/arc/data.trpfd", romfs_mount_name());
    snprintf(fs_path, sizeof fs_path, "%s:/arc/data.trpfs", romfs_mount_name());
    uint64_t hd, hs;
    if (romfs_open(&hd, fd_path, FS_OPEN_READ) || romfs_open(&hs, fs_path, FS_OPEN_READ))
        census_fail("cannot open arc/data.trpfd and arc/data.trpfs");
    int64_t fd_size = 0, fs_size = 0;
    romfs_size(&fd_size, hd), romfs_size(&fs_size, hs);
    uint8_t head[16], *fdb = read_file(hd, 0, (uint64_t)fd_size), *fsb = NULL;
    uint64_t got = 0, root_at = 0;
    if (fdb && !romfs_read(&got, hs, 0, head, 16) && got == 16 && !memcmp(head, "ONEPACK", 7) &&
        (root_at = u64(head + 8)) < (uint64_t)fs_size)
        fsb = read_file(hs, root_at, (uint64_t)fs_size - root_at);
    // data.trpfd root {file_hashes, pack_names, file_info, pack_info [{size, file_count}]}; data.trpfs root
    // {pack_hashes, pack_offsets}; a pack {file_hashes, files [{?, compression i8, level, size u64, data}]}
    Fb fd, fs;
    uint64_t names_v, names_n, info_v, info_n, ph_v, ph_n, po_v, po_n;
    if (!fsb || !fb_root(&fd, fdb, (uint64_t)fd_size) || !fb_root(&fs, fsb, (uint64_t)fs_size - root_at) ||
        !fb_vec(&fd, 1, &names_v, &names_n) || !fb_vec(&fd, 3, &info_v, &info_n) || !fb_vec(&fs, 0, &ph_v, &ph_n) ||
        !fb_vec(&fs, 1, &po_v, &po_n) || ph_n != po_n)
        census_fail("the archive's tables do not read as expected");
    rt_log("shader census: %llu packs, data.trpfs %.2f GiB", (unsigned long long)names_n, fs_size / 1073741824.0);

    uint64_t decodes = 0, failed = 0, too_big = 0, out_total = 0;
    for (uint64_t p = 0; p < names_n; p++) {
        uint32_t nl = 0;
        const char *nm = fb_string(&fd, names_v, p, &nl);
        Fb info;
        if (!nm || p >= info_n || !fb_table(&fd, info_v, p, &info)) continue;
        uint64_t psize = fb_u64(&info, 0), h = fnv1a64(nm, nl), poff = 0;
        int have = 0;
        for (uint64_t k = 0; k < ph_n && !have; k++)
            if (u64(fsb + ph_v + 8 * k) == h) poff = u64(fsb + po_v + 8 * k), have = 1;
        if (!have || !psize || psize > BUF_SIZE) continue;
        uint8_t *pk = read_file(hs, poff, psize);
        Fb pt;
        uint64_t fv, fn;
        if (!pk || !fb_root(&pt, pk, psize) || !fb_vec(&pt, 1, &fv, &fn)) { free(pk); continue; }
        for (uint64_t k = 0; k < fn; k++) {
            Fb e;
            uint64_t ds, dn;
            if (!fb_table(&pt, fv, k, &e) || !e.nslots || !fb_vec(&e, e.nslots - 1, &ds, &dn) || ds + dn > psize) continue;
            uint64_t unpacked = fb_u64(&e, 3);
            if (fb_i8(&e, 1, -1) == -1) { extract(pk + ds, dn); continue; }   // stored as is
            if (!unpacked || dn + 4096 + unpacked > BUF_SIZE) { too_big++; continue; }
            uint8_t *in = (uint8_t *)(uintptr_t)gbuf, *outp = in + ((dn + 4095) & ~4095ull);
            memcpy(in, pk + ds, dn);
            uint64_t r = rt_call(c, GUEST_OODLE, 8, (uint64_t)(uintptr_t)in, dn, (uint64_t)(uintptr_t)outp, unpacked,
                                 a[4], a[5], a[6], a[7]);
            decodes++;
            if (r != unpacked) { failed++; continue; }
            out_total += r;
            extract(outp, r);
        }
        free(pk);
        if (!(p % 2000))
            rt_log("shader census: %llu of %llu packs, %llu programs, %.0f s", (unsigned long long)p,
                   (unsigned long long)names_n, (unsigned long long)n_out, (plat_now_ns() - t0) / 1e9);
    }
    out_flush();
    int ok = !ferror(prg_f) && !ferror(ans_f);
    ok &= !fclose(prg_f) && !fclose(ans_f);
    rt_log("shader census: %llu packs, %llu Oodle decodes (%.1f GiB), %llu failed, %llu too big; %llu BNSH, %llu shader "
           "infos, %llu programs (vertex %llu, fragment %llu, compute %llu, other %llu), %.1f MiB; %llu left out (no "
           "shader header, no sane compute size), %llu not kept; %.0f s",
           (unsigned long long)names_n, (unsigned long long)decodes, out_total / 1073741824.0,
           (unsigned long long)failed, (unsigned long long)too_big, (unsigned long long)n_bnsh,
           (unsigned long long)n_infos, (unsigned long long)n_out, (unsigned long long)n_stage[GFX_STAGE_VS],
           (unsigned long long)n_stage[GFX_STAGE_FS], (unsigned long long)n_stage[GFX_STAGE_CS],
           (unsigned long long)(n_stage[1] + n_stage[2] + n_stage[3]), out_bytes / 1048576.0,
           (unsigned long long)n_not_shader, (unsigned long long)n_fail, (plat_now_ns() - t0) / 1e9);
    printf("shader census: %llu programs written (seed-programs.bin, seed-answers.bin)\n", (unsigned long long)n_out);
    fflush(NULL);
    exit(ok && n_out ? 0 : 2);
}

// ---- population
#define GEN_PACKED 157                       // object_generation_range: packed and unpacked size, FNV-1a 64 of the
#define GEN_UNPACKED 312                     // packed bytes
#define GEN_PRINT 0x8a61deb07c34472bull
static const struct { uint32_t at; char type; int32_t value; } gen_fields[] = {
    {0xd4, 'f', 20},   // wild Pokemon: nearest distance they appear at
    {0xd8, 'f', 60},   // farthest they appear at
    {0xdc, 'f', 80},   // nearest they go at
    {0xe0, 'f', 80},   // farthest they go at
    {0xe4, 'i', 0},    // how many at once: the level's count
    {0x3c, 'i', 5},    // values[8]: how many at once
};
static int wild_count = -1;   // 0: the game's own

static int population(void) {
    if (wild_count < 0) {
        const char *v = cfg_value("TOTK_POPULATION");
        wild_count = !RT_SV ? 0 : !strcmp(v, "switch2") ? 40 : !strcmp(v, "enhanced") ? 80 : !strcmp(v, "extreme") ? 120 : 0;
    }
    return wild_count;
}

static uint64_t fnv1a64_bytes(const uint8_t *p, uint64_t n) {
    uint64_t h = 0xCBF29CE484222645ull;
    for (uint64_t i = 0; i < n; i++) h = (h ^ p[i]) * 0x100000001B3ull;
    return h;
}

static void population_patch(const uint64_t a[8], uint64_t r) {
    if (r != GEN_UNPACKED || a[1] < GEN_PACKED || a[0] + GEN_PACKED > RT_ADDRESS_LIMIT || a[2] + r > RT_ADDRESS_LIMIT ||
        fnv1a64_bytes((const uint8_t *)(uintptr_t)a[0], GEN_PACKED) != GEN_PRINT)
        return;
    uint8_t *out = (uint8_t *)(uintptr_t)a[2];
    for (size_t i = 0; i < sizeof gen_fields / sizeof *gen_fields; i++) {
        int32_t v = gen_fields[i].value ? gen_fields[i].value : wild_count;
        if (gen_fields[i].type == 'f') {
            float f = (float)v;
            memcpy(out + gen_fields[i].at, &f, 4);
        } else {
            memcpy(out + gen_fields[i].at, &v, 4);
        }
    }
    static int logged;
    if (!logged++) rt_log("population: object_generation_range set to up to %d wild Pokemon at once", wild_count);
}

// ---- no version exclusives
static const struct { uint64_t unpacked, print; } excl_files[] = {   // Violet 4.0.0; the print over min(packed, 4096)
    {34724, 0x888111ef69f71aa3ull}, {36784, 0x8ae3f9b2548a5528ull}, {58984, 0x0a60102c8801df28ull},
};
#define EXCL_SLOT 28   // versiontable's slot in an entry (devid 0 .. bringItem 29)
#define EXCL_MAX 512
static int excl_on = -1;

static int no_exclusives(void) {
    if (excl_on < 0) excl_on = RT_SV && cfg_bool("TOTK_NO_EXCLUSIVES");
    return excl_on;
}

static uint16_t u16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static int64_t ex_vtable(const uint8_t *b, uint64_t pos) {
    int32_t d;
    memcpy(&d, b + pos, 4);
    return (int64_t)pos - d;
}
// The field's offset from the table's start, or 0 when its vtable has no slot for it or leaves it absent.
static uint16_t ex_slot(const uint8_t *b, uint64_t n, uint64_t pos, unsigned slot) {
    int64_t vt = ex_vtable(b, pos);
    if (vt < 0 || (uint64_t)vt + 4 > n) return 0;
    uint16_t size = u16(b + vt);
    uint64_t at = 4 + 2 * (uint64_t)slot;
    return at + 2 <= size && (uint64_t)vt + at + 2 <= n ? u16(b + vt + at) : 0;
}

static void excl_fix(uint8_t *b, uint64_t n) {
    if (n < 32) return;
    uint64_t root = u32(b);
    uint16_t vo = root + 8 <= n ? ex_slot(b, n, root, 0) : 0;   // the root's only field: the vector of entries
    if (!vo || root + vo + 4 > n) return;
    uint64_t v = root + vo;
    v += u32(b + v);
    if (v + 4 > n) return;
    uint32_t count = u32(b + v);
    if (!count || count > 100000 || v + 4 + 4 * (uint64_t)count > n) return;
    uint64_t pos[EXCL_MAX];
    int64_t vts[EXCL_MAX];
    int many = 0, both = 0;
    uint16_t off = 0;
    for (uint32_t k = 0; k < count; k++) {
        uint64_t e = v + 4 + 4 * (uint64_t)k;
        e += u32(b + e);
        if (e + 4 > n) return;
        uint16_t so = ex_slot(b, n, e, EXCL_SLOT);
        if (!so || e + so + 4 > n) continue;
        uint64_t vp = e + so;
        vp += u32(b + vp);
        if (vp + 4 > n) return;
        uint16_t a = ex_slot(b, n, vp, 0), bb = ex_slot(b, n, vp, 1);
        if (a && bb) { both++; continue; }
        if (!a && !bb) continue;                  // neither flag: the game's own default decides
        if (off && (a ? a : bb) != off) return;   // the stored flag must sit at one offset in all of them
        off = a ? a : bb;
        if (many == EXCL_MAX) return;
        pos[many] = vp, vts[many++] = ex_vtable(b, vp);
    }
    if (!many || !both) return;
    int64_t target = -1;   // a vtable with room for both slots: the one whose other edition's slot is only absent
    for (int i = 0; i < many; i++)
        if ((uint64_t)vts[i] + 8 <= n && u16(b + vts[i]) >= 8) target = vts[i];
    if (target < 0) return;
    uint16_t a = u16(b + target + 4), bb = u16(b + target + 6);
    if ((a != 0 && a != off) || (bb != 0 && bb != off) || (a && bb)) return;
    memcpy(b + target + (a ? 6 : 4), &off, 2);   // both slots now read the stored byte
    int moved = 0;
    for (int i = 0; i < many; i++) {
        if (vts[i] == target) continue;
        int32_t d = (int32_t)((int64_t)pos[i] - target);
        memcpy(b + pos[i], &d, 4);   // this entry's versiontable uses the vtable that has room
        moved++;
    }
    rt_log("no exclusives: %d of %u entries freed (%d pointed at the other vtable), flag at +%u", many, count, moved,
           off);
}

static void exclusives_patch(const uint64_t a[8], uint64_t r) {
    for (size_t i = 0; i < sizeof excl_files / sizeof *excl_files; i++) {
        if (excl_files[i].unpacked != r) continue;
        uint64_t head = a[1] < 4096 ? a[1] : 4096;
        if (a[0] + head > RT_ADDRESS_LIMIT || a[2] + r > RT_ADDRESS_LIMIT) return;
        uint64_t print = fnv1a64_bytes((const uint8_t *)(uintptr_t)a[0], head);
        if (print == excl_files[i].print) {
            excl_fix((uint8_t *)(uintptr_t)a[2], r);
        } else {
            static int told;
            if (told++ < 4)
                rt_log("no exclusives: a %llu-byte file with another fingerprint (%016llx), left alone",
                       (unsigned long long)r, (unsigned long long)print);
        }
        return;
    }
}

// ---- no trade evolutions
// The species table avalon/data/personal_array.bin (412572 bytes unpacked; Violet 4.0.0): a vector of 1424 entries,
// field 0 the species number and field 23 a vector of 16-byte evolution structs { u16 level, u16 method, u16 argument,
// u16 x3, u16 target species, u16 }. Every change fits inside a struct, so the file keeps its size.
// - Method 5 (by trading) becomes method 4 (on reaching the level in the first field), at the levels of trade_evo.
// - Method 6 (by trading holding an item, the argument a national item number) becomes method 19 (levels up by day
//   holding the item), whose argument is the item's plibID, the small item space of
//   world/data/battle/plib_item_conversion. Method 8 ("an item is used on it") does not do: these items can only be
//   held. Items with no plibID in the game's data get a substitute item (evo_item_own) or, where KarPur's tests found
//   no substitute that works, a level (evo_item_level).
#define EVO_SLOT 23
#define EVO_STRUCT 16
#define EVO_TRADE 5
#define EVO_LEVEL 4
#define EVO_TRADE_ITEM 6
#define EVO_HOLD_ITEM 19
static const struct { uint64_t unpacked, print; } evo_file[] = {{412572, 0x015856eb1c2b489aull}};
static const struct { uint16_t species, level; } trade_evo[] = {
    {64, 40},  {67, 39},  {75, 36},  {93, 42},   // Kadabra, Machoke, Graveler, Haunter
    {525, 35}, {533, 40}, {708, 30}, {710, 30},  // Boldore, Gurdurr, Phantump, Pumpkaboo
};
static const struct { uint16_t species, target, plib; } evo_item_own[] = {
    {61, 186, 87},    // Poliwhirl -> Politoed: plibID 87 (item 2345) for the King's Rock, which has none
    {79, 199, 87},    // Slowpoke -> Slowking: the same
    {366, 367, 111},  // Clamperl -> Huntail: plibID 111 (item 537) for the Deep Sea Tooth
    {366, 368, 27},   // Clamperl -> Gorebyss: plibID 27 (item 299), apart from Huntail's
};
static const struct { uint16_t species, target, level; } evo_item_level[] = {
    {95, 208, 40}, {123, 212, 30}, {682, 683, 20}, {684, 685, 20},   // Onix, Scyther, Spritzee, Swirlix
};
static const struct { uint16_t item, plib; } evo_item[] = {   // national item number -> plibID
    {235, 118}, {252, 113}, {321, 117}, {322, 115},   // Dragon Scale, Up-Grade, Protector, Electirizer
    {323, 116}, {324, 114}, {325, 112}, {537, 111},   // Magmarizer, Dubious Disc, Reaper Cloth, Prism Scale
};
static int evo_on = -1;

static int no_trade_evo(void) {
    if (evo_on < 0) evo_on = RT_SV && cfg_bool("TOTK_NO_TRADE_EVO");
    return evo_on;
}

static void put16(uint8_t *p, uint16_t v) { memcpy(p, &v, 2); }

static void evo_fix(uint8_t *b, uint64_t n) {
    if (n < 32) return;
    uint64_t root = u32(b);
    uint16_t vo = root + 8 <= n ? ex_slot(b, n, root, 0) : 0;
    if (!vo || root + vo + 4 > n) return;
    uint64_t v = root + vo;
    v += u32(b + v);
    if (v + 4 > n) return;
    uint32_t count = u32(b + v);
    if (!count || count > 100000 || v + 4 + 4 * (uint64_t)count > n) return;
    int levels = 0, items = 0, left = 0;
    for (uint32_t k = 0; k < count; k++) {
        uint64_t e = v + 4 + 4 * (uint64_t)k;
        e += u32(b + e);
        if (e + 4 > n) return;
        uint16_t so = ex_slot(b, n, e, 0), eo = ex_slot(b, n, e, EVO_SLOT);
        if (!so || !eo || e + so + 2 > n || e + eo + 4 > n) continue;
        uint16_t species = u16(b + e + so);
        uint64_t ev = e + eo;
        ev += u32(b + ev);
        if (ev + 4 > n) return;
        uint32_t m = u32(b + ev);
        if (m > 64 || ev + 4 + (uint64_t)m * EVO_STRUCT > n) continue;
        for (uint32_t j = 0; j < m; j++) {
            uint8_t *s = b + ev + 4 + (uint64_t)j * EVO_STRUCT;
            uint16_t method = u16(s + 2), target = u16(s + 12), level = 0, plib = 0;
            if (method == EVO_TRADE) {
                for (size_t i = 0; i < sizeof trade_evo / sizeof *trade_evo; i++)
                    if (trade_evo[i].species == species) level = trade_evo[i].level;
                if (!level) {   // a trade evolution with no level here (a later game version): said and left alone
                    left++;
                    rt_log("no trade evolutions: species %u evolves by trading into %u, no level here; left alone",
                           species, target);
                    continue;
                }
                put16(s, level), put16(s + 2, EVO_LEVEL), levels++;
            } else if (method == EVO_TRADE_ITEM) {
                for (size_t i = 0; i < sizeof evo_item_level / sizeof *evo_item_level; i++)
                    if (evo_item_level[i].species == species && evo_item_level[i].target == target)
                        level = evo_item_level[i].level;
                if (level) {
                    put16(s, level), put16(s + 2, EVO_LEVEL), put16(s + 4, 0), levels++;
                    continue;
                }
                for (size_t i = 0; i < sizeof evo_item_own / sizeof *evo_item_own; i++)
                    if (evo_item_own[i].species == species && evo_item_own[i].target == target)
                        plib = evo_item_own[i].plib;
                for (size_t i = 0; !plib && i < sizeof evo_item / sizeof *evo_item; i++)
                    if (evo_item[i].item == u16(s + 4)) plib = evo_item[i].plib;
                if (!plib) {
                    left++;
                    rt_log("no trade evolutions: species %u evolves by trading into %u holding item %u, which has no "
                           "plibID here; left alone", species, target, u16(s + 4));
                    continue;
                }
                put16(s + 2, EVO_HOLD_ITEM), put16(s + 4, plib), items++;
            }
        }
    }
    if (levels || items || left)
        rt_log("no trade evolutions: of %u species entries, %d evolve on a level and %d on a level by day holding an "
               "item, %d left alone", count, levels, items, left);
}

// The data patches of one decoded file: r unpacked bytes at a[2] from a[1] packed bytes at a[0].
static void data_patch(const uint64_t a[8], uint64_t r) {
    if (excl_on > 0) exclusives_patch(a, r);
    if (evo_on > 0 && r == evo_file[0].unpacked) {
        uint64_t head = a[1] < 4096 ? a[1] : 4096;
        if (a[0] + head > RT_ADDRESS_LIMIT || a[2] + r > RT_ADDRESS_LIMIT) return;
        uint64_t print = fnv1a64_bytes((const uint8_t *)(uintptr_t)a[0], head);
        if (print == evo_file[0].print) evo_fix((uint8_t *)(uintptr_t)a[2], r);
        else rt_log("no trade evolutions: a %llu-byte file with another fingerprint (%016llx), left alone",
                    (unsigned long long)r, (unsigned long long)print);
    }
}

// Every call of the game's Oodle decoder while the set is on: the guest decoder as before, then the data patches.
static void host_oodle(Cpu *c) {
    uint64_t a[8];
    memcpy(a, c->x, sizeof a);
    if (census_wanted()) {
        static int ran;
        if (!__atomic_exchange_n(&ran, 1, __ATOMIC_ACQ_REL)) census_run(c, a);   // does not return
    }
    uint64_t r = rt_call(c, GUEST_OODLE, 8, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);   // its result stays in x0
    if (wild_count > 0) population_patch(a, r);
    data_patch(a, r);
}

__attribute__((constructor)) static void sv_census_register(void) { rt_register_host(HOST_OODLE, host_oodle); }

// After rt_patches_init (boot.c): the hook on only when something wants the decodes.
void sv_oodle_init(void) {
    int set = rt_patch_set("sv_oodle");
    if (set < 0 || !RT_SV) return;
    int wild = population();
    if (wild) rt_log("population: up to %d wild Pokemon at once (the game's own: 15)", wild);
    if (no_exclusives()) rt_log("no exclusives: the wild Pokemon of both editions spawn");
    if (no_trade_evo()) rt_log("no trade evolutions: trade evolutions happen on a level or holding an item");
    if (census_wanted() || wild || excl_on || evo_on) rt_patch_switch(set, 1);
}

#else
void sv_oodle_init(void) {}
#endif
