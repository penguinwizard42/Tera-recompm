// Scarlet and Violet's Oodle decoder through the host: the shader census, and the wild Pokemon population.
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
// Adapted from KarPur's runtime/shader_census.c and runtime/data_watch.c (https://github.com/Grunttanamo/KarPur @
// c2b2c0d738f2c769c24e2ceddb60592aba6e7778, MIT, Copyright (c) 2026 KarPur contributors): the Oodle hook, the
// archive walk, the BNSH extraction, the seed records and the population fields. Left out: KarPur's search for programs recorded in play, its
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

// Every call of the game's Oodle decoder while the set is on: the guest decoder as before, then the population.
static void host_oodle(Cpu *c) {
    uint64_t a[8];
    memcpy(a, c->x, sizeof a);
    if (census_wanted()) {
        static int ran;
        if (!__atomic_exchange_n(&ran, 1, __ATOMIC_ACQ_REL)) census_run(c, a);   // does not return
    }
    uint64_t r = rt_call(c, GUEST_OODLE, 8, a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]);   // its result stays in x0
    if (wild_count > 0) population_patch(a, r);
}

__attribute__((constructor)) static void sv_census_register(void) { rt_register_host(HOST_OODLE, host_oodle); }

// After rt_patches_init (boot.c): the hook on only when something wants the decodes.
void sv_oodle_init(void) {
    int set = rt_patch_set("sv_oodle");
    if (set < 0 || !RT_SV) return;
    int wild = population();
    if (wild) rt_log("population: up to %d wild Pokemon at once (the game's own: 15)", wild);
    if (census_wanted() || wild) rt_patch_switch(set, 1);
}

#else
void sv_oodle_init(void) {}
#endif
