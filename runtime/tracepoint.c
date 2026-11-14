// Trace points: debugging aid for porting a game. A `trace <module> <address>` line in the patch file
// (recomp/patches.py) lifts rt_trace_point(c, address) in front of that instruction; this logs the guest registers
// there: every general register (x0-x7 arguments at a function entry, x0 the result after a call), the return
// address and the stack pointer. Each point logs its first TERA_TRACE_LIMIT passes (default 16), so a point inside a loop does not flood
// the log. It reads registers only; the game runs on unchanged.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "layout.h"
#include "platform/platform.h"

#define SLOTS 1024   // distinct trace points; more than any patch file should name

static struct { uint64_t pc; uint32_t n; } seen[SLOTS];
static PlatMutex lock = PLAT_MUTEX_INIT;
static int limit = -1;

// The text at a, when a points at a NUL-terminated run of at least 3 printable ASCII bytes (up to 96 shown).
static int text_at(uint64_t a, char *out, int cap) {
    if (a < 0x10000) return 0;
    uint8_t b[97];
    if (!plat_mem_read(a, b, sizeof b - 1)) return 0;
    int n = 0;
    while (n < 96 && n < cap - 1 && b[n] >= 0x20 && b[n] < 0x7f) out[n] = (char)b[n], n++;
    out[n] = 0;
    return n >= 3 && (n == 96 || b[n] == 0);
}
// TERA_TRACE_DUMP=<point>:x<reg> (debugging): at each pass of that point (named as the log names it, sdk+0x18de08),
// the memory x<reg> points at goes to dump/trace_<pass>.bin. A TrueType/OpenType font or collection is written whole
// (its size from its table directory); anything else as 1 MiB. For reading game data the game builds at run time.
static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

static uint64_t font_end(uint64_t a, uint64_t off) {   // end of the sfnt at a + off, 0 if it is not one
    uint8_t h[12];
    if (!plat_mem_read(a + off, h, sizeof h)) return 0;
    uint32_t tag = be32(h), n = (uint32_t)h[4] << 8 | h[5];
    if ((tag != 0x00010000 && tag != 0x4f54544f && tag != 0x74727565) || !n || n > 64) return 0;
    uint64_t end = off + 12 + 16ull * n;
    for (uint32_t i = 0; i < n; i++) {
        uint8_t r[16];
        if (!plat_mem_read(a + off + 12 + 16ull * i, r, sizeof r)) return 0;
        uint64_t e = (uint64_t)be32(r + 8) + be32(r + 12);
        if (e > end) end = e;
    }
    return end;
}

static void trace_dump(Cpu *c, const char *at, uint32_t n) {
    const char *v = cfg_get("TERA_TRACE_DUMP");
    if (!v || !*v) return;
    const char *colon = strrchr(v, ':');
    if (!colon || colon[1] != 'x' || strncmp(at, v, (size_t)(colon - v)) || at[colon - v] != ' ') return;
    int reg = atoi(colon + 2);
    if (reg < 0 || reg > 30) return;
    uint64_t a = c->x[reg], size = 0;
    uint8_t h[12];
    if (plat_mem_read(a, h, sizeof h) && be32(h) == 0x74746366) {   // 'ttcf': the end of its last font
        uint32_t fonts = be32(h + 8);
        for (uint32_t i = 0; i < fonts && i < 64; i++) {
            uint8_t o[4];
            if (!plat_mem_read(a + 12 + 4ull * i, o, 4)) break;
            uint64_t e = font_end(a, be32(o));
            if (e > size) size = e;
        }
    } else {
        size = font_end(a, 0);
    }
    if (!size || size > (64u << 20)) size = 1u << 20;
    uint8_t *buf = malloc(size);
    if (!buf || !plat_mem_read(a, buf, size)) { free(buf); rt_log("trace: dump of %#llx failed", (unsigned long long)a); return; }
    char path[64];
    snprintf(path, sizeof path, "dump/trace_%u.bin", n);
    plat_file_mkdir_p("dump");
    if (!plat_file_write(path, buf, size)) rt_log("trace: cannot write %s", path);
    else rt_log("trace: x%d %#llx: %llu bytes to %s", reg, (unsigned long long)a, (unsigned long long)size, path);
    free(buf);
}

static uint64_t sleep_pc;
static uint32_t sleep_ms;
static int sleep_done;

// TERA_DBG_WATCH=<point>:<reg>:<offset> (debugging, Windows): when trace point main+<point> first passes, the 8 bytes
// at x<reg> + <offset> are watched: their page is made inaccessible, and each access to the 8 bytes logs the lifted
// function that made it (once per host instruction), then the page is opened for one single-stepped instruction.
#ifdef _WIN32
#include <windows.h>
uint64_t rt_lifted_function(uintptr_t rip);
static uint64_t watch_point, watch_addr;
static unsigned watch_reg;
static uintptr_t watch_page;
static _Thread_local int watch_step;
static uintptr_t watch_seen[512];
static volatile LONG watch_n;

static LONG CALLBACK watch_veh(EXCEPTION_POINTERS *ep) {
    EXCEPTION_RECORD *er = ep->ExceptionRecord;
    DWORD old;
    if (er->ExceptionCode == EXCEPTION_SINGLE_STEP && watch_step) {
        watch_step = 0;
        VirtualProtect((void *)watch_page, 0x1000, PAGE_NOACCESS, &old);
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    if (er->ExceptionCode != EXCEPTION_ACCESS_VIOLATION || er->NumberParameters < 2 || !watch_page)
        return EXCEPTION_CONTINUE_SEARCH;
    uintptr_t a = (uintptr_t)er->ExceptionInformation[1];
    if ((a & ~(uintptr_t)0xfff) != watch_page) return EXCEPTION_CONTINUE_SEARCH;
    VirtualProtect((void *)watch_page, 0x1000, PAGE_READWRITE, &old);
    if (a + 8 > watch_addr && a < watch_addr + 8) {
        uintptr_t rip = (uintptr_t)ep->ContextRecord->Rip;
        int known = 0;
        LONG n = watch_n;
        for (LONG i = 0; i < n && i < 512; i++) known |= watch_seen[i] == rip;
        if (!known && n < 512) {
            watch_seen[InterlockedIncrement(&watch_n) - 1] = rip;
            uint64_t f = rt_lifted_function(rip);
            char d[128] = "host code";
            if (f) ldr_describe(f, d, sizeof d);
            rt_log("watch: %s of %#llx (+%lld) by %s, exe+%#llx", er->ExceptionInformation[0] == 1 ? "write" : "read",
                   (unsigned long long)watch_addr, (long long)(a - watch_addr), d,
                   (unsigned long long)(rip - plat_exe_base()));
        }
    }
    ep->ContextRecord->EFlags |= 0x100;
    watch_step = 1;
    return EXCEPTION_CONTINUE_EXECUTION;
}

static void watch_arm(Cpu *c, uint64_t pc) {
    static int parsed;
    if (!parsed) {
        parsed = 1;
        const char *v = cfg_get("TERA_DBG_WATCH");
        unsigned long long p = 0, off = 0;
        if (v && sscanf(v, "%llx:%u:%llx", &p, &watch_reg, &off) == 3 && watch_reg < 31)
            watch_point = MOD_MAIN_BASE + p, watch_addr = off;
    }
    if (!watch_point || pc != watch_point || watch_page) return;
    watch_addr += c->x[watch_reg];
    watch_page = (uintptr_t)watch_addr & ~(uintptr_t)0xfff;
    AddVectoredExceptionHandler(1, watch_veh);
    DWORD old;
    VirtualProtect((void *)watch_page, 0x1000, PAGE_NOACCESS, &old);
    rt_log("watch: armed on %#llx", (unsigned long long)watch_addr);
}
#else
static void watch_arm(Cpu *c, uint64_t pc) { (void)c, (void)pc; }
#endif

void rt_trace_point(Cpu *c, uint64_t pc) {
    plat_mutex_lock(&lock);
    if (limit < 0) {
        const char *v = cfg_get("TERA_TRACE_LIMIT");
        limit = v && *v ? atoi(v) : 16;
        const char *z = cfg_get("TERA_TRACE_SLEEP");
        if (z && *z) {
            char *end;
            uint64_t off = strtoull(z, &end, 16);
            if (*end == ':') {
                sleep_pc = MOD_MAIN_BASE + off;
                sleep_ms = (uint32_t)strtoul(end + 1, NULL, 10);
            }
        }
    }
    int do_sleep = pc == sleep_pc && !sleep_done;
    if (do_sleep) sleep_done = 1;
    uint32_t h = (uint32_t)(pc >> 2) & (SLOTS - 1), n = 0;
    for (int k = 0; k < SLOTS; k++, h = (h + 1) & (SLOTS - 1)) {
        if (!seen[h].pc) seen[h].pc = pc;
        if (seen[h].pc == pc) {
            n = ++seen[h].n;
            break;
        }
    }
    plat_mutex_unlock(&lock);
    watch_arm(c, pc);
    if (do_sleep) {
        rt_log("trace: sleeping %u ms at %#llx (TERA_TRACE_SLEEP), core given up", sleep_ms, (unsigned long long)pc);
        int away = kern_core_away();
        plat_sleep_ms(sleep_ms);
        kern_core_back(&away);
        rt_log("trace: woke at %#llx", (unsigned long long)pc);
    }
    if (!n || (int)n > limit) return;
    char at[128], lr[128];
    ldr_describe(pc, at, sizeof at);
    ldr_describe(c->x[30] - 4, lr, sizeof lr);
    rt_log("trace %s #%u: x0 %llx x1 %llx x2 %llx x3 %llx x4 %llx x5 %llx x6 %llx x7 %llx x8 %llx | x19 %llx x20 %llx "
           "x21 %llx x22 %llx x23 %llx x24 %llx x25 %llx x26 %llx x27 %llx x28 %llx | lr %s sp %llx",
           at, n, (unsigned long long)c->x[0], (unsigned long long)c->x[1], (unsigned long long)c->x[2],
           (unsigned long long)c->x[3], (unsigned long long)c->x[4], (unsigned long long)c->x[5],
           (unsigned long long)c->x[6], (unsigned long long)c->x[7], (unsigned long long)c->x[8],
           (unsigned long long)c->x[19], (unsigned long long)c->x[20], (unsigned long long)c->x[21],
           (unsigned long long)c->x[22], (unsigned long long)c->x[23], (unsigned long long)c->x[24],
           (unsigned long long)c->x[25], (unsigned long long)c->x[26], (unsigned long long)c->x[27],
           (unsigned long long)c->x[28], lr, (unsigned long long)c->sp);
    rt_log("trace   x9 %llx x10 %llx x11 %llx x12 %llx x13 %llx x14 %llx x15 %llx x16 %llx x17 %llx x18 %llx",
           (unsigned long long)c->x[9], (unsigned long long)c->x[10], (unsigned long long)c->x[11],
           (unsigned long long)c->x[12], (unsigned long long)c->x[13], (unsigned long long)c->x[14],
           (unsigned long long)c->x[15], (unsigned long long)c->x[16], (unsigned long long)c->x[17],
           (unsigned long long)c->x[18]);
    trace_dump(c, at, n);
    static const int text_regs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28};
    for (unsigned k = 0; k < sizeof text_regs / sizeof *text_regs; k++) {
        char t[100];
        if (text_at(c->x[text_regs[k]], t, sizeof t)) rt_log("trace   x%d -> \"%s\"", text_regs[k], t);
    }
    uint64_t obj[16];
    if (c->x[19] && plat_mem_read(c->x[19], obj, sizeof obj)) {
        char vt[128];
        ldr_describe(obj[0], vt, sizeof vt);
        rt_log("trace   [x19 %llx] vtable %s | +08 %llx +10 %llx +18 %llx +20 %llx +28 %llx +30 %llx +38 %llx +40 %llx "
               "+48 %llx +50 %llx +58 %llx +60 %llx +68 %llx +70 %llx +78 %llx",
               (unsigned long long)c->x[19], vt, (unsigned long long)obj[1], (unsigned long long)obj[2],
               (unsigned long long)obj[3], (unsigned long long)obj[4], (unsigned long long)obj[5],
               (unsigned long long)obj[6], (unsigned long long)obj[7], (unsigned long long)obj[8],
               (unsigned long long)obj[9], (unsigned long long)obj[10], (unsigned long long)obj[11],
               (unsigned long long)obj[12], (unsigned long long)obj[13], (unsigned long long)obj[14],
               (unsigned long long)obj[15]);
    }
}
