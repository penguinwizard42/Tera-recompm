// Scarlet and Violet 4.0.0: people and Pokemon keep their detailed meshes farther from the camera, and more town people
// are modeled at once.
//
// Adapted from KarPur's runtime/sv_detail.c (https://github.com/Grunttanamo/KarPur @
// c2b2c0d738f2c769c24e2ceddb60592aba6e7778, MIT, Copyright (c) 2026 KarPur contributors), itself based on PokeV-Recomp
// pull requests #16 (bf92855, the model LOD manager, its named entries and threshold offsets) and #15 (a00907a, the
// static town-person limit and the wrapped GlobalHeap counter correction), MIT.
//
// Once per presented frame (nvn_api_obj.c), with guest-memory checks before every pointer is followed, since a field
// reload leaves stale nodes:
// - people_detail / pokemon_detail: the model LOD manager (a global of main) holds a list of named entries; the three
//   distances at +0x58 of "chara_model" (people, 8 / 15 / 30 m in the game) and "pokemon" (10 / 20 / 30 m) become the
//   game's own times the level, at least. A distance the loaded data already has higher (a mod) is kept as it is.
// - people_shown: the static town-person manager's limit at +0xe8 (12 in the game) is raised to the setting. With more
//   than 12 the game's GlobalHeap can wrap its unsigned used counter below zero when it subtracts two allocators' free
//   sizes; only that wrapped case is set back to 0, with the last valid count of quality slots.
// No memory pool or capacity is changed.
#include <stdint.h>
#include <string.h>

#include "cfg.h"
#include "host.h"

#define LOD_GLOBAL 0x46d4e00ull
#define LOD_MGR_VTABLE 0x44a00b8ull
#define NPC_MGR_GLOBAL 0x473af30ull
#define NPC_MGR_VTABLE 0x4468fd8ull
#define GLOBAL_HEAP 0x4763918ull
#define GLOBAL_HEAP_VTABLE 0x44a6500ull

typedef struct {
    const char *name, *label;
    float baseline[3];   // the game's own full / middle / far distances
    uint64_t node;
    float original[3], written[3];
    int logged;
} Detail;

static Detail people = {.name = "chara_model", .label = "people", .baseline = {8, 15, 30}, .logged = -1};
static Detail pokemon = {.name = "pokemon", .label = "Pokemon", .baseline = {10, 20, 30}, .logged = -1};

static int accessible(uint64_t addr, uint64_t size, int write) {
    RtMemoryInfo info;
    if (!addr || addr >= RT_ADDRESS_LIMIT || size > RT_ADDRESS_LIMIT - addr || mem_query(addr, &info)) return 0;
    return addr >= info.base && size <= info.size && addr - info.base <= info.size - size &&
           (info.perm & (write ? 3u : 1u)) == (write ? 3u : 1u);
}

static uint64_t read_ptr(uint64_t addr) {
    uint64_t value = 0;
    if (accessible(addr, sizeof value, 0)) memcpy(&value, (const void *)(uintptr_t)addr, sizeof value);
    return value;
}

static void write_guest(uint64_t at, const void *v, size_t n) {
    mem_thaw(at, n);
    memcpy((void *)(uintptr_t)at, v, n);
}

static int level(const char *key, int maximum) {   // 1 (the game's own) or an even level up to maximum
    int value = cfg_int(key);
    if (value < 2 || (value & 1)) return 1;
    return value < maximum ? value : maximum;
}

static void fix_heap_counter(const RtLoaded *main, uint64_t manager, uint32_t limit, uint64_t *good_slots) {
    uint64_t heap = main->base + GLOBAL_HEAP, slots_at = manager + 0xc58;
    if (GLOBAL_HEAP + 0x40 > main->size || read_ptr(heap) != main->base + GLOBAL_HEAP_VTABLE ||
        !accessible(heap + 0x28, 0x18, 1) || !accessible(slots_at, 8, 1))
        return;
    uint64_t total = read_ptr(heap + 0x28), used = read_ptr(heap + 0x38), slots = read_ptr(slots_at);
    if (total < (128ull << 20) || total > (8ull << 30) || slots > 1024) return;
    if (used <= total) {
        if (slots) *good_slots = slots;
    } else if (used >= (1ull << 63)) {   // wrapped below zero; a real value above total is left alone
        uint64_t zero = 0, restored = *good_slots > slots ? *good_slots : slots;
        write_guest(heap + 0x38, &zero, sizeof zero);
        if (restored != slots) write_guest(slots_at, &restored, sizeof restored);
        static unsigned logged;
        if (logged++ < 8)
            rt_log("people shown: corrected the wrapped GlobalHeap used counter; quality slots %llu -> %llu",
                   (unsigned long long)slots, (unsigned long long)restored);
    }
    (void)limit;
}

static void update_people_shown(const RtLoaded *main) {
    static int wanted;
    static uint64_t manager, good_slots;
    static uint32_t original, written;
    static int logged = -1;
    if (!wanted) {
        wanted = cfg_int("TOTK_PEOPLE_SHOWN");
        if (wanted < 12 || wanted % 12) wanted = 12;
        if (wanted > 48) wanted = 48;
    }
    if (wanted == 12) return;
    if (NPC_MGR_GLOBAL + 8 > main->size || NPC_MGR_VTABLE >= main->size) return;
    uint64_t current_manager = read_ptr(main->base + NPC_MGR_GLOBAL);
    if (!current_manager || read_ptr(current_manager) != main->base + NPC_MGR_VTABLE ||
        !accessible(current_manager + 0xe8, 4, 1) || !accessible(current_manager + 0xc58, 8, 1)) {
        manager = 0;
        return;
    }
    uint32_t current;
    memcpy(&current, (const void *)(uintptr_t)(current_manager + 0xe8), sizeof current);
    if (current_manager != manager || current != written) {   // a new manager, or the game set its own value
        if (current == 0 || current > 1024) { manager = 0; return; }
        manager = current_manager;
        original = written = current;
    }
    uint32_t target = (uint32_t)wanted > original ? (uint32_t)wanted : original;
    if (current != target) write_guest(manager + 0xe8, &target, sizeof target), written = target;
    if (logged != (int)target) {
        logged = (int)target;
        rt_log("people shown: %u town people at once (the game's %u); no memory pool changed", target, original);
    }
    fix_heap_counter(main, manager, target, &good_slots);
}

static void update_entry(uint64_t map, Detail *entry, int multiplier) {
    uint64_t node = read_ptr(map + 0x38);
    const size_t name_size = strlen(entry->name);
    for (int i = 0; i < 64 && node; ++i) {
        if (!accessible(node, 0x64, 0)) return;
        uint64_t chars = read_ptr(node + 0x18), length = read_ptr(node + 0x20);
        if (length == name_size && accessible(chars, length, 0) &&
            !memcmp((const void *)(uintptr_t)chars, entry->name, name_size)) {
            uint64_t at = node + 0x58;
            if (!accessible(at, sizeof entry->written, 1)) return;
            float current[3];
            memcpy(current, (const void *)(uintptr_t)at, sizeof current);
            if (node != entry->node || memcmp(current, entry->written, sizeof current)) {
                if (!(current[0] > 0 && current[0] < current[1] && current[1] < current[2] && current[2] < 10000)) {
                    entry->node = 0;
                    return;
                }
                if (node != entry->node || memcmp(current, entry->original, sizeof current)) entry->logged = -1;
                entry->node = node;
                memcpy(entry->original, current, sizeof current);
                memcpy(entry->written, current, sizeof current);
            }
            float scaled[3];
            for (int j = 0; j < 3; ++j) {   // at least the game's own times the level; a higher loaded value stays
                float requested = entry->baseline[j] * (float)multiplier;
                scaled[j] = entry->original[j] > requested ? entry->original[j] : requested;
            }
            if (memcmp(current, scaled, sizeof scaled)) {
                write_guest(at, scaled, sizeof scaled);
                memcpy(entry->written, scaled, sizeof scaled);
            }
            if (entry->logged != multiplier) {
                entry->logged = multiplier;
                rt_log("detail: %s full model within %.0f m (loaded %.0f m); distances %.0f / %.0f / %.0f m",
                       entry->label, scaled[0], entry->original[0], scaled[0], scaled[1], scaled[2]);
            }
            return;
        }
        uint64_t next = read_ptr(node);
        if (next == node) return;
        node = next;
    }
}

void sv_detail_present(void) {
    if (!RT_SV) return;
    static int people_level, pokemon_level;
    if (!people_level) {
        people_level = level("TOTK_PEOPLE_DETAIL", 10);
        pokemon_level = level("TOTK_POKEMON_DETAIL", 8);
        if (people_level > 1 || pokemon_level > 1)
            rt_log("detail: people at least %.0f m, Pokemon at least %.0f m", people.baseline[0] * people_level,
                   pokemon.baseline[0] * pokemon_level);
    }
    const RtLoaded *main = ldr_module(0);
    if (!main) return;
    update_people_shown(main);
    if (people_level == 1 && pokemon_level == 1) return;
    if (LOD_GLOBAL + 8 > main->size || LOD_MGR_VTABLE >= main->size) return;
    uint64_t p = read_ptr(main->base + LOD_GLOBAL);
    p = read_ptr(p);
    p = read_ptr(p + 0x38);
    p = read_ptr(p + 0x50);
    if (!p || read_ptr(p) != main->base + LOD_MGR_VTABLE) return;
    uint64_t map = read_ptr(p + 0x30);
    if (!map) return;
    if (people_level > 1) update_entry(map, &people, people_level);
    if (pokemon_level > 1) update_entry(map, &pokemon, pokemon_level);
}
