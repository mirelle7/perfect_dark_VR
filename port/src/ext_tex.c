#include <dirent.h>
#include <sys/stat.h>

#define STB_IMAGE_IMPLEMENTATION
#include "external/stb_image.h"

#include "gbiex.h"
#include "types.h"
#include "system.h"
#include "fs.h"
#include "data.h"
#include "romdata.h"
#include "ext_tex.h"

#ifdef ANDROID
#include <android/log.h>
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  "PD-VR", __VA_ARGS__)
#else
#define LOGI(...) printf(__VA_ARGS__)
#endif

#include <pthread.h>
#include <stdatomic.h>
#include <unistd.h>
#include <video.h>
#include <menuimage.h>

// ============================================================================
// CONSTANTS & MACROS
// ============================================================================
#define MAX_EXT_TEX        8192
#define MAX_MODEL_FILES    4096
#define NUM_FONTS          5
#define FONT_OUTLINES_DIR  "outlines"
#define EXT_DECODE_THREADS 3

#define FONT_HANDELGOTHICSM 0
#define FONT_HANDELGOTHICMD 1
#define FONT_HANDELGOTHICXS 2
#define FONT_HANDELGOTHICLG 3
#define FONT_NUMERIC        4

#if VERSION == VERSION_PAL_FINAL
#define NCHARS 135
#else
#define NCHARS 94
#endif

const u16 IDMASK_FONT_OUTLINE = MASK_FONT_OUTLINE << 8;

// Lifecycle of one replacement texture:
//   IDLE   -> QUEUED  (requested by a prefetch or by a draw)
//   QUEUED -> READY   (decoded by a worker, pixels in texdata)
//   QUEUED -> FAILED  (missing or corrupt file, never retried)
//   READY  -> IDLE    (pixels taken by the renderer and uploaded to the GPU)
enum {
    EXT_IDLE = 0,
    EXT_QUEUED,
    EXT_READY,
    EXT_FAILED,
};

// ============================================================================
// TYPES & STRUCTURES
// ============================================================================
struct ExtTexture {
    u8 *texdata;
    s32 texnum;
    u32 width;
    u32 height;
    _Atomic(int) state;
    char extension[5];
};

struct ModelTextures {
    s16 fileNum;
    s16 numTextures;
    struct ExtTexture *textures;
};

struct ExtJob {
    u8 type;
    u16 id;
    s32 texnum;
};

struct JobRing {
    struct ExtJob *items;
    s32 cap;
    s32 head;
    s32 count;
};

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================
// Decode workers. Both rings are protected by jobMutex.
static pthread_mutex_t jobMutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  jobCond  = PTHREAD_COND_INITIALIZER;
static struct JobRing  decodeRing;
static struct JobRing  readyRing;
static pthread_t       decodeThreads[EXT_DECODE_THREADS];
static s32             numDecodeThreads = 0;
static _Atomic(int)    decodeThreadsRunning = 0;
static _Atomic(int)    inflight = 0; // queued + decoding + decoded but not yet uploaded

// Pack change requests, applied on the render thread.
static pthread_mutex_t packMutex = PTHREAD_MUTEX_INITIALIZER;
static char            pendingPackName[FS_MAXPATH] = "";
static s32             pendingPackChange = 0;

// Texture State
char g_ActiveExtTexPack[FS_MAXPATH] = ""; // Name of the chosen pack folder
static char extTexPath[FS_MAXPATH + 1];
static s32 packLoaded = 0;
static s32 numRegistered = 0;

static struct ExtTexture extTextures[MAX_EXT_TEX];
static struct ModelTextures *modelTextures;
static s32 numModels;
static s32 maxModels = 0;

static struct ExtTexture fontExtTextures[NUM_FONTS][NCHARS];
static struct ExtTexture fontOutlineExtTextures[NUM_FONTS][NCHARS];

// Everything the current stage has loaded, so a pack enabled mid-stage can
// prefetch what is already on screen.
static u8 recordedGeneral[MAX_EXT_TEX];
static u8 recordedModels[MAX_MODEL_FILES];

// Implemented by the renderer: true when the HD version of this texture is
// already on the GPU, so prefetching it again would only waste a decode.
extern s32 gfx_ext_tex_resident(u8 type, u16 id, s32 texnum);

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================
static void *extTexDecodeThreadFunc(void *arg);
static int findTruePackRoot(char *currentPath);
static int is_hex_string(const char *str);
u8 getTexPath(char *dst, u8 type, u16 id, s32 texnum);

// ============================================================================
// JOB RINGS
// ============================================================================

static void ringAlloc(struct JobRing *ring, s32 cap)
{
    ring->items = sysMemRealloc(ring->items, cap * sizeof(struct ExtJob));
    ring->cap = cap;
    ring->head = 0;
    ring->count = 0;
}

static s32 ringPush(struct JobRing *ring, u8 type, u16 id, s32 texnum)
{
    if (ring->count >= ring->cap) {
        return 0;
    }

    struct ExtJob *job = &ring->items[(ring->head + ring->count) % ring->cap];
    job->type = type;
    job->id = id;
    job->texnum = texnum;
    ring->count++;
    return 1;
}

static s32 ringPop(struct JobRing *ring, struct ExtJob *out)
{
    if (ring->count == 0) {
        return 0;
    }

    *out = ring->items[ring->head];
    ring->head = (ring->head + 1) % ring->cap;
    ring->count--;
    return 1;
}

// ============================================================================
// TEXTURE LOOKUP & PATH RESOLUTION
// ============================================================================

s32 fileInfo(const char *filename, s32 *texNum, char extension[5])
{
    const char *ext = strrchr(filename, '.');

    // No extension
    if (!ext || ext == filename) return 1;

    ++ext;
    if (strlen(ext) >= 5) return 1;
    strcpy(extension, ext);

    // Get the filename without extension. Only pure hex names are texture
    // numbers: "cover.png" would otherwise parse as texture 0x0c.
    char basename[256] = { 0 };
    size_t len = ext - filename - 1;
    if (len >= sizeof(basename)) return 1;
    memcpy(basename, filename, len);

    if (!is_hex_string(basename)) return 1;

    *texNum = strtol(basename, NULL, 16);

    return 0;
}

static struct ModelTextures *lookupModel(u16 fileNum)
{
    for (int i = 0; i < numModels; ++i) {
        if (modelTextures[i].fileNum == fileNum)
            return &modelTextures[i];
    }

    return NULL;
}

struct ExtTexture *lookupModelTex(u16 fileNum, s32 texNum)
{
    struct ModelTextures *modelTex = lookupModel(fileNum);

    if (modelTex == NULL)
        return NULL;

    for (int i = 0; i < modelTex->numTextures; ++i) {
        if (modelTex->textures[i].texnum == texNum)
            return &modelTex->textures[i];
    }

    return NULL;
}

struct ExtTexture *getExtTexture(u8 type, u16 id, s32 texnum)
{
    switch (type) {
        case G_TEXTYPE_NONE:
            return NULL;
        case G_TEXTYPE_GENERAL:
            if (texnum < 0 || texnum >= MAX_EXT_TEX)
                return NULL;
            return &extTextures[texnum];
        case G_TEXTYPE_MODEL:
            return lookupModelTex(id, texnum);
        case G_TEXTYPE_FONT: {
            u16 font = id & ~IDMASK_FONT_OUTLINE;
            if (font >= NUM_FONTS || texnum < 0 || texnum >= NCHARS)
                return NULL;

            if (id & IDMASK_FONT_OUTLINE)
                return &fontOutlineExtTextures[font][texnum];

            return &fontExtTextures[font][texnum];
        }
        default:
            sysLogPrintf(LOG_WARNING, "Invalid Texture type: %d, texnum: %04x", type, texnum);
            return NULL;
    }
}

u8 extTexExists(u8 type, u16 id, s32 texnum)
{
    struct ExtTexture *tex = getExtTexture(type, id, texnum);
    return tex && tex->texnum >= 0 && atomic_load(&tex->state) != EXT_FAILED;
}

char *resolveFontname(const u8 fontId)
{
    switch (fontId) {
        case FONT_HANDELGOTHICSM: return "fonthandelgothicsm";
        case FONT_HANDELGOTHICMD: return "fonthandelgothicmd";
        case FONT_HANDELGOTHICXS: return "fonthandelgothicxs";
        case FONT_HANDELGOTHICLG: return "fonthandelgothiclg";
        case FONT_NUMERIC:        return "fontnumeric";
        default:                  return "";
    }
}

u8 getTexPath(char *dst, u8 type, u16 id, s32 texnum)
{
    struct ExtTexture *tex = getExtTexture(type, id, texnum);
    const char *name;

    if (!tex) {
        return 1;
    }

    switch (type) {
        case G_TEXTYPE_GENERAL: {
            snprintf(dst, FS_MAXPATH, "%s/%04x.%s", extTexPath, texnum, tex->extension);
            return 0;
        }
        case G_TEXTYPE_FONT: {
            name = resolveFontname(id & ~IDMASK_FONT_OUTLINE);

            if (id & IDMASK_FONT_OUTLINE) {
                snprintf(dst, FS_MAXPATH, "%s/%s/" FONT_OUTLINES_DIR "/%02x.%s", extTexPath, name, texnum, tex->extension);
                return 0;
            }

            snprintf(dst, FS_MAXPATH, "%s/%s/%02x.%s", extTexPath, name, texnum, tex->extension);
            return 0;
        }
        case G_TEXTYPE_MODEL: {
            name = romdataFileGetName(id);
            snprintf(dst, FS_MAXPATH, "%s/%s/%05x.%s", extTexPath, name, texnum, tex->extension);
            return 0;
        }
        default: return 1;
    }
}

// ============================================================================
// ASYNC TEXTURE LOADING & THREADING
// ============================================================================

static void extTexEnsureThreadsStarted(void)
{
    if (numDecodeThreads > 0) {
        return;
    }

    atomic_store(&decodeThreadsRunning, 1);

    for (int i = 0; i < EXT_DECODE_THREADS; ++i) {
        if (pthread_create(&decodeThreads[i], NULL, extTexDecodeThreadFunc, NULL) == 0) {
            numDecodeThreads++;
        }
    }
}

void extTexAsyncShutdown(void)
{
    if (numDecodeThreads == 0)
        return;

    pthread_mutex_lock(&jobMutex);
    atomic_store(&decodeThreadsRunning, 0);
    pthread_cond_broadcast(&jobCond);
    pthread_mutex_unlock(&jobMutex);

    for (int i = 0; i < numDecodeThreads; ++i) {
        pthread_join(decodeThreads[i], NULL);
    }

    numDecodeThreads = 0;
}

static void extTexEnqueueLoad(u8 type, u16 id, s32 texnum)
{
    struct ExtTexture *tex = getExtTexture(type, id, texnum);

    if (!tex || tex->texnum < 0) {
        return;
    }

    // The state machine is the duplicate check: only an idle texture is queued.
    int expected = EXT_IDLE;
    if (!atomic_compare_exchange_strong(&tex->state, &expected, EXT_QUEUED)) {
        return;
    }

    extTexEnsureThreadsStarted();

    pthread_mutex_lock(&jobMutex);
    if (ringPush(&decodeRing, type, id, texnum)) {
        atomic_fetch_add(&inflight, 1);
        pthread_cond_signal(&jobCond);
    } else {
        atomic_store(&tex->state, EXT_IDLE);
    }
    pthread_mutex_unlock(&jobMutex);
}

// Moves a decoded texture's pixels to the caller. Returns NULL if someone else
// already took them.
static u8 *extTexTake(struct ExtTexture *tex, u32 *width, u32 *height)
{
    int expected = EXT_READY;
    if (!atomic_compare_exchange_strong(&tex->state, &expected, EXT_IDLE)) {
        return NULL;
    }

    u8 *pixels = tex->texdata;
    tex->texdata = NULL;
    *width = tex->width;
    *height = tex->height;
    atomic_fetch_sub(&inflight, 1);
    return pixels;
}

// Called by the renderer when it needs this texture now. Returns the decoded
// pixels if they are ready (the caller uploads them and frees them with
// extTexFreePixels), otherwise queues the decode and returns NULL.
u8 *extTexLoad(u8 type, u16 id, s32 texnum, u32 *width, u32 *height)
{
    struct ExtTexture *tex = getExtTexture(type, id, texnum);

    if (!tex || tex->texnum < 0) {
        return NULL;
    }

    u8 *pixels = extTexTake(tex, width, height);
    if (pixels) {
        return pixels;
    }

    extTexEnqueueLoad(type, id, texnum);
    return NULL;
}

void extTexFreePixels(u8 *pixels)
{
    if (pixels) {
        stbi_image_free(pixels);
    }
}

u8 *extTexTakeReady(u8 *outType, u16 *outId, s32 *outTexnum, u32 *outWidth, u32 *outHeight)
{
    struct ExtJob job;

    pthread_mutex_lock(&jobMutex);
    while (ringPop(&readyRing, &job)) {
        struct ExtTexture *tex = getExtTexture(job.type, job.id, job.texnum);
        // Skip entries whose pixels a draw already took.
        u8 *pixels = tex ? extTexTake(tex, outWidth, outHeight) : NULL;
        if (pixels) {
            pthread_mutex_unlock(&jobMutex);
            *outType = job.type;
            *outId = job.id;
            *outTexnum = job.texnum;
            return pixels;
        }
    }
    pthread_mutex_unlock(&jobMutex);

    return NULL;
}

s32 extTexInflight(void)
{
    return atomic_load(&inflight);
}

static void *extTexDecodeThreadFunc(void *arg)
{
    for (;;) {
        struct ExtJob job;

        pthread_mutex_lock(&jobMutex);
        while (atomic_load(&decodeThreadsRunning) && decodeRing.count == 0) {
            pthread_cond_wait(&jobCond, &jobMutex);
        }
        if (!atomic_load(&decodeThreadsRunning)) {
            pthread_mutex_unlock(&jobMutex);
            break;
        }
        ringPop(&decodeRing, &job);
        pthread_mutex_unlock(&jobMutex);

        struct ExtTexture *tex = getExtTexture(job.type, job.id, job.texnum);
        char path[FS_MAXPATH];
        u8 *pixels = NULL;
        int w = 0, h = 0, channels = 0;

        if (tex && getTexPath(path, job.type, job.id, job.texnum) == 0) {
            pixels = stbi_load(path, &w, &h, &channels, 4);
        }

        if (!pixels) {
            // The image does not exist or is corrupted: never try it again.
            if (tex) {
                atomic_store(&tex->state, EXT_FAILED);
            }
            atomic_fetch_sub(&inflight, 1);
            continue;
        }

        // Flip the texture to match the N64 rendering order!
        for (int y = 0; y < h / 2; y++) {
            u8 *a = pixels + (size_t)w * 4 * y;
            u8 *b = pixels + (size_t)w * 4 * (h - 1 - y);
            for (int x = 0; x < w * 4; x++) {
                u8 tmp = a[x];
                a[x] = b[x];
                b[x] = tmp;
            }
        }

        tex->texdata = pixels;
        tex->width = w;
        tex->height = h;
        atomic_store(&tex->state, EXT_READY);

        pthread_mutex_lock(&jobMutex);
        ringPush(&readyRing, job.type, job.id, job.texnum);
        pthread_mutex_unlock(&jobMutex);
    }

    return NULL;
}

// ============================================================================
// PREFETCHING
// ============================================================================

static s32 extTexPrefetchEnabled(void)
{
    return packLoaded && videoGetExternalTextures();
}

static void extTexPrefetchOne(u8 type, u16 id, s32 texnum)
{
    if (extTexExists(type, id, texnum) && !gfx_ext_tex_resident(type, id, texnum)) {
        extTexEnqueueLoad(type, id, texnum);
    }
}

void extTexStageBegin(void)
{
    memset(recordedGeneral, 0, sizeof(recordedGeneral));
    memset(recordedModels, 0, sizeof(recordedModels));
}

void extTexPrefetch(u8 type, u16 id, s32 texnum)
{
    if (type == G_TEXTYPE_GENERAL && texnum >= 0 && texnum < MAX_EXT_TEX) {
        recordedGeneral[texnum] = 1;
    }

    if (extTexPrefetchEnabled()) {
        extTexPrefetchOne(type, id, texnum);
    }
}

static void extTexPrefetchModelTextures(s32 fileNum)
{
    struct ModelTextures *modelTex = lookupModel(fileNum);

    if (modelTex == NULL) {
        return;
    }

    for (int i = 0; i < modelTex->numTextures; ++i) {
        extTexPrefetchOne(G_TEXTYPE_MODEL, fileNum, modelTex->textures[i].texnum);
    }
}

void extTexPrefetchModel(s32 fileNum)
{
    if (fileNum < 0 || fileNum >= MAX_MODEL_FILES) {
        return;
    }

    recordedModels[fileNum] = 1;

    if (extTexPrefetchEnabled()) {
        extTexPrefetchModelTextures(fileNum);
    }
}

// Queues the fonts and everything the current stage has loaded so far.
void extTexPrefetchRecorded(void)
{
    if (!extTexPrefetchEnabled()) {
        return;
    }

    for (int i = 0; i < NUM_FONTS; ++i) {
        for (int j = 0; j < NCHARS; ++j) {
            extTexPrefetchOne(G_TEXTYPE_FONT, i, j);
            extTexPrefetchOne(G_TEXTYPE_FONT, i | IDMASK_FONT_OUTLINE, j);
        }
    }

    for (int i = 0; i < MAX_EXT_TEX; ++i) {
        if (recordedGeneral[i]) {
            extTexPrefetchOne(G_TEXTYPE_GENERAL, 0, i);
        }
    }

    for (int i = 0; i < MAX_MODEL_FILES; ++i) {
        if (recordedModels[i]) {
            extTexPrefetchModelTextures(i);
        }
    }
}

// ============================================================================
// FONT & TEXTURE METADATA INITIALIZATION
// ============================================================================

u8 extTexFontID(struct font *font)
{
    if (font == g_FontHandelGothicSm)
        return FONT_HANDELGOTHICSM;
    else if (font == g_FontHandelGothicMd)
        return FONT_HANDELGOTHICMD;
    else if (font == g_FontHandelGothicXs)
        return FONT_HANDELGOTHICXS;
    else if (font == g_FontHandelGothicLg)
        return FONT_HANDELGOTHICLG;
    else if (font == g_FontNumeric)
        return FONT_NUMERIC;

    return 0xff;
}

u8 resolveFontID(const char *fontname)
{
    if (strcmp(fontname, "fonthandelgothicsm") == 0)
        return FONT_HANDELGOTHICSM;
    else if (strcmp(fontname, "fonthandelgothicmd") == 0)
        return FONT_HANDELGOTHICMD;
    else if (strcmp(fontname, "fonthandelgothicxs") == 0)
        return FONT_HANDELGOTHICXS;
    else if (strcmp(fontname, "fonthandelgothiclg") == 0)
        return FONT_HANDELGOTHICLG;
    else if (strcmp(fontname, "fontnumeric") == 0)
        return FONT_NUMERIC;

    return 0xff;
}

static void resetTex(struct ExtTexture *tex)
{
    if (tex->texdata) {
        stbi_image_free(tex->texdata);
    }

    tex->texdata = NULL;
    tex->texnum = -1;
    tex->width = 0;
    tex->height = 0;
    atomic_store(&tex->state, EXT_IDLE);
}

void setTex(struct ExtTexture *texlist, s32 index, s32 texNum, char extension[5])
{
    struct ExtTexture *tex = &texlist[index];
    tex->texdata = NULL;
    tex->texnum = texNum;
    tex->width = 0;
    tex->height = 0;
    atomic_store(&tex->state, EXT_IDLE);
    strcpy(tex->extension, extension);
    numRegistered++;
}

void readModelTextures(const char *path, s16 fileNum, struct ModelTextures *modelTex)
{
    DIR *dr = opendir(path);
    struct dirent *de;

    s32 maxTex = 16;
    modelTex->textures = sysMemAlloc(maxTex * sizeof(struct ExtTexture));
    modelTex->numTextures = 0;
    modelTex->fileNum = fileNum;

    char extension[5] = { 0 };

    /* The model's directory may not exist (no HD textures
     * are provided for this model): this is not an error, we
     * simply return with 0 textures. */
    if (dr == NULL) {
        return;
    }

    while ((de = readdir(dr)) != NULL) {
        const char *name = de->d_name;
        if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) continue;

        s32 texNum;
        if (fileInfo(name, &texNum, extension)) continue;

        // Grow before writing so the new entry always fits
        if (modelTex->numTextures >= maxTex) {
            maxTex *= 2;
            modelTex->textures = sysMemRealloc(modelTex->textures, maxTex * sizeof(struct ExtTexture));
        }

        setTex(modelTex->textures, modelTex->numTextures, texNum, extension);
        modelTex->numTextures++;
    }
    closedir(dr);
}

void readFontTextures(const char *path, u8 fontID)
{
    char outlinesPath[FS_MAXPATH];
    snprintf(outlinesPath, sizeof(outlinesPath), "%s/" FONT_OUTLINES_DIR, path);

    for (int outlines = 0; outlines < 2; ++outlines) {
        DIR *dr = opendir(outlines ? outlinesPath : path);
        struct dirent *de;
        char extension[5] = { 0 };

        /* The "outlines" subdirectory is optional */
        if (dr == NULL) continue;

        while ((de = readdir(dr)) != NULL) {
            const char *name = de->d_name;
            if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) continue;

            s32 texNum;
            if (fileInfo(name, &texNum, extension)) continue;
            if (texNum < 0 || texNum >= NCHARS) continue;

            if (outlines)
                setTex(fontOutlineExtTextures[fontID], texNum, texNum, extension);
            else
                setTex(fontExtTextures[fontID], texNum, texNum, extension);
        }

        closedir(dr);
    }
}

// ============================================================================
// CLEANUP
// ============================================================================

// Frees all decoded pixels and forgets the pack. The decode workers must be
// stopped first.
static void extTexResetTables(void)
{
    for (int i = 0; i < MAX_EXT_TEX; ++i) {
        resetTex(&extTextures[i]);
    }

    for (int i = 0; i < NUM_FONTS; ++i) {
        for (int j = 0; j < NCHARS; ++j) {
            resetTex(&fontExtTextures[i][j]);
            resetTex(&fontOutlineExtTextures[i][j]);
        }
    }

    for (int i = 0; i < numModels; ++i) {
        struct ModelTextures *modelTex = &modelTextures[i];
        for (int j = 0; j < modelTex->numTextures; ++j) {
            resetTex(&modelTex->textures[j]);
        }
        sysMemFree(modelTex->textures);
        modelTex->textures = NULL;
    }

    numModels = 0;
    numRegistered = 0;
    packLoaded = 0;
    atomic_store(&inflight, 0);

    pthread_mutex_lock(&jobMutex);
    decodeRing.head = decodeRing.count = 0;
    readyRing.head = readyRing.count = 0;
    pthread_mutex_unlock(&jobMutex);
}

// Frees decoded pixels that were never uploaded. Keeps the pack's file list.
void extTexFree(void)
{
    extTexAsyncShutdown();

    struct ExtTexture *tex;

    for (int i = 0; i < MAX_EXT_TEX; ++i) {
        tex = &extTextures[i];
        extTexFreePixels(tex->texdata);
        tex->texdata = NULL;
        if (atomic_load(&tex->state) != EXT_FAILED) atomic_store(&tex->state, EXT_IDLE);
    }

    for (int i = 0; i < NUM_FONTS; ++i) {
        for (int j = 0; j < NCHARS; ++j) {
            tex = &fontExtTextures[i][j];
            extTexFreePixels(tex->texdata);
            tex->texdata = NULL;
            if (atomic_load(&tex->state) != EXT_FAILED) atomic_store(&tex->state, EXT_IDLE);

            tex = &fontOutlineExtTextures[i][j];
            extTexFreePixels(tex->texdata);
            tex->texdata = NULL;
            if (atomic_load(&tex->state) != EXT_FAILED) atomic_store(&tex->state, EXT_IDLE);
        }
    }

    for (int i = 0; i < numModels; ++i) {
        struct ModelTextures *modelTex = &modelTextures[i];
        for (int j = 0; j < modelTex->numTextures; ++j) {
            tex = &modelTex->textures[j];
            extTexFreePixels(tex->texdata);
            tex->texdata = NULL;
            if (atomic_load(&tex->state) != EXT_FAILED) atomic_store(&tex->state, EXT_IDLE);
        }
    }

    atomic_store(&inflight, 0);

    pthread_mutex_lock(&jobMutex);
    decodeRing.head = decodeRing.count = 0;
    readyRing.head = readyRing.count = 0;
    pthread_mutex_unlock(&jobMutex);
}

// ============================================================================
// UTILITIES & PATH DISCOVERY
// ============================================================================

// Utility function to check if a filename contains only hexadecimal characters (ignores extensions)
static int is_hex_string(const char *str)
{
    if (!*str) return 0;
    while (*str) {
        char c = *str;
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
            return 0;
        }
        str++;
    }
    return 1;
}

// Recursive searcher that forces the discovery of the true folder containing the textures
static int findTruePackRoot(char *currentPath)
{
    DIR *dr = opendir(currentPath);
    if (!dr) return 0;

    struct dirent *de;
    int is_root = 0;
    char subdirs[16][FS_MAXPATH]; // Allows storing up to 16 subfolders to search
    int subdir_count = 0;

    while ((de = readdir(dr)) != NULL) {
        const char *name = de->d_name;

        // Ignore native folders and hidden OS remnants (e.g., macOS)
        if (name[0] == '.') continue;
        if (strncmp(name, "__MACOSX", 8) == 0) continue;

        char filepath[FS_MAXPATH];
        snprintf(filepath, sizeof(filepath), "%s/%s", currentPath, name);

        struct stat stbuf;
        if (stat(filepath, &stbuf) == -1) continue;

        if (S_ISDIR(stbuf.st_mode)) {
            char s = name[0];
            // If it is a TRUE model folder (P/C/G) validated by the game
            if ((s == 'P' || s == 'C' || s == 'G') && romdataFileGetNumForName(name) >= 0) {
                is_root = 1;
                break;
            }
            // If it is a TRUE font folder validated by the game
            if (s == 'f' && resolveFontID(name) != 0xff) {
                is_root = 1;
                break;
            }

            // It is not a game folder, keep it in memory to dig into it later
            if (subdir_count < 16) {
                strcpy(subdirs[subdir_count++], filepath);
            }
        } else {
            // It is a file: is it a true UI texture without a folder? (e.g., 0A3B.png)
            char *dot = strrchr(name, '.');
            if (dot) {
                char basename[256] = {0};
                int len = dot - name;
                if (len > 0 && len < 255) {
                    memcpy(basename, name, len);
                    if (is_hex_string(basename)) {
                        is_root = 1;
                        break;
                    }
                }
            }
        }
    }
    closedir(dr);

    if (is_root) {
        return 1; // Bingo, we found the root!
    }

    // Nothing found at the root, so automatically search subfolders
    for (int i = 0; i < subdir_count; i++) {
        if (findTruePackRoot(subdirs[i])) {
            strcpy(currentPath, subdirs[i]); // Update the main path with the winning folder!
            return 1;
        }
    }

    return 0; // Dead end
}

// ============================================================================
// SYSTEM INITIALIZATION
// ============================================================================

// Scans the active pack's files. The decode workers must be stopped.
s32 extTexInit(void)
{
    extTexResetTables();

    // 1. Add $S/ to force creation next to the executable
    const char *rootPath = fsFullPath("$S/texture-packs");
    struct stat stRoot;

    if (stat(rootPath, &stRoot) == -1) {
        fsCreateDir(rootPath);
        sysLogPrintf(LOG_NOTE, "ext_tex: Root folder created -> %s", rootPath);
    }

    if (g_ActiveExtTexPack[0] == '\0') {
        return 0;
    }

    // 2. Also add $S/ to load the pack subfolder
    char relPath[FS_MAXPATH];
    snprintf(relPath, sizeof(relPath), "$S/texture-packs/%s", g_ActiveExtTexPack);

    const char *path = fsFullPath(relPath);
    strcpy(extTexPath, path);

    if (findTruePackRoot(extTexPath)) {
        sysLogPrintf(LOG_NOTE, "ext_tex: True texture root found -> %s", extTexPath);
    }

    struct dirent *de;
    DIR *dr = opendir(extTexPath);
    char filepath[FS_MAXPATH];

    if (maxModels == 0) {
        maxModels = 16;
        modelTextures = sysMemAlloc(maxModels * sizeof(struct ModelTextures));
    }

    if (dr != NULL) {
        while ((de = readdir(dr)) != NULL) {
            const char *name = de->d_name;
            if (strcmp(name, ".") == 0 || strcmp(name, "..") == 0) continue;

            struct stat stbuf;
            snprintf(filepath, sizeof(filepath), "%s/%s", extTexPath, de->d_name);
            if (stat(filepath, &stbuf) == -1) continue;

            if (S_ISDIR(stbuf.st_mode)) {
                char s = name[0];
                if (s == 'P' || s == 'C' || s == 'G') {
                    s32 fileNum = romdataFileGetNumForName(name);
                    if (fileNum < 0) continue;

                    if (numModels >= maxModels) {
                        maxModels *= 2;
                        modelTextures = sysMemRealloc(modelTextures, maxModels * sizeof(struct ModelTextures));
                    }

                    readModelTextures(filepath, (s16)fileNum, &modelTextures[numModels++]);
                } else if (s == 'f') {
                    u8 fontID = resolveFontID(name);
                    if (fontID != 0xff) {
                        readFontTextures(filepath, fontID);
                    }
                }
            } else {
                s32 texNum = 0;
                char extension[5] = { 0 };
                if (!fileInfo(name, &texNum, extension) && texNum >= 0 && texNum < MAX_EXT_TEX) {
                    setTex(extTextures, texNum, texNum, extension);
                }
            }
        }
        closedir(dr);
    }

    // Every texture is in at most one ring at a time, plus stale ready entries
    // left behind when a draw takes pixels directly.
    s32 cap = numRegistered * 2 + 64;
    pthread_mutex_lock(&jobMutex);
    ringAlloc(&decodeRing, cap);
    ringAlloc(&readyRing, cap);
    pthread_mutex_unlock(&jobMutex);

    packLoaded = numRegistered > 0;
    sysLogPrintf(LOG_NOTE, "ext_tex: %d replacement textures in pack %s", numRegistered, g_ActiveExtTexPack);

    return 0;
}

void extTexSetPack(const char *newPackName)
{
    pthread_mutex_lock(&packMutex);

    if (newPackName != NULL) {
        strncpy(pendingPackName, newPackName, sizeof(pendingPackName) - 1);
        pendingPackName[sizeof(pendingPackName) - 1] = '\0';
    } else {
        pendingPackName[0] = '\0';
    }

    // The menus read the active name right after requesting a change
    strcpy(g_ActiveExtTexPack, pendingPackName);
    pendingPackChange = 1;

    pthread_mutex_unlock(&packMutex);
}

// Render thread only. Returns 1 if the pack changed, in which case the caller
// must drop every external texture from the GPU cache.
s32 extTexApplyPendingPack(void)
{
    pthread_mutex_lock(&packMutex);

    if (!pendingPackChange) {
        pthread_mutex_unlock(&packMutex);
        return 0;
    }

    strcpy(g_ActiveExtTexPack, pendingPackName);
    pendingPackChange = 0;

    pthread_mutex_unlock(&packMutex);

    extTexAsyncShutdown();
    extTexInit();

    return 1;
}
