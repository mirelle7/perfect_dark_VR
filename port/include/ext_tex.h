#ifndef _IN_EXT_TEX_H
#define _IN_EXT_TEX_H

#include <PR/ultratypes.h>

struct font;

#define MASK_FONT_OUTLINE 0x80
#define IDMASK_FONT_OUTLINE_BIT (MASK_FONT_OUTLINE << 8)

s32 extTexInit();
void extTexFree();
u8 *extTexLoad(u8 type, u16 id, s32 texnum, u32 *width, u32 *height);
void extTexFreePixels(u8 *pixels);
u8 extTexExists(u8 type, u16 id, s32 texnum);
u8 extTexFontID(struct font *font);

// Requests a pack change. Safe to call from any thread: the pack is swapped
// on the render thread at the start of the next frame (extTexApplyPendingPack).
void extTexSetPack(const char *newPackName);
s32 extTexApplyPendingPack(void);

// Prefetching: the game reports the textures and model files it loads so
// their HD replacements are decoded before they are first drawn.
void extTexStageBegin(void);
void extTexPrefetch(u8 type, u16 id, s32 texnum);
void extTexPrefetchModel(s32 fileNum);
void extTexPrefetchRecorded(void);

// Decoded textures waiting for upload. extTexTakeReady transfers ownership of
// the pixels to the caller, who frees them with extTexFreePixels.
s32 extTexInflight(void);
u8 *extTexTakeReady(u8 *outType, u16 *outId, s32 *outTexnum, u32 *outWidth, u32 *outHeight);

void extTexAsyncShutdown();
#endif
