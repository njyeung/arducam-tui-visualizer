/*
 * Fast half-block ANSI renderer for depth frames.
 *
 * Build:
 *   gcc -O2 -shared -fPIC -o _render.so _render.c
 */

#include <stdint.h>

static int u8_to_str(uint8_t val, char *buf) {
    if (val >= 100) {
        buf[0] = '0' + val / 100;
        buf[1] = '0' + (val / 10) % 10;
        buf[2] = '0' + val % 10;
        return 3;
    }
    if (val >= 10) {
        buf[0] = '0' + val / 10;
        buf[1] = '0' + val % 10;
        return 2;
    }
    buf[0] = '0' + val;
    return 1;
}

/*
 * Render paired top/bot RGB rows into a buffer of ANSI half-block sequences.
 *
 * top, bot: (n_rows, width, 3) C-contiguous uint8 arrays.
 * pad:      number of leading spaces per line (for centering).
 * out:      pre-allocated output buffer. Caller must ensure sufficient size
 *           (48 bytes per pixel + pad + overhead is safe).
 *
 * Returns number of bytes written.
 */
int render_halfblock(
    const uint8_t *top,
    const uint8_t *bot,
    int n_rows,
    int width,
    int pad,
    char *out)
{
    char *p = out;

    /* Cursor home: \033[H */
    *p++ = '\033'; *p++ = '['; *p++ = 'H';

    for (int r = 0; r < n_rows; r++) {
        const uint8_t *trow = top + r * width * 3;
        const uint8_t *brow = bot + r * width * 3;

        for (int i = 0; i < pad; i++)
            *p++ = ' ';

        for (int c = 0; c < width; c++) {
            int idx = c * 3;

            /* \033[38;2;R;G;B;48;2;R;G;Bm */
            *p++ = '\033'; *p++ = '[';
            *p++ = '3'; *p++ = '8'; *p++ = ';'; *p++ = '2'; *p++ = ';';
            p += u8_to_str(trow[idx],   p); *p++ = ';';
            p += u8_to_str(trow[idx+1], p); *p++ = ';';
            p += u8_to_str(trow[idx+2], p); *p++ = ';';
            *p++ = '4'; *p++ = '8'; *p++ = ';'; *p++ = '2'; *p++ = ';';
            p += u8_to_str(brow[idx],   p); *p++ = ';';
            p += u8_to_str(brow[idx+1], p); *p++ = ';';
            p += u8_to_str(brow[idx+2], p);
            *p++ = 'm';

            /* U+2580 UPPER HALF BLOCK = UTF-8 0xE2 0x96 0x80 */
            *p++ = (char)0xE2;
            *p++ = (char)0x96;
            *p++ = (char)0x80;
        }

        /* Reset: \033[0m + newline */
        *p++ = '\033'; *p++ = '['; *p++ = '0'; *p++ = 'm';
        *p++ = '\n';
    }

    return (int)(p - out);
}
