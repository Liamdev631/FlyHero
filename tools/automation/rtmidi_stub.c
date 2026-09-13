/*
 * libRtMidi.so stub for headless YARG/FlyHero.
 *
 * Why this exists: YARG's MIDI input backend (Minis -> RtMidi.Runtime.dll)
 * P/Invokes libRtMidi.so on every input update. On this host the real library
 * either segfaults (rtmidi_get_port_count with no usable /dev/snd/seq) or, if
 * removed, throws a DllNotFoundException every frame that destabilises
 * gameplay. Neither is acceptable for unattended runs.
 *
 * This stub satisfies every entry point the managed code resolves and reports
 * ZERO MIDI ports, so the backend comes up clean and enumerates nothing. MIDI
 * input is not needed for headless automation - the policy injects keyboard
 * input.
 *
 * Build:
 *   gcc -shared -fPIC -O2 -o libRtMidi.so rtmidi_stub.c
 * Install into <game>/YARG_Data/Plugins/libRtMidi.so
 *
 * Covers exactly the symbols found in YARG_Data/Managed/RtMidi.Runtime.dll.
 */

#include <stddef.h>

/* Opaque handles - the managed side only ever stores and passes these back. */
typedef void *RtMidiPtr;
typedef void *RtMidiInPtr;
typedef void *RtMidiOutPtr;
typedef void (*RtMidiCCallback)(double, void *, void *);

/* A non-NULL sentinel: some managed wrappers keep the handle and would throw
 * on a null check, so hand back a stable dummy rather than NULL. */
static int dummy_handle;
#define DUMMY ((void *) &dummy_handle)

static const char STUB_NAME[] = "stub";
static const char STUB_DISPLAY[] = "No MIDI (headless stub)";

/* ---- API enumeration ---------------------------------------------------- */

unsigned int rtmidi_num_compiled_apis(void)
{
    return 0;
}

int rtmidi_compiled_apis(int api, unsigned int *count)
{
    (void) api;
    if (count) *count = 0;
    return 0;
}

int rtmidi_get_compiled_api(int api, unsigned int *count)
{
    (void) api;
    if (count) *count = 0;
    return 0;
}

int rtmidi_compiled_api_by_name(const char *name)
{
    (void) name;
    return -1; /* RtMidi convention: -1 == not found */
}

const char *rtmidi_api_name(int api)
{
    (void) api;
    return STUB_NAME;
}

const char *rtmidi_api_display_name(int api)
{
    (void) api;
    return STUB_DISPLAY;
}

/* ---- Ports: the important ones ------------------------------------------ */

/* Returning 0 ports is what keeps the backend from ever opening a device. */
int rtmidi_get_port_count(RtMidiPtr device, unsigned int *count)
{
    (void) device;
    if (count) *count = 0;
    return 0;
}

int rtmidi_get_port_name(RtMidiPtr device, unsigned int port, char *buf,
                         unsigned int *size)
{
    (void) device;
    (void) port;
    if (buf && size && *size > 0) buf[0] = '\0';
    if (size) *size = 0;
    return 0;
}

int rtmidi_open_port(RtMidiPtr device, unsigned int port, const char *name)
{
    (void) device;
    (void) port;
    (void) name;
    return 0;
}

int rtmidi_open_virtual_port(RtMidiPtr device, const char *name)
{
    (void) device;
    (void) name;
    return 0;
}

int rtmidi_close_port(RtMidiPtr device)
{
    (void) device;
    return 0;
}

/* ---- Input -------------------------------------------------------------- */

RtMidiInPtr rtmidi_in_create_default(void)
{
    return DUMMY;
}

RtMidiInPtr rtmidi_in_create(int api, const char *client_name,
                             unsigned int queue_size_limit)
{
    (void) api;
    (void) client_name;
    (void) queue_size_limit;
    return DUMMY;
}

void rtmidi_in_free(RtMidiInPtr device)
{
    (void) device;
}

int rtmidi_in_get_current_api(RtMidiPtr device)
{
    (void) device;
    return 0;
}

int rtmidi_in_ignore_types(RtMidiInPtr device, int midi_sysex, int midi_time,
                           int midi_sense)
{
    (void) device;
    (void) midi_sysex;
    (void) midi_time;
    (void) midi_sense;
    return 0;
}

int rtmidi_in_set_callback(RtMidiInPtr device, RtMidiCCallback callback,
                           void *user_data)
{
    (void) device;
    (void) callback;
    (void) user_data;
    return 0;
}

int rtmidi_in_cancel_callback(RtMidiInPtr device)
{
    (void) device;
    return 0;
}

/* No queued messages, ever - a zero size means "nothing pending". */
int rtmidi_in_get_message(RtMidiInPtr device, unsigned char *message,
                          unsigned int *size)
{
    (void) device;
    (void) message;
    if (size) *size = 0;
    return 0;
}

/* ---- Output ------------------------------------------------------------- */

RtMidiOutPtr rtmidi_out_create_default(void)
{
    return DUMMY;
}

RtMidiOutPtr rtmidi_out_create(int api, const char *client_name)
{
    (void) api;
    (void) client_name;
    return DUMMY;
}

void rtmidi_out_free(RtMidiOutPtr device)
{
    (void) device;
}

int rtmidi_out_get_current_api(RtMidiPtr device)
{
    (void) device;
    return 0;
}

int rtmidi_out_send_message(RtMidiOutPtr device, const unsigned char *message,
                            int length)
{
    (void) device;
    (void) message;
    (void) length;
    return 0;
}
