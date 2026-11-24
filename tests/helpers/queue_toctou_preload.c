#define _GNU_SOURCE
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <unistd.h>

/*
 * Swap the queue directory for AW_TOCTOU_EVIL immediately before a path-based
 * O_CREAT inside it. A later open of the same pathname then creates the file
 * in the replacement directory. openat() on a directory fd the hook already
 * checked is left alone.
 *
 * bash on this system resolves the queue redirect through libc open()
 * (open@GLIBC_2.2.5). Worker processes are excluded so the drain script does
 * not swap the directory back.
 */

static int swapped = 0;
static int decide = -1;

static int cmdline_has(const char *needle) {
    char buf[4096];
    int fd = syscall(SYS_openat, AT_FDCWD, "/proc/self/cmdline", O_RDONLY);
    if (fd < 0)
        return 0;
    int n = syscall(SYS_read, fd, buf, sizeof(buf) - 1);
    syscall(SYS_close, fd);
    if (n < 0)
        n = 0;
    buf[n] = '\0';
    for (int i = 0; i < n; i++) {
        if (buf[i] == '\0')
            buf[i] = ' ';
    }
    return strstr(buf, needle) != NULL;
}

static int should_swap_process(void) {
    if (decide < 0)
        decide = !cmdline_has("worker");
    return decide;
}

static int path_is_queue_file(const char *path) {
    const char *queue = getenv("AW_TOCTOU_QUEUE");
    if (!queue || !path)
        return 0;
    size_t n = strlen(queue);
    return strncmp(path, queue, n) == 0 && path[n] == '/';
}

static void swap_once(void) {
    if (swapped || !should_swap_process())
        return;
    const char *queue = getenv("AW_TOCTOU_QUEUE");
    const char *evil = getenv("AW_TOCTOU_EVIL");
    const char *held = getenv("AW_TOCTOU_HELD");
    if (!queue || !evil || !held)
        return;
    if (rename(queue, held) != 0)
        return;
    if (rename(evil, queue) != 0) {
        rename(held, queue);
        return;
    }
    swapped = 1;
}

static int real_open(const char *pathname, int flags, mode_t mode, int has_mode) {
    int (*next)(const char *, int, ...) = dlsym(RTLD_NEXT, "open");
    if (has_mode)
        return next(pathname, flags, mode);
    return next(pathname, flags);
}

int open(const char *pathname, int flags, ...) {
    mode_t mode = 0;
    int has_mode = (flags & (O_CREAT | O_TMPFILE)) != 0;
    if (has_mode) {
        va_list ap;
        va_start(ap, flags);
        mode = (mode_t)va_arg(ap, int);
        va_end(ap);
    }
    if ((flags & O_CREAT) && path_is_queue_file(pathname))
        swap_once();
    return real_open(pathname, flags, mode, has_mode);
}

int open64(const char *pathname, int flags, ...) {
    mode_t mode = 0;
    int has_mode = (flags & (O_CREAT | O_TMPFILE)) != 0;
    va_list ap;
    va_start(ap, flags);
    if (has_mode)
        mode = (mode_t)va_arg(ap, int);
    va_end(ap);
    return open(pathname, flags, mode);
}
