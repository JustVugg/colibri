#ifndef COLI_SERVE_SIGNAL_H
#define COLI_SERVE_SIGNAL_H

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static volatile sig_atomic_t g_intr=0, g_term=0;

#if defined(__APPLE__) || defined(__linux__) || defined(__FreeBSD__)

#include <fcntl.h>
#include <unistd.h>

/*
 * SIGINT remains a turn-only soft stop. SIGTERM requests the same soft stop,
 * then exits the serve loop through its normal usage/KV cleanup path.
 *
 * Keep SA_RESTART for both signals: model reads may run on any engine thread
 * and must not acquire new EINTR behavior during shutdown. A nonblocking
 * self-pipe wakes the mux input select even when another thread gets SIGTERM.
 */
static int g_term_pipe_r=-1;
static volatile sig_atomic_t g_term_pipe_w=-1;

static void intr_sig(int s){ (void)s; g_intr=1; }

static void term_sig(int s){
    int saved_errno=errno;
    (void)s;
    g_intr=1;
    g_term=1;
    if(g_term_pipe_w>=0){
        unsigned char wake=1;
        ssize_t wake_result=write((int)g_term_pipe_w,&wake,1);
        (void)wake_result;
    }
    errno=saved_errno;
}

static int term_pipe_init(void){
    if(g_term_pipe_r>=0) return 0;
    int fds[2];
    if(pipe(fds)!=0) return -1;
    int flags=fcntl(fds[1],F_GETFL,0);
    if(flags<0 || fcntl(fds[1],F_SETFL,flags|O_NONBLOCK)<0 ||
       fcntl(fds[0],F_SETFD,FD_CLOEXEC)<0 ||
       fcntl(fds[1],F_SETFD,FD_CLOEXEC)<0){
        int saved_errno=errno;
        close(fds[0]); close(fds[1]);
        errno=saved_errno;
        return -1;
    }
    g_term_pipe_r=fds[0];
    g_term_pipe_w=fds[1];
    return 0;
}

static void intr_install(void){
    if(term_pipe_init()!=0){ perror("SIGTERM wake pipe"); exit(1); }
    struct sigaction sa; memset(&sa,0,sizeof(sa));
    sa.sa_handler=intr_sig; sigemptyset(&sa.sa_mask);
    sa.sa_flags=SA_RESTART;
    if(sigaction(SIGINT,&sa,NULL)!=0){ perror("sigaction(SIGINT)"); exit(1); }
    sa.sa_handler=term_sig;
    if(sigaction(SIGTERM,&sa,NULL)!=0){ perror("sigaction(SIGTERM)"); exit(1); }
}

static void term_pipe_drain(void){
    unsigned char wake;
    ssize_t wake_result=read(g_term_pipe_r,&wake,1);
    (void)wake_result;
}

#else

/* Windows behavior is intentionally unchanged; systemd/SIGTERM is POSIX-only. */
static void intr_install(void){}

#endif

#endif
