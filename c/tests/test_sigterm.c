#include <stdio.h>

#ifdef _WIN32

int main(void){
    puts("test_sigterm: skipped (POSIX-only)");
    return 0;
}

#else

#include <pthread.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/select.h>
#include <sys/wait.h>
#include <unistd.h>
#include "../serve_signal.h"

static int fails;

static void check(int cond,const char *what){
    printf("  %s %s\n",cond?"ok  ":"FAIL",what);
    if(!cond) fails++;
}

static int child_exited_zero(pid_t pid){
    for(int i=0;i<200;i++){
        int status=0;
        pid_t got=waitpid(pid,&status,WNOHANG);
        if(got==pid) return WIFEXITED(status)&&WEXITSTATUS(status)==0;
        if(got<0) return 0;
        usleep(10000);
    }
    kill(pid,SIGKILL);
    waitpid(pid,NULL,0);
    return 0;
}

static void *send_sigterm(void *unused){
    (void)unused;
    usleep(50000);
    raise(SIGTERM);
    return NULL;
}

static char *source_text(void){
    FILE *f=fopen("colibri.c","rb");
    if(!f) return NULL;
    if(fseek(f,0,SEEK_END)!=0){ fclose(f); return NULL; }
    long n=ftell(f);
    if(n<0 || fseek(f,0,SEEK_SET)!=0){ fclose(f); return NULL; }
    char *text=malloc((size_t)n+1);
    if(!text){ fclose(f); return NULL; }
    size_t got=fread(text,1,(size_t)n,f);
    fclose(f);
    text[got]=0;
    return got==(size_t)n?text:(free(text),NULL);
}

static int occurrences(const char *text,const char *needle){
    int count=0;
    for(const char *p=text;(p=strstr(p,needle));p+=strlen(needle)) count++;
    return count;
}

int main(void){
    struct sigaction sa;
    intr_install();

    check(sigaction(SIGTERM,NULL,&sa)==0,"SIGTERM disposition is readable");
    check(sa.sa_handler==term_sig,"SIGTERM uses the graceful-stop handler");
    check((sa.sa_flags&SA_RESTART)!=0,"SIGTERM preserves SA_RESTART for model I/O");

    g_intr=0; g_term=0;
    raise(SIGINT);
    check(g_intr==1 && g_term==0,"SIGINT remains a turn-only soft stop");

    int input[2];
    check(pipe(input)==0,"idle-shutdown fixture pipe opens");
    pid_t pid=fork();
    if(pid==0){
        close(input[1]);
        if(dup2(input[0],STDIN_FILENO)<0) _exit(10);
        close(input[0]);
        g_intr=0; g_term=0;
        pthread_t sender;
        if(pthread_create(&sender,NULL,send_sigterm,NULL)!=0) _exit(11);
        fd_set rfds; FD_ZERO(&rfds);
        FD_SET(STDIN_FILENO,&rfds); FD_SET(g_term_pipe_r,&rfds);
        int maxfd=STDIN_FILENO>g_term_pipe_r?STDIN_FILENO:g_term_pipe_r;
        int ready=select(maxfd+1,&rfds,NULL,NULL,NULL);
        pthread_join(sender,NULL);
        _exit(ready>0 && FD_ISSET(g_term_pipe_r,&rfds) &&
              g_intr==1 && g_term==1 ? 0 : 12);
    }
    close(input[0]);
    check(pid>0 && child_exited_zero(pid),
          "worker-thread SIGTERM wakes mux select and exits zero");
    close(input[1]);

    char *source=source_text();
    check(source!=NULL,"engine source is readable by the regression test");
    if(source){
        check(strstr(source,"#include \"serve_signal.h\"")!=NULL,
              "engine uses the tested signal module");
        check(occurrences(source,"if(g_term) break;")>=2,
              "both serve loops have a termination gate");
        check(strstr(source,"FD_SET(g_term_pipe_r,&rfds)")!=NULL,
              "mux select watches the SIGTERM wake pipe");
        check(strstr(source,"fread(raw,1,(size_t)sub.bytes,stdin)")!=NULL,
              "mux payload framing remains on the existing stdio path");
        free(source);
    }

    printf("test_sigterm: %s\n",fails?"FAILED":"ok");
    return fails?1:0;
}

#endif
