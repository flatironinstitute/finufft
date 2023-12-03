#ifndef _SCTL_STACKTRACE_H_
#define _SCTL_STACKTRACE_H_

#include <sctl/common.hpp>

#include <unistd.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <cxxabi.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace SCTL_NAMESPACE {

inline void print_stacktrace(FILE* out = stderr, int skip = 1) {
  // Get addresses
  void* addrlist[256];
  int addrlen = backtrace(addrlist, 255);
  for (int i = 0; i < addrlen; i++) addrlist[i] = (char*)addrlist[i] - 1;

  // Get symbols
  char** symbollist = backtrace_symbols(addrlist, addrlen);

  // Get filename
  char fname[10240];
#ifdef __APPLE__
  uint32_t size = sizeof(fname);
  _NSGetExecutablePath(fname, &size);
#elif __linux__
  ssize_t fname_len = ::readlink("/proc/self/exe", fname, sizeof(fname) - 1);
  fname[fname_len] = '\0';
#endif

  // Print
  for (int i = skip; i < addrlen; i++) {
    // Get command
    char cmd[10240+256+43];
#ifdef __APPLE__
    sprintf(cmd, "atos -o %s %p 2> /dev/null", fname, addrlist[i]); // on mac
#elif __linux__
    sprintf(cmd, "addr2line -f -C -i -e  %s  %p 2> /dev/null", fname, addrlist[i]);
#endif

    // Execute command
    FILE* pipe = popen(cmd, "r");
    if (!pipe) continue;
    char buffer0[10240];
    char buffer1[10240];
    fgets(buffer0, sizeof(buffer0) - 1, pipe);
    fgets(buffer1, sizeof(buffer1) - 1, pipe);
    for (int j = 0; j < (int)sizeof(buffer0) - 1; j++) {
      if (buffer0[j] == '\n') buffer0[j] = ' ';
    }
    for (int j = 0; j < (int)sizeof(buffer1) - 1; j++) {
      if (buffer1[j] == '\n') buffer1[j] = ' ';
    }
    pclose(pipe);

    // Print output
    if (buffer0[0] != '?' && buffer0[0] != '\0') {
      fprintf(out, "[%d] %s: %s\n", i - skip, buffer1, buffer0);
    } else {
      fprintf(out, "[%d] %p: %s\n", i - skip, addrlist[i], symbollist[i]);
    }
  }
  fprintf(stderr, "\n");
}

inline void abortHandler(int signum, siginfo_t* si, void* unused) {
  static bool first_time = true;
  SCTL_UNUSED(unused);
  SCTL_UNUSED(si);

#pragma omp critical(SCTL_STACK_TRACE)
  if (first_time) {
    first_time = false;
    const char* name = nullptr;
    switch (signum) {
      case SIGABRT:
        name = "SIGABRT";
        break;
      case SIGSEGV:
        name = "SIGSEGV";
        break;
      case SIGBUS:
        name = "SIGBUS";
        break;
      case SIGILL:
        name = "SIGILL";
        break;
      case SIGFPE:
        name = "SIGFPE";
        break;
    }

    if (name)
      fprintf(stderr, "\nCaught signal %d (%s)\n", signum, name);
    else
      fprintf(stderr, "\nCaught signal %d\n", signum);

    print_stacktrace(stderr, 2);
  }
  exit(signum);
}

inline int SetSigHandler() {
  struct sigaction sa;
  sa.sa_flags = SA_RESTART | SA_SIGINFO;
  sa.sa_sigaction = abortHandler;
  sigemptyset(&sa.sa_mask);

  sigaction(SIGABRT, &sa, nullptr);
  sigaction(SIGSEGV, &sa, nullptr);
  sigaction(SIGBUS, &sa, nullptr);
  sigaction(SIGILL, &sa, nullptr);
  sigaction(SIGFPE, &sa, nullptr);
  sigaction(SIGPIPE, &sa, nullptr);

  return 0;
}

}  // end namespace

#endif  // _SCTL_STACKTRACE_H_
